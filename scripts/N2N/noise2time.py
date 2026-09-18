"""Reproducible Noise2Time experiment. See REPRODUCING.md for assumptions.

No work is performed on import. Scientific outputs are lossless NumPy arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import sys
import tempfile
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import scipy
from scipy.signal import find_peaks
import torch
from torch import nn
from tqdm import tqdm

import warnings


@dataclass
class Config:
    seed: int = 2026
    history: int = 9
    base_channels: int = 32
    convlstm: bool = True
    block_size: int = 32
    blocks: int = 32
    objective: str = "article"  # Eq. 19; alternatives: l1, l2
    epochs: int = 250
    samples_per_epoch: int = 8000
    batch_size: int = 2
    validation_samples: int = 35  # Not specified by the article; explicit choice.
    learning_rate: float = 5e-5
    weight_decay: float = 0.01  # AdamW default in the supplied scripts.
    patience: int = 10
    validation_records: tuple = ()  # Empty => legacy fixed sequence split.
    input_mode: str = "patched"  # Or history_only: predict t from t-9,...,t-1.
    split_mode: str = "mixed"  # mixed, temporal, or record
    brightness_correction: bool = True
    validation_fraction: float = .5

    def validate(self):
        if self.input_mode not in ("patched", "history_only"):
            raise ValueError("input_mode must be patched or history_only")
        if self.split_mode not in ("mixed", "temporal", "record"):
            raise ValueError("split_mode must be mixed, temporal or record")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.split_mode == "record" and not self.validation_records:
            raise ValueError("record split requires validation_records")
        for key in ("history", "base_channels", "block_size", "blocks", "epochs",
                    "samples_per_epoch", "batch_size", "validation_samples", "patience"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 (GroupNorm)")
        if self.objective not in ("article", "l1", "l2"):
            raise ValueError("objective must be article, l1 or l2")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.objective == "article" and self.block_size < 3:
            raise ValueError("Use l1/l2 for tiny-block ablations; Hessian needs 3 pixels")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_frames(path):
    """Return normalized float32 T,H,W frames and video/container FPS if available."""
    path = Path(path)
    fps = None
    if path.suffix.lower() == ".npy":
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot read {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        buffer = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        finally:
            cap.release()
        if not buffer:
            raise ValueError(f"No frames in {path}")
        frames = np.stack(buffer)
    if frames.ndim != 3 or min(frames.shape) < 1:
        raise ValueError(f"Expected T,H,W: {path}")
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    elif np.issubdtype(frames.dtype, np.floating):
        frames = np.asarray(frames, dtype=np.float32)
    else:
        raise ValueError("Use uint8 or floating-point arrays in [0,1]; no automatic rescaling")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError(f"Input intensities must be finite and in [0,1]: {path}")
    return frames, fps


def circle_mask(height, width, cx, cy, radius):
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
    return mask.astype(bool)


def phases_from_peaks(count, peaks):
    """Legacy discrete phase: elapsed frames since peak, not fractional phase."""
    phase = np.full(count, -1, dtype=np.int64)
    for start, end in zip(peaks, list(peaks[1:]) + [count]):
        phase[start:end] = np.arange(end - start)
    return phase


def load_sibling(name):
    """Import a local workflow module, including when this script is loaded by tests."""
    import importlib.util
    module_name = "noise2time_" + name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(name+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def prepare(args):
    """Prepare one file, or all supported files directly inside an input folder."""
    if Path(args.input).is_dir():
        if any(p.is_dir() and list(p.glob("*.h5")) for p in Path(args.input).iterdir()):
            return load_sibling("dataset_workflow").prepare_dataset(args, sys.modules[__name__])
        if getattr(args,"avi",False) or getattr(args,"measures",None):
            raise ValueError("--avi and --measures require a dataset of HDF5 measurement folders")
        return prepare_folder(args)
    if getattr(args,"avi",False) or getattr(args,"measures",None):
        raise ValueError("--avi and --measures require a dataset of HDF5 measurement folders")
    return prepare_one(args)


def prepare_folder(args):
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise ValueError("Use separate input and prepared-output folders")
    pattern = getattr(args, "pattern", "*")
    if "/" in pattern or "\\" in pattern or "**" in pattern:
        raise ValueError("--pattern must select filenames in the input folder, without recursion")
    paths = sorted((p for p in source.glob(pattern)
                    if p.is_file() and p.suffix.lower() in (".avi", ".npy")),
                   key=lambda p:p.name.casefold())
    if not paths:
        raise ValueError(f"No AVI/NPY files matched in {source}")
    names = [p.stem.casefold() for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Multiple input files have the same stem; use --pattern '*.avi' or '*.npy'")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, path in enumerate(paths, 1):
        destination = output / path.stem
        row = dict(source=str(path), output=str(destination))
        print(f"[{number}/{len(paths)}] Preparing {path.name}", flush=True)
        try:
            if destination.exists():
                if getattr(args, "skip_existing", False):
                    results.append(dict(row, status="skipped", reason="Existing output not verified"))
                    print("  Skipped existing output (not verified)", flush=True)
                    continue
                raise FileExistsError(f"Output already exists: {destination}; use --skip-existing to leave it untouched")
            # Process one video at a time and publish only complete recordings.
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
                temporary_path = Path(temporary).resolve()
                if temporary_path.parent != output or destination.resolve().parent != output:
                    raise ValueError("Temporary/output path escaped prepared-output folder")
                staging = temporary_path / "record"
                options = argparse.Namespace(**vars(args))
                options.input, options.output = str(path), str(staging)
                prepare_one(options)
                staging.rename(destination)
            results.append(dict(row, status="prepared"))
        except Exception as exc:
            results.append(dict(row, status="error", error=f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR: {exc}", flush=True)
    counts = {key:sum(row["status"] == key for row in results) for key in ("prepared", "skipped", "error")}
    fd, report_path = tempfile.mkstemp(prefix="preparation_", suffix=".json", dir=output)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                       input=str(source), counts=counts, results=results), stream, indent=2)
    print(f"Preparation summary: {counts}\nReport: {report_path}", flush=True)
    return 1 if counts["error"] else 0


def prepare_one(args):
    frames, container_fps = load_frames(args.input)
    fps = args.fps if args.fps is not None else container_fps
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Supply --fps with the acquisition frame rate")
    if args.smooth_window < 1 or args.peak_distance < 1 or args.prominence <= 0 or args.radius < 1:
        raise ValueError("Invalid preprocessing parameters")
    roi = circle_mask(*frames.shape[1:], args.cx, args.cy, args.radius)
    if not roi.any():
        raise ValueError("Circular ROI is empty")
    frames = frames * roi
    # Article Eq. 2 uses all image pixels. Optional external vessel mask reproduces
    # the legacy preprocessing's different brightness definition.
    if args.brightness_mask:
        bm = np.load(args.brightness_mask, allow_pickle=False).astype(bool)
        if bm.shape != roi.shape or not (bm & roi).any():
            raise ValueError("Invalid brightness mask")
        raw = frames[:, bm & roi].mean(axis=1)
    else:
        raw = frames.mean(axis=(1, 2))
    win = args.smooth_window
    smooth = np.convolve(np.pad(raw, (win // 2, win - 1 - win // 2), mode="edge"),
                         np.ones(win) / win, mode="valid")
    artery_path = getattr(args, "artery_mask", None)
    method = getattr(args, "peak_method", "auto")
    method = ("arterial" if artery_path else "legacy") if method == "auto" else method
    peak_diagnostics = None
    if method == "arterial":
        if not artery_path:
            raise ValueError("Arterial peak detection requires --artery-mask")
        import importlib.util
        spec = importlib.util.spec_from_file_location("arterial_peaks", Path(__file__).with_name("arterial_peaks.py"))
        detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detector)
        # Strict geometry for a scientific signal; unlike visualization masks,
        # resizing without registration can select different vessels.
        if Path(artery_path).suffix.lower() == ".npy":
            mask_values = np.load(artery_path, allow_pickle=False)
        else:
            mask_values = cv2.imdecode(np.fromfile(artery_path, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask_values is None or mask_values.shape[:2] != roi.shape:
            raise ValueError("Artery mask must match the prepared video geometry; no automatic resizing")
        artery_mask = load_evaluation_mask(artery_path, roi.shape) & roi
        if not artery_mask.any():
            raise ValueError("Artery mask has no pixels inside the ROI")
        arterial_signal = np.array([frame[artery_mask].mean(dtype=np.float64) for frame in frames])
        peak_diagnostics = detector.detect_arterial_peaks(arterial_signal, float(fps),
            min_hz=getattr(args, "peak_min_hz", .5), max_hz=getattr(args, "peak_max_hz", 2.5))
        peaks = peak_diagnostics["peaks"]
        for message in peak_diagnostics["warnings"]:
            warnings.warn(message)
    else:
        amplitude = np.percentile(smooth, 95) - np.percentile(smooth, 5)
        peaks, _ = find_peaks(smooth, distance=args.peak_distance,
                             prominence=args.prominence * amplitude)
        if amplitude <= 0:
            raise ValueError("Brightness signal has no usable variation")
    if len(peaks) < 2:
        raise ValueError("Need at least two reliable detected peaks for donor pairing")
    first = int(peaks[0])
    phase = phases_from_peaks(len(frames) - first, peaks - first)
    if any(size % 2 for size in frames.shape[1:]):
        raise ValueError("AVI previews require even image dimensions to avoid codec cropping")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "frames.npy", frames[first:].astype(np.float32))
    np.save(output / "roi.npy", roi)
    np.save(output / "brightness.npy", smooth[first:].astype(np.float32))
    np.save(output / "brightness_raw.npy", raw[first:].astype(np.float32))
    np.save(output / "phase.npy", phase)
    if peak_diagnostics is not None:
        np.savez_compressed(output/"arterial_peak_diagnostics.npz", raw=arterial_signal,
                           **{k:v for k,v in peak_diagnostics.items() if isinstance(v,np.ndarray)})
        peak_summary = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in peak_diagnostics.items()
                        if k not in ("smoothed","detection_signal","repaired","artifact_mask","acf")}
        peak_summary.update(method="arterial", mask_sha256=sha256(artery_path),
                            min_hz=getattr(args,"peak_min_hz",.5),max_hz=getattr(args,"peak_max_hz",2.5),
                            frames=len(frames),fps=float(fps),indices="original input frames, before trimming",
                            artifact_frames=np.flatnonzero(peak_diagnostics["artifact_mask"]).tolist(),
                            detector_sha256=sha256(Path(__file__).with_name("arterial_peaks.py")))
        write_json(output/"arterial_peaks.json",peak_summary)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure=Figure(figsize=(12,4),layout="constrained");FigureCanvasAgg(figure)
        ax=figure.subplots();times=np.arange(len(frames))/fps
        ax.plot(times,arterial_signal,color=".65",label="Raw arterial mean")
        ax.plot(times,peak_diagnostics["smoothed"],color="#147c91",label="Detection smoothing")
        ax.scatter(times[peaks],peak_diagnostics["smoothed"][peaks],color="red",marker="v",label="Selected maxima")
        ax.set(xlabel="Time from original input start (s)",ylabel="Arterial mean intensity",
               title="Arterial peaks — inspect warnings in arterial_peaks.json")
        ax.legend();figure.savefig(output/"arterial_peaks.png",dpi=130);figure.clear()
    save_preparation_previews(output, frames[first:], raw[first:], smooth[first:],
                              phase, peaks-first, first, float(fps), Path(args.input).stem)
    metadata = dict(source=str(Path(args.input).resolve()), source_sha256=sha256(args.input),
                    first_original_frame=first, original_frame_count=len(frames), fps=float(fps),
                    peaks=(peaks - first).tolist(), phase_definition="frames_since_peak",
                    peak_method=method,
                    brightness_definition="vessel_mask" if args.brightness_mask else "whole_masked_frame",
                    brightness_mask_sha256=sha256(args.brightness_mask) if args.brightness_mask else None,
                    smooth_window=win, peak_distance=args.peak_distance, prominence_ratio=args.prominence,
                    circle=dict(cx=args.cx, cy=args.cy, radius=args.radius),
                    normalization="uint8 / 255 or supplied [0,1] floats; no contrast enhancement",
                    preview=dict(video="prepared.avi", brightness_plot="brightness.png",
                                 brightness_table="brightness.csv",
                                 video_conversion="MJPG; clip [0,1], round to uint8; no contrast normalization",
                                 table_indices="zero-based; times in seconds"))
    write_json(output / "metadata.json", metadata)
    print(f"Prepared {len(frames)-first} frames; peaks={len(peaks)}; original offset={first}")


def save_preparation_previews(output, frames, raw, smooth, phase, peaks, first, fps, title):
    """Viewable, aligned artifacts; NPY remains the scientific training input."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(str(output/"prepared.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (width,height), True)
    try:
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize prepared MJPG AVI writer")
        for frame in tqdm(frames, desc="Saving prepared AVI", unit="frame",
                          file=sys.stdout, dynamic_ncols=True, mininterval=1.0):
            view = np.rint(np.clip(frame,0,1)*255).astype(np.uint8)
            writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    indices = np.arange(len(frames))
    is_peak = np.zeros(len(frames), dtype=np.uint8)
    is_peak[peaks] = 1
    times = indices/fps
    table = np.column_stack((indices, indices+first, times, (indices+first)/fps,
                             raw, smooth, phase, is_peak))
    np.savetxt(output/"brightness.csv", table, delimiter=",", comments="",
               header="frame_index,original_frame_index,time_seconds,original_time_seconds,raw_brightness,smooth_brightness,phase,is_peak",
               fmt=["%d","%d","%.9g","%.9g","%.9g","%.9g","%d","%d"])
    figure = Figure(figsize=(12,6), constrained_layout=True)
    FigureCanvasAgg(figure)  # Headless: no GUI windows or global backend changes.
    curve, phase_axis = figure.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[3,1]})
    curve.plot(times,raw,label="Raw brightness",color="#8597a6",linewidth=1)
    curve.plot(times,smooth,label="Smoothed brightness",color="#126782",linewidth=1.6)
    curve.scatter(times[peaks],smooth[peaks],label="Detected peaks",color="#cf5735",s=28,zorder=3)
    for peak in peaks:
        curve.axvline(times[peak],color="#cf5735",alpha=.18,linewidth=.8)
    curve.set_ylabel("Mean intensity [0,1]")
    curve.set_title(f"{title}\nPrepared frames; original starting frame {first} (zero-based)")
    curve.legend(loc="best")
    curve.grid(alpha=.2)
    phase_axis.step(times,phase,where="post",color="#126782",linewidth=1)
    phase_axis.set_ylabel("Phase\n(frames since peak)")
    phase_axis.set_xlabel("Time since prepared video start (s)")
    phase_axis.grid(alpha=.2)
    figure.savefig(output/"brightness.png",dpi=160)
    figure.clear()


class Record:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.name = self.path.name
        self.frames = np.load(self.path / "frames.npy", mmap_mode="r", allow_pickle=False)
        self.roi = np.load(self.path / "roi.npy", allow_pickle=False).astype(bool)
        self.brightness = np.load(self.path / "brightness.npy", allow_pickle=False)
        self.phase = np.load(self.path / "phase.npy", allow_pickle=False)
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        if self.frames.ndim != 3 or self.roi.shape != self.frames.shape[1:]:
            raise ValueError(f"Invalid frame/ROI dimensions: {self.path}")
        if self.frames.dtype != np.float32 or any(s % 32 for s in self.frames.shape[1:]):
            raise ValueError("Prepared frames must be float32 with H,W divisible by 32")
        if self.brightness.shape != (len(self.frames),) or self.phase.shape != (len(self.frames),):
            raise ValueError("Frame/brightness/phase lengths differ")
        dataset_record = self.metadata.get("schema") == "noise2time.dataset.v1"
        if (not np.isfinite(self.frames).all() or self.frames.min() < 0 or self.frames.max() > 1
                or not np.isfinite(self.brightness).all() or np.any(self.brightness < 0)
                or not np.issubdtype(self.phase.dtype, np.integer) or np.any(self.phase < (-1 if dataset_record else 0))):
            raise ValueError("Invalid prepared intensities, brightness or phase")
        self.valid = np.load(self.path/"valid_frames.npy", allow_pickle=False).astype(bool) if dataset_record else np.ones(len(self.frames),bool)
        if self.valid.shape != (len(self.frames),):
            raise ValueError("Invalid valid_frames dimensions")
        self.donors = {int(p): np.flatnonzero((self.phase == p) & self.valid & (self.phase >= 0) & (self.brightness > 1e-8))
                       for p in np.unique(self.phase)}

    def eligible(self, history):
        return [t for t in range(history, len(self.frames))
                if self.valid[t] and self.phase[t] >= 0 and self.brightness[t] > 1e-8 and np.any(self.donors[int(self.phase[t])] != t)]


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                    nn.GroupNorm(8, outgoing), nn.SiLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                    nn.GroupNorm(8, outgoing), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class ConvLSTM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)

    def forward(self, x, state):
        h, c = (torch.zeros_like(x), torch.zeros_like(x)) if state is None else state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c)


class Noise2Time(nn.Module):
    """Architecture of the supplied ConvLSTM scripts; 512 -> 16 spatial bottleneck."""
    def __init__(self, base_channels=32, convlstm=True):
        super().__init__()
        widths = [base_channels * s for s in (1, 2, 4, 8, 16, 16)]
        self.encoders = nn.ModuleList([ConvBlock(1, widths[0])] +
                                      [ConvBlock(c, c) for c in widths[1:]])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths[:-1], widths[1:])])
        self.bottleneck = ConvBlock(widths[-1], widths[-1])
        self.memory = ConvLSTM(widths[-1]) if convlstm else None
        self.decoders = nn.ModuleList([ConvBlock(2*c, c) for c in widths])
        self.up = nn.ModuleList([nn.ConvTranspose2d(b, a, 2, stride=2)
                                for a, b in zip(widths[:-1], widths[1:])])
        self.final = nn.Conv2d(widths[0], 1, 3, padding=1)

    def forward(self, sequence):
        state = None  # Reset for each independent history window.
        # Historical decoder outputs are unused; avoid computing them. Encoder and
        # recurrent gradients still propagate through the entire history.
        indices = range(sequence.shape[1]) if self.memory is not None else [sequence.shape[1]-1]
        for t in indices:
            current = sequence[:, t:t+1]
            x = current
            skips = []
            for level, encoder in enumerate(self.encoders):
                x = encoder(x)
                skips.append(x)
                if level < len(self.down):
                    x = self.down[level](x)
            x = self.bottleneck(x)
            if self.memory is not None:
                x, state = self.memory(x, state)
        for level in range(5, -1, -1):
            x = self.decoders[level](torch.cat((x, skips[level]), dim=1))
            if level:
                x = self.up[level-1](x)
        return current - self.final(x)


def replacement(record, t, cfg, rng, stage=None):
    if cfg.split_mode == "temporal":
        if stage not in ("train", "valid"):
            raise ValueError("Temporal replacement requires an explicit train/valid stage")
        record = record.stage_views[stage]
    sequence = np.array(record.frames[t-cfg.history:t+1], copy=True)
    target = sequence[-1].copy()
    mask = np.zeros(record.roi.shape, bool)
    occupied = np.zeros_like(mask)
    donors = record.donors[int(record.phase[t])]
    donors = donors[donors != t]
    if cfg.split_mode == "temporal":
        if stage not in ("train", "valid"):
            raise ValueError("Temporal replacement requires an explicit train/valid stage")
            # Validation targets use the training partition's donor pool.
            donor_stage = "train" if stage == "valid" else stage
            lo, hi = record.split_ranges[donor_stage]
            target_lo, target_hi = record.split_ranges[stage]
            donors = donors[(donors >= lo) & (donors < hi)]
            if not target_lo <= t-cfg.history <= t < target_hi:
                raise ValueError("Target history crosses temporal partition")
    if len(donors) == 0 or record.brightness[t] <= 1e-8:
        raise ValueError("No eligible donor for target")
    h, w = mask.shape
    size = cfg.block_size
    if size > min(h, w):
        raise ValueError("Block exceeds frame size")
    accepted = 0
    for _ in range(cfg.blocks * 100):
        y, x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
        region = np.s_[y:y+size, x:x+size]
        if not record.roi[region].all() or occupied[region].any():
            continue
        donor = int(rng.choice(donors))
        ratio = float(record.brightness[t] / record.brightness[donor]) if cfg.brightness_correction else 1.
        sequence[-1][region] = np.clip(record.frames[donor][region] * ratio, 0, 1)
        mask[region] = occupied[region] = True
        accepted += 1
        if accepted == cfg.blocks:
            break
    if accepted != cfg.blocks:
        raise ValueError(f"Placed {accepted}/{cfg.blocks} blocks; reduce blocks/size or check ROI")
    # Keep the same loss locations and sampling as the baseline. The target frame
    # (including its replaced patches) is entirely absent in history-only mode.
    if cfg.input_mode == "history_only":
        sequence = sequence[:-1]
    return sequence, target[None], mask[None].astype(np.float32)


def loss_terms(prediction, target, mask, objective="article"):
    """Equations 6, 14, 18, 19, with every derivative stencil fully masked."""
    dims = (1, 2, 3)
    count = mask.sum(dims)
    if torch.any(count == 0):
        raise ValueError("Empty reconstruction mask")
    error = prediction - target
    reconstruction = ((error.square() if objective == "l2" else error.abs()) * mask).sum(dims) / count
    zero = torch.zeros_like(reconstruction)
    gradient, hessian = zero, zero
    if objective == "article":
        mx = mask[..., 1:] * mask[..., :-1]
        my = mask[..., 1:, :] * mask[..., :-1, :]
        gx = error[..., 1:] - error[..., :-1]
        gy = error[..., 1:, :] - error[..., :-1, :]
        gradient = ((gx.abs()*mx).sum(dims) + (gy.abs()*my).sum(dims)) / (mx.sum(dims)+my.sum(dims)+1e-8)
        mxx = mask[..., 2:]*mask[..., 1:-1]*mask[..., :-2]
        myy = mask[..., 2:, :]*mask[..., 1:-1, :]*mask[..., :-2, :]
        mxy = mask[..., 1:, 1:]*mask[..., :-1, 1:]*mask[..., 1:, :-1]*mask[..., :-1, :-1]
        xx = error[..., 2:] - 2*error[..., 1:-1] + error[..., :-2]
        yy = error[..., 2:, :] - 2*error[..., 1:-1, :] + error[..., :-2, :]
        xy = error[..., 1:, 1:] - error[..., :-1, 1:] - error[..., 1:, :-1] + error[..., :-1, :-1]
        hessian = ((xx.abs()*mxx).sum(dims)+(yy.abs()*myy).sum(dims)+2*(xy.abs()*mxy).sum(dims)) / (
            mxx.sum(dims)+myy.sum(dims)+2*mxy.sum(dims)+1e-8)
    total = reconstruction + .10*gradient + .05*hessian
    return dict(total=total, reconstruction=reconstruction, gradient=gradient, hessian=hessian)


def split_samples(records, cfg):
    if cfg.split_mode == "temporal":
        if cfg.validation_records:
            raise ValueError("Temporal and video-disjoint validation cannot be combined")
        return load_sibling("benchmark_splits").temporal_split(records, cfg, sys.modules[__name__])
    rng = np.random.default_rng(cfg.seed)
    pools = [[(i, t) for t in rec.eligible(cfg.history)] for i, rec in enumerate(records)]
    if any(not pool for pool in pools):
        raise ValueError("Each record needs an eligible target with another same-phase donor")
    if cfg.validation_records:
        unknown = set(cfg.validation_records) - {r.name for r in records}
        if unknown:
            raise ValueError(f"Unknown validation records: {unknown}")
        train = [s for i,pool in enumerate(pools) if records[i].name not in cfg.validation_records for s in pool]
        candidates = [s for i,pool in enumerate(pools) if records[i].name in cfg.validation_records for s in pool]
    else:
        # Reserve one sample per recording for training, as in the existing scripts.
        reserved = [int(rng.integers(len(pool))) for pool in pools]
        train = [pool[j] for pool,j in zip(pools,reserved)]
        candidates = [s for pool,j in zip(pools,reserved) for k,s in enumerate(pool) if k != j]
    if not train or len(candidates) < cfg.validation_samples:
        raise ValueError("Not enough independent pools/samples for the requested split")
    selected = set(rng.choice(len(candidates), cfg.validation_samples, replace=False).tolist())
    valid = [s for i,s in enumerate(candidates) if i in selected]
    if not cfg.validation_records:
        train += [s for i,s in enumerate(candidates) if i not in selected]
    return train, valid


def resolve_training_records(paths):
    """Expand a prepared-record parent folder into its direct child records."""
    resolved = []
    for path in map(Path, paths):
        if (path / "frames.npy").is_file():
            resolved.append(path)
        elif path.is_dir():
            children = sorted(
                (child for child in path.iterdir()
                 if child.is_dir() and (child / "frames.npy").is_file()),
                key=lambda child: child.name.casefold(),
            )
            if not children:
                raise ValueError(f"No prepared records found in {path}")
            resolved.extend(children)
        else:
            raise ValueError(f"Not a prepared record or record folder: {path}")
    paths_by_identity = [path.resolve() for path in resolved]
    if len(set(paths_by_identity)) != len(paths_by_identity):
        raise ValueError("The same prepared record was supplied more than once")
    return resolved


def select_train_samples_for_epoch(training, samples_per_epoch, rng):
    """Choose at least one target per record, then fill from the pooled split."""
    by_record = {}
    for index, (record_index, _) in enumerate(training):
        by_record.setdefault(record_index, []).append(index)
    if len(by_record) > samples_per_epoch:
        raise ValueError(
            f"{len(by_record)} training records exceed the "
            f"{samples_per_epoch} samples-per-epoch budget"
        )
    mandatory = [int(rng.choice(indices)) for _, indices in sorted(by_record.items())]
    mandatory_set = set(mandatory)
    remaining_pool = [i for i in range(len(training)) if i not in mandatory_set]
    remaining_count = samples_per_epoch - len(mandatory)
    if remaining_count:
        if not remaining_pool:
            remaining_pool = list(range(len(training)))
        extra = rng.choice(
            remaining_pool, size=remaining_count,
            replace=remaining_count > len(remaining_pool),
        ).tolist()
    else:
        extra = []
    choices = np.asarray(mandatory + extra, dtype=np.int64)
    rng.shuffle(choices)
    return choices


def get_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def train(args):
    support = load_sibling("training_monitor")
    output = Path(args.output)
    resume = getattr(args, "resume", False)
    saved = torch.load(output/"last.pt", map_location="cpu", weights_only=True) if resume else None
    cfg = (Config(**json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config
           else Config(**saved["config"]) if saved else Config())
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    requested_config = json.loads(json.dumps(asdict(cfg)))
    if saved and any(requested_config[key] != value for key,value in
                     json.loads(json.dumps(saved["config"])).items() if key != "epochs"):
        raise ValueError("Resume must keep the saved configuration; only total --epochs may change")
    cfg.validate()
    seed_all(cfg.seed)
    device = get_device(args.device)
    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    record_paths = resolve_training_records(args.records)
    records = []
    for index, path in enumerate(record_paths, 1):
        print(f"Loading recording {index}/{len(record_paths)}: {path}", flush=True)
        records.append(Record(path))
    if len({r.name for r in records}) != len(records):
        raise ValueError("Record folder names must be unique")
    if len({r.frames.shape[1:] for r in records}) != 1:
        raise ValueError("All training records must have the same spatial dimensions")
    training, validation = split_samples(records, cfg)
    if len({i for i, _ in training}) > cfg.samples_per_epoch:
        raise ValueError("samples_per_epoch must be at least the number of training records")
    print(f"Split: {len(training)} training targets, {len(validation)} validation targets. "
          f"Each epoch: {cfg.samples_per_epoch} training samples, "
          f"{(cfg.samples_per_epoch + cfg.batch_size - 1) // cfg.batch_size} batches.", flush=True)
    print("Checking replacement patch placement...", flush=True)
    # Validate block placement before creating outputs or spending time on training.
    for stage, samples in (("train", training[:1]), ("valid", validation)):
        for i,t in samples:
            replacement(records[i], t, cfg, np.random.default_rng(cfg.seed), stage)
    split = dict(training=training, validation=validation,
               policy="record" if cfg.validation_records else "overlapping_sequences",
               records=[str(r.path) for r in records])
    if cfg.split_mode == "temporal":
        split.update(policy="disjoint_temporal_blocks", ranges={r.name:r.split_ranges for r in records},
                     partition_signals={r.name:r.split_signals for r in records})
    manifest = []
    for index, record in enumerate(records, 1):
        print(f"Hashing input files {index}/{len(records)}: {record.name}", flush=True)
        files = ["frames.npy","roi.npy","phase.npy","brightness.npy","metadata.json"]
        if (record.path/"valid_frames.npy").exists():
            files.append("valid_frames.npy")
        manifest.append(dict(path=str(record.path), metadata=record.metadata,
                             hashes={name:sha256(record.path/name) for name in files}))
    if resume:
        if json.loads((output/"provenance.json").read_text())["records"] != manifest:
            raise ValueError("Prepared records changed since training; cannot resume")
        if json.loads((output/"split.json").read_text()) != json.loads(json.dumps(split)):
            raise ValueError("Training/validation split changed; cannot resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "split.json", split)
        write_json(output / "provenance.json", dict(records=manifest, script_sha256=sha256(__file__),
               python=sys.version, platform=platform.platform(), torch=torch.__version__,
               numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__, device=str(device)))
    write_json(output / "config.json", asdict(cfg))
    monitors = {}
    if not getattr(args, "no_epoch_metrics", False):
        for record in records[:1]:
            if "dataset_measure" in record.metadata:
                monitors[record.name] = support.Monitor(record, cfg.history, sys.modules[__name__],
                                                        getattr(args, "background_dilation_radius", 2))
    monitor_provenance = {name: monitor.provenance for name,monitor in monitors.items()}
    monitor_path = output/"monitor_masks.json"
    if resume and monitor_path.exists():
        previous_monitors = json.loads(monitor_path.read_text())
        if any(name in previous_monitors and previous_monitors[name] != value
               for name,value in monitor_provenance.items()):
            raise ValueError("Monitoring masks/settings changed; keep them unchanged when resuming")
        # Retain provenance for old curves when reducing monitoring to one video.
        monitor_provenance = dict(previous_monitors, **monitor_provenance)
    write_json(monitor_path, monitor_provenance)
    if not cfg.validation_records and cfg.split_mode == "mixed":
        print("Validation uses overlapping sequences and shared donor pools; it is not an independent test.", flush=True)
    print("Initializing model and optimizer...", flush=True)
    model = Noise2Time(cfg.base_channels, cfg.convlstm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best, stale = float("inf"), 0
    rows, first_epoch = [], 1
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch = saved["epoch"] + 1
        best = saved["best_validation"]
        history_path = output/"metrics.jsonl"
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: continue  # Old interrupted append may be incomplete.
                if row["epoch"] <= saved["epoch"]: rows.append(row)
        by_epoch = {row["epoch"]:row for row in rows}
        rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
        if not rows or rows[-1]["epoch"] != saved["epoch"]:
            rows.append(dict(epoch=saved["epoch"], **saved["metrics"]))
        stale = saved.get("stale", 0)
        if "stale" not in saved:
            for row in reversed(rows):
                if row["valid"]["total"] <= best: break
                stale += 1
        if "rng" in saved:
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        else:
            warnings.warn("Older checkpoint: model/optimizer restored; random sampling cannot be restored exactly")
        support.save_history(output, rows)
        print(f"Resuming after epoch {saved['epoch']}; target total: {cfg.epochs} epochs", flush=True)
    for epoch in range(first_epoch, cfg.epochs+1):
        if stale >= cfg.patience:
            print("Saved run already reached early stopping.", flush=True)
            break
        epoch_started = time.monotonic()
        summaries = {}
        for stage, pool in (("train", training), ("valid", validation)):
            model.train(stage == "train")
            choices = (select_train_samples_for_epoch(pool, cfg.samples_per_epoch, rng)
                       if stage == "train" else np.arange(len(pool)))
            sums = {key:0. for key in ("total","reconstruction","gradient","hessian")}
            seen = 0
            batch_size = cfg.batch_size if stage == "train" else 1
            batches = tqdm(range(0, len(choices), batch_size),
                           desc=f"Epoch {epoch}/{cfg.epochs} {stage}", unit="batch",
                           file=sys.stdout, dynamic_ncols=True, mininterval=1.0)
            for start in batches:
                samples = []
                for index in choices[start:start+batch_size]:
                    i,t = pool[int(index)]
                    # Fixed target, donor and patch choices on every validation pass.
                    sampler = rng if stage == "train" else np.random.default_rng(10000+int(index))
                    samples.append(replacement(records[i],t,cfg,sampler,stage))
                sequence, target, mask = [torch.from_numpy(np.stack(items)).to(device) for items in zip(*samples)]
                with torch.set_grad_enabled(stage == "train"):
                    parts = loss_terms(model(sequence), target, mask, cfg.objective)
                    loss = parts["total"].sum()  # Paper specifies sum across batch.
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
                    if stage == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                for key in sums:
                    sums[key] += float(parts[key].detach().sum())
                seen += len(samples)
                batches.set_postfix(loss=f"{sums['total']/seen:.6g}",
                                    samples=f"{seen}/{len(choices)}", refresh=False)
            if seen == 0:
                raise ValueError(f"No {stage} samples evaluated")
            summaries[stage] = {key:value/seen for key,value in sums.items()}
        value = summaries["valid"]["total"]
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale+1)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=asdict(cfg),
                          epoch=epoch, best_validation=best, metrics=summaries,
                          stale=stale, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                          python_rng=random.getstate(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                          script_sha256=sha256(__file__))
        print(f"Epoch {epoch}/{cfg.epochs}: saving checkpoints...", flush=True)
        if improved:
            support.atomic_checkpoint(torch, checkpoint, output / "best.pt")
        support.atomic_checkpoint(torch, checkpoint, output / "last.pt")
        rows.append(dict(epoch=epoch, **summaries))
        support.save_history(output, rows)
        if monitors or not getattr(args, "no_epoch_previews", False):
            preview_dir = output / "previews" / f"epoch_{epoch:03d}"
            print(f"Epoch {epoch}/{cfg.epochs}: computing diagnostics/exporting previews...", flush=True)
            diagnostics = {}
            for index, record in enumerate(records[:1]):
                monitor = monitors.get(record.name)
                if monitor: monitor.reset()
                preview_path = None if getattr(args, "no_epoch_previews", False) else preview_dir / f"{record.name}.avi"
                if preview_path is None and monitor is None: continue
                export_denoised(model, record, cfg, device, avi_path=preview_path,
                                frame_callback=monitor.update if monitor else None)
                if monitor: diagnostics[record.name] = monitor.result()
                if preview_path is not None: write_json(preview_path.with_suffix(".json"), dict(
                    epoch=epoch, record=str(record.path), config=asdict(cfg),
                    record_sha256=manifest[index]["hashes"]["frames.npy"],
                    first_original_frame=record.metadata["first_original_frame"],
                    fps=record.metadata["fps"], copied_prefix=cfg.history,
                    intensities="MJPG preview: clip [0,1], round to uint8; no contrast normalization",
                    weights_source="Current epoch model; only best.pt and last.pt weights retained"))
            rows[-1]["diagnostics"] = diagnostics
            support.save_history(output, rows)
        print(f"Epoch {epoch}/{cfg.epochs}: train={summaries['train']['total']:.6g}, "
              f"valid={value:.6g}, best={best:.6g}, "
              f"no improvement={stale}/{cfg.patience}, "
              f"elapsed={time.monotonic()-epoch_started:.1f}s", flush=True)
        if stale >= cfg.patience:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True)
            break
    print(f"Training finished. Best checkpoint: {output / 'best.pt'}", flush=True)


def export_denoised(model, record, cfg, device, npy_path=None, avi_path=None, original_avi_path=None, frame_callback=None):
    """One inference pass; stream float NPY and/or fixed-scale MJPG AVI."""
    paths = [Path(p) for p in (npy_path, avi_path, original_avi_path) if p is not None]
    if not paths and frame_callback is None:
        raise ValueError("At least one output is required")
    if len(record.frames) <= cfg.history:
        raise ValueError("Record shorter than history")
    if any(p.exists() for p in paths):
        raise FileExistsError(next(p for p in paths if p.exists()))
    if paths and len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError("AVI and NPY outputs must share a directory")
    fps = float(record.metadata["fps"])
    if (avi_path is not None or original_avi_path is not None) and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("AVI output requires a finite positive acquisition FPS")
    parent = paths[0].parent.resolve() if paths else Path(tempfile.gettempdir()).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    previous_mode = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory(prefix=".denoise-", dir=parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != parent:
                raise ValueError("Temporary output escaped output directory")
            restored, writer, original_writer = None, None, None
            try:
                if npy_path is not None:
                    restored = np.lib.format.open_memmap(staging/"frames.npy", mode="w+",
                                                        dtype=np.float32, shape=record.frames.shape)
                if avi_path is not None:
                    height, width = record.frames.shape[1:]
                    writer = cv2.VideoWriter(str(staging/"preview.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                             fps, (width,height), True)
                    if not writer.isOpened():
                        raise RuntimeError("Cannot initialize MJPG AVI writer")
                if original_avi_path is not None:
                    height, width = record.frames.shape[1:]
                    original_writer = cv2.VideoWriter(str(staging/"original.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                                     fps, (width,height), True)
                    if not original_writer.isOpened():
                        raise RuntimeError("Cannot initialize original MJPG AVI writer")
                with torch.inference_mode():
                    for t in tqdm(range(len(record.frames)), desc=f"Denoising {record.name}",
                                  unit="frame", file=sys.stdout, dynamic_ncols=True, mininterval=1.0):
                        if t < cfg.history:
                            prediction = record.frames[t]
                        else:
                            stop = t if cfg.input_mode == "history_only" else t+1
                            sequence = torch.from_numpy(np.array(record.frames[t-cfg.history:stop], copy=True))[None].to(device)
                            prediction = model(sequence)[0,0].cpu().numpy()
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError(f"Non-finite output at frame {t}")
                        if record.metadata.get("diaphragm_mask_applied", False):
                            # Convolutions may predict nonzero values outside the aperture.
                            prediction = np.where(record.roi, prediction, 0)
                        if frame_callback is not None:
                            frame_callback(t, prediction)
                        if restored is not None:
                            restored[t] = prediction  # Preserve unclipped scientific intensities.
                        if writer is not None:
                            view = np.rint(np.clip(prediction,0,1)*255).astype(np.uint8)
                            writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
                        if original_writer is not None:
                            view = np.rint(np.clip(record.frames[t],0,1)*255).astype(np.uint8)
                            original_writer.write(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
            finally:
                if writer is not None:
                    writer.release()
                if original_writer is not None:
                    original_writer.release()
                if restored is not None:
                    try:
                        restored.flush()
                    finally:
                        restored._mmap.close()  # Windows requires closure before rename.
            if npy_path is not None:
                (staging/"frames.npy").rename(npy_path)
            if avi_path is not None:
                (staging/"preview.avi").rename(avi_path)
            if original_avi_path is not None:
                (staging/"original.avi").rename(original_avi_path)
    finally:
        model.train(previous_mode)


def denoise(args):
    requested = Path(args.output)
    folder_mode = requested.is_dir() or requested.suffix.lower() not in (".npy", ".avi")
    record = Record(args.record)
    if folder_mode:
        measure = record.name if record.metadata.get("schema") == "noise2time.dataset.v1" else record.name.removesuffix("_HD_M0")
        if not measure or measure in (".", ".."):
            raise ValueError("Invalid measurement folder name")
        destination = requested.resolve() / measure
        if destination.exists():
            raise FileExistsError(destination)
        output, avi_path = destination/"denoised.npy", destination/"denoised.avi"
    else:
        output = requested.with_suffix(".npy") if requested.suffix.lower() == ".avi" else requested
        avi_path = requested if requested.suffix.lower() == ".avi" else None
    for path in (output, output.with_suffix(".json"), avi_path):
        if path is not None and path.exists():
            raise FileExistsError(path)
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    cfg.validate()
    seed_all(cfg.seed)
    if len(record.frames) <= cfg.history:
        raise ValueError("Record shorter than history")
    model = Noise2Time(cfg.base_channels, cfg.convlstm).to(device)
    model.load_state_dict(checkpoint["model"])
    metadata = dict(record=str(record.path), record_sha256=sha256(record.path/"frames.npy"),
               checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
               first_original_frame=record.metadata["first_original_frame"], fps=record.metadata["fps"],
               copied_prefix=cfg.history, epoch=checkpoint["epoch"],
               inference="previous_frames_only" if cfg.input_mode == "history_only" else "fully_visible_sliding_window",
               intensities="float32, unclipped",
               intensity_scale=record.metadata.get("intensity_scale",1.),
               input_mode=record.metadata.get("input_mode","legacy"),
               diaphragm_mask_applied=record.metadata.get("diaphragm_mask_applied",False),
               avi=str(avi_path.resolve()) if avi_path is not None else None,
               avi_conversion="MJPG; clip [0,1], round to uint8; no contrast normalization" if avi_path is not None else None)
    if folder_mode:
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata.update(original_avi=str(destination/"original.avi"),
                        original_definition=("Full-length selected input, fixed-scale preview; diaphragm masking follows preparation metadata; no trimming" if record.metadata.get("schema") == "noise2time.dataset.v1" else
                        "Prepared input before denoising; circularly masked and trimmed to first peak; aligned with denoised output"))
        with tempfile.TemporaryDirectory(prefix=".denoise-record-", dir=destination.parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != destination.parent.resolve() or destination.resolve().parent != staging.parent:
                raise ValueError("Output escaped destination parent")
            bundle = staging/"record"
            export_denoised(model, record, cfg, device, npy_path=bundle/"denoised.npy",
                            avi_path=bundle/"denoised.avi", original_avi_path=bundle/"original.avi")
            write_json(bundle/"denoised.json", metadata)
            bundle.rename(destination)
    else:
        export_denoised(model, record, cfg, device, npy_path=output, avi_path=avi_path)
        write_json(output.with_suffix(".json"), metadata)
    print(f"Saved {output}" + (f" and {avi_path}" if avi_path is not None else "")
          + f"; omit first {cfg.history} copied frames from metrics")


def load_evaluation_mask(path, expected_shape):
    """Load a binary NPY or PNG mask without changing polarity."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.dtype.kind not in "buif" or not np.isfinite(values).all():
            raise ValueError(f"Mask must be a finite numeric 2D array: {path}")
        mask = values != 0
    elif path.suffix.lower() == ".png":
        # imdecode supports Unicode paths on Windows via NumPy file reading.
        encoded = np.fromfile(path, dtype=np.uint8)
        values = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        if values is None:
            raise ValueError(f"Cannot decode PNG mask: {path}")
        if values.ndim == 2:
            mask = values != 0
        elif values.ndim == 3 and values.shape[2] in (3, 4):
            mask = np.any(values[..., :3] != 0, axis=2)
            if values.shape[2] == 4:
                mask &= values[..., 3] != 0  # Ignore fully transparent pixels.
        else:
            raise ValueError(f"Unsupported PNG mask dimensions: {path}")
    else:
        raise ValueError(f"Mask must be .npy or .png: {path}")
    if mask.shape != tuple(expected_shape):
        mask = cv2.resize(mask.astype(np.uint8), tuple(reversed(expected_shape)), interpolation=cv2.INTER_NEAREST).astype(bool)
        warnings.warn(f"Mask {path} resized to match ROI shape {expected_shape}")
    return mask


def evaluate_regional(args, record, restored, metadata):
    import importlib.util
    spec = importlib.util.spec_from_file_location("noise2time_report", Path(__file__).with_name("noise2time_report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    if args.vessel_mask:
        raise ValueError("Use either --vessel-mask or the three regional vessel masks")
    paths = dict(retinal_artery=args.retinal_artery_mask, retinal_vein=args.retinal_vein_mask,
                 choroidal=args.choroidal_masks)
    if not all(paths.values()):
        raise ValueError("Regional evaluation requires retinal artery, retinal vein and choroidal masks")
    if args.background_mask or args.background_masks:
        raise ValueError("Regional background is automatic; omit --background-mask/--background-masks")
    raw, provenance = {}, {}
    for name, sources in paths.items():
        raw[name] = np.zeros(record.roi.shape, bool)
        provenance[name] = []
        for source in sources:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                raw[name] |= load_evaluation_mask(source, record.roi.shape)
            messages = [str(notice.message) for notice in notices]
            for message in messages:
                warnings.warn(message)
            provenance[name].append(dict(path=str(Path(source).resolve()), sha256=sha256(source), warnings=messages))
    raw["background"] = report.derive_background(raw, record.roi, args.background_dilation_radius)
    provenance["background"] = dict(
        method="ROI & ~(dilate(original retinal artery | original retinal vein) | original choroidal)",
        retinal_dilation_radius_pixels=args.background_dilation_radius,
        kernel="Euclidean disk", choroidal_dilated=False,
        source_masks="Original input unions before overlap removal; dilation before ROI clipping")
    first = metadata.get("copied_prefix")
    if not isinstance(first, int) or not 0 <= first < len(restored):
        raise ValueError("Invalid copied_prefix in denoised metadata")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, denoised_sha256=sha256(args.denoised),
                    denoised_path=str(Path(args.denoised).resolve()))
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=destination.parent) as staging:
        output = Path(staging)/"report"
        report.build_report(record, restored, metadata, raw, provenance, args, output)
        output.rename(destination)
    print(f"Saved evaluation report: {destination/'report.html'}")


def evaluate(args):
    record = Record(args.record)
    restored = np.load(args.denoised, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(Path(args.denoised).with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["record_sha256"] != sha256(record.path/"frames.npy"):
        raise ValueError("Output does not correspond to this source array")
    if restored.shape != record.frames.shape or not np.isfinite(restored).all():
        raise ValueError("Output dimensions/values invalid")
    if any(getattr(args, key, None) for key in
           ("retinal_artery_mask", "retinal_vein_mask", "choroidal_masks", "background_masks")):
        return evaluate_regional(args, record, restored, metadata)
    if not args.vessel_mask or not args.background_mask:
        raise ValueError("Supply the regional masks, or legacy --vessel-mask and --background-mask")
    vessel = load_evaluation_mask(args.vessel_mask, record.roi.shape)
    background = load_evaluation_mask(args.background_mask, record.roi.shape)
    if (not vessel.any()):
        raise ValueError("Vessel mask must be non-empty")
    if (not background.any()):
        raise ValueError("Background mask must be non-empty")
    if (np.any(vessel & background)):
        vessel &= ~(vessel & background)
        background &= ~(vessel & background)
        warnings.warn("Vessel and background masks must be non-overlapping")
    print(f"Vessel mask: {vessel.sum()} pixels; background mask: {background.sum()} pixels")
    print(f"ROI mask: {record.roi.sum()} pixels; vessel & background overlap: {(vessel & background).sum()} pixels")
    warnings.warn("Vessel/background masks must be non-empty, non-overlapping, and match ROI shape")

    vessel &= record.roi
    background &= record.roi

    first = metadata["copied_prefix"]
    original = record.frames[first:]
    denoised = restored[first:]
    fps = record.metadata["fps"]
    if len(original) < 4 or not 0 < args.min_hz < args.max_hz < fps/2:
        raise ValueError("Need >=4 scored frames and 0 < min_hz < max_hz < Nyquist")
    before, after = [x[:,vessel].mean(axis=1, dtype=np.float64) for x in (original,denoised)]
    freq = np.fft.rfftfreq(len(before), d=1/fps)
    candidates = np.flatnonzero((freq >= args.min_hz) & (freq <= args.max_hz))
    if not len(candidates):
        raise ValueError("Recording too short for requested frequency band")
    k = candidates[np.argmax(np.abs(np.fft.rfft(before-before.mean()))[candidates])]
    f0 = float(freq[k])  # Select once from original; evaluate BOTH at that frequency.
    time = np.arange(len(before))/fps
    design = np.column_stack([np.ones(len(time)), np.cos(2*np.pi*f0*time), np.sin(2*np.pi*f0*time)])
    amplitudes = [float(np.linalg.norm(np.linalg.lstsq(design, curve, rcond=None)[0][1:])) for curve in (before,after)]
    bg_std = [float(x[:,background].std(axis=0, ddof=0, dtype=np.float64).mean()) for x in (original,denoised)]
    correlation = float(np.corrcoef(before,after)[0,1]) if min(before.std(),after.std()) > 1e-12 else None
    result = dict(record=record.name, frames_scored=len(before), excluded_prefix=first,
                  background_std_original=bg_std[0], background_std_denoised=bg_std[1],
                  NRR=1-bg_std[1]/bg_std[0] if bg_std[0] > 1e-12 else None,
                  temporal_correlation=correlation, cardiac_frequency_hz=f0,
                  amplitude_original=amplitudes[0], amplitude_denoised=amplitudes[1],
                  amplitude_ratio=amplitudes[1]/amplitudes[0] if amplitudes[0] > 1e-12 else None,
                  vessel_mean_original=float(before.mean()), vessel_mean_denoised=float(after.mean()),
                  vessel_mask_sha256=sha256(args.vessel_mask), background_mask_sha256=sha256(args.background_mask),
                  mask_interpretation="Nonzero pixels selected; for PNG any nonzero color channel with nonzero alpha when present",
                  caveat="No clean reference: these are fluctuation and waveform diagnostics, not accuracy scores")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


def summarize(args):
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.metrics]
    if len({row["record"] for row in rows}) != len(rows):
        raise ValueError("Duplicate record names in summary")
    keys = ("background_std_original", "background_std_denoised", "NRR",
            "temporal_correlation", "amplitude_ratio")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        if not all(np.isfinite(values)):
            raise ValueError(f"Non-finite metric: {key}")
        summary[key] = dict(n=len(values), missing=len(rows)-len(values),
                            mean=float(np.mean(values)) if values else None,
                            sample_sd=float(np.std(values, ddof=1)) if len(values)>1 else None)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, dict(record_count=len(rows), records=rows, summary=summary))
    print(json.dumps(summary,indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Find arterial peaks and brightness tables from an HDF5 dataset; cache selected input and preview")
    p.add_argument("--input", required=True, help="Dataset folder containing measurement folders; legacy AVI/NPY input also accepted")
    p.add_argument("--output", required=True,help="Dataset mode: workflow root, writes prepared/; legacy mode: prepared destination")
    p.add_argument("--measures",nargs="+",help="Dataset mode: selected measurement folder names; default all")
    p.add_argument("--avi",action="store_true",help="Dataset mode: encode/decode MJPG before peak detection and training; default raw HDF5")
    p.add_argument("--pattern", default="*", help="Folder mode: filename glob, e.g. '*.avi' or '*.npy'")
    p.add_argument("--skip-existing", action="store_true", help="Folder mode: leave existing outputs untouched (not verified)")
    p.add_argument("--fps", type=float, help="Effective M0 FPS; dataset default 37000/256 = 144.53125")
    p.add_argument("--cx", type=int, default=255)
    p.add_argument("--cy", type=int, default=255)
    p.add_argument("--radius", type=int, default=260)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--peak-distance", type=int, default=15)
    p.add_argument("--prominence", type=float, default=.10)
    p.add_argument("--brightness-mask", help="Optional NPY mask for legacy vessel-mean brightness")
    p.add_argument("--artery-mask", help="Manual retinal artery PNG/NPY used for robust peak timing; same geometry as input")
    p.add_argument("--peak-method", choices=("auto","arterial","legacy"),default="auto",
                   help="Auto uses arterial detection when --artery-mask is supplied, otherwise legacy detection")
    p.add_argument("--peak-min-hz",type=float,default=.5)
    p.add_argument("--peak-max-hz",type=float,default=2.5)
    p.set_defaults(func=prepare)
    p = commands.add_parser("train", help="Train on one or multiple prepared records")
    p.add_argument("--records", nargs="+",
                   help="Prepared record directories and/or a folder containing prepared records")
    p.add_argument("--config", help="JSON configuration; unspecified keys use defaults")
    p.add_argument("--output", required=True, help="Dataset mode: workflow root; legacy mode: new experiment directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-epoch-previews", action="store_true", help="Skip the default end-of-epoch AVI export for every supplied record")
    p.add_argument("--resume", action="store_true", help="Resume output/runs/last.pt (legacy: output/last.pt), from the last completed epoch")
    p.add_argument("--epochs", type=int, help="Total epoch target, including already completed epochs")
    p.add_argument("--no-epoch-metrics", action="store_true", help="Skip full-video regional diagnostics; loss history is always saved")
    p.add_argument("--background-dilation-radius", type=int, default=2)
    p.add_argument("--input",help="Dataset folder; use --output as workflow root (training writes runs/)")
    p.add_argument("--measures",nargs="+")
    p.set_defaults(func=train)
    p = commands.add_parser("denoise", help="Infer from a checkpoint, without retraining")
    p.add_argument("--checkpoint", help="Dataset mode default: output/runs/best.pt")
    p.add_argument("--record")
    p.add_argument("--input",help="Dataset folder; use --output as workflow root")
    p.add_argument("--measures",nargs="+")
    p.add_argument("--output", required=True, help="Parent folder for a measurement bundle; alternatively .npy or .avi output filename")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=denoise)
    p = commands.add_parser("evaluate", help="Calculate the article's per-recording metrics")
    p.add_argument("--record")
    p.add_argument("--denoised")
    p.add_argument("--input",help="Dataset folder: auto-load manual retinal/pseudo choroidal masks; writes evaluation/")
    p.add_argument("--measures",nargs="+")
    p.add_argument("--checkpoint",help="Dataset mode default: output/runs/best.pt; infer missing denoised outputs")
    p.add_argument("--device",default="auto")
    p.add_argument("--vessel-mask", help="Legacy single-region NPY or PNG mask")
    p.add_argument("--background-mask", help="Background NPY or PNG for legacy single-region evaluation only")
    p.add_argument("--retinal-artery-mask", "--retinal-artery", nargs="+", help="One or more artery masks, combined by union")
    p.add_argument("--retinal-vein-mask", "--retinal-vein", nargs="+", help="One or more vein masks, combined by union")
    p.add_argument("--choroidal-masks", "--choroidal-mask", nargs="+", help="One or more choroidal masks")
    p.add_argument("--background-masks", nargs="+", help=argparse.SUPPRESS)
    p.add_argument("--background-dilation-radius", type=int, default=2,
                   help="Regional background: retinal dilation disk radius in prepared-image pixels (default: 2; 0 disables dilation)")
    p.add_argument("--cardiac-hz", type=float, help="Regional report: override common cardiac frequency")
    p.add_argument("--local-size", type=int, default=32, help="Regional report: local grid patch width in pixels")
    p.add_argument("--local-count", type=int, default=3, help="Regional report: maximum local patches per vessel group")
    p.add_argument("--max-lag-seconds", type=float, default=.25)
    p.add_argument("--profiles", help="Regional report: JSON mapping region to start/end [x,y] profile coordinates")
    p.add_argument("--min-hz", type=float, default=.5)
    p.add_argument("--max-hz", type=float, default=3.)
    p.add_argument("--output", required=True, help="New report folder for regional masks; JSON file for legacy evaluation")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("summarize", help="Aggregate metrics across recordings (mean and sample SD)")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    if args.command in ("train","denoise","evaluate") and args.input:
        if any(getattr(args,key,None) for key in ("record","records","denoised","vessel_mask","retinal_artery_mask","retinal_vein_mask","choroidal_masks","background_mask","background_masks")):
            parser.error("Dataset --input mode automatically resolves records and masks; do not mix legacy input flags")
        return load_sibling("dataset_workflow").run_stage(args,sys.modules[__name__]) or 0
    if args.command == "train" and not args.records:
        parser.error("train requires --input dataset or --records")
    if args.command == "denoise" and (not args.record or not args.checkpoint):
        parser.error("denoise requires --input dataset or both --record and --checkpoint")
    if args.command == "evaluate" and (not args.record or not args.denoised):
        parser.error("evaluate requires --input dataset or both --record and --denoised")
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
