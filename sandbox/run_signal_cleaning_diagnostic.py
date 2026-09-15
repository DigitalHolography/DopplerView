"""Inspect cardiac signal cleaning on one DopplerView measure.

Example::

    python sandbox/run_signal_cleaning_diagnostic.py \
        D:/dataset_choroid/260622_DUM_L_1 \
        benchmark/260622_DUM_L_1_signal_cleaning
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from dopplerview.segmentation import pulse_analysis as pa
from dopplerview.segmentation import signal_processing
from sandbox.signal_preprocessing import preprocess_cardiac_signal
from sandbox.run_choroid_global_benchmark import _notebook_candidate_mask


def _find_h5(measure_folder):
    candidates = sorted(measure_folder.rglob("*.h5"))
    for path in candidates:
        try:
            with h5py.File(path, "r") as h5:
                container = h5["doppler_signal"] if "doppler_signal" in h5 else h5
                if "M0_ff" in container:
                    return path
        except OSError:
            continue
    raise FileNotFoundError(f"no HDF5 file containing M0_ff under {measure_folder}")


def _find_mask(measure_folder, names):
    for directory in (measure_folder / "ground_truths", measure_folder / "manual", measure_folder):
        for name in names:
            path = directory / name
            if path.is_file():
                return path
    return None


def _load_mask(path, spatial_shape):
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize(
                (spatial_shape[1], spatial_shape[0]),
                Image.Resampling.NEAREST,
            )
        ) > 0


def load_reference_signal(measure_folder, mask_source):
    """Load M0 and return a research reference without keeping the video alive."""
    measure_folder = Path(measure_folder).expanduser().resolve()
    h5_path = _find_h5(measure_folder)
    with h5py.File(h5_path, "r") as h5:
        container = h5["doppler_signal"] if "doppler_signal" in h5 else h5
        video = container["M0_ff"][()]

    if mask_source == "full-frame":
        mask = np.ones(video.shape[1:], dtype=bool)
        mask_paths = []
    elif mask_source == "candidate":
        mask = _notebook_candidate_mask(video)
        mask_paths = []
    elif mask_source == "retina-artery":
        path = _find_mask(measure_folder, ("retina_artery_mask.png", "retina_artery.png"))
        if path is None:
            raise FileNotFoundError("retinal artery mask was not found")
        mask = _load_mask(path, video.shape[1:])
        mask_paths = [path]
    else:
        mask_paths = []
        masks = []
        for names in (
            ("choroid_artery_mask.png", "choroidal_artery_mask.png", "choroid_artery.png"),
            ("choroid_vein_mask.png", "choroidal_vein_mask.png", "choroid_vein.png"),
            (
                "choroid_aliased_artery_mask.png",
                "choroidal_aliased_artery_mask.png",
                "choroid_aliased_artery.png",
            ),
        ):
            path = _find_mask(measure_folder, names)
            if path is not None:
                mask_paths.append(path)
                masks.append(_load_mask(path, video.shape[1:]))
        if not masks:
            raise FileNotFoundError("no choroidal partial masks were found")
        mask = np.any(masks, axis=0)
    if not np.any(mask):
        raise ValueError("selected reference mask is empty")
    signal = signal_processing.get_pulse_from_mask(video, mask)
    return np.asarray(signal, dtype=float), h5_path, mask_paths


def _normalized_cycles(result):
    cycles = np.asarray(
        [result.cleaned_signal[start:stop] for start, stop in result.cycle_bounds]
    )
    cycles -= np.mean(cycles, axis=1, keepdims=True)
    scales = np.std(cycles, axis=1, keepdims=True)
    return np.divide(cycles, scales, out=np.zeros_like(cycles), where=scales > 0)


def save_diagnostic(result, output_folder, metadata):
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    summary = {**metadata, **result.summary()}
    (output_folder / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(result.cycle_rows()).to_csv(output_folder / "cycles.csv", index=False)
    np.savez_compressed(
        output_folder / "cleaning_arrays.npz",
        raw_signal=result.raw_signal,
        cleaned_signal=result.cleaned_signal,
        frame_artifact_mask=result.frame_artifact_mask,
        cycle_bounds=result.cycle_bounds,
        cycle_valid=result.cycle_valid,
        valid_cycle_frame_mask=result.valid_cycle_frame_mask,
        fit_frame_mask=result.fit_frame_mask,
    )

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    frames = np.arange(len(result.raw_signal))
    axes[0].plot(frames, result.raw_signal, color="0.55", linewidth=1, label="raw")
    axes[0].plot(frames, result.cleaned_signal, color="tab:blue", linewidth=1.4, label="cleaned reference")
    artifact_frames = np.flatnonzero(result.frame_artifact_mask)
    if len(artifact_frames):
        axes[0].scatter(
            artifact_frames,
            result.raw_signal[artifact_frames],
            color="tab:red",
            marker="x",
            s=35,
            label="detected artifact",
            zorder=4,
        )
    for valid, (start, stop) in zip(result.cycle_valid, result.cycle_bounds):
        axes[0].axvspan(start, stop, color=("tab:green" if valid else "tab:red"), alpha=0.07)
    axes[0].set(title="Raw signal, conservative impulse repair, and cycle decisions", xlabel="frame", ylabel="M0")
    axes[0].legend(loc="best")

    cycle_indices = np.arange(result.n_cycles)
    colors = np.where(result.cycle_valid, "tab:green", "tab:red")
    axes[1].bar(cycle_indices, result.cycle_correlations, color=colors, alpha=0.75)
    axes[1].axhline(0.5, color="0.25", linestyle="--", linewidth=1, label="nominal correlation floor")
    axes[1].set(title="Cycle similarity to the robust median shape", xlabel="cycle", ylabel="Pearson correlation", ylim=(-1, 1.05))
    axes[1].legend(loc="best")

    normalized = _normalized_cycles(result)
    phase = np.arange(result.beat_period) / result.beat_period
    for index, cycle in enumerate(normalized):
        axes[2].plot(
            phase,
            cycle,
            color=("tab:green" if result.cycle_valid[index] else "tab:red"),
            alpha=0.38,
            linewidth=1,
        )
    accepted = normalized[result.cycle_valid]
    axes[2].plot(phase, np.median(accepted, axis=0), color="black", linewidth=2.5, label="accepted-cycle median")
    axes[2].set(title="Phase-aligned cycle shapes", xlabel="normalized cardiac phase", ylabel="standardized M0")
    axes[2].legend(loc="best")
    figure.suptitle(metadata["measure"] + " signal-cleaning diagnostic", fontsize=15)
    figure.savefig(output_folder / "signal_cleaning.png", dpi=160)
    plt.close(figure)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measure_folder")
    parser.add_argument("output_folder")
    parser.add_argument(
        "--mask-source",
        choices=("candidate", "choroid", "retina-artery", "full-frame"),
        default="candidate",
        help="Reference used only to detect global temporal artifacts/cycles",
    )
    parser.add_argument("--sampling-frequency", type=float)
    parser.add_argument("--camera-frequency", type=float, default=37037.0)
    parser.add_argument("--temporal-window-size", type=int, default=256)
    parser.add_argument("--beat-period", type=int)
    parser.add_argument("--derivative-z-threshold", type=float, default=6.0)
    parser.add_argument("--maximum-cycle-artifact-fraction", type=float, default=0.15)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    signal, h5_path, mask_paths = load_reference_signal(args.measure_folder, args.mask_source)
    sampling_frequency = args.sampling_frequency or pa.get_effective_sampling_frequency(
        args.camera_frequency, args.temporal_window_size
    )
    beat_period = args.beat_period or pa.compute_period(signal, sampling_frequency)
    if beat_period is None:
        raise RuntimeError("cardiac period could not be estimated")
    result = preprocess_cardiac_signal(
        signal,
        sampling_frequency,
        int(beat_period),
        derivative_z_threshold=args.derivative_z_threshold,
        maximum_cycle_artifact_fraction=args.maximum_cycle_artifact_fraction,
    )
    summary = save_diagnostic(
        result,
        args.output_folder,
        {
            "measure": Path(args.measure_folder).resolve().name,
            "h5_path": str(h5_path),
            "mask_source": args.mask_source,
            "mask_paths": [str(path) for path in mask_paths],
            "sampling_frequency": float(sampling_frequency),
        },
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
