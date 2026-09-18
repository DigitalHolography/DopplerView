"""Dataset-oriented Noise2Time workflow. Source HDF5 files are read-only.

One output root contains prepared/, runs/, and evaluation/. Raw preparation
zeros pixels outside the diaphragm and preserves frame alignment. A fixed multiplicative
scale makes network inputs compatible with the existing [0,1] model convention.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

import cv2
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

KEY = "doppler_signal/M0_ff"
SCHEMA = "noise2time.dataset.v1"


def measurements(dataset, names=None):
    root = Path(dataset).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    found = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("*.h5")))
    if names:
        missing = set(names) - {p.name for p in found}
        if missing:
            raise ValueError(f"Unknown measurements: {sorted(missing)}")
        found = [p for p in found if p.name in names]
    if not found:
        raise ValueError(f"No measurement folders with HDF5 files in {root}")
    return found


def single_h5(folder):
    paths = sorted(folder.glob("*.h5"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one HDF5 file in {folder}; found {len(paths)}")
    return paths[0]


def manual_mask(folder, kind):
    names = {
        "artery": ("retinal_artery_mask.png", "retina_artery_mask.png"),
        "vein": ("retinal_vein_mask.png", "retina_vein_mask.png"),
    }[kind]
    matches = [folder/"manual"/name for name in names if (folder/"manual"/name).exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected one manual retinal {kind} mask in {folder/'manual'}")
    return matches[0]


def strict_mask(path, shape, api):
    values = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if values is None or values.shape[:2] != tuple(shape):
        raise ValueError(f"Mask must match video geometry {shape}: {path}")
    mask = api.load_evaluation_mask(path, shape)
    if not mask.any():
        raise ValueError(f"Empty mask: {path}")
    return mask


def write_preview(path, frames, fps):
    """Fixed-scale MJPG encoding; the raw path never reads this display file."""
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), True)
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create AVI: {path}")
        for frame in frames:
            view = np.rint(np.clip(frame, 0, 1)*255).astype(np.uint8)
            writer.write(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def decode_into(path, frames, roi):
    cap = cv2.VideoCapture(str(path))
    try:
        for index in range(len(frames)):
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != frames.shape[1:]:
                raise ValueError(f"AVI frame count/geometry changed at frame {index}: {path}")
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            # MJPG can introduce ringing outside the aperture; restore exact zeros.
            frames[index, ~roi] = 0
        if cap.read()[0]:
            raise ValueError(f"AVI contains unexpected extra frames: {path}")
    finally:
        cap.release()


def prepare_measure(folder, output, args, api):
    source = single_h5(folder)
    artery_path = manual_mask(folder, "artery")
    fps = args.fps if args.fps is not None else 37000/256
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Effective M0 FPS must be finite and positive")
    detector = api.load_sibling("arterial_peaks")
    output.mkdir()
    with h5py.File(source, "r") as h5:
        data = h5[KEY]
        if data.ndim != 3 or any(s % 32 for s in data.shape[1:]):
            raise ValueError("M0_ff must be (time,height,width), spatial dimensions divisible by 32")
        roi = api.circle_mask(*data.shape[1:], args.cx, args.cy, args.radius)
        if not roi.any():
            raise ValueError("Diaphragm ROI is empty")
        mask = strict_mask(artery_path, data.shape[1:], api) & roi
        if not mask.any():
            raise ValueError("Retinal artery mask is empty inside the diaphragm")
        # Scan in bounded blocks. No percentile clipping or framewise normalization.
        scale = 0.
        for start in range(0, len(data), 16):
            block = data[start:start+16]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("M0_ff must contain finite nonnegative intensities")
            scale = max(scale, float(block[:, roi].max()))
        if scale <= 0:
            raise ValueError("M0_ff is entirely zero inside the diaphragm")
        frames = np.lib.format.open_memmap(output/"frames.npy", mode="w+", dtype=np.float32, shape=data.shape)
        try:
            for start in range(0, len(data), 16):
                frames[start:start+16] = np.where(roi, data[start:start+16].astype(np.float64)/scale, 0)
            # Both paths display the same mapping. --avi alone reads the encoded
            # file back, introducing 8-bit quantization AND MJPG compression.
            write_preview(output/"prepared.avi", frames, fps)
            if args.avi:
                decode_into(output/"prepared.avi", frames, roi)
            arterial = np.array([frame[mask].mean(dtype=np.float64) for frame in frames])
            result = detector.detect_arterial_peaks(arterial, fps, args.peak_min_hz, args.peak_max_hz)
            peaks = result["peaks"]
            if len(peaks) < 3:
                raise ValueError("Need at least three detected peaks to form two complete donor cycles")
            # Brightness is measured, not interpolated: repair is for peak timing.
            brightness = gaussian_filter1d(arterial, max(.5, .02*fps), mode="reflect")
            phase = api.phases_from_peaks(len(frames), peaks)
            phase[peaks[-1]:] = -1  # No complete final cycle; keep frames but exclude pairing.
            valid = (phase >= 0) & ~result["artifact_mask"]
            np.save(output/"roi.npy", roi)
            np.save(output/"artery_mask.npy", mask)
            np.save(output/"phase.npy", phase)
            np.save(output/"valid_frames.npy", valid)
            np.save(output/"brightness.npy", brightness.astype(np.float32))
            np.save(output/"brightness_raw.npy", arterial.astype(np.float32))
            np.savez_compressed(output/"arterial_peak_diagnostics.npz", raw=arterial,
                               **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
            table = np.column_stack((np.arange(len(frames)),np.arange(len(frames))/fps,
                                     arterial*scale,brightness*scale,result["smoothed"]*scale,
                                     phase,valid,result["artifact_mask"]))
            np.savetxt(output/"brightness.csv",table,delimiter=",",comments="",
                       header="frame,time_seconds,raw_arterial_M0,brightness_smoothed_M0,detection_smoothed_M0,phase,valid_target_or_donor,artifact")
            summary = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()
                       if k not in ("smoothed","repaired","detection_signal","artifact_mask","acf")}
            api.write_json(output/"arterial_peaks.json", dict(summary,fps=fps,indices="original frames; no trimming"))
            plot_cycles(output/"brightness.png", arterial*scale, brightness*scale, result, scale, fps)
            metadata = dict(schema=SCHEMA,record=folder.name,dataset_measure=str(folder.resolve()),
                source=str(source.resolve()),source_sha256=api.sha256(source),h5_key=KEY,frame_axis=0,
                shape=list(frames.shape),fps=fps,first_original_frame=0,original_frame_count=len(frames),
                peaks=peaks.tolist(),phase_definition="frames_since_peak; -1 outside complete cycles",
                peak_method="arterial",peak_min_hz=args.peak_min_hz,peak_max_hz=args.peak_max_hz,
                artery_mask=str(artery_path.resolve()),artery_mask_sha256=api.sha256(artery_path),
                brightness_definition="manual retinal artery intersected with diaphragm; mean then Gaussian sigma 20 ms; no artifact interpolation",
                input_mode="avi" if args.avi else "raw",intensity_scale=scale,
                normalization="one fixed division by full-recording raw maximum inside diaphragm; multiply by intensity_scale for M0 units inside diaphragm",
                avi_conversion="MJPG, round(255*raw/scale); AVI mode includes quantization plus codec effects",
                diaphragm_mask_applied=True,
                spatial_processing="zero outside circular diaphragm before encoding and after AVI decoding; artery mask intersected with diaphragm",
                valid_frames_definition="inside complete peak-to-peak intervals and not flagged as artifacts",
                circle=dict(cx=args.cx,cy=args.cy,radius=args.radius),
                detector_sha256=api.sha256(Path(__file__).with_name("arterial_peaks.py")),warnings=result["warnings"])
            api.write_json(output/"metadata.json", metadata)
            frames.flush()
        finally:
            frames._mmap.close()


def plot_cycles(path, raw, brightness, result, scale, fps):
    figure = Figure(figsize=(13,5), layout="constrained"); FigureCanvasAgg(figure)
    ax = figure.subplots(); t=np.arange(len(raw))/fps
    peaks=result["peaks"]
    for index,(start,stop) in enumerate(zip(peaks[:-1],peaks[1:])):
        ax.axvspan(start/fps,stop/fps,color=("#d7edf3" if index%2==0 else "#f4ead5"),alpha=.5)
        ax.text((start+stop)/2/fps,.98,f"Cycle {index+1}",transform=ax.get_xaxis_transform(),ha="center",va="top")
    ax.plot(t,raw,color=".65",lw=.8,label="Raw arterial mean (selected input)")
    ax.plot(t,brightness,color="#9467bd",lw=1,label="Measured brightness smoothing")
    ax.plot(t,result["smoothed"]*scale,color="#087f8c",label="Robust detection smoothing")
    ax.scatter(t[peaks],result["smoothed"][peaks]*scale,color="red",marker="v",label="Peaks",zorder=5)
    ax.set(xlabel="Time from original start (s)",ylabel="M0 intensity",title="Arterial signal, peaks and complete cycles — no video trimming")
    ax.legend(fontsize=8);figure.savefig(path,dpi=140);figure.clear()


def prepare_dataset(args, api):
    if args.peak_method == "legacy" or args.artery_mask or args.brightness_mask:
        raise ValueError("Dataset preparation uses automatic manual-artery discovery and arterial detection; omit legacy mask/method flags")
    root=Path(args.output).resolve(); prepared=root/"prepared";prepared.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in measurements(args.input,args.measures):
        target=prepared/folder.name
        print(f"Preparing {folder.name} ({'AVI round trip' if args.avi else 'raw HDF5'})",flush=True)
        try:
            if target.exists():
                if not args.skip_existing: raise FileExistsError(target)
                metadata=json.loads((target/"metadata.json").read_text())
                expected_fps=args.fps if args.fps is not None else 37000/256
                if (metadata.get("diaphragm_mask_applied") is not True
                    or metadata.get("input_mode") != ("avi" if args.avi else "raw") or metadata.get("fps") != expected_fps
                    or metadata.get("source_sha256") != api.sha256(single_h5(folder))
                    or metadata.get("artery_mask_sha256") != api.sha256(manual_mask(folder,"artery"))
                    or metadata.get("detector_sha256") != api.sha256(Path(__file__).with_name("arterial_peaks.py"))
                    or metadata.get("peak_min_hz") != args.peak_min_hz or metadata.get("peak_max_hz") != args.peak_max_hz
                    or metadata.get("circle") != dict(cx=args.cx,cy=args.cy,radius=args.radius)):
                    raise ValueError("Existing preparation differs from requested source/settings; use a new output root")
                rows.append(dict(record=folder.name,status="skipped"));continue
            with tempfile.TemporaryDirectory(prefix=".prepare-",dir=prepared) as staging:
                temporary=Path(staging)/"record"
                prepare_measure(folder,temporary,args,api)
                temporary.rename(target)
            rows.append(dict(record=folder.name,status="prepared"))
        except Exception as exc:
            print(f"  ERROR: {exc}",flush=True)
            rows.append(dict(record=folder.name,status="error",error=str(exc)))
    api.write_json(root/"preparation_summary.json",rows)
    return int(any(row["status"]=="error" for row in rows))


def choroidal_masks(folder):
    # User-selected policy: never use manual choroidal annotations. Prefer the
    # canonical unsuffixed output; do not silently choose among versioned copies.
    paths=sorted((folder/"pseudo").glob("*choroidal_vessel_mask.png"))
    if len(paths)==1:
        return paths,"pseudo choroidal vessel mask (not handmade)"
    raise ValueError(f"Expected one canonical pseudo/*choroidal_vessel_mask.png in {folder}")


def run_stage(args, api):
    root=Path(args.output).resolve()
    folders=measurements(args.input,args.measures)
    records=[root/"prepared"/p.name for p in folders]
    for folder,record in zip(folders,records):
        if not (record/"metadata.json").exists():
            raise ValueError(f"Missing preparation: {record}. Run prepare with the same dataset/output root first.")
        metadata=json.loads((record/"metadata.json").read_text())
        if metadata.get("schema")!=SCHEMA or Path(metadata["dataset_measure"])!=folder:
            raise ValueError(f"Preparation does not correspond to dataset: {record}")
        if metadata.get("diaphragm_mask_applied") is not True:
            raise ValueError(f"Preparation lacks diaphragm masking: {record}. Run prepare in a new output root.")
    modes={json.loads((record/"metadata.json").read_text())["input_mode"] for record in records}
    if len(modes)>1:
        raise ValueError("Do not mix raw and AVI preparations in one workflow root")
    if args.command=="train":
        return api.train(SimpleNamespace(records=records,output=root/"runs",config=args.config,
                                        device=args.device,no_epoch_previews=args.no_epoch_previews))
    checkpoint=Path(args.checkpoint) if args.checkpoint else root/"runs"/"best.pt"
    results=[]
    for folder,record in zip(folders,records):
        try:
            destination=root/"runs"/"denoised"/folder.name
            denoised=destination/"denoised.npy"
            if args.command=="denoise":
                api.denoise(SimpleNamespace(checkpoint=checkpoint,record=record,output=root/"runs"/"denoised",device=args.device))
            else:
                arteries=[manual_mask(folder,"artery")];veins=[manual_mask(folder,"vein")]
                choroid,origin=choroidal_masks(folder)
                shape=np.load(record/"roi.npy").shape
                for path in arteries+veins+choroid: strict_mask(path,shape,api)
                target=root/"evaluation"/folder.name
                if target.exists(): raise FileExistsError(target)
                if not denoised.exists():
                    api.denoise(SimpleNamespace(checkpoint=checkpoint,record=record,output=root/"runs"/"denoised",device=args.device))
                else:
                    provenance=json.loads(denoised.with_suffix(".json").read_text())
                    if provenance["checkpoint_sha256"]!=api.sha256(checkpoint):
                        raise ValueError("Existing denoised output uses a different checkpoint; use a new output root")
                options=vars(args).copy()
                options.update(record=record,denoised=denoised,output=target,vessel_mask=None,
                               background_mask=None,background_masks=None,retinal_artery_mask=arteries,
                               retinal_vein_mask=veins,choroidal_masks=choroid)
                api.evaluate(SimpleNamespace(**options))
                api.write_json(target/"dataset_sources.json",dict(record=folder.name,choroidal_mask_origin=origin,
                               choroidal_masks=[str(p) for p in choroid],intensity_scale=json.loads((record/"metadata.json").read_text())["intensity_scale"]))
                # Make mask origin visible, especially when pseudo masks were allowed.
                report=target/"report.html"
                import html
                metadata=json.loads((record/"metadata.json").read_text())
                unit_note=(f"Input mode: {metadata['input_mode']}. Intensities are M0 / {metadata['intensity_scale']:.8g}; "
                           "multiply means, amplitudes and standard deviations by this scale to recover M0 units.")
                report.write_text(report.read_text(encoding="utf-8").replace("<h1>",
                    "<p><strong>Dataset mask source:</strong> "+html.escape(origin)+"<br>"+html.escape(unit_note)+"</p><h1>",1),encoding="utf-8")
            results.append(dict(record=folder.name,status="complete"))
        except Exception as exc:
            print(f"{folder.name}: ERROR: {exc}",flush=True)
            results.append(dict(record=folder.name,status="error",error=str(exc)))
    api.write_json(root/(args.command+"_summary.json"),results)
    return int(any(row["status"]=="error" for row in results))
