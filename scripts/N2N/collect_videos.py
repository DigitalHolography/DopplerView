"""Collect requested LDH measurements using bounded directory listings.

Example:
  python scripts/N2N/collect_videos.py --measures measures.txt \
      --folders Y:/folder1 Y:/folder2 --output D:/N2T/videos

No recursive glob or file traversal. --max-depth counts directory levels from
each supplied search folder to a measurement folder (default 2).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time


def read_lines(path):
    """UTF-8/BOM text; blank lines and whole-line # comments are ignored."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def read_measures(path):
    measures, seen = [], set()
    for name in read_lines(path):
        if name in (".", "..") or any(c in name for c in '<>:"/\\|?*') or name.endswith((".", " ")):
            raise ValueError(f"Expected a measurement folder name, not a path: {name!r}")
        if name.casefold() not in seen:
            measures.append(name)
            seen.add(name.casefold())
    if not measures:
        raise ValueError("Measurement list is empty")
    return measures


def scan_folder(path):
    # DirEntry directory flags normally come from the SMB directory listing;
    # do not stat every contained file. Do not follow symlink directories.
    with os.scandir(path) as entries:
        return sorted((Path(entry.path) for entry in entries
                       if entry.is_dir(follow_symlinks=False)), key=lambda p:p.name.casefold())


def find_measurements(measures, folders, max_depth=2, workers=4):
    """List each searched directory once; never descend into a matched measure."""
    requested = {name.casefold():name for name in measures}
    matches = {name:[] for name in measures}
    errors, visited = [], set()
    pending = [(Path(folder).absolute(), 0) for folder in folders]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            futures = {}
            for path, depth in pending:
                key = os.path.normcase(os.path.normpath(str(path)))
                if key in visited:
                    continue
                visited.add(key)
                if path.name.casefold() in requested:
                    matches[requested[path.name.casefold()]].append(path)
                elif depth < max_depth:
                    futures[pool.submit(scan_folder, path)] = (path, depth)
            pending = []
            for future in as_completed(futures):
                path, depth = futures[future]
                try:
                    pending.extend((child, depth+1) for child in future.result())
                except OSError as exc:
                    errors.append(dict(path=str(path), error=str(exc)))
    for paths in matches.values():
        paths.sort(key=lambda p:str(p).casefold())
    return matches, errors


def source_for(measurement, h5=False):
    name = measurement.name
    hd = measurement / f"{name}_HD"
    return hd / ("h5" if h5 else "avi") / (f"{name}_HD_output.h5" if h5 else f"{name}_HD_M0.avi")


def extract_h5(source, destination, frame_axis, chunk_mb=64):
    """Stream moment0ff to a T,H,W NPY, retaining dtype and intensities."""
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        if "moment0ff" not in handle or not isinstance(handle["moment0ff"], h5py.Dataset):
            raise ValueError("H5 has no dataset at key 'moment0ff'")
        dataset = handle["moment0ff"]
        if dataset.ndim != 3 or min(dataset.shape) == 0 or dataset.dtype.kind not in "uif":
            raise ValueError(f"Expected nonempty real numeric 3D moment0ff, got {dataset.shape}, {dataset.dtype}")
        axis = frame_axis % 3
        spatial = [i for i in range(3) if i != axis]
        shape = (dataset.shape[axis], *(dataset.shape[i] for i in spatial))
        bytes_per_frame = shape[1]*shape[2]*dataset.dtype.itemsize
        count = max(1, chunk_mb*1024*1024 // bytes_per_frame)
        result = np.lib.format.open_memmap(destination, mode="w+", dtype=dataset.dtype, shape=shape)
        try:
            for start in range(0, shape[0], count):
                stop = min(start+count, shape[0])
                slices = [slice(None)]*3
                slices[axis] = slice(start,stop)
                result[start:stop] = np.moveaxis(dataset[tuple(slices)], axis, 0)
            result.flush()
        finally:
            # Close mmap explicitly so Windows can rename/delete the temporary file.
            result._mmap.close()
        return dict(dataset="moment0ff", source_shape=list(dataset.shape),
                    shape=list(shape), dtype=str(dataset.dtype), frame_axis=axis,
                    normalization="none", axes="T,H,W (remaining spatial axes retain their order)")


def collect_one(name, paths, output, h5=False, frame_axis=None, overwrite=False, dry_run=False, chunk_mb=64):
    row = dict(measure=name, matches=[str(p) for p in paths])
    if not paths:
        return dict(row, status="missing", error="Measurement folder not found within search depth")
    # Check only the single exact expected path per matched measurement.
    candidates = []
    for path in paths:
        source = source_for(path,h5)
        try:
            info = source.stat()
            if stat.S_ISREG(info.st_mode):
                candidates.append((source,info))
        except FileNotFoundError:
            continue
        except OSError as exc:
            return dict(row,status="error",source=str(source),error=str(exc))
    if not candidates:
        return dict(row,status="missing",error="Measurement found, but expected source file is missing")
    if len(candidates) > 1:
        return dict(row,status="ambiguous",sources=[str(p) for p,_ in candidates],
                    error="Multiple source files; narrow --folders to choose one")
    source, info = candidates[0]
    destination = output / (name + "_HD_M0" + (".npy" if h5 else ".avi"))
    row.update(source=str(source), destination=str(destination), source_bytes=info.st_size,
               source_mtime_ns=info.st_mtime_ns)
    if destination.exists() and not overwrite:
        return dict(row,status="exists",error="Existing destination left untouched; not verified")
    if dry_run:
        return dict(row,status="planned")
    temporary = None
    try:
        if source.absolute() == destination.absolute():
            raise ValueError("Source and destination are identical")
        fd, temporary = tempfile.mkstemp(prefix=".collect-",suffix=".partial",dir=output)
        os.close(fd)
        if h5:
            row.update(extract_h5(source,temporary,frame_axis,chunk_mb))
        else:
            shutil.copyfile(source,temporary)
            if Path(temporary).stat().st_size != info.st_size:
                raise OSError("Copied file size differs from source")
        after = source.stat()
        if (after.st_size,after.st_mtime_ns) != (info.st_size,info.st_mtime_ns):
            raise OSError("Source changed during copying; retry after acquisition/export finishes")
        if overwrite:
            os.replace(temporary,destination)
        elif os.name == "nt":
            # Windows rename refuses an existing destination, including on SMB.
            os.rename(temporary,destination)
        else:
            # POSIX rename overwrites; link instead for atomic no-clobber behavior.
            os.link(temporary,destination)
            os.unlink(temporary)
        temporary = None
        return dict(row,status="copied",output_bytes=destination.stat().st_size)
    except Exception as exc:
        return dict(row,status="error",error=f"{type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measures",required=True,help="UTF-8 txt: one exact measurement folder name per line")
    parser.add_argument("--folders",nargs="+",default=[],help="Folders containing measurement folders")
    parser.add_argument("--folders-file",help="Optional UTF-8 txt: one search folder per line")
    parser.add_argument("--output",required=True)
    parser.add_argument("--h5",action="store_true",help="Extract moment0ff losslessly into .npy instead of copying AVI")
    parser.add_argument("--frame-axis",type=int,choices=(0,1,2),help="H5 time axis; required with --h5 (no shape guessing)")
    parser.add_argument("--max-depth",type=int,default=2,help="Maximum depth to measure folders (default: 2); 1 for direct children")
    parser.add_argument("--scan-workers",type=int,default=4)
    parser.add_argument("--copy-workers",type=int,default=2)
    parser.add_argument("--chunk-mb",type=int,default=64,help="Approximate H5 read-buffer budget per worker")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--dry-run",action="store_true",help="Find sources and write report without copying/extracting")
    args = parser.parse_args(argv)
    if min(args.max_depth,args.scan_workers,args.copy_workers,args.chunk_mb) < 1:
        parser.error("Depth, workers and chunk size must be positive")
    if args.h5 and args.frame_axis is None:
        parser.error("--h5 requires --frame-axis 0, 1 or 2")
    folders = args.folders + (read_lines(args.folders_file) if args.folders_file else [])
    if not folders:
        parser.error("Supply --folders and/or --folders-file")
    measures = read_measures(args.measures)
    started = time.monotonic()
    output = Path(args.output).absolute()
    output.mkdir(parents=True,exist_ok=True)
    print(f"Searching {len(folders)} roots for {len(measures)} measurements (depth {args.max_depth})...",flush=True)
    matches, errors = find_measurements(measures,folders,args.max_depth,args.scan_workers)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.copy_workers) as pool:
        futures = {pool.submit(collect_one,name,matches[name],output,args.h5,args.frame_axis,
                               args.overwrite,args.dry_run,args.chunk_mb):name for name in measures}
        for future in as_completed(futures):
            row = future.result()
            rows[row["measure"]] = row
            print(f"[{len(rows)}/{len(measures)}] {row['measure']}: {row['status']}",flush=True)
    counts = {}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"],0)+1
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-started, folders=folders,
                  options=vars(args), counts=counts, scan_errors=errors,
                  results=[rows[name] for name in measures])
    fd, report_path = tempfile.mkstemp(prefix="collection_",suffix=".json",dir=output)
    with os.fdopen(fd,"w",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)
    print(f"Summary: {counts}; scan errors: {len(errors)}\nReport: {report_path}")
    return 1 if errors or any(r["status"] in ("missing","ambiguous","error") for r in rows.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
