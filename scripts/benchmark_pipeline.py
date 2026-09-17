"""Run reproducible DopplerView execution-profile baselines.

This runner deliberately creates a fresh Pipeline for every sample so context
caches and loaded model instances cannot make later samples look artificially
fast.  Profile order alternates between repetitions to reduce filesystem-cache
and thermal ordering bias.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import matplotlib
import psutil

from dopplerview._version import __version__
from dopplerview.input_output import user_config
from dopplerview.input_output.output_manager import OutputManager
from dopplerview.pipeline.pipeline import Pipeline


PROFILES = ("sequential_reference", "default")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sequential-reference and default pipeline profiles."
    )
    parser.add_argument("input", type=Path, help="A .holo file, batch folder, or input-list file.")
    parser.add_argument("--params", type=Path, help="DopplerView parameter JSON file.")
    parser.add_argument(
        "--config-mode",
        choices=("default", "local"),
        default="default",
        help="Configuration source when --params is omitted.",
    )
    parser.add_argument("--targets", nargs="+", help="Optional target step names.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pipeline_baseline.json"),
        help="Destination for raw benchmark results.",
    )
    parser.add_argument(
        "--diagnostic-images",
        action="store_true",
        help="Also render optional PNG diagnostics (disabled by default).",
    )
    return parser


def _new_pipeline(args: argparse.Namespace, profile: str) -> Pipeline:
    schema_path = user_config.ensure_config_file("h5_schema.json")
    output_config_path = user_config.ensure_config_file("output_config.json")
    manager = OutputManager(
        schema_path=schema_path,
        output_config_path=output_config_path,
        output_enabled=args.diagnostic_images,
    )
    pipeline = Pipeline(output_manager=manager, execution_profile=profile)
    if args.params is not None:
        pipeline.load_dopplerview_config(args.params)
    else:
        pipeline.set_config_mode(args.config_mode)
        if args.config_mode == "default":
            pipeline.load_dopplerview_config(
                user_config.ensure_config_file("default_DV_params.json")
            )
    pipeline.load_input(args.input)
    return pipeline


def _run_sample(args: argparse.Namespace, profile: str, repetition: int) -> dict:
    pipeline = _new_pipeline(args, profile)
    started = time.perf_counter()
    try:
        results = pipeline.run_batch(targets=args.targets)
        wall_seconds = time.perf_counter() - started
        metrics = pipeline.ctx.runtime_metrics.snapshot()
        policy = pipeline.ctx.execution_policy
        return {
            "profile": profile,
            "repetition": repetition,
            "status": "success",
            "wall_seconds": wall_seconds,
            "batch_results": results,
            "execution_policy": policy.describe(),
            "metrics": metrics,
        }
    except BaseException as error:
        return {
            "profile": profile,
            "repetition": repetition,
            "status": "failed",
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "metrics": pipeline.ctx.runtime_metrics.snapshot(),
        }
    finally:
        pipeline.close()


def _profile_order(profiles: list[str], repetition: int) -> list[str]:
    return profiles if repetition % 2 else list(reversed(profiles))


def _summarize(samples: list[dict]) -> dict:
    summary = {}
    for profile in PROFILES:
        successful = [
            sample
            for sample in samples
            if sample["profile"] == profile and sample["status"] == "success"
        ]
        if not successful:
            continue
        step_values: dict[str, dict[str, list[float]]] = {}
        for sample in successful:
            for metric in sample["metrics"]:
                if metric.get("kind") != "step" or metric.get("status") != "success":
                    continue
                values = step_values.setdefault(
                    metric["step"], {"duration_s": [], "process_peak_rss_mb": []}
                )
                values["duration_s"].append(metric["duration_s"])
                values["process_peak_rss_mb"].append(metric["process_peak_rss_mb"])
        summary[profile] = {
            "successful_samples": len(successful),
            "wall_seconds_median": statistics.median(
                sample["wall_seconds"] for sample in successful
            ),
            "steps": {
                step: {
                    "observations": len(values["duration_s"]),
                    "duration_s_median": statistics.median(values["duration_s"]),
                    "process_peak_rss_mb_max": max(values["process_peak_rss_mb"]),
                }
                for step, values in sorted(step_values.items())
            },
        }
    if all(profile in summary for profile in PROFILES):
        default_wall = summary["default"]["wall_seconds_median"]
        if default_wall:
            summary["default_vs_sequential_speedup"] = (
                summary["sequential_reference"]["wall_seconds_median"]
                / default_wall
            )
    return summary


def main() -> int:
    matplotlib.use("Agg")
    args = _parser().parse_args()
    args.input = args.input.resolve()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.params is not None:
        args.params = args.params.resolve()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    samples = []
    for repetition in range(1, args.repetitions + 1):
        for profile in _profile_order(args.profiles, repetition):
            sample = _run_sample(args, profile, repetition)
            samples.append(sample)
            print(
                f"{profile} repetition {repetition}: {sample['status']} "
                f"({sample['wall_seconds']:.3f}s)",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dopplerview_version": __version__,
        "input": str(args.input),
        "params": str(args.params) if args.params is not None else None,
        "config_mode": args.config_mode,
        "targets": args.targets,
        "repetitions": args.repetitions,
        "diagnostic_images": args.diagnostic_images,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "available_cpus": len(psutil.Process().cpu_affinity())
            if hasattr(psutil.Process(), "cpu_affinity")
            else os.cpu_count(),
            "memory_mb": psutil.virtual_memory().total / (1024 * 1024),
        },
        "summary": _summarize(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Raw baseline written to {args.output.resolve()}")
    return 1 if any(sample["status"] != "success" for sample in samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
