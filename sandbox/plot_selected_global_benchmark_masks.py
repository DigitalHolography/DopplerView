"""Plot selected global-benchmark masks in a consistent branch-mapped format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandbox import choroid_benchmark
from sandbox import run_choroid_global_benchmark as global_benchmark


PRIMARY = "heldout_partial_weighted_ARI"
MEASURES = (
    "260310_AUZ0752_4",
    "260622_DUM_R_3",
    "260626_COY_choroid_6",
)
CLASS_COLORS = {
    "artery": np.asarray((1.0, 0.05, 0.05)),
    "vein": np.asarray((0.05, 0.12, 1.0)),
    "aliased_artery": np.asarray((0.05, 0.9, 0.12)),
}
DISPLAY_NAMES = {
    "Best correlation stack": "Three-band correlation + Ward (k=3)",
    "Best one-step": "Complex Fourier + trimmed K-means (k=3)",
    "Best two-step": "Gradient-PCA: GMM → gradient-PCA K-means",
}


def _choose_global_methods(summary_path: Path):
    summary = pd.read_csv(summary_path)

    def best(predicate):
        candidates = summary.loc[predicate(summary)].copy()
        if candidates.empty:
            raise ValueError("no configuration matches a requested gallery category")
        return candidates.sort_values(
            ["primary_rank", f"{PRIMARY}_median"], ascending=[True, False]
        ).iloc[0]

    configuration = summary["configuration"].astype(str)
    correlation = best(lambda _: configuration.str.startswith("correlation_HF_M0_LF / "))
    two_step = best(lambda _: configuration.str.startswith("two_step"))
    one_step = best(
        lambda _: ~configuration.str.startswith("two_step")
        & ~configuration.str.startswith("correlation_HF_M0_LF / ")
    )

    def split(row):
        representation, method = str(row["configuration"]).split(" / ", 1)
        return {
            "representation": representation,
            "method": method,
            "median": float(row[f"{PRIMARY}_median"]),
            "rank": float(row["primary_rank"]),
        }

    return {
        "Best correlation stack": split(correlation),
        "Best one-step": split(one_step),
        "Best two-step": split(two_step),
        "Threshold": {
            "representation": "precomputed_thresholds",
            "method": "threshold_current_pixel_pipeline",
            "median": np.nan,
            "rank": np.nan,
        },
    }


def _safe_key(representation: str, method: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in f"{representation}_{method}"
    )


def _branch_masks(labels, targets, labeled_vessels):
    labels = np.asarray(labels, dtype=int)
    if labels.shape != targets.labels.shape:
        raise ValueError("archived labels and reconstructed branches do not align")
    return {
        class_name: np.isin(
            labeled_vessels,
            targets.branch_ids[labels == class_index],
        )
        for class_index, class_name in enumerate(targets.class_names)
    }


def _overlay(image, masks, alpha=0.78):
    gray = np.asarray(image, dtype=float)
    gray -= np.nanmin(gray)
    maximum = np.nanmax(gray)
    if maximum > 0:
        gray /= maximum
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for class_name, mask in masks.items():
        color = CLASS_COLORS[class_name]
        mask = np.asarray(mask, dtype=bool)
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0, 1)


def _load_archive(benchmark_root, measure, specification, resources):
    archive_path = (
        benchmark_root
        / measure
        / "clusters"
        / _safe_key(specification["representation"], specification["method"])
        / "clusters.npz"
    )
    archive = choroid_benchmark.load_cluster_archive(archive_path)
    if not np.array_equal(archive["branch_ids"], resources.partial_targets.branch_ids):
        raise ValueError(f"branch IDs changed since benchmark creation: {archive_path}")
    labels = archive["semantic_labels"]
    if not len(labels):
        labels = archive["deployment_labels"]
    return _branch_masks(
        labels,
        resources.partial_targets,
        resources.labeled_vessels,
    )


def _prepare_measure(measure_folder: Path, scratch_output: Path):
    parser = global_benchmark.build_parser()
    arguments = parser.parse_args([str(measure_folder.parent), str(scratch_output)])
    files, reason = global_benchmark.discover_measure(measure_folder)
    if files is None:
        raise RuntimeError(f"cannot prepare {measure_folder.name}: {reason}")
    return global_benchmark.prepare_measure(files, arguments)


def make_gallery(dataset_root, benchmark_root, summary_path, output_path):
    selected = _choose_global_methods(summary_path)
    metrics = pd.read_csv(
        benchmark_root / "global_metrics.csv",
        usecols=("measure", "representation", "method", PRIMARY),
    )
    figure, axes = plt.subplots(
        len(MEASURES), 5, figsize=(22, 12.8), constrained_layout=True
    )
    method_items = list(selected.items())

    for row, measure in enumerate(MEASURES):
        resources = _prepare_measure(
            dataset_root / measure,
            output_path.parent / "_unused",
        )
        for column, (heading, specification) in enumerate(method_items):
            masks = _load_archive(
                benchmark_root, measure, specification, resources
            )
            axis = axes[row, column]
            axis.imshow(_overlay(resources.visualization_image, masks))
            score = metrics.loc[
                (metrics["measure"] == measure)
                & (metrics["representation"] == specification["representation"])
                & (metrics["method"] == specification["method"]),
                PRIMARY,
            ]
            if len(score) == 1:
                axis.text(
                    0.02, 0.03, f"weighted ARI = {float(score.iloc[0]):.3f}",
                    transform=axis.transAxes, color="white", fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.58, "pad": 3, "edgecolor": "none"},
                )
            if row == 0:
                if heading == "Threshold":
                    title = "Threshold reference\n(branch mapped)"
                else:
                    title = (
                        f"{heading}\n{DISPLAY_NAMES[heading]}\n"
                        f"global mean rank {specification['rank']:.1f}"
                    )
                axis.set_title(title, fontsize=10.5, weight="bold")
            axis.axis("off")

        ground_truth_masks = _branch_masks(
            resources.partial_targets.labels,
            resources.partial_targets,
            resources.labeled_vessels,
        )
        ground_truth_axis = axes[row, -1]
        ground_truth_axis.imshow(
            _overlay(resources.visualization_image, ground_truth_masks)
        )
        if row == 0:
            ground_truth_axis.set_title(
                "Partial ground truth\n(reliable labels expanded to branches)",
                fontsize=10.5, weight="bold",
            )
        ground_truth_axis.axis("off")
        axes[row, 0].text(
            -0.08, 0.5, measure, transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=12, weight="bold",
        )

    legend = [
        Patch(facecolor=CLASS_COLORS["artery"], label="artery"),
        Patch(facecolor=CLASS_COLORS["vein"], label="vein"),
        Patch(facecolor=CLASS_COLORS["aliased_artery"], label="aliased artery"),
    ]
    figure.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.018),
        ncol=3, frameon=False, fontsize=11,
    )
    figure.suptitle(
        "Branch-mapped comparison of globally selected choroid methods",
        fontsize=17, weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/dataset_choroid"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("D:/global_choroid_benchmark"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "sandbox" / "choroid_presentation_assets"
        / "global_benchmark_configuration_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "sandbox" / "choroid_presentation_assets"
        / "selected_measure_branch_mask_comparison.png",
    )
    args = parser.parse_args(argv)
    selected = make_gallery(
        args.dataset_root.resolve(), args.benchmark_root.resolve(),
        args.summary.resolve(), args.output.resolve(),
    )
    print(f"Saved {args.output.resolve()}")
    for heading, specification in selected.items():
        print(f"{heading}: {specification['representation']} / {specification['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
