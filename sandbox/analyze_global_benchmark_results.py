"""Summarize the completed choroid global benchmark for the presentation.

The script is intentionally read-only with respect to benchmark outputs.  It
writes reproducible tables and figures into the presentation asset directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = PROJECT_ROOT / "sandbox" / "choroid_presentation_assets"

PRIMARY = "heldout_partial_weighted_ARI"
OFFLINE = "heldout_mapped_macro_f1"
DEPLOYMENT = "heldout_physiology_macro_f1"
COVERAGE = "heldout_physiology_mapped_coverage"
LABELED_COVERAGE = "heldout_partial_labeled_coverage"


def _configuration(table: pd.DataFrame) -> pd.Series:
    return table["representation"].astype(str) + " / " + table["method"].astype(str)


def _shorten(name: str) -> str:
    return (
        name.replace("correlation_HF_M0_LF", "three-band correlation")
        .replace("two_step_gradient_pca3", "two-step gradient PCA")
        .replace("two_step_fourier3", "two-step complex Fourier")
        .replace("cycle_templates", "cycle template")
        .replace("soft_dtw", "Soft-DTW")
        .replace("agglomerative", "agglom.")
    )


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def load_results(root: Path):
    columns = [
        "measure", "method", "representation", "temporal_leakage", "error",
        PRIMARY, OFFLINE, DEPLOYMENT, COVERAGE, LABELED_COVERAGE,
        "all_partial_branch_count", "all_partial_labeled_branch_count",
        "all_partial_ambiguous_branch_count", "all_partial_unlabeled_branch_count",
    ]
    metrics = pd.read_csv(
        root / "global_metrics.csv", usecols=columns, low_memory=False
    ).copy()
    status = pd.read_csv(root / "global_measure_status.csv")
    metrics["configuration"] = _configuration(metrics)
    for column in (PRIMARY, OFFLINE, DEPLOYMENT, COVERAGE):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    eligible = status.loc[status["status"].isin(("completed", "resumed")), "measure"]
    row_counts = metrics.groupby("measure").size().reindex(eligible, fill_value=0)
    expected = int(row_counts.max())
    exhaustive_measures = row_counts.index[row_counts == expected].tolist()
    incomplete_measures = row_counts.index[row_counts != expected].tolist()

    clean = metrics.loc[
        metrics["measure"].isin(exhaustive_measures)
        & ~metrics["temporal_leakage"].fillna(False).astype(bool)
        & metrics["error"].isna()
    ].copy()
    completeness = clean.groupby("configuration")["measure"].nunique()
    complete_configurations = completeness[completeness == len(exhaustive_measures)].index
    analysis = clean.loc[clean["configuration"].isin(complete_configurations)].copy()
    return metrics, status, row_counts, expected, exhaustive_measures, incomplete_measures, analysis


def summarize(analysis: pd.DataFrame) -> pd.DataFrame:
    grouped = analysis.groupby("configuration")
    summary = grouped[[PRIMARY, OFFLINE, DEPLOYMENT, COVERAGE]].agg(
        ["median", lambda values: values.quantile(0.25), lambda values: values.quantile(0.75)]
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "q25").replace("<lambda_1>", "q75"))

    ranks = analysis.copy()
    ranks["primary_rank"] = ranks.groupby("measure")[PRIMARY].rank(
        ascending=False, method="average"
    )
    ranks["offline_rank"] = ranks.groupby("measure")[OFFLINE].rank(
        ascending=False, method="average"
    )
    rank_summary = ranks.groupby("configuration")[["primary_rank", "offline_rank"]].mean()
    summary = summary.join(rank_summary)
    summary["primary_wins"] = ranks.loc[
        ranks["primary_rank"] == 1
    ].groupby("configuration").size().reindex(summary.index, fill_value=0)
    return summary.sort_values(["primary_rank", f"{PRIMARY}_median"], ascending=[True, False])


def make_support_figure(metrics: pd.DataFrame, status: pd.DataFrame, output_dir: Path) -> None:
    support_columns = {
        "reliably labeled": "all_partial_labeled_branch_count",
        "ambiguous": "all_partial_ambiguous_branch_count",
        "unlabeled": "all_partial_unlabeled_branch_count",
    }
    support = metrics.groupby("measure")[list(support_columns.values())].first()
    support = support.reindex(status.loc[status["status"].isin(("completed", "resumed")), "measure"])
    support = support.sort_index()
    figure, axis = plt.subplots(figsize=(13.5, 7.2))
    bottom = np.zeros(len(support))
    palette = ("#2f855a", "#d69e2e", "#94a3b8")
    for (label, column), color in zip(support_columns.items(), palette):
        values = support[column].to_numpy(float)
        axis.bar(np.arange(len(support)), values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(np.arange(len(support)), support.index, rotation=42, ha="right")
    axis.set_ylabel("candidate branches")
    axis.set_title("Partial-label support across the 15 evaluated measures", weight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    _save(figure, output_dir / "global_benchmark_label_support.png")


def make_endpoint_figure(summary: pd.DataFrame, analysis: pd.DataFrame, output_dir: Path) -> None:
    x = f"{PRIMARY}_median"
    y = f"{OFFLINE}_median"
    top = summary.head(10)
    figure, (axis, key_axis) = plt.subplots(
        1, 2, figsize=(15.5, 7.6), gridspec_kw={"width_ratios": (2.5, 1.35)}
    )
    axis.scatter(summary[x], summary[y], s=28, color="#64748b", alpha=0.28, edgecolor="none")
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    for number, (color, (name, row)) in enumerate(zip(colors, top.iterrows()), start=1):
        axis.scatter(row[x], row[y], s=75, color=color, edgecolor="white", linewidth=0.7)
        axis.annotate(str(number), (row[x], row[y]), xytext=(5, 4), textcoords="offset points", weight="bold")
    axis.set_xlabel("median held-out partial weighted ARI")
    axis.set_ylabel("median held-out mapped macro F1")
    axis.set_title(
        f"Complete-case global benchmark: {analysis['measure'].nunique()} measures, "
        f"{analysis['configuration'].nunique()} configurations",
        weight="bold",
    )
    axis.grid(alpha=0.22)
    key_axis.axis("off")
    key_axis.text(0, 0.99, "Best mean per-measure ARI ranks", va="top", weight="bold", fontsize=12)
    for offset, (name, row) in enumerate(top.iterrows()):
        key_axis.text(0, 0.92 - 0.087 * offset, f"{offset + 1}.", color=colors[offset], weight="bold", va="top")
        key_axis.text(
            0.08, 0.92 - 0.087 * offset,
            f"{_shorten(name)}\nmean rank {row['primary_rank']:.1f}",
            va="top", fontsize=8.2,
        )
    figure.tight_layout()
    _save(figure, output_dir / "global_benchmark_endpoint_scatter.png")


def make_heatmap(summary: pd.DataFrame, analysis: pd.DataFrame, output_dir: Path) -> None:
    top_names = summary.head(10).index
    pivot = analysis.loc[analysis["configuration"].isin(top_names)].pivot(
        index="configuration", columns="measure", values=PRIMARY
    ).reindex(top_names)
    figure, axis = plt.subplots(figsize=(16, 7.8))
    image = axis.imshow(pivot.to_numpy(), vmin=-0.05, vmax=0.75, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=42, ha="right", fontsize=8)
    axis.set_yticks(range(len(pivot.index)), [_shorten(name) for name in pivot.index], fontsize=7.8)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iat[row, column]
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value < 0.43 else "black",
            )
    figure.colorbar(image, ax=axis, label="held-out partial weighted ARI")
    axis.set_title("Top configurations remain heterogeneous between measures", weight="bold")
    axis.set_xlabel("measure")
    axis.set_ylabel("configuration")
    figure.tight_layout()
    _save(figure, output_dir / "global_benchmark_per_measure_heatmap.png")


def make_deployment_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    offline = f"{OFFLINE}_median"
    deployment = f"{DEPLOYMENT}_median"
    coverage = f"{COVERAGE}_median"
    figure, axis = plt.subplots(figsize=(9.2, 7.4))
    points = axis.scatter(
        summary[offline], summary[deployment], c=summary[coverage], cmap="plasma",
        vmin=0, vmax=1, s=38, alpha=0.62, edgecolor="none",
    )
    bounds = [
        np.nanmin([summary[offline].min(), summary[deployment].min()]),
        np.nanmax([summary[offline].max(), summary[deployment].max()]),
    ]
    axis.plot(bounds, bounds, linestyle="--", color="#64748b", linewidth=1, label="equal offline/deployment score")
    axis.set_xlabel("median offline mapped macro F1")
    axis.set_ylabel("median label-free physiology macro F1")
    axis.set_title("Offline cluster naming does not guarantee deployable naming", weight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.colorbar(points, ax=axis, label="median physiology-mapped coverage")
    figure.tight_layout()
    _save(figure, output_dir / "global_benchmark_deployment_gap.png")


def make_threshold_comparison(
    metrics: pd.DataFrame,
    exhaustive_measures: list[str],
    analysis: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare the leaky threshold reference with three clustering landmarks."""
    selected = {
        "Threshold reference\n(temporal leakage)": (
            "precomputed_thresholds", "threshold_current_pixel_pipeline"
        ),
        "PCA + trimmed\nK-means, k=3": ("PCA_M0", "trimmed_kmeans_k3"),
        "Complex Fourier + trimmed\nK-means, k=3": (
            "complex_fourier_M0", "trimmed_kmeans_k3"
        ),
        "Three-band correlation +\nCOP-KMeans, k=3": (
            "correlation_HF_M0_LF", "cop_kmeans_k3"
        ),
    }
    frames = []
    for label, (representation, method) in selected.items():
        source = metrics if representation == "precomputed_thresholds" else analysis
        rows = source.loc[
            source["measure"].isin(exhaustive_measures)
            & (source["representation"] == representation)
            & (source["method"] == method)
        ].copy()
        rows["comparison_method"] = label
        frames.append(rows)
    comparison = pd.concat(frames, ignore_index=True)

    metrics_to_plot = {
        "Weighted ARI\n(partition)": PRIMARY,
        "Direct/physiology macro F1\n(semantic labels)": DEPLOYMENT,
        "Held-out labeled coverage": LABELED_COVERAGE,
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharex=True)
    rng = np.random.default_rng(0)
    labels = list(selected)
    palette = ("#d97706", "#2563eb", "#7c3aed", "#059669")
    for axis, (title, column) in zip(axes, metrics_to_plot.items()):
        values = [
            comparison.loc[comparison["comparison_method"] == label, column]
            .dropna().to_numpy(float)
            for label in labels
        ]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
        for patch, color in zip(boxes["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for position, (group, color) in enumerate(zip(values, palette), start=1):
            jitter = rng.uniform(-0.10, 0.10, len(group))
            axis.scatter(position + jitter, group, color=color, s=24, alpha=0.75, edgecolor="none")
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right", fontsize=8)
        axis.set_title(title, weight="bold")
        axis.set_ylim(-0.08 if column == PRIMARY else -0.02, 1.03)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Threshold pipeline versus selected clustering landmarks\n"
        "Threshold values are descriptive only: temporal leakage flag is set",
        weight="bold", fontsize=15,
    )
    figure.tight_layout()
    _save(figure, output_dir / "global_benchmark_threshold_comparison.png")

    summary_rows = []
    for label in labels:
        group = comparison.loc[comparison["comparison_method"] == label]
        row = {"comparison_method": label.replace("\n", " "), "measure_count": group["measure"].nunique()}
        for column in metrics_to_plot.values():
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_q25"] = group[column].quantile(0.25)
            row[f"{column}_q75"] = group[column].quantile(0.75)
        summary_rows.append(row)
    result = pd.DataFrame(summary_rows)
    result.to_csv(output_dir / "global_benchmark_threshold_comparison.csv", index=False)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path, nargs="?", default=Path("D:/global_choroid_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args(argv)
    root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, status, row_counts, expected, exhaustive, incomplete, analysis = load_results(root)
    summary = summarize(analysis)
    summary.to_csv(output_dir / "global_benchmark_configuration_summary.csv")
    row_counts.rename("result_rows").to_csv(output_dir / "global_benchmark_measure_row_counts.csv")
    make_support_figure(metrics, status, output_dir)
    make_endpoint_figure(summary, analysis, output_dir)
    make_heatmap(summary, analysis, output_dir)
    make_deployment_figure(summary, output_dir)
    threshold_summary = make_threshold_comparison(
        metrics, exhaustive, analysis, output_dir
    )

    print(f"Eligible measures: {len(row_counts)}")
    print(f"Expected rows per exhaustive measure: {expected}")
    print(f"Exhaustive measures ({len(exhaustive)}): {', '.join(exhaustive)}")
    print(f"Incomplete measures ({len(incomplete)}): {', '.join(incomplete)}")
    print(f"Complete non-leaking configurations: {analysis['configuration'].nunique()}")
    print(summary.head(12)[[
        f"{PRIMARY}_median", f"{PRIMARY}_q25", f"{PRIMARY}_q75", "primary_rank",
        f"{OFFLINE}_median", f"{DEPLOYMENT}_median", f"{COVERAGE}_median", "primary_wins",
    ]].to_string())
    print("\nThreshold comparison:\n" + threshold_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
