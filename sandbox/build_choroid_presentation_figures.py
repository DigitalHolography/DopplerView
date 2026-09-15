"""Build reproducible figures for CHOROID_SEGMENTATION_PRESENTATION.md.

The script deliberately reads benchmark results and one development measure but
never modifies them.  Generated assets are kept under ``sandbox/`` so the
Markdown presentation remains portable with the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandbox import experimental_clustering
from sandbox import run_choroid_global_benchmark as global_benchmark
from sandbox import signal_preprocessing


CLASS_COLORS = {"artery": "#e41a1c", "vein": "#2445ef", "aliased artery": "#23d52c"}
CLUSTER_COLORS = ("#d62728", "#1f77b4", "#9467bd", "#ff7f0e")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        (0.4, 5.4, 2.7, 1.35, "Doppler hologram", "M0 / LF / HF videos", "#e8f1fb"),
        (3.8, 5.4, 2.7, 1.35, "Candidate vessels", "Frangi + retina removal\n+ connected branches", "#f1edf8"),
        (7.2, 5.4, 2.7, 1.35, "Branch signals", "quality-controlled cycles\n+ one template / branch", "#e8f6ee"),
        (10.6, 5.4, 3.0, 1.35, "Embedding + clustering", "1-step or physiology-guided\n+ 2-step partition", "#fff2d9"),
        (10.6, 2.1, 3.0, 1.35, "Semantic pre-masks", "artery / vein /\n+ aliased artery", "#fde8e8"),
        (6.9, 2.1, 2.9, 1.35, "PU evaluation", "partial branch labels\n+ signal plausibility", "#e8f1fb"),
        (3.2, 2.1, 2.9, 1.35, "Future deep model", "shared encoder, class heads\n+ supervised + PU losses", "#eeeeee"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6, edgecolor="#374151", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=10.5)

    def arrow(start, end, *, dashed=False):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=17, linewidth=1.7,
            color="#4b5563", linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0",
        ))

    arrow((3.1, 6.08), (3.8, 6.08))
    arrow((6.5, 6.08), (7.2, 6.08))
    arrow((9.9, 6.08), (10.6, 6.08))
    arrow((12.1, 5.4), (12.1, 3.45))
    arrow((10.6, 2.78), (9.8, 2.78))
    arrow((6.9, 2.78), (6.1, 2.78), dashed=True)
    arrow((4.65, 3.45), (4.65, 5.4), dashed=True)
    ax.text(7.5, 0.8, "Current work: pre-mask generation and validation", ha="center", fontsize=14, weight="bold", color="#9a3412")
    ax.text(4.65, 3.9, "later training loop", ha="center", fontsize=9.5, color="#4b5563")
    ax.set_title("Physiology-guided choroid segmentation strategy", fontsize=19, pad=10, weight="bold")
    _save(fig, path)


def _z_normalize_rows(values):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def make_kshape_centroid_figure(measure_folder, benchmark_visualization, path):
    parser = global_benchmark.build_parser()
    args = parser.parse_args([str(measure_folder.parent), str(path.parent / "_unused")])
    args.temporal_protocol = "alternating"
    files, reason = global_benchmark.discover_measure(measure_folder)
    if files is None:
        raise RuntimeError(reason)
    resources = global_benchmark.prepare_measure(files, args)

    # Legacy visualization archives intentionally contain labels/masks only, so
    # they cannot reconstruct a centroid after the branch pre-mask changes.  Fit
    # a small deterministic demonstration on the currently reconstructed sample.
    # Keep the benchmark path argument in the CLI as a provenance check.
    if not (benchmark_visualization / "labels_and_masks.npz").is_file():
        raise FileNotFoundError(benchmark_visualization / "labels_and_masks.npz")
    templates = _z_normalize_rows(resources.cycle_templates)
    weights = np.asarray(resources.branch_weights, dtype=float)
    labels = experimental_clustering.kshape_cluster(
        templates,
        n_clusters=3,
        sample_weight=weights,
        n_init=3,
        max_iter=30,
        random_state=0,
    )

    cluster_ids = sorted(cluster_id for cluster_id in np.unique(labels) if cluster_id >= 0)
    fig, axes = plt.subplots(len(cluster_ids), 1, figsize=(11.5, 2.75 * len(cluster_ids)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    phase = np.linspace(0, 1, templates.shape[1], endpoint=False)
    for axis, cluster_id in zip(axes, cluster_ids):
        members = templates[labels == cluster_id]
        member_weights = weights[labels == cluster_id]
        reference = members[np.argmax(member_weights)]
        centroid = experimental_clustering._extract_shape(members, reference, member_weights)
        aligned = np.asarray([
            experimental_clustering.shape_based_distance(centroid, member)[1]
            for member in members
        ])
        maximum = 80
        if len(aligned) > maximum:
            selected = np.linspace(0, len(aligned) - 1, maximum, dtype=int)
            aligned = aligned[selected]
        axis.plot(phase, aligned.T, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)], alpha=0.09, linewidth=0.8)
        axis.plot(phase, centroid, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)], linewidth=3.0, label="shape centroid")
        axis.axhline(0, color="#9ca3af", linewidth=0.7)
        axis.set_ylabel(f"cluster {cluster_id}\n$z$-score")
        axis.legend(loc="upper right", frameon=False)
        axis.text(0.01, 0.91, f"n = {len(members)} branches", transform=axis.transAxes, fontsize=10)
    axes[-1].set_xlabel("normalized cardiac-cycle phase")
    fig.suptitle(
        f"K-Shape illustrative refit: aligned branch cycles and extracted centroids\n{measure_folder.name}",
        fontsize=15, weight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def _read_benchmark_tables(benchmark_root):
    csv_paths = sorted(benchmark_root.rglob("*single_sample_benchmark.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no benchmark CSV found under {benchmark_root}")
    tables = []
    for csv_path in csv_paths:
        table = pd.read_csv(csv_path).copy()
        sample = csv_path.name.replace("_choroid_clean_min25_single_sample_benchmark.csv", "")
        table = pd.concat(
            [pd.DataFrame({"sample": np.repeat(sample, len(table))}), table], axis=1
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True), csv_paths


def make_benchmark_summary(benchmark_root, output_dir):
    table, csv_paths = _read_benchmark_tables(benchmark_root)
    table = table.loc[~table["temporal_leakage"].fillna(False).astype(bool)].copy()
    table["configuration"] = table["representation"].astype(str) + " / " + table["method"].astype(str)
    endpoints = {
        "weighted ARI": "heldout_partial_weighted_ARI",
        "weighted macro F1": "heldout_weighted_mapped_macro_f1",
    }
    for column in endpoints.values():
        table[column] = pd.to_numeric(table[column], errors="coerce")

    complete_counts = table.groupby("configuration")["sample"].nunique()
    required = table["sample"].nunique()
    complete = complete_counts[complete_counts == required].index
    table = table[table["configuration"].isin(complete)].copy()
    medians = table.groupby("configuration")[[*endpoints.values()]].median()
    for column in endpoints.values():
        medians[f"rank_{column}"] = medians[column].rank(ascending=False, method="average")
    medians["mean_rank"] = medians[[f"rank_{column}" for column in endpoints.values()]].mean(axis=1)
    leaders = medians.sort_values("mean_rank").head(12).copy()
    leaders.to_csv(output_dir / "benchmark_snapshot_top_methods.csv")

    fig, (axis, key_axis) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": (2.4, 1.15)}
    )
    axis.scatter(
        medians[endpoints["weighted ARI"]], medians[endpoints["weighted macro F1"]],
        s=30, alpha=0.24, color="#64748b", edgecolor="none", label="all complete configurations",
    )
    highlighted = leaders.head(8)
    palette = plt.cm.tab10(np.linspace(0, 1, len(highlighted)))
    leader_key = []
    for number, (color, (name, row)) in enumerate(zip(palette, highlighted.iterrows()), start=1):
        axis.scatter(row[endpoints["weighted ARI"]], row[endpoints["weighted macro F1"]], s=72, color=color, edgecolor="white", linewidth=0.8)
        short = name.replace("correlation_HF_M0_LF", "corr").replace("two_step_gradient_pca3", "2-step grad-PCA").replace("two_step_fourier3", "2-step Fourier")
        axis.annotate(str(number), (row[endpoints["weighted ARI"]], row[endpoints["weighted macro F1"]]), xytext=(5, 4), textcoords="offset points", fontsize=9, weight="bold")
        leader_key.append((number, short, color))
    axis.set_xlabel("Median held-out partial weighted ARI")
    axis.set_ylabel("Median held-out weighted mapped macro F1")
    axis.set_title(f"Exploratory benchmark snapshot ({required} result files)", fontsize=15, weight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    key_axis.axis("off")
    key_axis.text(0, 0.98, "Highlighted configurations", va="top", fontsize=13, weight="bold")
    for row, (number, short, color) in enumerate(leader_key):
        y = 0.91 - row * 0.073
        key_axis.text(0.0, y, str(number), color=color, fontsize=11, weight="bold", va="top")
        key_axis.text(0.09, y, short, fontsize=8.6, va="top", wrap=True)
    _save(fig, output_dir / "benchmark_endpoint_scatter.png")

    top = leaders.head(10).index
    pivot = table[table["configuration"].isin(top)].pivot(index="configuration", columns="sample", values=endpoints["weighted ARI"])
    pivot = pivot.loc[top]
    fig, axis = plt.subplots(figsize=(11.5, 7.2))
    image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=max(0.7, np.nanmax(pivot.to_numpy())), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [name.replace("260", "…260") for name in pivot.columns], rotation=20, ha="right")
    labels = [name.replace("correlation_HF_M0_LF", "corr").replace("two_step_gradient_pca3", "2-step grad-PCA").replace("two_step_fourier3", "2-step Fourier") for name in pivot.index]
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iat[row, column]
            axis.text(column, row, "—" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", color="white" if value < 0.48 else "black", fontsize=9)
    fig.colorbar(image, ax=axis, label="held-out partial weighted ARI")
    axis.set_title("The same method can behave very differently between measures", fontsize=15, weight="bold")
    axis.set_xlabel("measure")
    axis.set_ylabel("configuration")
    _save(fig, output_dir / "benchmark_per_sample_heatmap.png")
    return table, csv_paths


def _load_m0_video(measure_folder):
    h5_paths = sorted(measure_folder.rglob("*.h5"))
    for h5_path in h5_paths:
        try:
            with h5py.File(h5_path, "r") as h5:
                container = h5["doppler_signal"] if "doppler_signal" in h5 else h5
                if "M0_ff" in container:
                    return np.asarray(container["M0_ff"][()]), h5_path
        except OSError:
            continue
    raise FileNotFoundError(f"no HDF5 file containing M0_ff under {measure_folder}")


def _finite_work_signal(raw):
    raw = np.asarray(raw, dtype=float)
    work = raw.copy()
    finite = np.isfinite(raw)
    if not finite.all():
        indices = np.arange(len(raw))
        work[~finite] = np.interp(indices[~finite], indices[finite], raw[finite])
    return work


def _normalized_cycles(cleaned, bounds):
    cycles = np.asarray([cleaned[start:stop] for start, stop in bounds], dtype=float)
    centered = cycles - cycles.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _hard_dtw_path(local_cost, window):
    """Return accumulated hard-DTW cost and its single optimal path."""
    n, m = local_cost.shape
    accumulated = np.full((n, m), np.inf)
    predecessor = np.full((n, m, 2), -1, dtype=int)
    for i in range(n):
        for j in range(max(0, i - window), min(m, i + window + 1)):
            if i == 0 and j == 0:
                accumulated[i, j] = local_cost[i, j]
                continue
            candidates = []
            if i > 0 and np.isfinite(accumulated[i - 1, j]):
                candidates.append((accumulated[i - 1, j], i - 1, j))
            if j > 0 and np.isfinite(accumulated[i, j - 1]):
                candidates.append((accumulated[i, j - 1], i, j - 1))
            if i > 0 and j > 0 and np.isfinite(accumulated[i - 1, j - 1]):
                candidates.append((accumulated[i - 1, j - 1], i - 1, j - 1))
            if candidates:
                cost, previous_i, previous_j = min(candidates)
                accumulated[i, j] = local_cost[i, j] + cost
                predecessor[i, j] = (previous_i, previous_j)
    path = []
    i, j = n - 1, m - 1
    if not np.isfinite(accumulated[i, j]):
        raise ValueError("DTW window excludes every complete alignment path")
    while i >= 0 and j >= 0:
        path.append((i, j))
        previous_i, previous_j = predecessor[i, j]
        if previous_i < 0:
            break
        i, j = int(previous_i), int(previous_j)
    return accumulated, np.asarray(path[::-1], dtype=int)


def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return -np.inf
    maximum = np.max(values[finite])
    return float(maximum + np.log(np.sum(np.exp(values[finite] - maximum))))


def _soft_alignment_occupancy(local_cost, gamma, window):
    """Expected cell occupancy under the Soft-DTW Gibbs path distribution."""
    n, m = local_cost.shape
    log_weight = -local_cost / gamma
    forward = np.full((n, m), -np.inf)
    for i in range(n):
        for j in range(max(0, i - window), min(m, i + window + 1)):
            if i == 0 and j == 0:
                forward[i, j] = log_weight[i, j]
                continue
            predecessors = []
            if i > 0:
                predecessors.append(forward[i - 1, j])
            if j > 0:
                predecessors.append(forward[i, j - 1])
            if i > 0 and j > 0:
                predecessors.append(forward[i - 1, j - 1])
            forward[i, j] = log_weight[i, j] + _logsumexp(predecessors)
    backward = np.full((n, m), -np.inf)
    for i in range(n - 1, -1, -1):
        for j in range(min(m - 1, i + window), max(-1, i - window - 1), -1):
            if i == n - 1 and j == m - 1:
                backward[i, j] = log_weight[i, j]
                continue
            successors = []
            if i + 1 < n:
                successors.append(backward[i + 1, j])
            if j + 1 < m:
                successors.append(backward[i, j + 1])
            if i + 1 < n and j + 1 < m:
                successors.append(backward[i + 1, j + 1])
            backward[i, j] = log_weight[i, j] + _logsumexp(successors)
    log_partition = forward[-1, -1]
    log_occupancy = forward + backward - log_weight - log_partition
    occupancy = np.zeros_like(local_cost)
    finite = np.isfinite(log_occupancy)
    occupancy[finite] = np.exp(np.minimum(log_occupancy[finite], 0.0))
    return occupancy


def make_dtw_comparison_figure(diagnostic_folder, output_dir):
    """Compare hard and soft temporal alignment on real DUM_L_1 cycles."""
    with np.load(diagnostic_folder / "cleaning_arrays.npz") as arrays:
        cleaned = np.asarray(arrays["cleaned_signal"], dtype=float)
        bounds = np.asarray(arrays["cycle_bounds"], dtype=int)
    cycles = _normalized_cycles(cleaned, bounds)
    # Cycle 0 is accepted and cycle 2 is the amplitude-rejected example used in
    # the quality-control section. Standardization isolates their shapes.
    source_phase = np.linspace(0, 1, cycles.shape[1])
    target_phase = np.linspace(0, 1, 32)
    first = np.interp(target_phase, source_phase, cycles[0])
    second = np.interp(target_phase, source_phase, cycles[2])
    local_cost = (first[:, None] - second[None, :]) ** 2
    window = 4
    gamma = 0.1
    accumulated, hard_path = _hard_dtw_path(local_cost, window)
    occupancy = _soft_alignment_occupancy(local_cost, gamma, window)

    fig = plt.figure(figsize=(15, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1, 1), wspace=0.28)
    signal_axis = fig.add_subplot(grid[0, 0])
    hard_axis = fig.add_subplot(grid[0, 1])
    soft_axis = fig.add_subplot(grid[0, 2])
    signal_axis.plot(target_phase, first, color="#16a34a", linewidth=2.2, label="cycle 0 (accepted)")
    signal_axis.plot(target_phase, second, color="#ef4444", linewidth=2.2, label="cycle 2 (amplitude-rejected)")
    signal_axis.set(xlabel="normalized phase", ylabel="standardized M0", title="Two real cycle shapes")
    signal_axis.grid(alpha=0.2)
    signal_axis.legend(frameon=False)

    hard_image = hard_axis.imshow(local_cost, origin="lower", cmap="magma_r", aspect="equal")
    hard_axis.plot(hard_path[:, 1], hard_path[:, 0], color="#00e5ff", linewidth=2.2, label="single minimum-cost path")
    hard_axis.set(xlabel="cycle 2 sample", ylabel="cycle 0 sample", title=f"DTW: one optimal path\ncost = {accumulated[-1, -1]:.2f}")
    hard_axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.colorbar(hard_image, ax=hard_axis, fraction=0.046, pad=0.04, label="local squared difference")

    soft_image = soft_axis.imshow(occupancy, origin="lower", cmap="viridis", aspect="equal", vmin=0, vmax=1)
    soft_axis.set(xlabel="cycle 2 sample", ylabel="cycle 0 sample", title=f"Soft-DTW: expected path occupancy\n$\\gamma$={gamma}, window=±{window}")
    fig.colorbar(soft_image, ax=soft_axis, fraction=0.046, pad=0.04, label="alignment occupancy")
    fig.suptitle("Hard DTW chooses one alignment; Soft-DTW averages plausible alignments", fontsize=15, weight="bold")
    _save(fig, output_dir / "dtw_vs_soft_dtw_alignment.png")


def make_quality_control_figures(measure_folder, diagnostic_folder, output_dir):
    """Visualize every temporal-cleaning decision on one real measure."""
    summary = json.loads((diagnostic_folder / "summary.json").read_text(encoding="utf-8"))
    cycle_table = pd.read_csv(diagnostic_folder / "cycles.csv")
    with np.load(diagnostic_folder / "cleaning_arrays.npz") as arrays:
        raw = np.asarray(arrays["raw_signal"], dtype=float)
        cleaned = np.asarray(arrays["cleaned_signal"], dtype=float)
        artifacts = np.asarray(arrays["frame_artifact_mask"], dtype=bool)
        bounds = np.asarray(arrays["cycle_bounds"], dtype=int)
        valid = np.asarray(arrays["cycle_valid"], dtype=bool)
        valid_cycle_frames = np.asarray(arrays["valid_cycle_frame_mask"], dtype=bool)
        fit_frames = np.asarray(arrays["fit_frame_mask"], dtype=bool)
    frames = np.arange(len(raw))
    initial_period = int(summary["initial_beat_period"])
    beat_period = int(summary["beat_period"])
    sampling_frequency = float(summary["sampling_frequency"])

    # 1. Spatial support used to extract the global reference signal.
    video, _ = _load_m0_video(measure_folder)
    mean_m0 = np.mean(video, axis=0)
    candidate_mask = global_benchmark._notebook_candidate_mask(video)
    low, high = np.percentile(mean_m0[np.isfinite(mean_m0)], (1, 99))
    display = np.clip((mean_m0 - low) / max(high - low, 1e-12), 0, 1)
    overlay = plt.cm.gray(display)[..., :3]
    overlay[candidate_mask] = 0.45 * overlay[candidate_mask] + 0.55 * np.array([1.0, 0.52, 0.0])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axes[0].imshow(display, cmap="gray")
    axes[0].set_title("Temporal mean M0")
    axes[1].imshow(candidate_mask, cmap="gray")
    axes[1].set_title("Candidate-vessel support")
    axes[2].imshow(overlay)
    axes[2].set_title("Pixels averaged for reference signal")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"Step 1 — global candidate-mask reference: {measure_folder.name}", fontsize=15, weight="bold")
    _save(fig, output_dir / "quality_01_reference_mask.png")
    del video

    # 2. Conservative paired-derivative impulse detector.
    work = _finite_work_signal(raw)
    differences = np.diff(work)
    center = np.median(differences)
    scale = signal_preprocessing._robust_scale(differences)
    derivative_z = (differences - center) / max(scale, np.finfo(float).eps)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True, gridspec_kw={"height_ratios": (1.45, 1)})
    axes[0].plot(frames, raw, color="#4b5563", linewidth=1.2)
    axes[0].scatter(frames[artifacts], raw[artifacts], color="#dc2626", marker="x", s=35, label="marked for repair", zorder=3)
    axes[0].set_ylabel("candidate-mask M0")
    axes[0].legend(frameon=False)
    axes[1].plot(frames[1:], derivative_z, color="#2563eb", linewidth=1)
    axes[1].axhline(6, color="#dc2626", linestyle="--", label="±6 robust-z candidate threshold")
    axes[1].axhline(-6, color="#dc2626", linestyle="--")
    axes[1].fill_between(frames[1:], -6, 6, color="#dbeafe", alpha=0.35)
    axes[1].set(xlabel="frame", ylabel="derivative robust z")
    axes[1].legend(frameon=False)
    fig.suptitle("Step 2 — detect short impulse-and-return artifacts", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_02_impulse_detection.png")

    # 3. Interpolation is applied to extracted signals, never to the source video.
    difference = np.abs(raw - cleaned)
    focus = int(np.nanargmax(difference)) if np.any(np.isfinite(difference)) else len(raw) // 2
    left, right = max(0, focus - 28), min(len(raw), focus + 29)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2))
    axes[0].plot(frames, raw, color="0.58", linewidth=1, label="raw reference")
    axes[0].plot(frames, cleaned, color="#2563eb", linewidth=1.45, label="linearly repaired reference")
    axes[0].scatter(frames[artifacts], raw[artifacts], color="#dc2626", marker="x", s=24)
    axes[0].set(xlabel="frame", ylabel="M0", title="Full reference signal")
    axes[0].legend(frameon=False)
    axes[1].plot(frames[left:right], raw[left:right], color="0.45", marker=".", label="raw")
    axes[1].plot(frames[left:right], cleaned[left:right], color="#2563eb", marker=".", label="interpolated")
    axes[1].scatter(frames[left:right][artifacts[left:right]], raw[left:right][artifacts[left:right]], color="#dc2626", marker="x", s=55, zorder=4)
    axes[1].set(xlabel="frame", ylabel="M0", title=f"Largest repaired event (frames {left}–{right - 1})")
    axes[1].legend(frameon=False)
    fig.suptitle("Step 3 — repair only marked samples in extracted temporal signals", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_03_impulse_repair.png")

    # 4. Reproduce the overlap-normalized, detrended autocorrelation used for refinement.
    indices = np.arange(len(cleaned), dtype=float)
    slope, intercept = np.polyfit(indices, cleaned, 1)
    centered = cleaned - (slope * indices + intercept)
    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    autocorrelation /= np.arange(len(centered), 0, -1)
    autocorrelation /= autocorrelation[0]
    lower = max(4, int(round(0.55 * initial_period)))
    upper = min(len(cleaned) - 2, int(round(1.5 * initial_period)))
    lags = np.arange(lower, upper + 1)
    fig, axis = plt.subplots(figsize=(12.5, 5.4))
    axis.plot(lags, autocorrelation[lags], color="#1d4ed8", linewidth=2)
    axis.axvline(initial_period, color="#f59e0b", linestyle="--", linewidth=2, label=f"initial spectral estimate: {initial_period} frames")
    axis.axvline(beat_period, color="#16a34a", linewidth=2.4, label=f"accepted autocorrelation period: {beat_period} frames")
    axis.scatter([beat_period], [autocorrelation[beat_period]], color="#16a34a", s=65, zorder=3)
    axis.text(beat_period + 3, autocorrelation[beat_period], f"r = {autocorrelation[beat_period]:.2f}\n{sampling_frequency / beat_period:.2f} Hz", va="center")
    axis.set(xlabel="lag (frames)", ylabel="detrended autocorrelation", title="Nearby repetition peak corrects a biased short-record spectral estimate")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.suptitle("Step 4 — refine the cardiac period", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_04_period_refinement.png")

    # 5. Transition anchors choose polarity; the final fixed grid maximizes cycle agreement.
    positive, _ = signal_preprocessing._candidate_anchors(cleaned, beat_period, "positive")
    negative, _ = signal_preprocessing._candidate_anchors(cleaned, beat_period, "negative")
    selected_anchors = negative if summary["anchor_polarity"] == "negative" else positive
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.plot(frames, cleaned, color="#2563eb", linewidth=1.4, label="cleaned reference")
    axis.scatter(positive, cleaned[positive], marker="^", color="#f59e0b", s=55, label="positive-gradient anchors")
    axis.scatter(negative, cleaned[negative], marker="v", color="#7c3aed", s=55, label="negative-gradient anchors")
    for cycle_index, ((start, stop), is_valid) in enumerate(zip(bounds, valid)):
        axis.axvspan(start, stop, color=("#22c55e" if is_valid else "#ef4444"), alpha=0.10)
        axis.axvline(start, color="#334155", linewidth=0.8)
        axis.text((start + stop) / 2, axis.get_ylim()[1], f"cycle {cycle_index}", ha="center", va="top", fontsize=9)
    axis.axvline(bounds[-1, 1], color="#334155", linewidth=0.8)
    axis.set(xlabel="frame", ylabel="M0", title=f"Selected polarity: {summary['anchor_polarity']}; fixed-grid phase offset: {summary['phase_offset']} frames")
    axis.legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("Step 5 — select transition polarity and phase-align complete cycles", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_05_cycle_alignment.png")

    # 6. Independent robust criteria decide whether a complete cycle is usable.
    cycle_ids = cycle_table["cycle"].to_numpy(dtype=int)
    colors = np.where(valid, "#22c55e", "#ef4444")
    correlations = cycle_table["shape_correlation"].to_numpy(dtype=float)
    effective_correlation_floor = max(
        0.5,
        float(
            np.median(correlations)
            - 3.5 * signal_preprocessing._robust_scale(correlations)
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.3), sharex=True)
    axes[0].bar(cycle_ids, correlations, color=colors, alpha=0.82)
    axes[0].axhline(
        effective_correlation_floor,
        color="#334155",
        linestyle="--",
        label=f"effective correlation floor = {effective_correlation_floor:.2f}",
    )
    axes[0].set(ylabel="correlation", ylim=(0, 1.05))
    axes[0].legend(frameon=False)
    axes[1].bar(cycle_ids, cycle_table["amplitude_robust_z"], color=colors, alpha=0.82)
    axes[1].axhline(3.5, color="#334155", linestyle="--", label="amplitude |z| limit = 3.5")
    axes[1].axhline(-3.5, color="#334155", linestyle="--")
    axes[1].set(ylabel="amplitude robust z")
    axes[1].legend(frameon=False)
    axes[2].bar(cycle_ids, cycle_table["artifact_fraction"], color=colors, alpha=0.82)
    axes[2].axhline(0.15, color="#334155", linestyle="--", label="artifact-fraction limit = 0.15")
    axes[2].set(xlabel="complete cycle", ylabel="artifact fraction")
    axes[2].set_xticks(cycle_ids)
    axes[2].legend(frameon=False)
    fig.suptitle("Step 6 — reject cycles by shape, amplitude, and artifact burden", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_06_cycle_selection_metrics.png")

    # 7. Accepted-cycle median is the robust branch-template target.
    standardized = _normalized_cycles(cleaned, bounds)
    phase = np.arange(beat_period) / beat_period
    fig, axis = plt.subplots(figsize=(12.5, 5.4))
    for cycle_index, cycle in enumerate(standardized):
        axis.plot(phase, cycle, color=("#22c55e" if valid[cycle_index] else "#ef4444"), alpha=0.48, linewidth=1.5, label=("accepted cycle" if valid[cycle_index] else "rejected cycle"))
    median = np.median(standardized[valid], axis=0)
    axis.plot(phase, median, color="black", linewidth=3, label="median of accepted cycles")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False)
    axis.set(xlabel="normalized cardiac-cycle phase", ylabel="standardized M0", title="Accepted cycles define the robust template used downstream")
    fig.suptitle("Step 7 — standardize cycles and form the median template", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_07_accepted_cycle_template.png")

    # 8. Make the two distinct downstream temporal masks explicit.
    rows = np.vstack((artifacts, valid_cycle_frames, fit_frames)).astype(float)
    fig, axis = plt.subplots(figsize=(14, 3.8))
    axis.imshow(rows, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1, extent=(0, len(raw), 2.5, -0.5))
    axis.set_yticks((0, 1, 2), ("artifact frame", "accepted-cycle frame", "fit frame"))
    axis.set_xlabel("frame")
    axis.set_title("fit frame = accepted-cycle frame AND NOT artifact frame")
    fig.suptitle("Step 8 — expose auditable frame masks for downstream computations", fontsize=15, weight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "quality_08_final_frame_masks.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-folder", type=Path, default=Path("D:/dataset_choroid/260310_AUZ0752_16"))
    parser.add_argument(
        "--kshape-visualization", type=Path,
        default=Path("D:/benchmarks/260310_AUZ0752_16_choroid_clean_min25_visualizations/cycle_templates_kshape_k3"),
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("D:/benchmarks"))
    parser.add_argument(
        "--quality-control-measure-folder", type=Path,
        default=Path("D:/dataset_choroid/260622_DUM_L_1"),
    )
    parser.add_argument(
        "--quality-control-diagnostic", type=Path,
        default=PROJECT_ROOT / "benchmark" / "260622_DUM_L_1_candidate_signal_cleaning",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "sandbox" / "choroid_presentation_assets")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_pipeline_figure(args.output_dir / "pipeline_overview.png")
    make_kshape_centroid_figure(
        args.measure_folder.resolve(), args.kshape_visualization.resolve(),
        args.output_dir / "kshape_real_sample_centroids.png",
    )
    make_quality_control_figures(
        args.quality_control_measure_folder.resolve(),
        args.quality_control_diagnostic.resolve(),
        args.output_dir,
    )
    make_dtw_comparison_figure(
        args.quality_control_diagnostic.resolve(),
        args.output_dir,
    )
    table, csv_paths = make_benchmark_summary(args.benchmark_root.resolve(), args.output_dir)
    print(f"Generated figures in {args.output_dir.resolve()}")
    print(f"Benchmark rows: {len(table)} from {len(csv_paths)} CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
