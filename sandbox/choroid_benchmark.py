"""Leakage-aware, single-sample benchmark for choroidal branch clustering."""

from dataclasses import dataclass, replace
import logging
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dopplerview.segmentation import clustering, signal_processing

from .clustering_stability import run_resampled_clustering_stability
from .evaluation import evaluate_partial_branch_clustering
from .experimental_clustering import (
    cop_kmeans_cluster,
    kshape_cluster,
    soft_dtw_kmedoids_cluster,
)
from .partial_branch_evaluation import UNLABELED
from .signal_evaluation import evaluate_mask_signal_similarity


logger = logging.getLogger(__name__)

CLUSTER_COLORS = (
    "tab:red",
    "tab:blue",
    "tab:purple",
    "tab:green",
    "tab:orange",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)


@dataclass(frozen=True)
class SingleSampleBenchmarkResult:
    table: pd.DataFrame
    cluster_labels: dict
    mapped_class_labels: dict
    constraint_branch_ids: np.ndarray
    evaluation_branch_ids: np.ndarray
    csv_path: Path | None = None
    physiology_mapped_class_labels: dict | None = None


@dataclass(frozen=True)
class CorrelationPhysiologyMapping:
    """Diagnostic result of label-free correlation-profile assignment."""

    mapped_labels: np.ndarray
    cluster_ids: np.ndarray
    class_names: tuple
    cluster_centroids: np.ndarray
    class_prototypes: np.ndarray
    similarity_matrix: np.ndarray
    cluster_to_class: dict


@dataclass(frozen=True)
class BenchmarkEmbeddingView:
    """An embedding/partition pair to include in a saved method diagnostic."""

    name: str
    X: np.ndarray
    cluster_labels: np.ndarray
    component_names: tuple | None = None
    partial_labels: np.ndarray | None = None
    masks: tuple = ()


@dataclass(frozen=True)
class BenchmarkPrediction:
    """Rich result returned by a composite benchmark method.

    ``cluster_labels`` must be aligned with the original branch map.  A
    two-step method may additionally return its label-free semantic decision
    in ``deployment_labels``, its stage-specific embeddings for plotting, and
    its native pixel-level ``class_masks``. Native masks are preferred for
    visualization and signal evaluation; branch-aligned labels remain the
    common representation for partial-label metrics.
    """

    cluster_labels: np.ndarray
    deployment_labels: np.ndarray | None = None
    embedding_views: tuple = ()
    class_masks: dict | None = None


@dataclass(frozen=True)
class BenchmarkCandidate:
    """A custom method evaluated alongside the built-in one-step methods."""

    name: str
    representation: str
    X: np.ndarray | None
    run: object
    subsample_safe: bool = True
    temporal_leakage: bool = False
    component_names: tuple | None = None


def stratified_partial_label_split(targets, *, constraint_fraction=0.5, random_state=0):
    """Split known branch labels per class into constraint and evaluation sets."""
    if not 0 < constraint_fraction < 1:
        raise ValueError("constraint_fraction must lie in (0, 1)")
    rng = np.random.default_rng(random_state)
    constraint = np.zeros(len(targets.branch_ids), dtype=bool)
    evaluation = np.zeros(len(targets.branch_ids), dtype=bool)
    for class_label in range(len(targets.class_names)):
        indices = np.flatnonzero(targets.labels == class_label)
        indices = rng.permutation(indices)
        if len(indices) == 1:
            constraint[indices] = True
            continue
        if len(indices) > 1:
            count = int(round(constraint_fraction * len(indices)))
            count = min(max(count, 1), len(indices) - 1)
            constraint[indices[:count]] = True
            evaluation[indices[count:]] = True
    if np.count_nonzero(evaluation) < 2:
        raise ValueError(
            "partial labels do not leave two held-out branches after a stratified split"
        )
    return constraint, evaluation


def constraints_from_partial_targets(targets, selected):
    """Build pairwise sample-index constraints from selected reliable targets."""
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != targets.labels.shape:
        raise ValueError("selected must contain one value per branch")
    indices = np.flatnonzero(selected & targets.labeled_mask)
    must_link = []
    cannot_link = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            destination = (
                must_link
                if targets.labels[left] == targets.labels[right]
                else cannot_link
            )
            destination.append((int(left), int(right)))
    return np.asarray(must_link, dtype=int).reshape(-1, 2), np.asarray(
        cannot_link, dtype=int
    ).reshape(-1, 2)


def _masked_targets(targets, selected):
    labels = np.full_like(targets.labels, UNLABELED)
    labels[selected & targets.labeled_mask] = targets.labels[selected & targets.labeled_mask]
    confidence = np.where(labels >= 0, targets.confidence, 0.0)
    return replace(targets, labels=labels, confidence=confidence)


def map_clusters_to_classes(cluster_labels, targets, fit_mask, sample_weight=None):
    """One-to-one Hungarian cluster alignment using training labels only.

    The score of a cluster/class pair is the sum of branch weights multiplied
    by partial-label confidence.  Only positive-evidence assignments are kept;
    noise, clusters without evidence, and surplus clusters remain ``UNLABELED``.
    This function is intended for evaluation alignment, not deployment.
    """
    cluster_labels = np.asarray(cluster_labels)
    fit_mask = np.asarray(fit_mask, dtype=bool) & targets.labeled_mask
    if cluster_labels.shape != targets.labels.shape or fit_mask.shape != targets.labels.shape:
        raise ValueError("cluster labels, fit mask, and targets must align")
    weights = (
        np.ones(len(cluster_labels), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weights.shape != cluster_labels.shape:
        raise ValueError("sample_weight must contain one value per branch")
    cluster_ids = np.unique(cluster_labels[cluster_labels >= 0])
    mapped = np.full(len(cluster_labels), UNLABELED, dtype=int)
    if not len(cluster_ids):
        return mapped
    scores = np.zeros((len(cluster_ids), len(targets.class_names)), dtype=float)
    for row, cluster_id in enumerate(cluster_ids):
        evidence = fit_mask & (cluster_labels == cluster_id)
        if not np.any(evidence):
            continue
        scores[row] = np.bincount(
            targets.labels[evidence],
            weights=weights[evidence] * targets.confidence[evidence],
            minlength=len(targets.class_names),
        )
    rows, classes = linear_sum_assignment(-scores)
    for row, class_label in zip(rows, classes):
        if scores[row, class_label] > 0:
            mapped[cluster_labels == cluster_ids[row]] = int(class_label)
    return mapped


def _weighted_feature_median(values, weights):
    """Return a feature-wise weighted median."""
    medians = np.empty(values.shape[1], dtype=float)
    for feature in range(values.shape[1]):
        order = np.argsort(values[:, feature], kind="stable")
        ordered_values = values[order, feature]
        ordered_weights = weights[order]
        cutoff = 0.5 * np.sum(ordered_weights)
        medians[feature] = ordered_values[
            np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")
        ]
    return medians


def map_clusters_by_correlation_physiology(
    cluster_labels,
    correlation_features,
    *,
    sample_weight=None,
    class_names=("artery", "vein", "aliased_artery"),
    class_prototypes=None,
    feature_weights=(2.0, 1.0, 1.0),
    min_similarity=0.0,
    min_profile_norm=0.1,
):
    """Name anonymous clusters without choroidal labels using HF/M0/LF signs.

    Rows of ``correlation_features`` must be raw Pearson correlations ordered
    as HF, M0, and LF relative to the retinal arterial signal.  The default
    prototypes encode provisional physiological hypotheses::

        artery          (+1, +1, +1)
        vein             (0, -1, -1)
        aliased artery   (-1, -1, -1)

    Cluster profiles are feature-wise weighted medians.  Weighted cosine
    similarities to the prototypes form a rectangular assignment matrix, and
    Hungarian matching selects at most one cluster per class.  Surplus clusters,
    matches below ``min_similarity``, and near-zero profiles below
    ``min_profile_norm`` remain ``UNLABELED``.

    These prototypes are deliberately configurable: they are a testable
    physiological prior, not a learned or clinically validated classifier.
    """
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    correlation_features = np.asarray(correlation_features, dtype=float)
    if correlation_features.ndim != 2 or correlation_features.shape[1] != 3:
        raise ValueError("correlation_features must have shape (n_branches, 3)")
    if cluster_labels.shape != (len(correlation_features),):
        raise ValueError("cluster_labels and correlation_features must align")
    if not np.all(np.isfinite(correlation_features)):
        raise ValueError("correlation_features must be finite raw correlations")

    weights = (
        np.ones(len(cluster_labels), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weights.shape != cluster_labels.shape:
        raise ValueError("sample_weight must contain one value per branch")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sample_weight must contain positive finite values")

    class_names = tuple(class_names)
    if class_prototypes is None:
        default_prototypes = {
            "artery": (1.0, 1.0, 1.0),
            "vein": (0.0, -1.0, -1.0),
            "aliased_artery": (-1.0, -1.0, -1.0),
        }
        try:
            class_prototypes = [default_prototypes[name] for name in class_names]
        except KeyError as error:
            raise ValueError(
                "custom class_names require matching class_prototypes"
            ) from error
    prototypes = np.asarray(class_prototypes, dtype=float)
    if prototypes.shape != (len(class_names), 3):
        raise ValueError("class_prototypes must have shape (n_classes, 3)")
    feature_weights = np.asarray(feature_weights, dtype=float)
    if feature_weights.shape != (3,) or np.any(feature_weights < 0):
        raise ValueError("feature_weights must contain three non-negative values")
    if not np.any(feature_weights > 0):
        raise ValueError("at least one feature weight must be positive")
    if not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must lie in [-1, 1]")
    if not np.isfinite(min_profile_norm) or min_profile_norm < 0:
        raise ValueError("min_profile_norm must be finite and non-negative")

    cluster_ids = np.unique(cluster_labels[cluster_labels >= 0])
    mapped = np.full(len(cluster_labels), UNLABELED, dtype=int)
    if not len(cluster_ids):
        empty = np.empty((0, 3), dtype=float)
        return CorrelationPhysiologyMapping(
            mapped, cluster_ids, class_names, empty, prototypes, empty, {}
        )

    centroids = np.vstack(
        [
            _weighted_feature_median(
                correlation_features[cluster_labels == cluster_id],
                weights[cluster_labels == cluster_id],
            )
            for cluster_id in cluster_ids
        ]
    )
    scale = np.sqrt(feature_weights)
    weighted_centroids = centroids * scale
    weighted_prototypes = prototypes * scale
    centroid_norms = np.linalg.norm(weighted_centroids, axis=1, keepdims=True)
    prototype_norms = np.linalg.norm(weighted_prototypes, axis=1, keepdims=True).T
    denominator = centroid_norms * prototype_norms
    similarities = np.divide(
        weighted_centroids @ weighted_prototypes.T,
        denominator,
        out=np.full((len(cluster_ids), len(class_names)), -1.0),
        where=denominator > np.finfo(float).eps,
    )

    rows, classes = linear_sum_assignment(-similarities)
    cluster_to_class = {}
    for row, class_label in zip(rows, classes):
        if (
            similarities[row, class_label] >= min_similarity
            and centroid_norms[row, 0] >= min_profile_norm
        ):
            cluster_id = int(cluster_ids[row])
            mapped[cluster_labels == cluster_id] = int(class_label)
            cluster_to_class[cluster_id] = class_names[class_label]
    return CorrelationPhysiologyMapping(
        mapped,
        cluster_ids,
        class_names,
        centroids,
        prototypes,
        similarities,
        cluster_to_class,
    )


def _predicted_masks(mapped_labels, targets, labeled_vessels):
    masks = {}
    for class_label, class_name in enumerate(targets.class_names):
        branch_ids = targets.branch_ids[mapped_labels == class_label]
        masks[class_name] = np.isin(labeled_vessels, branch_ids)
    return masks


def _validated_class_masks(class_masks, targets, labeled_vessels):
    """Validate method-native semantic masks without changing their support."""
    if class_masks is None:
        return None
    labeled_vessels = np.asarray(labeled_vessels)
    masks = {}
    for class_name in targets.class_names:
        if class_name not in class_masks:
            raise ValueError(f"class_masks is missing {class_name!r}")
        mask = np.asarray(class_masks[class_name], dtype=bool)
        if mask.shape != labeled_vessels.shape:
            raise ValueError(
                "class_masks and labeled_vessels must have matching spatial shapes"
            )
        masks[class_name] = mask
    overlap_count = np.sum(list(masks.values()), axis=0)
    if np.any(overlap_count > 1):
        raise ValueError("class_masks must be mutually exclusive")
    return masks


def class_masks_to_branch_labels(
    labeled_vessels,
    class_masks,
    class_names=("artery", "vein", "aliased_artery"),
    *,
    min_overlap_fraction=0.5,
):
    """Convert pixel class masks into labels aligned with the original branches.

    This is useful for two-step pipelines, whose second stage relabels a subset
    of the vessel tree.  A branch is assigned only when its best class covers
    at least ``min_overlap_fraction`` of its pixels; unresolved branches remain
    ``UNLABELED`` instead of being forced into a class.
    """
    labeled_vessels = np.asarray(labeled_vessels)
    if labeled_vessels.ndim != 2:
        raise ValueError("labeled_vessels must be a 2-D label image")
    if not 0 <= min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must lie in [0, 1]")
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    masks = []
    for class_name in class_names:
        if class_name not in class_masks:
            raise ValueError(f"missing class mask {class_name!r}")
        mask = np.asarray(class_masks[class_name], dtype=bool)
        if mask.shape != labeled_vessels.shape:
            raise ValueError("class masks and labeled_vessels must have the same shape")
        masks.append(mask)
    fractions = np.asarray(
        [
            [np.mean(mask[labeled_vessels == branch_id]) for mask in masks]
            for branch_id in branch_ids
        ]
    )
    labels = np.argmax(fractions, axis=1).astype(int)
    labels[np.max(fractions, axis=1) < min_overlap_fraction] = UNLABELED
    return labels


def _safe_output_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "method"


def _embedding_projection(X, component_names=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X):
        raise ValueError("visualized embeddings must be non-empty 2-D arrays")
    component_names = None if component_names is None else tuple(component_names)
    if component_names is not None and len(component_names) != X.shape[1]:
        raise ValueError("component_names must name every embedding component")
    if X.shape[1] == 1:
        name = component_names[0] if component_names is not None else "component 1"
        return np.column_stack([X[:, 0], np.zeros(len(X))]), (name, ""), False
    if X.shape[1] == 2:
        names = component_names or ("component 1", "component 2")
        return X, names, False
    if X.shape[1] == 3:
        names = component_names or ("component 1", "component 2", "component 3")
        return X, names, True
    pca = PCA(n_components=3).fit(X)
    projected = pca.transform(X)
    names = tuple(
        f"visualization PC{index + 1} ({100 * ratio:.1f}%)"
        for index, ratio in enumerate(pca.explained_variance_ratio_)
    )
    return projected, names, True


def _normalized_rgb(image):
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    else:
        raise ValueError("visualization_image must be 2-D grayscale or RGB(A)")
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=float)
    low, high = np.nanpercentile(image[finite], (1, 99))
    if high <= low:
        high = low + 1.0
    return np.clip((np.nan_to_num(image, nan=low) - low) / (high - low), 0, 1)


def _overlay_masks(base_rgb, masks, colors):
    overlay = np.array(base_rgb, copy=True)
    for mask, color in zip(masks, colors):
        overlay[np.asarray(mask, dtype=bool)] = color
    return overlay


def _atomic_save_figure(figure, path, *, dpi=140):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        figure.savefig(temporary_path, format=path.suffix.lstrip("."), dpi=dpi)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        figure.clear()


def _plot_partition(
    axis,
    coordinates,
    labels,
    axis_names,
    *,
    label_names=None,
    semantic=False,
):
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (len(coordinates),):
        raise ValueError("plot labels and embedding rows must align")
    semantic_colors = ("tab:red", "tab:blue", "tab:green")
    for label in np.unique(labels):
        if semantic and label < 0:
            continue
        selected = labels == label
        color = (
            "0.65"
            if label < 0
            else semantic_colors[label % len(semantic_colors)]
            if semantic
            else CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
        )
        label_name = (
            "unassigned/noise"
            if label < 0
            else label_names[label].replace("_", " ")
            if semantic and label_names is not None and label < len(label_names)
            else f"cluster {label}"
        )
        if coordinates.shape[1] == 3:
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                coordinates[selected, 2],
                color=color,
                s=18,
                label=label_name,
            )
        else:
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                color=color,
                s=18,
                label=label_name,
            )
    axis.set_xlabel(axis_names[0] if axis_names else "")
    if len(axis_names) >= 2:
        axis.set_ylabel(axis_names[1])
    if coordinates.shape[1] == 3:
        axis.set_zlabel(axis_names[2])
    if len(axis.collections):
        axis.legend(fontsize="small")
    axis.grid(alpha=0.3)


def _cycle_template(video, mask, sampling_frequency, beat_period):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    pulse = signal_processing.get_pulse_from_mask(video, mask)
    pulse = signal_processing.get_filtered_pulse(pulse, sampling_frequency)
    cycle_count = len(pulse) // beat_period
    if cycle_count == 0:
        return None
    template = np.median(
        pulse[: cycle_count * beat_period].reshape(cycle_count, beat_period), axis=0
    )
    template = template - np.mean(template)
    scale = np.std(template)
    if scale <= np.finfo(float).eps:
        return np.zeros_like(template)
    return template / scale


def _save_clustering_view(view, path, targets, title):
    """Save one stage's cluster/partial-label comparison."""
    coordinates, axis_names, use_3d = _embedding_projection(
        view.X, view.component_names
    )
    projection = "3d" if use_3d else None
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    figure.suptitle(title)
    predicted_axis = figure.add_subplot(1, 2, 1, projection=projection)
    partial_axis = figure.add_subplot(1, 2, 2, projection=projection)
    _plot_partition(
        predicted_axis,
        coordinates,
        view.cluster_labels,
        axis_names,
    )
    predicted_axis.set_title("Anonymous clusters")
    partial_labels = view.partial_labels
    if partial_labels is None and len(coordinates) == len(targets.labels):
        partial_labels = targets.labels
    if partial_labels is None:
        partial_axis.text(
            0.5,
            0.5,
            "No aligned partial labels",
            ha="center",
            va="center",
            transform=partial_axis.transAxes,
        )
    else:
        _plot_partition(
            partial_axis,
            coordinates,
            partial_labels,
            axis_names,
            label_names=targets.class_names,
            semantic=True,
        )
    partial_axis.set_title("Partial labels")
    _atomic_save_figure(figure, path)


def _save_stage_masks(view, path, title):
    """Save one binary panel for each intermediate mask of a pipeline stage."""
    masks = tuple(view.masks)
    if not masks:
        raise ValueError("a staged embedding view must provide masks")
    figure = Figure(figsize=(5 * len(masks), 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    figure.suptitle(title)
    axes = figure.subplots(1, len(masks), squeeze=False)[0]
    expected_shape = None
    for axis, (name, mask) in zip(axes, masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("stage masks must be 2-D")
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError("all masks in one stage must have the same shape")
        axis.imshow(mask, cmap="gray")
        axis.set_title(str(name).replace("_", " "))
        axis.axis("off")
    _atomic_save_figure(figure, path)


def _ordinal_step(index):
    if index == 1:
        return "1st_step"
    if index == 2:
        return "2nd_step"
    if index == 3:
        return "3rd_step"
    return f"{index}th_step"


def _save_method_artifacts(
    output_directory,
    key,
    prediction,
    candidate,
    semantic_labels,
    targets,
    labeled_vessels,
    visualization_image,
    partial_reference_masks,
    signal_videos,
    sampling_frequency,
    beat_period,
):
    """Save gallery-compatible figures and arrays without using pyplot."""
    method_directory = Path(output_directory) / _safe_output_name(key)
    method_directory.mkdir(parents=True, exist_ok=True)

    cluster_labels = prediction.cluster_labels
    predicted_masks = prediction.class_masks
    if predicted_masks is None:
        predicted_masks = _predicted_masks(
            semantic_labels, targets, labeled_vessels
        )
    temporary_arrays = method_directory / "labels_and_masks.npz.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(
            stream,
            branch_ids=targets.branch_ids,
            cluster_labels=cluster_labels,
            semantic_labels=semantic_labels,
            **{f"mask_{name}": mask for name, mask in predicted_masks.items()},
        )
    temporary_arrays.replace(method_directory / "labels_and_masks.npz")

    views = tuple(prediction.embedding_views)
    if not views and candidate.X is not None:
        views = (
            BenchmarkEmbeddingView(
                candidate.representation,
                candidate.X,
                cluster_labels,
                component_names=candidate.component_names,
                partial_labels=targets.labels,
            ),
        )
    if not views:
        raise ValueError("at least one embedding view is required for visualization")

    is_staged = len(views) > 1 and all(view.masks for view in views)
    if is_staged:
        for step_index, view in enumerate(views, start=1):
            step_directory = method_directory / _ordinal_step(step_index)
            step_directory.mkdir(parents=True, exist_ok=True)
            _save_clustering_view(
                view,
                step_directory / "clusters.png",
                targets,
                f"{key} - {view.name}",
            )
            _save_stage_masks(
                view,
                step_directory / "masks.png",
                f"{key} - {view.name}",
            )
        overlay_path = method_directory / "final_overlays.png"
        signal_path = method_directory / "signals.png"
    else:
        _save_clustering_view(
            views[0], method_directory / "clustering.png", targets, key
        )
        overlay_path = method_directory / "mask_overlays.png"
        signal_path = method_directory / "signal_comparisons.png"

    # Figure 2: predicted semantic masks beside the sparse partial annotations.
    base_rgb = _normalized_rgb(visualization_image)
    semantic_colors = np.asarray(
        [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    )
    predicted_overlay = _overlay_masks(
        base_rgb,
        [predicted_masks[name] for name in targets.class_names],
        semantic_colors,
    )
    reference_overlay = _overlay_masks(
        base_rgb,
        [partial_reference_masks[name] for name in targets.class_names],
        semantic_colors,
    )
    overlay_figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(overlay_figure)
    overlay_figure.suptitle(key)
    predicted_axis, reference_axis = overlay_figure.subplots(1, 2)
    predicted_axis.imshow(predicted_overlay)
    predicted_axis.set_title("Predicted masks")
    predicted_axis.axis("off")
    reference_axis.imshow(reference_overlay)
    reference_axis.set_title("Partial ground truth")
    reference_axis.axis("off")
    _atomic_save_figure(overlay_figure, overlay_path)

    # Figure 3: predicted and partial-reference cycle templates for every band.
    signal_figure = Figure(
        figsize=(5 * len(signal_videos), 3 * len(targets.class_names)),
        constrained_layout=True,
    )
    FigureCanvasAgg(signal_figure)
    signal_axes = signal_figure.subplots(
        len(targets.class_names), len(signal_videos), squeeze=False,
        sharex=True, sharey=True,
    )
    for row, class_name in enumerate(targets.class_names):
        for column, (band_name, video) in enumerate(signal_videos.items()):
            axis = signal_axes[row, column]
            predicted_template = _cycle_template(
                video, predicted_masks[class_name], sampling_frequency, beat_period
            )
            reference_template = _cycle_template(
                video,
                partial_reference_masks[class_name],
                sampling_frequency,
                beat_period,
            )
            if predicted_template is not None:
                axis.plot(predicted_template, label="predicted", linewidth=2)
            if reference_template is not None:
                axis.plot(
                    reference_template,
                    label="partial reference",
                    linestyle="--",
                )
            axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
            axis.set_title(f"{class_name.replace('_', ' ')} - {band_name}")
            if row == len(targets.class_names) - 1:
                axis.set_xlabel("cardiac-cycle sample")
            if column == 0:
                axis.set_ylabel("standardized amplitude")
    signal_axes[0, 0].legend()
    signal_figure.suptitle(f"{key}: median class signals")
    _atomic_save_figure(signal_figure, signal_path)

    obsolete_files = [method_directory / "diagnostic.png"]
    if is_staged:
        obsolete_files.extend(
            [
                method_directory / "clustering.png",
                method_directory / "mask_overlays.png",
                method_directory / "signal_comparisons.png",
            ]
        )
    for obsolete_file in obsolete_files:
        obsolete_file.unlink(missing_ok=True)
    return method_directory


def _candidate_set(
    embeddings,
    temporal_signals,
    threshold_labelings,
    cluster_counts,
    must_link,
    cannot_link,
    include_adaptive,
    branch_weights,
    soft_dtw_max_pairwise_samples,
    soft_dtw_max_template_length,
    soft_dtw_window,
):
    candidates = []
    for representation, values in embeddings.items():
        values = np.asarray(values, dtype=float)
        for count in cluster_counts:
            candidates.extend(
                [
                    BenchmarkCandidate(
                        f"kmeans_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, sample_weight=None, count=count: KMeans(
                            n_clusters=count, n_init=20, random_state=random_state
                        ).fit(X, sample_weight=sample_weight).labels_,
                    ),
                    BenchmarkCandidate(
                        f"gmm_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, count=count, **_: GaussianMixture(
                            n_components=count, n_init=5, random_state=random_state
                        ).fit_predict(X),
                    ),
                    *[
                        BenchmarkCandidate(
                            f"agglomerative_{linkage}_k{count}",
                            representation,
                            values,
                            lambda X, count=count, linkage=linkage, **_: AgglomerativeClustering(
                                n_clusters=count,
                                linkage=linkage,
                                metric="euclidean",
                            ).fit_predict(X),
                        )
                        for linkage in ("ward", "average", "complete", "single")
                    ],
                    BenchmarkCandidate(
                        f"trimmed_kmeans_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, count=count, **_: clustering.trimmed_kmeans_cluster(
                            X,
                            n_clusters=count,
                            trim_fraction=0.05,
                            random_state=random_state,
                        ),
                    ),
                    BenchmarkCandidate(
                        f"cop_kmeans_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, sample_weight=None, count=count: cop_kmeans_cluster(
                            X,
                            n_clusters=count,
                            must_link=must_link,
                            cannot_link=cannot_link,
                            sample_weight=sample_weight,
                            random_state=random_state,
                        ),
                        subsample_safe=False,
                    ),
                ]
            )
            if branch_weights is not None:
                candidates.extend(
                    [
                        BenchmarkCandidate(
                            f"weighted_kmeans_k{count}",
                            representation,
                            values,
                            lambda X, random_state=0, sample_weight=None, count=count: KMeans(
                                n_clusters=count, n_init=20, random_state=random_state
                            ).fit(X, sample_weight=sample_weight).labels_,
                        ),
                        BenchmarkCandidate(
                            f"weighted_agglomerative_ward_k{count}",
                            representation,
                            values,
                            lambda X, sample_weight=None, count=count, **_: clustering.weighted_agglomerative_cluster(
                                X, n_clusters=count, sample_weight=sample_weight
                            ),
                        ),
                        BenchmarkCandidate(
                            f"trimmed_weighted_kmeans_k{count}",
                            representation,
                            values,
                            lambda X, random_state=0, sample_weight=None, count=count: clustering.trimmed_kmeans_cluster(
                                X,
                                n_clusters=count,
                                trim_fraction=0.05,
                                sample_weight=sample_weight,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
        if include_adaptive:
            candidates.extend(
                [
                    BenchmarkCandidate(
                        "bayesian_gmm_adaptive",
                        representation,
                        values,
                        lambda X, random_state=0, **_: clustering.bayesian_gmm_cluster(
                            X, random_state=random_state
                        ),
                    ),
                    BenchmarkCandidate(
                        "hdbscan_adaptive",
                        representation,
                        values,
                        lambda X, **_: clustering.hdbscan_cluster(X),
                    ),
                ]
            )
    if temporal_signals is not None:
        temporal_signals = np.asarray(temporal_signals, dtype=float)
        for count in cluster_counts:
            candidates.extend(
                [
                    BenchmarkCandidate(
                        f"kshape_k{count}",
                        "cycle_templates",
                        temporal_signals,
                        lambda X, random_state=0, sample_weight=None, count=count: kshape_cluster(
                            X,
                            n_clusters=count,
                            sample_weight=sample_weight,
                            random_state=random_state,
                        ),
                    ),
                    BenchmarkCandidate(
                        f"soft_dtw_clara_kmedoids_k{count}",
                        "cycle_templates",
                        temporal_signals,
                        lambda X, random_state=0, sample_weight=None, count=count: soft_dtw_kmedoids_cluster(
                            X,
                            n_clusters=count,
                            sample_weight=sample_weight,
                            random_state=random_state,
                            max_pairwise_samples=soft_dtw_max_pairwise_samples,
                            max_template_length=soft_dtw_max_template_length,
                            window=soft_dtw_window,
                        ),
                    ),
                ]
            )
    for name, labels in (threshold_labelings or {}).items():
        labels = np.asarray(labels, dtype=int)
        candidates.append(
            BenchmarkCandidate(
                f"threshold_{name}",
                "precomputed_thresholds",
                np.arange(len(labels), dtype=float)[:, None],
                lambda X, labels=labels, **_: BenchmarkPrediction(
                    cluster_labels=labels.copy(),
                    deployment_labels=labels.copy(),
                ),
                subsample_safe=False,
                temporal_leakage=True,
            )
        )
    return candidates


def run_single_sample_benchmark(
    embeddings,
    partial_targets,
    *,
    custom_candidates=(),
    custom_candidates_first=False,
    embedding_component_names=None,
    temporal_signals=None,
    threshold_labelings=None,
    cluster_counts=(2, 3),
    branch_weights=None,
    deployment_correlation_features=None,
    labeled_vessels=None,
    signal_videos=None,
    signal_reference_masks=None,
    sampling_frequency=None,
    beat_period=None,
    signal_frame_mask=None,
    constraint_fraction=0.5,
    random_state=0,
    stability_runs=0,
    stability_sample_fraction=0.8,
    include_adaptive=True,
    candidate_keys=None,
    csv_path=None,
    visualization_dir=None,
    visualization_image=None,
    soft_dtw_max_pairwise_samples=96,
    soft_dtw_max_template_length=32,
    soft_dtw_window=4,
):
    """Benchmark method families on one sample without consuming held-out labels.

    Expensive execution is explicit: pass ``stability_runs >= 2`` to enable
    repeated subsampling. The default Soft-DTW candidate uses a 96-branch
    CLARA-style medoid search and 32-sample templates. Set either limit to
    ``None`` only for deliberately small exact experiments.
    """
    if (
        not embeddings
        and temporal_signals is None
        and not threshold_labelings
        and not custom_candidates
    ):
        raise ValueError("at least one representation or threshold labeling is required")
    n_branches = len(partial_targets.branch_ids)
    for name, values in embeddings.items():
        if np.asarray(values).shape[0] != n_branches:
            raise ValueError(f"embedding {name!r} is not branch-aligned")
    if temporal_signals is not None and np.asarray(temporal_signals).shape[0] != n_branches:
        raise ValueError("temporal_signals is not branch-aligned")
    if threshold_labelings and any(
        np.asarray(labels).shape != (n_branches,) for labels in threshold_labelings.values()
    ):
        raise ValueError("every threshold labeling must contain one label per branch")
    if branch_weights is not None:
        branch_weights = np.asarray(branch_weights, dtype=float)
        if branch_weights.shape != (n_branches,):
            raise ValueError("branch_weights must contain one value per branch")
    if deployment_correlation_features is not None:
        deployment_correlation_features = np.asarray(
            deployment_correlation_features, dtype=float
        )
        if deployment_correlation_features.shape != (n_branches, 3):
            raise ValueError(
                "deployment_correlation_features must have shape (n_branches, 3)"
            )

    constraint_mask, evaluation_mask = stratified_partial_label_split(
        partial_targets,
        constraint_fraction=constraint_fraction,
        random_state=random_state,
    )
    must_link, cannot_link = constraints_from_partial_targets(
        partial_targets, constraint_mask
    )
    held_out_targets = _masked_targets(partial_targets, evaluation_mask)
    candidates = _candidate_set(
        embeddings,
        temporal_signals,
        threshold_labelings,
        tuple(cluster_counts),
        must_link,
        cannot_link,
        include_adaptive,
        branch_weights,
        soft_dtw_max_pairwise_samples,
        soft_dtw_max_template_length,
        soft_dtw_window,
    )
    custom_candidates = tuple(custom_candidates)
    if any(not isinstance(candidate, BenchmarkCandidate) for candidate in custom_candidates):
        raise TypeError("custom_candidates must contain BenchmarkCandidate instances")
    for candidate in custom_candidates:
        if candidate.X is not None and np.asarray(candidate.X).shape[0] != n_branches:
            raise ValueError(
                f"custom candidate {candidate.name!r} has a non-aligned X"
            )
    candidates = (
        [*custom_candidates, *candidates]
        if custom_candidates_first
        else [*candidates, *custom_candidates]
    )
    embedding_component_names = embedding_component_names or {}
    candidates = [
        replace(
            candidate,
            component_names=tuple(embedding_component_names[candidate.representation]),
        )
        if candidate.component_names is None
        and candidate.representation in embedding_component_names
        else candidate
        for candidate in candidates
    ]
    if candidate_keys is not None:
        candidate_keys = set(candidate_keys)
        available_keys = {
            f"{candidate.representation}/{candidate.name}" for candidate in candidates
        }
        unknown = candidate_keys - available_keys
        if unknown:
            raise ValueError(f"unknown candidate keys: {sorted(unknown)}")
        candidates = [
            candidate
            for candidate in candidates
            if f"{candidate.representation}/{candidate.name}" in candidate_keys
        ]
    logger.info(
        "Starting choroid benchmark: %d methods, %d branches, %d constraint labels, "
        "%d held-out labels",
        len(candidates),
        n_branches,
        np.count_nonzero(constraint_mask),
        np.count_nonzero(evaluation_mask),
    )

    use_signal_metrics = any(
        value is not None
        for value in (
            signal_videos,
            signal_reference_masks,
            sampling_frequency,
            beat_period,
        )
    )
    if use_signal_metrics and not all(
        value is not None
        for value in (
            labeled_vessels,
            signal_videos,
            signal_reference_masks,
            sampling_frequency,
            beat_period,
        )
    ):
        raise ValueError("all signal-evaluation inputs must be supplied together")
    resolved_visualization_dir = None
    if visualization_dir is not None:
        if labeled_vessels is None or visualization_image is None or not use_signal_metrics:
            raise ValueError(
                "visualization_dir requires visualization_image and all "
                "signal-evaluation inputs"
            )
        resolved_visualization_dir = Path(visualization_dir).expanduser().resolve()
        resolved_visualization_dir.mkdir(parents=True, exist_ok=True)
        if np.asarray(labeled_vessels).shape != np.asarray(visualization_image).shape[-2:]:
            image = np.asarray(visualization_image)
            if image.ndim == 3 and image.shape[:2] == np.asarray(labeled_vessels).shape:
                pass
            else:
                raise ValueError(
                    "visualization_image and labeled_vessels must have matching spatial shapes"
                )

    rows = []
    label_results = {}
    mapped_results = {}
    physiology_mapped_results = {}
    resolved_csv_path = None
    if csv_path is not None:
        resolved_csv_path = Path(csv_path).expanduser().resolve()
        resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint():
        if resolved_csv_path is None:
            return
        checkpoint = pd.DataFrame(rows)
        temporary_path = resolved_csv_path.with_suffix(
            resolved_csv_path.suffix + ".tmp"
        )
        try:
            checkpoint.to_csv(temporary_path, index=False)
            temporary_path.replace(resolved_csv_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    for candidate_index, candidate in enumerate(candidates, start=1):
        key = f"{candidate.representation}/{candidate.name}"
        row = {
            "method": candidate.name,
            "representation": candidate.representation,
            "temporal_leakage": candidate.temporal_leakage,
        }
        method_weights = branch_weights if "weighted" in candidate.name else None
        logger.info(
            "[%d/%d] Running %s",
            candidate_index,
            len(candidates),
            key,
        )
        start = perf_counter()
        try:
            raw_prediction = candidate.run(
                candidate.X,
                random_state=random_state,
                sample_weight=method_weights,
            )
            prediction = (
                raw_prediction
                if isinstance(raw_prediction, BenchmarkPrediction)
                else BenchmarkPrediction(np.asarray(raw_prediction, dtype=int))
            )
            labels = np.asarray(prediction.cluster_labels, dtype=int)
            if labels.shape != (n_branches,):
                raise ValueError("method did not return one label per branch")
            deployment_labels = prediction.deployment_labels
            if deployment_labels is not None:
                deployment_labels = np.asarray(deployment_labels, dtype=int)
                if deployment_labels.shape != (n_branches,):
                    raise ValueError(
                        "deployment_labels must contain one label per branch"
                    )
            if prediction.class_masks is not None and labeled_vessels is None:
                raise ValueError("class_masks require labeled_vessels")
            class_masks = (
                _validated_class_masks(
                    prediction.class_masks,
                    partial_targets,
                    np.asarray(labeled_vessels),
                )
                if prediction.class_masks is not None
                else None
            )
            prediction = replace(
                prediction,
                cluster_labels=labels,
                deployment_labels=deployment_labels,
                class_masks=class_masks,
            )
            row["clustering_seconds"] = perf_counter() - start
            label_results[key] = labels

            all_metrics = evaluate_partial_branch_clustering(
                labels,
                partial_targets,
                sample_weight=branch_weights,
            )
            held_out_metrics = evaluate_partial_branch_clustering(
                labels,
                held_out_targets,
                sample_weight=branch_weights,
            )
            row.update({f"all_{name}": value for name, value in all_metrics.items()})
            row.update(
                {
                    f"heldout_{name}": value
                    for name, value in held_out_metrics.items()
                    if "resubstitution" not in name
                }
            )

            mapped = map_clusters_to_classes(
                labels,
                partial_targets,
                constraint_mask,
                sample_weight=branch_weights,
            )
            mapped_results[key] = mapped
            held_out_true = partial_targets.labels[evaluation_mask]
            held_out_predicted = mapped[evaluation_mask]
            held_out_weights = (
                partial_targets.confidence[evaluation_mask]
                if branch_weights is None
                else branch_weights[evaluation_mask]
                * partial_targets.confidence[evaluation_mask]
            )
            classes = np.arange(len(partial_targets.class_names))
            row.update(
                {
                    "heldout_mapped_accuracy": accuracy_score(
                        held_out_true, held_out_predicted
                    ),
                    "heldout_mapped_balanced_accuracy": recall_score(
                        held_out_true,
                        held_out_predicted,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "heldout_mapped_macro_f1": f1_score(
                        held_out_true,
                        held_out_predicted,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "heldout_weighted_mapped_accuracy": accuracy_score(
                        held_out_true,
                        held_out_predicted,
                        sample_weight=held_out_weights,
                    ),
                    "heldout_weighted_mapped_macro_f1": f1_score(
                        held_out_true,
                        held_out_predicted,
                        labels=classes,
                        average="macro",
                        sample_weight=held_out_weights,
                        zero_division=0,
                    ),
                }
            )
            if deployment_labels is not None:
                physiology_mapped = deployment_labels
                row["deployment_mapping"] = "method_assignment"
            elif deployment_correlation_features is not None:
                physiology_mapped = map_clusters_by_correlation_physiology(
                    labels,
                    deployment_correlation_features,
                    sample_weight=branch_weights,
                    class_names=partial_targets.class_names,
                ).mapped_labels
                row["deployment_mapping"] = "correlation_prototypes"
            else:
                physiology_mapped = None
            if physiology_mapped is not None:
                physiology_mapped_results[key] = physiology_mapped
                physiology_held_out = physiology_mapped[evaluation_mask]
                row.update(
                    {
                        "heldout_physiology_accuracy": accuracy_score(
                            held_out_true, physiology_held_out
                        ),
                        "heldout_physiology_balanced_accuracy": recall_score(
                            held_out_true,
                            physiology_held_out,
                            labels=classes,
                            average="macro",
                            zero_division=0,
                        ),
                        "heldout_physiology_macro_f1": f1_score(
                            held_out_true,
                            physiology_held_out,
                            labels=classes,
                            average="macro",
                            zero_division=0,
                        ),
                        "heldout_weighted_physiology_accuracy": accuracy_score(
                            held_out_true,
                            physiology_held_out,
                            sample_weight=held_out_weights,
                        ),
                        "heldout_physiology_mapped_coverage": np.average(
                            physiology_held_out >= 0,
                            weights=held_out_weights,
                        ),
                    }
                )
            semantic_labels = (
                deployment_labels
                if deployment_labels is not None
                else physiology_mapped
                if physiology_mapped is not None
                else mapped
            )
            if use_signal_metrics:
                predicted_masks = (
                    class_masks
                    if class_masks is not None
                    else _predicted_masks(
                        semantic_labels,
                        partial_targets,
                        np.asarray(labeled_vessels),
                    )
                )
                signal_metrics = evaluate_mask_signal_similarity(
                    signal_videos,
                    predicted_masks,
                    signal_reference_masks,
                    sampling_frequency=sampling_frequency,
                    beat_period=beat_period,
                    frame_mask=signal_frame_mask,
                    exclude_reference_pixels=True,
                )
                row.update(signal_metrics)

            if stability_runs >= 2 and candidate.subsample_safe:
                logger.info(
                    "[%d/%d] Computing %d stability runs for %s",
                    candidate_index,
                    len(candidates),
                    stability_runs,
                    key,
                )
                stability = run_resampled_clustering_stability(
                    candidate.X,
                    candidate.run,
                    n_runs=stability_runs,
                    sample_fraction=stability_sample_fraction,
                    random_state=random_state,
                    sample_weight=method_weights,
                )
                row.update(stability.metrics)
            if resolved_visualization_dir is not None:
                try:
                    artifact_directory = _save_method_artifacts(
                        resolved_visualization_dir,
                        key,
                        prediction,
                        candidate,
                        semantic_labels,
                        partial_targets,
                        np.asarray(labeled_vessels),
                        visualization_image,
                        signal_reference_masks,
                        signal_videos,
                        sampling_frequency,
                        beat_period,
                    )
                    row["artifact_directory"] = str(artifact_directory)
                    logger.info(
                        "[%d/%d] Saved visualization for %s to %s",
                        candidate_index,
                        len(candidates),
                        key,
                        artifact_directory,
                    )
                except Exception as visualization_error:
                    row["visualization_error"] = (
                        f"{type(visualization_error).__name__}: {visualization_error}"
                    )
                    logger.warning(
                        "[%d/%d] Could not save visualization for %s: %s",
                        candidate_index,
                        len(candidates),
                        key,
                        visualization_error,
                    )
            row["runtime_seconds"] = perf_counter() - start
            row["evaluation_seconds"] = (
                row["runtime_seconds"] - row["clustering_seconds"]
            )
            logger.info(
                "[%d/%d] Finished %s in %.2f s (held-out weighted ARI=%.3f, coverage=%.3f)",
                candidate_index,
                len(candidates),
                key,
                row["runtime_seconds"],
                row.get("heldout_partial_weighted_ARI", np.nan),
                row.get("heldout_partial_labeled_coverage", np.nan),
            )
        except Exception as error:  # Preserve other methods in long research runs.
            row["runtime_seconds"] = perf_counter() - start
            row["error"] = f"{type(error).__name__}: {error}"
            logger.error(
                "[%d/%d] Failed %s after %.2f s",
                candidate_index,
                len(candidates),
                key,
                row["runtime_seconds"],
            )
            logger.debug("Failure details for %s", key, exc_info=True)
        rows.append(row)
        save_checkpoint()
        if resolved_csv_path is not None:
            logger.info(
                "Checkpointed %d/%d rows to %s",
                len(rows),
                len(candidates),
                resolved_csv_path,
            )

    table = pd.DataFrame(rows)
    if resolved_csv_path is not None:
        logger.info(
            "Saved %d benchmark rows to %s",
            len(table),
            resolved_csv_path,
        )
    logger.info(
        "Choroid benchmark complete: %d succeeded, %d failed",
        len(label_results),
        len(table) - len(label_results),
    )
    return SingleSampleBenchmarkResult(
        table=table,
        cluster_labels=label_results,
        mapped_class_labels=mapped_results,
        constraint_branch_ids=partial_targets.branch_ids[constraint_mask],
        evaluation_branch_ids=partial_targets.branch_ids[evaluation_mask],
        csv_path=resolved_csv_path,
        physiology_mapped_class_labels=physiology_mapped_results,
    )
