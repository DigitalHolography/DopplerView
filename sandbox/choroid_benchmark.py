"""Leakage-aware, single-sample benchmark for choroidal branch clustering."""

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.mixture import GaussianMixture

from dopplerview.segmentation import clustering

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


@dataclass(frozen=True)
class SingleSampleBenchmarkResult:
    table: pd.DataFrame
    cluster_labels: dict
    mapped_class_labels: dict
    constraint_branch_ids: np.ndarray
    evaluation_branch_ids: np.ndarray
    csv_path: Path | None = None


@dataclass(frozen=True)
class _Candidate:
    name: str
    representation: str
    X: np.ndarray
    run: object
    subsample_safe: bool = True
    temporal_leakage: bool = False


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
    """Map every cluster to its weighted-majority class using training labels only."""
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
    mapped = np.full(len(cluster_labels), UNLABELED, dtype=int)
    for cluster_id in np.unique(cluster_labels[cluster_labels >= 0]):
        evidence = fit_mask & (cluster_labels == cluster_id)
        if not np.any(evidence):
            continue
        scores = np.bincount(
            targets.labels[evidence],
            weights=weights[evidence] * targets.confidence[evidence],
            minlength=len(targets.class_names),
        )
        mapped[cluster_labels == cluster_id] = int(np.argmax(scores))
    return mapped


def _predicted_masks(mapped_labels, targets, labeled_vessels):
    masks = {}
    for class_label, class_name in enumerate(targets.class_names):
        branch_ids = targets.branch_ids[mapped_labels == class_label]
        masks[class_name] = np.isin(labeled_vessels, branch_ids)
    return masks


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
                    _Candidate(
                        f"kmeans_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, sample_weight=None, count=count: KMeans(
                            n_clusters=count, n_init=20, random_state=random_state
                        ).fit(X, sample_weight=sample_weight).labels_,
                    ),
                    _Candidate(
                        f"gmm_k{count}",
                        representation,
                        values,
                        lambda X, random_state=0, count=count, **_: GaussianMixture(
                            n_components=count, n_init=5, random_state=random_state
                        ).fit_predict(X),
                    ),
                    _Candidate(
                        f"hierarchical_ward_k{count}",
                        representation,
                        values,
                        lambda X, count=count, **_: AgglomerativeClustering(
                            n_clusters=count, linkage="ward"
                        ).fit_predict(X),
                    ),
                    _Candidate(
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
                    _Candidate(
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
                        _Candidate(
                            f"weighted_kmeans_k{count}",
                            representation,
                            values,
                            lambda X, random_state=0, sample_weight=None, count=count: KMeans(
                                n_clusters=count, n_init=20, random_state=random_state
                            ).fit(X, sample_weight=sample_weight).labels_,
                        ),
                        _Candidate(
                            f"weighted_hierarchical_k{count}",
                            representation,
                            values,
                            lambda X, sample_weight=None, count=count, **_: clustering.weighted_agglomerative_cluster(
                                X, n_clusters=count, sample_weight=sample_weight
                            ),
                        ),
                        _Candidate(
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
                    _Candidate(
                        "bayesian_gmm_adaptive",
                        representation,
                        values,
                        lambda X, random_state=0, **_: clustering.bayesian_gmm_cluster(
                            X, random_state=random_state
                        ),
                    ),
                    _Candidate(
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
                    _Candidate(
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
                    _Candidate(
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
            _Candidate(
                f"threshold_{name}",
                "precomputed_thresholds",
                np.arange(len(labels), dtype=float)[:, None],
                lambda X, labels=labels, **_: labels.copy(),
                subsample_safe=False,
                temporal_leakage=True,
            )
        )
    return candidates


def run_single_sample_benchmark(
    embeddings,
    partial_targets,
    *,
    temporal_signals=None,
    threshold_labelings=None,
    cluster_counts=(2, 3),
    branch_weights=None,
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
    if not embeddings and temporal_signals is None and not threshold_labelings:
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
            labeled_vessels,
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

    rows = []
    label_results = {}
    mapped_results = {}
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
            labels = candidate.run(
                candidate.X,
                random_state=random_state,
                sample_weight=method_weights,
            )
            labels = np.asarray(labels, dtype=int)
            if labels.shape != (n_branches,):
                raise ValueError("method did not return one label per branch")
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
            if use_signal_metrics:
                predicted_masks = _predicted_masks(
                    mapped, partial_targets, np.asarray(labeled_vessels)
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
    )
