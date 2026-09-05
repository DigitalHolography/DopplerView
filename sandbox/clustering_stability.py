"""Permutation-invariant stability diagnostics for branch clustering."""

import inspect
import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .partial_branch_evaluation import weighted_adjusted_rand_score


logger = logging.getLogger(__name__)


MISSING = -2
NOISE = -1


@dataclass(frozen=True)
class ClusteringStabilityResult:
    """Repeated labels, pairwise co-assignment evidence, and summary metrics."""

    label_runs: np.ndarray
    coassignment_probability: np.ndarray
    cooccurrence_count: np.ndarray
    metrics: dict


def _mean_std_min(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
        }
    values = values[finite]
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
    }


def evaluate_clustering_stability(
    label_runs,
    *,
    sample_weight=None,
    missing_label=MISSING,
    noise_label=NOISE,
):
    """Evaluate repeated clusterings without matching numeric cluster labels.

    ``missing_label`` represents a branch omitted by a resample. ``noise_label``
    represents an observed branch rejected by the clustering method. The main ARI
    includes noise as an assignment, while ``assigned_ARI`` reports the less
    conservative score after excluding noise from both compared runs.
    """
    labels = np.asarray(label_runs)
    if labels.ndim != 2 or labels.shape[0] < 2 or labels.shape[1] < 2:
        raise ValueError("label_runs must contain at least two runs and two samples")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label_runs must contain integer labels")
    if missing_label == noise_label:
        raise ValueError("missing_label and noise_label must differ")

    if sample_weight is None:
        weights = np.ones(labels.shape[1], dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (labels.shape[1],):
            raise ValueError("sample_weight must contain one value per sample")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("sample_weight must be finite and strictly positive")

    ari = []
    weighted_ari = []
    assigned_ari = []
    assigned_weighted_ari = []
    valid_pairs = 0
    for left in range(labels.shape[0]):
        for right in range(left + 1, labels.shape[0]):
            observed = (labels[left] != missing_label) & (labels[right] != missing_label)
            if np.count_nonzero(observed) < 2:
                continue
            valid_pairs += 1
            ari.append(adjusted_rand_score(labels[left, observed], labels[right, observed]))
            weighted_ari.append(
                weighted_adjusted_rand_score(
                    labels[left, observed], labels[right, observed], weights[observed]
                )
            )
            assigned = observed & (labels[left] != noise_label) & (labels[right] != noise_label)
            if np.count_nonzero(assigned) >= 2:
                assigned_ari.append(
                    adjusted_rand_score(labels[left, assigned], labels[right, assigned])
                )
                assigned_weighted_ari.append(
                    weighted_adjusted_rand_score(
                        labels[left, assigned], labels[right, assigned], weights[assigned]
                    )
                )

    observed = labels != missing_label
    assigned = observed & (labels != noise_label)
    cooccurrence = np.zeros((labels.shape[1], labels.shape[1]), dtype=int)
    coassigned = np.zeros_like(cooccurrence, dtype=int)
    for run in range(labels.shape[0]):
        run_assigned = assigned[run]
        both_assigned = np.outer(run_assigned, run_assigned)
        cooccurrence += both_assigned
        same_cluster = labels[run, :, None] == labels[run, None, :]
        coassigned += both_assigned & same_cluster
    coassignment = np.full(cooccurrence.shape, np.nan, dtype=float)
    np.divide(coassigned, cooccurrence, out=coassignment, where=cooccurrence > 0)

    cluster_counts = []
    noise_fractions = []
    observation_fractions = []
    for run in range(labels.shape[0]):
        run_observed = observed[run]
        run_assigned = assigned[run]
        cluster_counts.append(np.unique(labels[run, run_assigned]).size)
        observation_fractions.append(float(np.mean(run_observed)))
        noise_fractions.append(
            float(np.mean(labels[run, run_observed] == noise_label))
            if np.any(run_observed)
            else np.nan
        )

    metrics = {
        "stability_run_count": int(labels.shape[0]),
        "stability_valid_run_pair_count": int(valid_pairs),
        "stability_observation_fraction_mean": float(np.mean(observation_fractions)),
        "stability_cluster_count_mean": float(np.mean(cluster_counts)),
        "stability_cluster_count_std": float(np.std(cluster_counts)),
        "stability_noise_fraction_mean": float(np.nanmean(noise_fractions)),
    }
    metrics.update(_mean_std_min(ari, "stability_ARI"))
    metrics.update(_mean_std_min(weighted_ari, "stability_weighted_ARI"))
    metrics.update(_mean_std_min(assigned_ari, "stability_assigned_ARI"))
    metrics.update(
        _mean_std_min(assigned_weighted_ari, "stability_assigned_weighted_ARI")
    )
    return ClusteringStabilityResult(labels, coassignment, cooccurrence, metrics)


def _call_clusterer(clusterer, X, seed, sample_weight):
    try:
        parameters = inspect.signature(clusterer).parameters.values()
    except (TypeError, ValueError) as error:
        raise TypeError("clusterer must expose an inspectable call signature") from error
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    kwargs = {}
    if "random_state" in names or accepts_kwargs:
        kwargs["random_state"] = int(seed)
    if sample_weight is not None:
        if "sample_weight" not in names and not accepts_kwargs:
            raise TypeError("sample weights were supplied but clusterer does not accept them")
        kwargs["sample_weight"] = sample_weight
    result = clusterer(X, **kwargs)
    if hasattr(result, "cluster_labels"):
        result = result.cluster_labels
    result = np.asarray(result)
    if result.shape != (len(X),) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("clusterer must return one integer label per input sample")
    return result.astype(int, copy=False)


def run_resampled_clustering_stability(
    X,
    clusterer,
    *,
    n_runs=20,
    sample_fraction=0.8,
    random_state=0,
    sample_weight=None,
    missing_label=MISSING,
    noise_label=NOISE,
):
    """Repeatedly cluster branch subsamples and compare their shared branches."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite 2-D matrix with at least two samples")
    if not isinstance(n_runs, (int, np.integer)) or n_runs < 2:
        raise ValueError("n_runs must be an integer of at least two")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must lie in (0, 1]")
    sample_count = max(2, int(np.ceil(sample_fraction * len(X))))
    sample_count = min(sample_count, len(X))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (len(X),):
            raise ValueError("sample_weight must contain one value per sample")

    rng = np.random.default_rng(random_state)
    label_runs = np.full((n_runs, len(X)), missing_label, dtype=int)
    started = perf_counter()
    progress_interval = max(1, n_runs // 10)
    for run in range(n_runs):
        if sample_count == len(X):
            indices = np.arange(len(X))
        else:
            indices = np.sort(rng.choice(len(X), sample_count, replace=False))
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        run_weights = None if sample_weight is None else sample_weight[indices]
        label_runs[run, indices] = _call_clusterer(
            clusterer, X[indices], seed, run_weights
        )
        if (run + 1) % progress_interval == 0 or run + 1 == n_runs:
            logger.info(
                "Stability progress: %d/%d runs (%.1f%%, %.2f s elapsed)",
                run + 1,
                n_runs,
                100 * (run + 1) / n_runs,
                perf_counter() - started,
            )

    return evaluate_clustering_stability(
        label_runs,
        sample_weight=sample_weight,
        missing_label=missing_label,
        noise_label=noise_label,
    )
