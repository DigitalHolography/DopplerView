import heapq
import inspect
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import RobustScaler

import dopplerview.segmentation.pulse_analysis as pa

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # HDBSCAN was added to scikit-learn in version 1.3.
    HDBSCAN = None


def _validated_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if len(X) < 2:
        raise ValueError("X must contain at least two samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def _validated_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=float)
    if sample_weight.shape != (n_samples,):
        raise ValueError("sample_weight must contain exactly one value per sample")
    if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0):
        raise ValueError("sample_weight values must be finite and strictly positive")
    return sample_weight


def prepare_clustering_features(
    X,
    *,
    robust_scale=False,
    clip_quantiles=None,
):
    """Optionally winsorize and robustly scale an embedding.

    Feature-wise processing is useful for heterogeneous correlation features, but
    should be disabled for embeddings whose geometry must remain isotropic (for
    example paired sine/cosine Fourier coordinates).
    """
    X = _validated_features(X).copy()
    if clip_quantiles is not None:
        if len(clip_quantiles) != 2:
            raise ValueError("clip_quantiles must contain a lower and upper quantile")
        low, high = clip_quantiles
        if not 0 <= low < high <= 1:
            raise ValueError("clip_quantiles must be increasing values in [0, 1]")
        bounds_low, bounds_high = np.quantile(X, (low, high), axis=0)
        X = np.clip(X, bounds_low, bounds_high)
    if robust_scale:
        X = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(X)
    return X


def branch_size_weights(
    labeled_vessels,
    *,
    mode="sqrt_area",
    clip_quantiles=(0.05, 0.95),
    normalize=True,
):
    """Return one branch weight in ascending positive branch-ID order.

    ``sqrt_area`` reduces the variance of mean signals from tiny branches without
    letting a very large trunk dominate as strongly as raw pixel area would.
    """
    labeled_vessels = np.asarray(labeled_vessels)
    if labeled_vessels.ndim != 2 or not np.issubdtype(labeled_vessels.dtype, np.integer):
        raise ValueError("labeled_vessels must be a two-dimensional integer label image")
    branch_ids, counts = np.unique(labeled_vessels, return_counts=True)
    counts = counts[branch_ids > 0].astype(float)
    branch_ids = branch_ids[branch_ids > 0]
    if not len(branch_ids):
        raise ValueError("labeled_vessels does not contain a positive branch ID")

    if mode == "uniform":
        weights = np.ones_like(counts)
    elif mode == "area":
        weights = counts
    elif mode == "sqrt_area":
        weights = np.sqrt(counts)
    elif mode == "log_area":
        weights = np.log1p(counts)
    else:
        raise ValueError("mode must be uniform, area, sqrt_area, or log_area")

    if clip_quantiles is not None and len(weights) > 1:
        if len(clip_quantiles) != 2:
            raise ValueError("clip_quantiles must contain two values")
        low, high = clip_quantiles
        if not 0 <= low <= high <= 1:
            raise ValueError("clip_quantiles must be ordered values in [0, 1]")
        lower, upper = np.quantile(weights, (low, high))
        weights = np.clip(weights, lower, upper)
    if normalize:
        weights = weights / np.mean(weights)
    return weights


def kmeans_cluster(X, n_clusters=2, sample_weight=None):
    X = _validated_features(X)
    sample_weight = _validated_sample_weight(sample_weight, len(X))
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        random_state=0,
        algorithm="lloyd",
    )
    return model.fit(X, sample_weight=sample_weight).labels_


def agglomerative_cluster(X, n_clusters=2):
    X = _validated_features(X)
    return AgglomerativeClustering(
        n_clusters=n_clusters
    ).fit_predict(X)


def weighted_agglomerative_cluster(
    X,
    n_clusters=2,
    *,
    sample_weight=None,
    max_merge_cost=None,
    robust_scale=False,
    clip_quantiles=None,
):
    """Weighted Ward agglomeration with an optional merge-cost cutoff.

    The Ward cost of merging clusters A and B is the increase in weighted
    within-cluster squared error::

        weight_A * weight_B / (weight_A + weight_B) * ||mean_A - mean_B||²

    Specify either ``n_clusters`` or ``max_merge_cost``. The latter avoids
    fixing the final number of clusters, but its scale depends on the embedding,
    preprocessing, and normalized branch weights.
    """
    X = prepare_clustering_features(
        X,
        robust_scale=robust_scale,
        clip_quantiles=clip_quantiles,
    )
    sample_weight = _validated_sample_weight(sample_weight, len(X))
    weights = np.ones(len(X), dtype=float) if sample_weight is None else sample_weight

    if max_merge_cost is not None:
        if n_clusters is not None:
            raise ValueError("specify n_clusters or max_merge_cost, not both")
        if not np.isfinite(max_merge_cost) or max_merge_cost < 0:
            raise ValueError("max_merge_cost must be finite and non-negative")
    else:
        if not isinstance(n_clusters, (int, np.integer)):
            raise ValueError("n_clusters must be an integer")
        if not 1 <= n_clusters <= len(X):
            raise ValueError("n_clusters must lie between 1 and the sample count")

    cluster_weights = {index: weights[index] for index in range(len(X))}
    centroids = {index: X[index].copy() for index in range(len(X))}
    members = {index: [index] for index in range(len(X))}
    active = set(range(len(X)))
    merge_heap = []

    def ward_cost(left, right):
        left_weight = cluster_weights[left]
        right_weight = cluster_weights[right]
        difference = centroids[left] - centroids[right]
        return float(
            left_weight
            * right_weight
            / (left_weight + right_weight)
            * np.dot(difference, difference)
        )

    for left in range(len(X)):
        for right in range(left + 1, len(X)):
            heapq.heappush(merge_heap, (ward_cost(left, right), left, right))

    next_cluster_id = len(X)
    target_count = n_clusters if max_merge_cost is None else 1
    while len(active) > target_count:
        while merge_heap:
            cost, left, right = heapq.heappop(merge_heap)
            if left in active and right in active:
                break
        else:
            break
        if max_merge_cost is not None and cost > max_merge_cost:
            break

        merged_weight = cluster_weights[left] + cluster_weights[right]
        merged_centroid = (
            cluster_weights[left] * centroids[left]
            + cluster_weights[right] * centroids[right]
        ) / merged_weight
        active.remove(left)
        active.remove(right)
        merged = next_cluster_id
        next_cluster_id += 1
        cluster_weights[merged] = merged_weight
        centroids[merged] = merged_centroid
        members[merged] = members[left] + members[right]
        active.add(merged)

        for other in active:
            if other == merged:
                continue
            first, second = sorted((merged, other))
            heapq.heappush(
                merge_heap,
                (ward_cost(first, second), first, second),
            )

    labels = np.empty(len(X), dtype=int)
    ordered_clusters = sorted(active, key=lambda cluster: min(members[cluster]))
    for label_id, cluster_id in enumerate(ordered_clusters):
        labels[members[cluster_id]] = label_id
    return labels


def gmm_cluster(X, n_clusters=2):
    X = _validated_features(X)
    return GaussianMixture(
        n_components=n_clusters,
        random_state=0
    ).fit(X).predict(X)


def trimmed_kmeans_cluster(
    X,
    n_clusters=2,
    *,
    trim_fraction=0.05,
    sample_weight=None,
    robust_scale=False,
    clip_quantiles=None,
    max_trim_iterations=20,
    n_init=20,
    random_state=0,
):
    """Weighted K-means that labels the farthest samples as noise (``-1``)."""
    X = prepare_clustering_features(
        X,
        robust_scale=robust_scale,
        clip_quantiles=clip_quantiles,
    )
    sample_weight = _validated_sample_weight(sample_weight, len(X))
    if not 0 <= trim_fraction < 1:
        raise ValueError("trim_fraction must lie in [0, 1)")
    trim_count = int(np.floor(trim_fraction * len(X)))
    retained_count = len(X) - trim_count
    if retained_count < n_clusters:
        raise ValueError("trim_fraction leaves fewer samples than clusters")
    if not isinstance(max_trim_iterations, (int, np.integer)) or max_trim_iterations < 1:
        raise ValueError("max_trim_iterations must be a positive integer")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= retained_count:
        raise ValueError("n_clusters must be between 1 and the retained sample count")

    weights = np.ones(len(X)) if sample_weight is None else sample_weight
    if trim_count == 0:
        return KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
            algorithm="lloyd",
        ).fit(X, sample_weight=weights).labels_

    rng = np.random.default_rng(random_state)
    best_objective = np.inf
    best_labels = None
    best_retained = None

    for _ in range(n_init):
        centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
        retained = np.ones(len(X), dtype=bool)

        for _ in range(max_trim_iterations):
            squared_distances = np.sum(
                (X[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            )
            labels = np.argmin(squared_distances, axis=1)
            residuals = squared_distances[np.arange(len(X)), labels]
            next_retained = np.zeros(len(X), dtype=bool)
            retained_indices = np.argsort(residuals, kind="stable")[:retained_count]
            next_retained[retained_indices] = True

            next_centers = centers.copy()
            valid = True
            for cluster_id in range(n_clusters):
                members = next_retained & (labels == cluster_id)
                if not np.any(members):
                    valid = False
                    break
                next_centers[cluster_id] = np.average(
                    X[members],
                    axis=0,
                    weights=weights[members],
                )
            if not valid:
                break
            converged = np.array_equal(next_retained, retained) and np.allclose(
                next_centers,
                centers,
            )
            retained = next_retained
            centers = next_centers
            if converged:
                break
        else:
            valid = True

        if not valid:
            continue
        squared_distances = np.sum(
            (X[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(squared_distances, axis=1)
        residuals = squared_distances[np.arange(len(X)), labels]
        objective = float(np.sum(weights[retained] * residuals[retained]))
        if objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
            best_retained = retained.copy()

    if best_labels is None:
        raise RuntimeError("trimmed K-means could not initialize every cluster")
    best_labels = best_labels.astype(int)
    best_labels[~best_retained] = -1
    return best_labels


def hdbscan_cluster(
    X,
    *,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    robust_scale=False,
    clip_quantiles=None,
    allow_single_cluster=False,
):
    """Infer density-based clusters and retain uncertain branches as noise."""
    if HDBSCAN is None:
        raise ImportError("hdbscan_cluster requires scikit-learn >= 1.3")
    X = prepare_clustering_features(
        X,
        robust_scale=robust_scale,
        clip_quantiles=clip_quantiles,
    )
    parameters = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "allow_single_cluster": allow_single_cluster,
    }
    if "copy" in inspect.signature(HDBSCAN).parameters:
        parameters["copy"] = False
    return HDBSCAN(
        **parameters,
    ).fit_predict(X)


def bayesian_gmm_cluster(
    X,
    *,
    max_components=6,
    min_component_weight=0.02,
    weight_concentration_prior=0.1,
    covariance_type="full",
    robust_scale=False,
    clip_quantiles=None,
    random_state=0,
):
    """Infer an effective component count below ``max_components``.

    Samples assigned to posterior components below ``min_component_weight`` are
    labeled as noise. This is a finite variational approximation, not proof of the
    true physiological class count.
    """
    X = prepare_clustering_features(
        X,
        robust_scale=robust_scale,
        clip_quantiles=clip_quantiles,
    )
    if not isinstance(max_components, (int, np.integer)) or max_components < 1:
        raise ValueError("max_components must be a positive integer")
    if not 0 <= min_component_weight < 1:
        raise ValueError("min_component_weight must lie in [0, 1)")
    n_components = min(max_components, len(X))
    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        n_init=5,
        max_iter=1000,
        random_state=random_state,
    ).fit(X)
    original_labels = model.predict(X)
    active_components = np.flatnonzero(model.weights_ >= min_component_weight)
    mapping = {component: index for index, component in enumerate(active_components)}
    return np.array(
        [mapping.get(component, -1) for component in original_labels],
        dtype=int,
    )


def correlation_clustering(corr_stacks, thresholds=[0,0]):
    """
    Cluster correlation stacks based on a threshold for each axis.

    Parameters
    ----------
    corr_stacks : ndarray, shape (n_samples, n_features)
        Correlation stacks for each sample.
    thresholds : list of float
        Thresholds for clustering. If a sample's correlation stack is below the threshold for a given axis, it is assigned to one cluster; otherwise, it is assigned to another cluster.

    Returns
    -------
    cluster_labels : ndarray, shape (n_samples,)
        Cluster labels for each sample.
    """
    cluster_labels = np.zeros(corr_stacks.shape[0], dtype=int)
    for i in range(corr_stacks.shape[0]):
        if all(corr_stacks[i] < thresholds):
            cluster_labels[i] = 0
        else:
            cluster_labels[i] = 1
    return cluster_labels


@dataclass
class ClusteringResult:
    templates: np.ndarray
    periods: np.ndarray

    X: np.ndarray

    cluster_labels: np.ndarray
    mask_labels: np.ndarray

    artery_mask: np.ndarray
    vein_mask: np.ndarray

    branch_ids: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    outlier_mask: Optional[np.ndarray] = None


def _cluster_with_optional_weights(clustering_func, X, sample_weight):
    if sample_weight is None:
        return clustering_func(X)
    try:
        parameters = inspect.signature(clustering_func).parameters.values()
    except (TypeError, ValueError) as error:
        raise TypeError(
            "cannot determine whether clustering_func supports sample_weight"
        ) from error
    supports_weights = any(
        parameter.name == "sample_weight"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not supports_weights:
        raise TypeError(
            "branch weights were requested, but clustering_func does not accept "
            "sample_weight"
        )
    return clustering_func(X, sample_weight=sample_weight)

def run_clustering_pipeline(
    signals,
    labeled_vessels,
    sampling_frequency,
    embedding_func,
    clustering_func,
    video,
    correct_signals=False,
    beat_period=None,
    assign_to_av=True,
    sample_weight=None,
    branch_weight_mode=None,
    branch_weight_clip_quantiles=(0.05, 0.95),
    noise_policy="error",
):
    """
    Complete clustering pipeline.
    signals: array of shape (n_branches, n_timepoints)
    labeled_vessels: array of shape (height, width) with branch labels
    sampling_frequency: sampling frequency of the signals
    embedding_func: function to embed the signals (e.g., PCA). If the signals are already embedded, this can be None.
    clustering_func: function to perform clustering
    video: the video data
    correct_signals: whether to correct the signals
    beat_period: the period of the heartbeats
    assign_to_av: whether to assign clusters to artery/vein

    Returns
    -------
    ClusteringResult
    """
    signals = np.asarray(signals)
    labeled_vessels = np.asarray(labeled_vessels)
    video = np.asarray(video)
    if signals.ndim != 2:
        raise ValueError("signals must be a 2-D branch-by-time array")
    if labeled_vessels.ndim != 2:
        raise ValueError("labeled_vessels must be a 2-D label image")
    if video.ndim != 3 or video.shape[1:] != labeled_vessels.shape:
        raise ValueError("video and labeled_vessels must have matching spatial shapes")
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if len(signals) != len(branch_ids):
        raise ValueError("signals must contain exactly one row per labeled branch")
    if sample_weight is not None and branch_weight_mode is not None:
        raise ValueError("provide sample_weight or branch_weight_mode, not both")
    if branch_weight_mode is not None:
        sample_weight = branch_size_weights(
            labeled_vessels,
            mode=branch_weight_mode,
            clip_quantiles=branch_weight_clip_quantiles,
        )
    sample_weight = _validated_sample_weight(sample_weight, len(branch_ids))
    if noise_policy not in {"error", "unassigned"}:
        raise ValueError("noise_policy must be 'error' or 'unassigned'")

    if embedding_func is not None:
        if correct_signals:
            if beat_period is None:
                beat_period = pa.compute_period(signals, sampling_frequency)
            if beat_period is None:
                raise ValueError("Unable to estimate a cardiac period for signal correction")
            corrected = [
                pa.remove_bad_beats(
                    branch,
                    beat_period,
                )[0]
                for branch in signals
            ]

            cycle_templates = [
                pa.get_cycle_template(
                    branch,
                    sampling_frequency,
                    return_period=True,
                )
                for branch in corrected
            ]

        else:

            cycle_templates = [
                pa.get_cycle_template(
                    branch,
                    sampling_freq=sampling_frequency,
                    beat_period=beat_period,
                    return_period=True,
                )
                for branch in signals
            ]

        templates, periods = zip(*cycle_templates)

        templates = np.asarray(templates)
        periods = np.asarray(periods)

        X = embedding_func(templates)

    else:
        X = signals
        templates = None
        periods = np.asarray([beat_period] * len(signals))

    cluster_labels = _cluster_with_optional_weights(
        clustering_func,
        X,
        sample_weight,
    )
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(branch_ids):
        raise ValueError("clustering must return exactly one label per branch")
    if not np.issubdtype(cluster_labels.dtype, np.integer):
        raise ValueError("clustering labels must be integers")
    outlier_mask = cluster_labels < 0
    non_outlier_clusters = np.unique(cluster_labels[~outlier_mask])
    if non_outlier_clusters.size == 2:
        canonical = pa.canonicalize_binary_cluster_labels(
            cluster_labels[~outlier_mask],
            X[~outlier_mask],
        )
        cluster_labels = cluster_labels.copy()
        cluster_labels[~outlier_mask] = canonical

    if assign_to_av:
        if np.any(outlier_mask) and noise_policy == "error":
            raise ValueError(
                "clustering returned noise labels; use noise_policy='unassigned' "
                "or assign_to_av=False"
            )
        if non_outlier_clusters.size != 2:
            raise ValueError(
                "artery/vein assignment requires exactly two non-noise clusters"
            )
        if np.any(outlier_mask):
            retained_branch_ids = branch_ids[~outlier_mask]
            retained_vessels = np.where(
                np.isin(labeled_vessels, retained_branch_ids),
                labeled_vessels,
                0,
            )
            artery_mask, vein_mask, retained_mask_labels = pa.assign_clusters_to_av(
                cluster_labels[~outlier_mask],
                video,
                periods[~outlier_mask],
                retained_vessels,
                sampling_freq=sampling_frequency,
            )
            mask_labels = np.zeros_like(cluster_labels, dtype=int)
            mask_labels[~outlier_mask] = retained_mask_labels
        else:
            artery_mask, vein_mask, mask_labels = pa.assign_clusters_to_av(
                cluster_labels,
                video,
                periods,
                labeled_vessels,
                sampling_freq=sampling_frequency,
            )
    else:
        mask_labels = np.zeros_like(cluster_labels, dtype=int)
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)

    return ClusteringResult(
        templates=templates,
        periods=periods,
        X=X,
        cluster_labels=cluster_labels,
        mask_labels=mask_labels,
        artery_mask=artery_mask,
        vein_mask=vein_mask,
        branch_ids=branch_ids,
        sample_weight=sample_weight,
        outlier_mask=outlier_mask,
    )
