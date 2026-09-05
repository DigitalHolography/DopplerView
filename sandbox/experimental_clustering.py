"""Experimental time-series and constrained clustering for choroidal branches."""

import logging
from time import perf_counter

import numpy as np

from .signal_evaluation import soft_dtw_divergence


logger = logging.getLogger(__name__)


def _validated_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X must be a two-dimensional matrix with at least two rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def _validated_weights(sample_weight, n_samples):
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError("sample_weight must contain one value per sample")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sample_weight must be finite and strictly positive")
    return weights


def _z_normalize_rows(X):
    centered = X - np.mean(X, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _shift_with_zeros(signal, lag):
    shifted = np.zeros_like(signal)
    if lag == 0:
        shifted[:] = signal
    elif lag > 0:
        shifted[lag:] = signal[:-lag]
    else:
        shifted[:lag] = signal[-lag:]
    return shifted


def shape_based_distance(reference, candidate):
    """k-Shape shape-based distance and optimally shifted candidate."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must be matching 1-D signals")
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if denominator <= 1e-12:
        return 1.0, candidate.copy()
    correlations = np.correlate(reference, candidate, mode="full") / denominator
    best = int(np.argmax(correlations))
    lag = best - (len(candidate) - 1)
    return float(1.0 - correlations[best]), _shift_with_zeros(candidate, lag)


def _extract_shape(members, reference, weights):
    aligned = np.array([shape_based_distance(reference, row)[1] for row in members])
    centered = aligned - np.mean(aligned, axis=1, keepdims=True)
    scatter = centered.T @ (weights[:, None] * centered)
    projector = np.eye(centered.shape[1]) - np.ones(
        (centered.shape[1], centered.shape[1])
    ) / centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(projector @ scatter @ projector)
    centroid = eigenvectors[:, np.argmax(eigenvalues)]
    if np.linalg.norm(reference + centroid) < np.linalg.norm(reference - centroid):
        centroid = -centroid
    normalized = _z_normalize_rows(centroid[None])[0]
    return normalized


def kshape_cluster(
    signals,
    n_clusters=3,
    *,
    sample_weight=None,
    n_init=10,
    max_iter=100,
    random_state=0,
):
    """Cluster z-normalized cycle shapes with the k-Shape objective.

    Phase alignment uses normalized cross-correlation, and centroids use the
    eigenvector shape-extraction update from k-Shape. Branch weights affect both
    the centroid scatter matrix and the selected objective.
    """
    signals = _z_normalize_rows(_validated_matrix(signals))
    weights = _validated_weights(sample_weight, len(signals))
    if not isinstance(n_clusters, (int, np.integer)) or not 1 < n_clusters <= len(signals):
        raise ValueError("n_clusters must lie between two and the sample count")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    rng = np.random.default_rng(random_state)
    best_labels = None
    best_objective = np.inf
    for _ in range(n_init):
        centers = signals[rng.choice(len(signals), n_clusters, replace=False)].copy()
        previous_labels = None
        for _ in range(max_iter):
            distances = np.array(
                [[shape_based_distance(center, row)[0] for center in centers] for row in signals]
            )
            labels = np.argmin(distances, axis=1)
            empty = set(range(n_clusters)) - set(labels)
            if empty:
                residual = distances[np.arange(len(signals)), labels].copy()
                for cluster_id in empty:
                    selected = int(np.argmax(residual))
                    labels[selected] = cluster_id
                    residual[selected] = -np.inf
            next_centers = np.array(
                [
                    _extract_shape(
                        signals[labels == cluster_id],
                        centers[cluster_id],
                        weights[labels == cluster_id],
                    )
                    for cluster_id in range(n_clusters)
                ]
            )
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                centers = next_centers
                break
            previous_labels = labels.copy()
            centers = next_centers
        distances = np.array(
            [[shape_based_distance(center, row)[0] for center in centers] for row in signals]
        )
        labels = np.argmin(distances, axis=1)
        objective = float(np.sum(weights * distances[np.arange(len(signals)), labels]))
        if len(np.unique(labels)) == n_clusters and objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("k-Shape could not produce all requested clusters")
    return best_labels


def pairwise_soft_dtw_divergence(signals, *, gamma=0.1, window=None):
    """Return a symmetric matrix of Soft-DTW divergences."""
    signals = _z_normalize_rows(_validated_matrix(signals))
    distances = np.zeros((len(signals), len(signals)), dtype=float)
    pair_count = len(signals) * (len(signals) - 1) // 2
    completed_pairs = 0
    progress_interval = max(1, len(signals) // 10)
    started = perf_counter()
    logger.info(
        "Computing Soft-DTW matrix for %d branches (%d pairs)",
        len(signals),
        pair_count,
    )
    for left in range(len(signals)):
        for right in range(left + 1, len(signals)):
            distance = soft_dtw_divergence(
                signals[left], signals[right], gamma=gamma, window=window
            )
            distances[left, right] = distance
            distances[right, left] = distance
            completed_pairs += 1
        if (left + 1) % progress_interval == 0 or left + 1 == len(signals):
            logger.info(
                "Soft-DTW progress: %d/%d rows, %d/%d pairs (%.1f%%, %.2f s elapsed)",
                left + 1,
                len(signals),
                completed_pairs,
                pair_count,
                100 * completed_pairs / max(pair_count, 1),
                perf_counter() - started,
            )
    return distances


def _resample_templates(signals, max_template_length):
    if max_template_length is None or signals.shape[1] <= max_template_length:
        return signals
    if not isinstance(max_template_length, (int, np.integer)) or max_template_length < 4:
        raise ValueError("max_template_length must be an integer of at least four")
    original = np.linspace(0.0, 1.0, signals.shape[1])
    target = np.linspace(0.0, 1.0, max_template_length)
    logger.info(
        "Resampling Soft-DTW templates from %d to %d samples",
        signals.shape[1],
        max_template_length,
    )
    return _z_normalize_rows(
        np.asarray([np.interp(target, original, row) for row in signals])
    )


def _fit_kmedoids_distance_matrix(distances, weights, n_clusters, n_init, max_iter, rng):
    best_labels = None
    best_medoids = None
    best_objective = np.inf
    for _ in range(n_init):
        medoids = [int(rng.integers(len(distances)))]
        while len(medoids) < n_clusters:
            nearest = np.min(distances[:, medoids], axis=1)
            nearest[medoids] = 0
            probabilities = weights * nearest
            available = np.setdiff1d(np.arange(len(distances)), medoids)
            if np.sum(probabilities) <= 0:
                selected = int(rng.choice(available))
            else:
                selected = int(
                    rng.choice(len(distances), p=probabilities / np.sum(probabilities))
                )
            medoids.append(selected)
        medoids = np.asarray(medoids, dtype=int)
        for _ in range(max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)
            next_medoids = medoids.copy()
            for cluster_id in range(n_clusters):
                members = np.flatnonzero(labels == cluster_id)
                if not len(members):
                    residual = np.min(distances[:, medoids], axis=1)
                    next_medoids[cluster_id] = int(np.argmax(residual))
                    continue
                costs = distances[np.ix_(members, members)] @ weights[members]
                next_medoids[cluster_id] = members[int(np.argmin(costs))]
            if np.array_equal(next_medoids, medoids):
                break
            medoids = next_medoids
        labels = np.argmin(distances[:, medoids], axis=1)
        objective = float(
            np.sum(weights * distances[np.arange(len(distances)), medoids[labels]])
        )
        if len(np.unique(labels)) == n_clusters and objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
            best_medoids = medoids.copy()
    return best_labels, best_medoids


def soft_dtw_kmedoids_cluster(
    signals,
    n_clusters=3,
    *,
    gamma=0.1,
    window=None,
    sample_weight=None,
    n_init=5,
    max_iter=100,
    random_state=0,
    max_pairwise_samples=None,
    max_template_length=None,
):
    """PAM-style weighted clustering using Soft-DTW divergence.

    This deliberately uses observed medoids rather than Soft-DTW barycenters: it
    is slower than Euclidean methods but avoids inventing an unvalidated waveform.
    """
    signals = _resample_templates(
        _z_normalize_rows(_validated_matrix(signals)), max_template_length
    )
    weights = _validated_weights(sample_weight, len(signals))
    if not isinstance(n_clusters, (int, np.integer)) or not 1 < n_clusters <= len(signals):
        raise ValueError("n_clusters must lie between two and the sample count")
    rng = np.random.default_rng(random_state)
    if max_pairwise_samples is not None:
        if (
            not isinstance(max_pairwise_samples, (int, np.integer))
            or max_pairwise_samples < n_clusters
        ):
            raise ValueError("max_pairwise_samples must be at least n_clusters")
        sample_count = min(int(max_pairwise_samples), len(signals))
    else:
        sample_count = len(signals)

    if sample_count < len(signals):
        probabilities = weights / np.sum(weights)
        sample_indices = np.sort(
            rng.choice(len(signals), sample_count, replace=False, p=probabilities)
        )
        logger.info(
            "Using CLARA-style Soft-DTW approximation: %d/%d branches in medoid search",
            sample_count,
            len(signals),
        )
    else:
        sample_indices = np.arange(len(signals))

    sample_distances = pairwise_soft_dtw_divergence(
        signals[sample_indices], gamma=gamma, window=window
    )
    _, local_medoids = _fit_kmedoids_distance_matrix(
        sample_distances,
        weights[sample_indices],
        n_clusters,
        n_init,
        max_iter,
        rng,
    )
    best_labels = None
    if local_medoids is not None:
        medoids = sample_indices[local_medoids]
        logger.info(
            "Assigning %d branches to %d Soft-DTW medoids",
            len(signals),
            n_clusters,
        )
        assignment_distances = np.empty((len(signals), n_clusters), dtype=float)
        progress_interval = max(1, len(signals) // 10)
        started = perf_counter()
        for row, signal in enumerate(signals):
            for column, medoid in enumerate(medoids):
                assignment_distances[row, column] = soft_dtw_divergence(
                    signal, signals[medoid], gamma=gamma, window=window
                )
            if (row + 1) % progress_interval == 0 or row + 1 == len(signals):
                logger.info(
                    "Soft-DTW assignment: %d/%d branches (%.1f%%, %.2f s elapsed)",
                    row + 1,
                    len(signals),
                    100 * (row + 1) / len(signals),
                    perf_counter() - started,
                )
        best_labels = np.argmin(assignment_distances, axis=1)
    if best_labels is None:
        raise RuntimeError("Soft-DTW k-medoids could not produce all requested clusters")
    return best_labels


class _DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return int(value)

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _validated_constraints(pairs, n_samples, name):
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{name} must be an array of index pairs")
    if np.any(pairs < 0) or np.any(pairs >= n_samples):
        raise ValueError(f"{name} contains an out-of-range sample index")
    return pairs


def cop_kmeans_cluster(
    X,
    n_clusters=3,
    *,
    must_link=(),
    cannot_link=(),
    sample_weight=None,
    n_init=20,
    max_iter=100,
    random_state=0,
):
    """Weighted COP-KMeans with transitive must-link components."""
    X = _validated_matrix(X)
    weights = _validated_weights(sample_weight, len(X))
    must_link = _validated_constraints(must_link, len(X), "must_link")
    cannot_link = _validated_constraints(cannot_link, len(X), "cannot_link")
    disjoint = _DisjointSet(len(X))
    for left, right in must_link:
        disjoint.union(int(left), int(right))
    roots = np.array([disjoint.find(index) for index in range(len(X))])
    _, component_index = np.unique(roots, return_inverse=True)
    n_components = np.max(component_index) + 1
    if not 1 < n_clusters <= n_components:
        raise ValueError("n_clusters must lie between two and the must-link component count")

    component_weights = np.bincount(component_index, weights=weights)
    component_X = np.vstack(
        [
            np.average(X[component_index == component], axis=0, weights=weights[component_index == component])
            for component in range(n_components)
        ]
    )
    conflicts = [set() for _ in range(n_components)]
    for left, right in cannot_link:
        left_component = int(component_index[left])
        right_component = int(component_index[right])
        if left_component == right_component:
            raise ValueError("a cannot-link pair belongs to one must-link component")
        conflicts[left_component].add(right_component)
        conflicts[right_component].add(left_component)

    rng = np.random.default_rng(random_state)
    best_labels = None
    best_objective = np.inf
    order = np.array(
        sorted(range(n_components), key=lambda item: (-len(conflicts[item]), -component_weights[item]))
    )
    for _ in range(n_init):
        center_indices = [int(rng.integers(n_components))]
        while len(center_indices) < n_clusters:
            squared = np.min(
                np.sum((component_X[:, None] - component_X[center_indices][None]) ** 2, axis=2),
                axis=1,
            )
            squared[center_indices] = 0
            probability = component_weights * squared
            available = np.setdiff1d(np.arange(n_components), center_indices)
            selected = (
                int(rng.choice(available))
                if np.sum(probability) <= 0
                else int(rng.choice(n_components, p=probability / np.sum(probability)))
            )
            center_indices.append(selected)
        centers = component_X[center_indices].copy()
        previous = None
        feasible = True
        for _ in range(max_iter):
            component_labels = np.full(n_components, -1, dtype=int)
            squared = np.sum((component_X[:, None] - centers[None]) ** 2, axis=2)
            for component in order:
                for cluster_id in np.argsort(squared[component]):
                    if all(component_labels[other] != cluster_id for other in conflicts[component]):
                        component_labels[component] = int(cluster_id)
                        break
                if component_labels[component] < 0:
                    feasible = False
                    break
            if not feasible or len(np.unique(component_labels)) != n_clusters:
                feasible = False
                break
            next_centers = np.vstack(
                [
                    np.average(
                        component_X[component_labels == cluster_id],
                        axis=0,
                        weights=component_weights[component_labels == cluster_id],
                    )
                    for cluster_id in range(n_clusters)
                ]
            )
            if previous is not None and np.array_equal(component_labels, previous):
                centers = next_centers
                break
            previous = component_labels.copy()
            centers = next_centers
        if not feasible:
            continue
        squared = np.sum((component_X - centers[component_labels]) ** 2, axis=1)
        objective = float(np.sum(component_weights * squared))
        if objective < best_objective:
            best_objective = objective
            best_labels = component_labels[component_index]
    if best_labels is None:
        raise RuntimeError("COP-KMeans found no feasible assignment; inspect constraints")
    return best_labels.astype(int)
