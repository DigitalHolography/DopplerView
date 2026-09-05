import numpy as np
import pytest
from sklearn.cluster import KMeans

from sandbox.clustering_stability import (
    MISSING,
    evaluate_clustering_stability,
    run_resampled_clustering_stability,
)


def test_stability_is_invariant_to_numeric_cluster_permutations():
    result = evaluate_clustering_stability(
        np.array([[0, 0, 1, 1], [8, 8, 3, 3], [1, 1, 0, 0]])
    )

    assert result.metrics["stability_ARI_mean"] == 1.0
    assert result.metrics["stability_weighted_ARI_mean"] == 1.0
    np.testing.assert_allclose(
        result.coassignment_probability,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=float,
        ),
    )


def test_missing_samples_are_compared_only_on_intersection():
    labels = np.array(
        [
            [0, 0, 1, MISSING],
            [4, 4, MISSING, 2],
            [MISSING, 7, 3, 3],
        ]
    )
    result = evaluate_clustering_stability(labels)

    assert result.metrics["stability_valid_run_pair_count"] == 3
    assert result.metrics["stability_observation_fraction_mean"] == 0.75
    assert result.cooccurrence_count[0, 3] == 1
    assert result.cooccurrence_count[0, 1] == 2


def test_noise_has_conservative_and_assigned_only_scores():
    labels = np.array([[0, 0, 1, 1], [5, 5, -1, 8]])
    result = evaluate_clustering_stability(labels)

    assert result.metrics["stability_ARI_mean"] < 1.0
    assert result.metrics["stability_assigned_ARI_mean"] == 1.0
    assert result.metrics["stability_noise_fraction_mean"] > 0


def test_resampled_runner_returns_full_branch_aligned_matrix():
    X = np.r_[
        np.column_stack((np.linspace(-2, -1, 10), np.zeros(10))),
        np.column_stack((np.linspace(1, 2, 10), np.zeros(10))),
    ]

    def clusterer(values, random_state):
        return KMeans(n_clusters=2, n_init=5, random_state=random_state).fit_predict(values)

    result = run_resampled_clustering_stability(
        X,
        clusterer,
        n_runs=8,
        sample_fraction=0.8,
        random_state=4,
    )

    assert result.label_runs.shape == (8, 20)
    assert np.any(result.label_runs == MISSING)
    assert result.metrics["stability_ARI_mean"] == pytest.approx(1.0)


def test_resampled_runner_forwards_branch_weights():
    X = np.arange(12, dtype=float).reshape(6, 2)
    received = []

    def clusterer(values, random_state, sample_weight):
        received.append(sample_weight.copy())
        return (values[:, 0] > np.median(values[:, 0])).astype(int)

    weights = np.arange(1, 7, dtype=float)
    run_resampled_clustering_stability(
        X,
        clusterer,
        n_runs=3,
        sample_fraction=0.5,
        sample_weight=weights,
    )

    assert len(received) == 3
    assert all(len(run_weights) == 3 for run_weights in received)
