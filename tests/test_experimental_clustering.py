import numpy as np
from sklearn.metrics import adjusted_rand_score

from sandbox.experimental_clustering import (
    cop_kmeans_cluster,
    kshape_cluster,
    shape_based_distance,
    soft_dtw_kmedoids_cluster,
)


def test_shape_based_distance_aligns_a_shifted_waveform():
    signal = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    shifted = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    distance, aligned = shape_based_distance(signal, shifted)

    assert distance == 0.0
    np.testing.assert_array_equal(aligned, signal)


def _waveform_dataset():
    time = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rng = np.random.default_rng(3)
    first = np.array([np.sin(time) + rng.normal(0, 0.02, len(time)) for _ in range(4)])
    second = np.array([
        np.sign(np.sin(time)) + rng.normal(0, 0.02, len(time)) for _ in range(4)
    ])
    return np.vstack((first, second)), np.repeat([0, 1], 4)


def test_kshape_separates_distinct_cycle_shapes():
    signals, truth = _waveform_dataset()
    labels = kshape_cluster(signals, n_clusters=2, n_init=4, random_state=2)

    assert adjusted_rand_score(truth, labels) == 1.0


def test_soft_dtw_kmedoids_separates_distinct_cycle_shapes():
    signals, truth = _waveform_dataset()
    labels = soft_dtw_kmedoids_cluster(
        signals, n_clusters=2, gamma=0.1, window=4, n_init=3, random_state=2
    )

    assert adjusted_rand_score(truth, labels) == 1.0


def test_soft_dtw_clara_mode_returns_labels_for_every_branch():
    signals, _ = _waveform_dataset()
    labels = soft_dtw_kmedoids_cluster(
        signals,
        n_clusters=2,
        max_pairwise_samples=5,
        max_template_length=8,
        window=2,
        n_init=2,
        random_state=5,
    )

    assert labels.shape == (len(signals),)
    assert len(np.unique(labels)) == 2


def test_cop_kmeans_obeys_must_and_cannot_links():
    X = np.array([[0.0], [0.1], [9.9], [10.0]])
    labels = cop_kmeans_cluster(
        X,
        n_clusters=2,
        must_link=[(0, 3)],
        cannot_link=[(0, 1)],
        n_init=10,
        random_state=4,
    )

    assert labels[0] == labels[3]
    assert labels[0] != labels[1]


def test_cop_kmeans_rejects_contradictory_constraints():
    X = np.arange(4, dtype=float)[:, None]
    try:
        cop_kmeans_cluster(
            X,
            n_clusters=2,
            must_link=[(0, 1)],
            cannot_link=[(0, 1)],
        )
    except ValueError as error:
        assert "cannot-link" in str(error)
    else:
        raise AssertionError("contradictory constraints should fail")
