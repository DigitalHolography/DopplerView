import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from dopplerview.segmentation import clustering


def test_weighted_agglomerative_with_uniform_weights_matches_ward_partition():
    X = np.array([[0.0], [0.2], [5.0], [5.2]])

    expected = clustering.agglomerative_cluster(X, n_clusters=2)
    actual = clustering.weighted_agglomerative_cluster(
        X,
        n_clusters=2,
        sample_weight=np.ones(len(X)),
    )

    assert adjusted_rand_score(expected, actual) == 1.0


def test_weighted_agglomerative_weights_change_the_merge_tree():
    X = np.array([[0.0], [4.0], [5.0], [9.0]])

    unweighted = clustering.weighted_agglomerative_cluster(X, n_clusters=2)
    weighted = clustering.weighted_agglomerative_cluster(
        X,
        n_clusters=2,
        sample_weight=np.array([10.0, 1.0, 1.0, 1.0]),
    )

    assert unweighted[0] == unweighted[1] == unweighted[2]
    assert unweighted[3] != unweighted[0]
    assert weighted[1] == weighted[2] == weighted[3]
    assert weighted[0] != weighted[1]


def test_weighted_agglomerative_merge_cost_can_infer_cluster_count():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])

    labels = clustering.weighted_agglomerative_cluster(
        X,
        n_clusters=None,
        max_merge_cost=0.01,
    )

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_weighted_agglomerative_rejects_two_stopping_rules():
    with pytest.raises(ValueError, match="n_clusters or max_merge_cost"):
        clustering.weighted_agglomerative_cluster(
            np.array([[0.0], [1.0]]),
            n_clusters=2,
            max_merge_cost=1.0,
        )


def test_branch_size_weights_follow_present_ids_and_sqrt_area():
    labeled = np.array(
        [
            [2, 2, 0, 5, 5, 5],
            [2, 2, 0, 5, 5, 5],
            [0, 0, 0, 5, 5, 5],
        ]
    )
    weights = clustering.branch_size_weights(
        labeled,
        mode="sqrt_area",
        clip_quantiles=None,
        normalize=False,
    )

    np.testing.assert_allclose(weights, [2.0, 3.0])


def test_pipeline_computes_and_passes_branch_weights():
    labeled = np.array([[2, 2, 0], [5, 5, 5]])
    captured = {}

    def weighted_cluster(X, sample_weight=None):
        captured["sample_weight"] = sample_weight.copy()
        return np.array([0, 1])

    result = clustering.run_clustering_pipeline(
        signals=np.array([[0.0, 1.0], [2.0, 3.0]]),
        labeled_vessels=labeled,
        sampling_frequency=40,
        embedding_func=None,
        clustering_func=weighted_cluster,
        video=np.ones((2, 2, 3)),
        assign_to_av=False,
        branch_weight_mode="sqrt_area",
        branch_weight_clip_quantiles=None,
    )

    expected = np.sqrt([2.0, 3.0])
    expected /= expected.mean()
    np.testing.assert_allclose(captured["sample_weight"], expected)
    np.testing.assert_allclose(result.sample_weight, expected)
    np.testing.assert_array_equal(result.branch_ids, [2, 5])


def test_pipeline_rejects_weights_for_an_unsupported_method():
    labeled = np.array([[1, 0], [0, 2]])
    with pytest.raises(TypeError, match="does not accept sample_weight"):
        clustering.run_clustering_pipeline(
            signals=np.array([[0.0, 1.0], [2.0, 3.0]]),
            labeled_vessels=labeled,
            sampling_frequency=40,
            embedding_func=None,
            clustering_func=lambda X: np.array([0, 1]),
            video=np.ones((2, 2, 2)),
            assign_to_av=False,
            branch_weight_mode="sqrt_area",
        )


def test_trimmed_kmeans_marks_a_remote_branch_as_noise():
    X = np.array([[-0.1], [0.0], [0.1], [4.9], [5.0], [5.1], [100.0]])
    labels = clustering.trimmed_kmeans_cluster(
        X,
        n_clusters=2,
        trim_fraction=1 / len(X),
        sample_weight=np.ones(len(X)),
    )

    assert labels[-1] == -1
    assert set(labels[:-1]) == {0, 1}


def test_hdbscan_infers_two_groups_and_keeps_remote_sample_as_noise():
    first = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    second = first + 5.0
    X = np.vstack((first, second, [[20.0, 20.0]]))
    labels = clustering.hdbscan_cluster(
        X,
        min_cluster_size=3,
        min_samples=2,
    )

    assert labels[-1] == -1
    assert len(set(labels[labels >= 0])) == 2


def test_bayesian_gmm_is_deterministic_and_respects_component_upper_bound():
    rng = np.random.default_rng(5)
    X = np.vstack(
        (
            rng.normal(loc=-2.0, scale=0.1, size=(20, 2)),
            rng.normal(loc=2.0, scale=0.1, size=(20, 2)),
        )
    )
    kwargs = {
        "max_components": 5,
        "min_component_weight": 0.02,
        "weight_concentration_prior": 0.05,
        "random_state": 7,
    }

    labels_first = clustering.bayesian_gmm_cluster(X, **kwargs)
    labels_second = clustering.bayesian_gmm_cluster(X, **kwargs)

    np.testing.assert_array_equal(labels_first, labels_second)
    assert 1 <= len(set(labels_first[labels_first >= 0])) <= 5


def test_robust_feature_preparation_clips_extreme_values():
    X = np.array([[0.0], [1.0], [2.0], [1000.0]])
    unclipped = clustering.prepare_clustering_features(X, robust_scale=True)
    prepared = clustering.prepare_clustering_features(
        X,
        robust_scale=True,
        clip_quantiles=(0.0, 0.75),
    )

    assert np.all(np.isfinite(prepared))
    assert prepared[-1, 0] < unclipped[-1, 0]


def test_noise_policy_can_leave_outlier_branches_unassigned(monkeypatch):
    labeled = np.array([[2, 0, 5], [2, 9, 5]])

    def fake_assignment(labels, video, periods, retained_vessels, sampling_freq):
        assert set(np.unique(retained_vessels)) == {0, 5, 9}
        np.testing.assert_array_equal(labels, [0, 1])
        return retained_vessels == 5, retained_vessels == 9, np.array([1, 2])

    monkeypatch.setattr(clustering.pa, "assign_clusters_to_av", fake_assignment)
    result = clustering.run_clustering_pipeline(
        signals=np.array([[0.0], [1.0], [2.0]]),
        labeled_vessels=labeled,
        sampling_frequency=40,
        embedding_func=None,
        clustering_func=lambda X: np.array([-1, 0, 1]),
        video=np.ones((1, 2, 3)),
        assign_to_av=True,
        noise_policy="unassigned",
    )

    np.testing.assert_array_equal(result.mask_labels, [0, 1, 2])
    np.testing.assert_array_equal(result.outlier_mask, [True, False, False])
    assert not np.any(result.artery_mask & (labeled == 2))
    assert not np.any(result.vein_mask & (labeled == 2))
