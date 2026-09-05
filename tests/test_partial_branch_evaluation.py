from types import SimpleNamespace

import numpy as np

from sandbox import evaluation
from sandbox.partial_branch_evaluation import (
    AMBIGUOUS,
    UNLABELED,
    build_partial_branch_targets,
    evaluate_partial_branch_clustering,
)


def _targets():
    labeled = np.array(
        [
            [2, 2, 5, 5],
            [7, 7, 9, 9],
        ]
    )
    masks = {
        "artery": np.array(
            [[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool
        ),
        "vein": np.array(
            [[0, 0, 1, 1], [0, 1, 0, 0]], dtype=bool
        ),
        "aliased_artery": np.zeros_like(labeled, dtype=bool),
    }
    return labeled, masks


def test_partial_targets_keep_actual_ids_and_separate_ambiguous_unlabeled():
    labeled, masks = _targets()
    targets = build_partial_branch_targets(
        labeled,
        masks,
        min_annotated_pixels=1,
        min_dominance=0.8,
        evidence_saturation_pixels=2,
    )

    np.testing.assert_array_equal(targets.branch_ids, [2, 5, 7, 9])
    np.testing.assert_array_equal(targets.labels, [0, 1, AMBIGUOUS, UNLABELED])
    np.testing.assert_array_equal(targets.branch_sizes, [2, 2, 2, 2])
    assert targets.confidence[0] == 1.0
    assert targets.records()[2]["target_status"] == "ambiguous"


def test_overlapping_annotation_pixels_do_not_create_two_class_evidence():
    labeled = np.array([[3, 3]])
    overlap = np.array([[1, 0]], dtype=bool)
    targets = build_partial_branch_targets(
        labeled,
        {"artery": overlap, "vein": overlap},
        min_annotated_pixels=1,
    )

    assert targets.labels[0] == AMBIGUOUS
    assert targets.ambiguous_pixels[0] == 1
    assert targets.annotated_pixels[0] == 0


def test_partial_clustering_is_permutation_invariant_and_supports_weights():
    labeled = np.array([[2, 2, 5, 5, 8, 8]])
    masks = {
        "artery": labeled == 2,
        "vein": labeled == 5,
        "aliased_artery": labeled == 8,
    }
    targets = build_partial_branch_targets(labeled, masks, min_annotated_pixels=1)
    metrics = evaluate_partial_branch_clustering(
        np.array([12, 4, 9]),
        targets,
        branch_ids=np.array([2, 5, 8]),
        sample_weight=np.array([1.0, 2.0, 4.0]),
    )

    assert metrics["partial_ARI"] == 1.0
    assert metrics["partial_NMI"] == 1.0
    assert metrics["partial_weighted_ARI"] == 1.0
    assert metrics["partial_mapped_macro_f1_resubstitution"] == 1.0
    assert metrics["partial_labeled_coverage"] == 1.0


def test_noise_is_penalized_and_assigned_only_scores_are_reported_separately():
    labeled = np.array([[2, 2, 5, 5, 8, 8]])
    masks = {
        "artery": labeled == 2,
        "vein": labeled == 5,
        "aliased_artery": labeled == 8,
    }
    targets = build_partial_branch_targets(labeled, masks, min_annotated_pixels=1)
    metrics = evaluate_partial_branch_clustering(np.array([0, 1, -1]), targets)

    assert metrics["partial_labeled_coverage"] == 2 / 3
    assert metrics["partial_noise_branch_count"] == 1
    assert metrics["partial_assigned_ARI"] == 1.0
    assert metrics["partial_mapped_macro_f1_resubstitution"] < 1.0


def test_evaluate_experiment_includes_partial_branch_metrics():
    labeled = np.array([[2, 2, 5, 5, 8, 8]])
    masks = {
        "artery": labeled == 2,
        "vein": labeled == 5,
        "aliased_artery": labeled == 8,
    }
    targets = build_partial_branch_targets(labeled, masks, min_annotated_pixels=1)
    result = SimpleNamespace(
        X=np.array([[0.0], [1.0], [2.0]]),
        cluster_labels=np.array([2, 0, 1]),
        branch_ids=np.array([2, 5, 8]),
    )

    metrics = evaluation.evaluate_experiment(
        result,
        partial_branch_targets=targets,
        partial_branch_weights=np.array([1.0, 2.0, 3.0]),
        decimals=None,
    )

    assert metrics["partial_ARI"] == 1.0
    assert metrics["partial_weighted_NMI"] == 1.0
