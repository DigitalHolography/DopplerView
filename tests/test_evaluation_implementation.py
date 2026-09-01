from types import SimpleNamespace

import numpy as np
import pytest

from sandbox import benchmark, evaluation


def test_benchmark_uses_the_shared_evaluation_module():
    assert benchmark.evaluation is evaluation


def test_empty_mask_scores_have_explicit_semantics():
    empty = np.zeros((4, 4), dtype=bool)
    nonempty = empty.copy()
    nonempty[1, 1] = True

    assert evaluation.dice_score(empty, empty) == 1.0
    assert evaluation.iou_score(empty, empty) == 1.0
    assert evaluation.cldice_score(empty, empty) == 1.0
    assert evaluation.hd95_score(empty, empty) == 0.0
    assert evaluation.dice_score(nonempty, empty) == 0.0
    assert evaluation.iou_score(nonempty, empty) == 0.0
    assert evaluation.cldice_score(nonempty, empty) == 0.0
    assert np.isinf(evaluation.hd95_score(nonempty, empty))


def test_evaluate_experiment_excludes_unannotated_branches_and_keeps_precision():
    artery = np.array([[1, 0], [0, 0]], dtype=bool)
    vein = np.array([[0, 1], [0, 0]], dtype=bool)
    result = SimpleNamespace(
        X=np.array([[0.0], [1.0], [2.0]]),
        cluster_labels=np.array([0, 1, 1]),
        mask_labels=np.array([1, 2, 2]),
        artery_mask=artery,
        vein_mask=vein,
    )

    metrics = evaluation.evaluate_experiment(
        result,
        gt_branch_labels=np.array([1, 2, 0]),
        gt_artery_mask=artery,
        gt_vein_mask=vein,
    )

    assert metrics["annotated_branch_count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["hd95_mean"] == 0.0


def test_positive_unlabeled_metrics_ignore_ambiguous_annotation_pixels():
    known_artery = np.zeros((3, 3), dtype=bool)
    known_vein = np.zeros((3, 3), dtype=bool)
    known_artery[0, 0] = True
    known_vein[0, 1] = True
    known_artery[2, 2] = True
    known_vein[2, 2] = True

    predicted_artery = np.zeros((3, 3), dtype=bool)
    predicted_artery[0, 0] = True
    predicted_artery[0, 1] = True
    predicted_vein = np.zeros((3, 3), dtype=bool)
    predicted_vein[0, 1] = True

    metrics = evaluation.evaluate_positive_unlabeled_masks(
        {"artery": predicted_artery, "vein": predicted_vein},
        {"artery": known_artery, "vein": known_vein},
        skeleton_tolerance=0,
    )

    assert metrics["positive_recall_artery"] == 1.0
    assert metrics["contamination_artery"] == 1.0
    assert metrics["positive_recall_vein"] == 1.0
    assert metrics["contamination_vein"] == 0.0


def test_positive_unlabeled_metrics_report_undefined_empty_denominators():
    empty = np.zeros((3, 3), dtype=bool)
    artery = empty.copy()
    artery[1, 1] = True

    metrics = evaluation.evaluate_positive_unlabeled_masks(
        {"artery": artery, "vein": empty},
        {"artery": artery, "vein": empty},
    )

    assert metrics["positive_recall_artery"] == 1.0
    assert np.isnan(metrics["contamination_artery"])
    assert np.isnan(metrics["positive_recall_vein"])


def test_branch_pu_metrics_remove_branches_annotated_as_multiple_classes():
    labeled = np.array([[2, 2, 5, 5], [7, 7, 7, 7]])
    known_artery = np.zeros_like(labeled, dtype=bool)
    known_vein = np.zeros_like(labeled, dtype=bool)
    known_artery[0, 0] = True
    known_vein[0, 2] = True
    known_artery[1, 0] = True
    known_vein[1, 3] = True
    predicted_artery = labeled == 2
    predicted_vein = labeled == 5

    metrics = evaluation.evaluate_positive_unlabeled_masks(
        {"artery": predicted_artery, "vein": predicted_vein},
        {"artery": known_artery, "vein": known_vein},
        labeled_vessels=labeled,
    )

    assert metrics["branch_positive_recall_artery"] == 1.0
    assert metrics["branch_contamination_artery"] == 0.0
    assert metrics["branch_positive_recall_vein"] == 1.0
    assert metrics["branch_contamination_vein"] == 0.0


def test_correlation_assignment_uses_noncontiguous_branch_ids():
    labeled = np.array([[2, 2, 0], [0, 5, 5]])
    artery, vein, labels = evaluation.assign_clusters_to_correlation_stack(
        cluster_labels=np.array([0, 1]),
        X=np.array([[0.8, 0.2], [-0.4, -0.1]]),
        labeled_vessels=labeled,
    )

    assert np.array_equal(artery, labeled == 2)
    assert np.array_equal(vein, labeled == 5)
    assert np.array_equal(labels, [1, 2])


def test_mask_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        evaluation.dice_score(np.zeros((2, 2)), np.zeros((3, 3)))
