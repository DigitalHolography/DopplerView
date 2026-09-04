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
    assert metrics["known_precision_artery"] == 0.5
    assert metrics["known_precision_vein"] == 1.0


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


def test_pu_metrics_expose_trivial_predict_everything_behavior():
    known_artery = np.zeros((3, 3), dtype=bool)
    known_vein = np.zeros((3, 3), dtype=bool)
    known_artery[0, 0] = True
    known_vein[0, 1] = True
    predicted_everywhere = np.ones((3, 3), dtype=bool)

    metrics = evaluation.evaluate_positive_unlabeled_masks(
        {"artery": predicted_everywhere, "vein": predicted_everywhere},
        {"artery": known_artery, "vein": known_vein},
        skeleton_tolerance=0,
    )

    assert metrics["positive_recall_artery"] == 1.0
    assert metrics["contamination_artery"] == 1.0
    assert metrics["known_precision_artery"] == 0.5
    assert metrics["unlabeled_prediction_rate_artery"] == 1.0
    assert metrics["prediction_rate_artery"] == 1.0
    assert metrics["positive_recall_macro"] == 1.0
    assert metrics["contamination_macro"] == 1.0


def test_pu_evaluation_domain_excludes_irrelevant_background():
    known_artery = np.zeros((4, 4), dtype=bool)
    known_vein = np.zeros((4, 4), dtype=bool)
    known_artery[0, 0] = True
    known_vein[0, 1] = True
    evaluation_mask = np.zeros((4, 4), dtype=bool)
    evaluation_mask[0, :3] = True
    predicted_artery = np.zeros((4, 4), dtype=bool)
    predicted_artery[0, 0] = True
    predicted_artery[0, 2] = True
    predicted_artery[3, 3] = True

    metrics = evaluation.evaluate_positive_unlabeled_masks(
        {"artery": predicted_artery, "vein": known_vein},
        {"artery": known_artery, "vein": known_vein},
        evaluation_mask=evaluation_mask,
        skeleton_tolerance=0,
    )

    assert metrics["evaluation_pixel_count"] == 3
    assert metrics["unlabeled_prediction_rate_artery"] == 1.0
    assert metrics["prediction_rate_artery"] == pytest.approx(2 / 3)
    assert metrics["predicted_count_artery"] == 2


def test_pu_evaluation_domain_must_retain_all_annotations():
    artery = np.zeros((2, 2), dtype=bool)
    vein = np.zeros((2, 2), dtype=bool)
    artery[0, 0] = True
    domain = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="include every annotated pixel"):
        evaluation.evaluate_positive_unlabeled_masks(
            {"artery": artery, "vein": vein},
            {"artery": artery, "vein": vein},
            evaluation_mask=domain,
        )


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
    assert metrics["branch_known_precision_macro"] == 1.0


def test_evaluate_experiment_adds_three_class_pu_metrics_without_fake_dice():
    shape = (3, 3)
    known = {
        "artery": np.zeros(shape, dtype=bool),
        "aliased_artery": np.zeros(shape, dtype=bool),
        "vein": np.zeros(shape, dtype=bool),
    }
    known["artery"][0, 0] = True
    known["aliased_artery"][1, 1] = True
    known["vein"][2, 2] = True
    predicted = {name: mask.copy() for name, mask in known.items()}
    metrics = evaluation.evaluate_experiment(
        pu_predicted_masks=predicted,
        pu_known_positive_masks=known,
        decimals=None,
    )

    assert metrics["pu_positive_recall_macro"] == 1.0
    assert metrics["pu_contamination_macro"] == 0.0
    assert metrics["pu_known_precision_macro"] == 1.0
    assert metrics["pu_unlabeled_prediction_rate_macro"] == 0.0
    assert "dice_mean" not in metrics
    assert "silhouette" not in metrics


def test_evaluate_experiment_can_derive_two_class_pu_predictions_from_result():
    artery = np.array([[1, 0], [0, 0]], dtype=bool)
    vein = np.array([[0, 1], [0, 0]], dtype=bool)
    result = SimpleNamespace(
        X=np.array([[0.0], [1.0]]),
        cluster_labels=np.array([0, 1]),
        artery_mask=artery,
        vein_mask=vein,
    )

    metrics = evaluation.evaluate_experiment(
        result,
        pu_known_positive_masks={"artery": artery, "vein": vein},
        decimals=None,
    )

    assert metrics["pu_positive_recall_macro"] == 1.0
    assert metrics["pu_known_precision_macro"] == 1.0


def test_evaluate_experiment_rejects_one_complete_mask_without_the_other():
    result = SimpleNamespace(
        X=np.array([[0.0], [1.0]]),
        cluster_labels=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        evaluation.evaluate_experiment(
            result,
            gt_artery_mask=np.zeros((2, 2), dtype=bool),
        )


def test_intrinsic_metrics_exclude_noise_and_report_abstention():
    result = SimpleNamespace(
        X=np.array([[0.0], [0.1], [5.0], [5.1], [100.0]]),
        cluster_labels=np.array([0, 0, 1, 1, -1]),
    )

    metrics = evaluation.evaluate_experiment(result, decimals=None)

    assert metrics["cluster_count"] == 2
    assert metrics["noise_count"] == 1
    assert metrics["noise_fraction"] == 0.2
    assert metrics["silhouette"] > 0.9


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
