"""Evaluation utilities for supervised and positive-unlabeled vessel masks."""

import h5py
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)

from .partial_branch_evaluation import (
    PartialBranchTargets,
    build_partial_branch_targets,
    evaluate_partial_branch_clustering,
)
from .signal_evaluation import evaluate_mask_signal_similarity


def _as_bool_mask(mask, name):
    mask = np.asarray(mask)
    if mask.ndim < 1:
        raise ValueError(f"{name} must be an array")
    return mask.astype(bool)


def _mask_pair(pred, gt):
    pred = _as_bool_mask(pred, "pred")
    gt = _as_bool_mask(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError("pred and gt masks must have the same shape")
    return pred, gt


def _safe_fraction(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def dice_score(pred, gt):
    """Sørensen-Dice score, with two empty masks treated as a perfect match."""
    pred, gt = _mask_pair(pred, gt)
    denominator = np.count_nonzero(pred) + np.count_nonzero(gt)
    if denominator == 0:
        return 1.0
    intersection = np.count_nonzero(pred & gt)
    return 2.0 * intersection / denominator


def iou_score(pred, gt):
    """Intersection-over-union, with two empty masks treated as a perfect match."""
    pred, gt = _mask_pair(pred, gt)
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    return np.count_nonzero(pred & gt) / union


def cldice_score(pred, gt):
    """Topology-aware clDice score for two binary masks."""
    pred, gt = _mask_pair(pred, gt)
    if not np.any(pred) and not np.any(gt):
        return 1.0
    if not np.any(pred) or not np.any(gt):
        return 0.0

    pred_skeleton = skeletonize(pred)
    gt_skeleton = skeletonize(gt)
    topology_precision = _safe_fraction(
        np.count_nonzero(pred_skeleton & gt),
        np.count_nonzero(pred_skeleton),
    )
    topology_sensitivity = _safe_fraction(
        np.count_nonzero(gt_skeleton & pred),
        np.count_nonzero(gt_skeleton),
    )
    denominator = topology_precision + topology_sensitivity
    if denominator == 0:
        return 0.0
    return 2.0 * topology_precision * topology_sensitivity / denominator


def hd95_score(pred, gt, spacing=None):
    """Symmetric 95th-percentile Hausdorff distance between mask surfaces."""
    pred, gt = _mask_pair(pred, gt)
    if not np.any(pred) and not np.any(gt):
        return 0.0
    if not np.any(pred) or not np.any(gt):
        return np.inf

    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distance_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    distances = np.concatenate(
        (distance_to_gt[pred_surface], distance_to_pred[gt_surface])
    )
    return float(np.percentile(distances, 95))


def exclusive_known_masks(known_positive_masks):
    """Remove pixels annotated as more than one class from incomplete labels."""
    masks, _, _ = _partition_known_masks(known_positive_masks)
    return masks


def _partition_known_masks(known_positive_masks):
    """Return exclusive positives, ambiguous pixels, and genuinely unlabeled pixels."""
    if not known_positive_masks:
        raise ValueError("known_positive_masks cannot be empty")

    masks = {
        name: _as_bool_mask(mask, f"known_positive_masks[{name!r}]")
        for name, mask in known_positive_masks.items()
    }
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError("all known-positive masks must have the same shape")

    overlap_count = np.sum(np.stack(list(masks.values())), axis=0)
    unambiguous = overlap_count == 1
    exclusive = {name: mask & unambiguous for name, mask in masks.items()}
    return exclusive, overlap_count > 1, overlap_count == 0


def _dilated_skeleton(mask, tolerance):
    skeleton = skeletonize(mask)
    if tolerance:
        skeleton = binary_dilation(skeleton, iterations=tolerance)
    return skeleton


def _positive_recall_and_contamination(pred, positive, known_negative):
    recall = _safe_fraction(
        np.count_nonzero(pred & positive),
        np.count_nonzero(positive),
    )
    contamination = _safe_fraction(
        np.count_nonzero(pred & known_negative),
        np.count_nonzero(known_negative),
    )
    return recall, contamination


def _known_precision_and_f1(pred, positive, known_negative, recall):
    """Precision restricted to annotated positives and annotated negatives."""
    true_positive = np.count_nonzero(pred & positive)
    known_false_positive = np.count_nonzero(pred & known_negative)
    precision = _safe_fraction(true_positive, true_positive + known_false_positive)
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, f1


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else np.nan


def _branch_sets(masks, labeled_vessels, branch_overlap_threshold):
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    selected = {name: set() for name in masks}
    for branch_id in branch_ids:
        branch = labeled_vessels == branch_id
        branch_size = np.count_nonzero(branch)
        for name, mask in masks.items():
            overlap = np.count_nonzero(mask & branch) / branch_size
            if overlap >= branch_overlap_threshold:
                selected[name].add(int(branch_id))
    return selected


def evaluate_positive_unlabeled_masks(
    predicted_masks,
    known_positive_masks,
    labeled_vessels=None,
    evaluation_mask=None,
    skeleton_tolerance=1,
    branch_overlap_threshold=0.5,
):
    """Evaluate incomplete multiclass masks without treating unlabeled pixels as negatives.

    For every class, annotated-positive recall measures coverage of that class's
    exclusive annotations. Contamination measures coverage of exclusive annotations
    belonging to all other classes. ``known_precision`` is precision restricted to
    these annotated positive/negative pixels; it is not population precision.
    ``unlabeled_prediction_rate`` is descriptive, not an error rate. Undefined
    metrics have value ``numpy.nan``.
    """
    if set(predicted_masks) != set(known_positive_masks):
        raise ValueError("predicted and known-positive masks must use the same classes")
    if not isinstance(skeleton_tolerance, (int, np.integer)) or skeleton_tolerance < 0:
        raise ValueError("skeleton_tolerance must be a non-negative integer")
    if not 0 < branch_overlap_threshold <= 1:
        raise ValueError("branch_overlap_threshold must lie in (0, 1]")

    known, ambiguous_pixels, unlabeled_pixels = _partition_known_masks(
        known_positive_masks
    )
    predicted = {
        name: _as_bool_mask(mask, f"predicted_masks[{name!r}]")
        for name, mask in predicted_masks.items()
    }
    expected_shape = next(iter(known.values())).shape
    if any(mask.shape != expected_shape for mask in predicted.values()):
        raise ValueError("predicted and known-positive masks must have the same shape")

    annotated_pixels = ambiguous_pixels | np.any(
        np.stack(list(known.values())), axis=0
    )
    if evaluation_mask is None:
        if labeled_vessels is None:
            evaluation_mask = np.ones(expected_shape, dtype=bool)
        else:
            labeled_vessels_array = np.asarray(labeled_vessels)
            if labeled_vessels_array.shape != expected_shape:
                raise ValueError("labeled_vessels must have the same shape as the masks")
            evaluation_mask = (labeled_vessels_array > 0) | annotated_pixels
    else:
        evaluation_mask = _as_bool_mask(evaluation_mask, "evaluation_mask")
        if evaluation_mask.shape != expected_shape:
            raise ValueError("evaluation_mask must have the same shape as the masks")
        if np.any(annotated_pixels & ~evaluation_mask):
            raise ValueError("evaluation_mask must include every annotated pixel")

    unlabeled_pixels &= evaluation_mask
    evaluable_pixels = evaluation_mask & ~ambiguous_pixels
    metrics = {"evaluation_pixel_count": int(np.count_nonzero(evaluable_pixels))}

    for name, pred in predicted.items():
        pred = pred & evaluation_mask
        positive = known[name]
        other_known_masks = [
            mask for other, mask in known.items() if other != name
        ]
        known_negative = (
            np.any(np.stack(other_known_masks), axis=0)
            if other_known_masks
            else np.zeros(expected_shape, dtype=bool)
        )
        recall, contamination = _positive_recall_and_contamination(
            pred,
            positive,
            known_negative,
        )
        metrics[f"positive_recall_{name}"] = recall
        metrics[f"contamination_{name}"] = contamination
        known_precision, known_f1 = _known_precision_and_f1(
            pred,
            positive,
            known_negative,
            recall,
        )
        metrics[f"known_precision_{name}"] = known_precision
        metrics[f"known_f1_{name}"] = known_f1
        metrics[f"unlabeled_prediction_rate_{name}"] = _safe_fraction(
            np.count_nonzero(pred & unlabeled_pixels),
            np.count_nonzero(unlabeled_pixels),
        )
        metrics[f"prediction_rate_{name}"] = _safe_fraction(
            np.count_nonzero(pred & evaluable_pixels),
            np.count_nonzero(evaluable_pixels),
        )
        metrics[f"known_positive_count_{name}"] = int(np.count_nonzero(positive))
        metrics[f"known_negative_count_{name}"] = int(
            np.count_nonzero(known_negative)
        )
        metrics[f"predicted_count_{name}"] = int(
            np.count_nonzero(pred & evaluable_pixels)
        )

        pred_skeleton = _dilated_skeleton(pred, skeleton_tolerance)
        positive_skeleton = skeletonize(positive)
        negative_skeleton = skeletonize(known_negative)
        skeleton_recall, skeleton_contamination = (
            _positive_recall_and_contamination(
                pred_skeleton,
                positive_skeleton,
                negative_skeleton,
            )
        )
        metrics[f"skeleton_positive_recall_{name}"] = skeleton_recall
        metrics[f"skeleton_contamination_{name}"] = skeleton_contamination
        skeleton_precision, skeleton_f1 = _known_precision_and_f1(
            pred_skeleton,
            positive_skeleton,
            negative_skeleton,
            skeleton_recall,
        )
        metrics[f"skeleton_known_precision_{name}"] = skeleton_precision
        metrics[f"skeleton_known_f1_{name}"] = skeleton_f1

    if labeled_vessels is not None:
        labeled_vessels = np.asarray(labeled_vessels)
        if labeled_vessels.shape != expected_shape:
            raise ValueError("labeled_vessels must have the same shape as the masks")
        if not np.issubdtype(labeled_vessels.dtype, np.integer):
            raise ValueError("labeled_vessels must contain integer branch IDs")

        predicted_branches = _branch_sets(
            predicted,
            labeled_vessels,
            branch_overlap_threshold,
        )
        # A partial annotation identifies a branch even when it covers less than
        # half of that branch. Branches annotated as several classes stay unknown.
        known_branches = {
            name: set(np.unique(labeled_vessels[mask])) - {0}
            for name, mask in known.items()
        }
        ambiguous = set()
        names = tuple(known_branches)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                ambiguous |= known_branches[name] & known_branches[other]
        known_branches = {
            name: branches - ambiguous for name, branches in known_branches.items()
        }
        all_branches = set(int(branch_id) for branch_id in np.unique(labeled_vessels))
        all_branches.discard(0)
        known_branch_union = set().union(*known_branches.values())
        unlabeled_branches = all_branches - known_branch_union - ambiguous
        evaluable_branches = all_branches - ambiguous

        for name, pred_branches in predicted_branches.items():
            positive_branches = known_branches[name]
            negative_branches = set().union(
                *(branches for other, branches in known_branches.items() if other != name)
            )
            branch_recall = _safe_fraction(
                len(pred_branches & positive_branches),
                len(positive_branches),
            )
            branch_contamination = _safe_fraction(
                len(pred_branches & negative_branches),
                len(negative_branches),
            )
            branch_true_positive = len(pred_branches & positive_branches)
            branch_known_false_positive = len(pred_branches & negative_branches)
            branch_precision = _safe_fraction(
                branch_true_positive,
                branch_true_positive + branch_known_false_positive,
            )
            if np.isnan(branch_precision) or np.isnan(branch_recall):
                branch_f1 = np.nan
            elif branch_precision + branch_recall == 0:
                branch_f1 = 0.0
            else:
                branch_f1 = (
                    2.0
                    * branch_precision
                    * branch_recall
                    / (branch_precision + branch_recall)
                )
            metrics[f"branch_positive_recall_{name}"] = branch_recall
            metrics[f"branch_contamination_{name}"] = branch_contamination
            metrics[f"branch_known_precision_{name}"] = branch_precision
            metrics[f"branch_known_f1_{name}"] = branch_f1
            metrics[f"branch_unlabeled_prediction_rate_{name}"] = _safe_fraction(
                len(pred_branches & unlabeled_branches),
                len(unlabeled_branches),
            )
            metrics[f"branch_prediction_rate_{name}"] = _safe_fraction(
                len(pred_branches & evaluable_branches),
                len(evaluable_branches),
            )
            metrics[f"known_positive_branch_count_{name}"] = len(positive_branches)
            metrics[f"known_negative_branch_count_{name}"] = len(negative_branches)
            metrics[f"predicted_branch_count_{name}"] = len(
                pred_branches & evaluable_branches
            )

    class_names = tuple(predicted)
    metric_families = [
        "positive_recall",
        "contamination",
        "known_precision",
        "known_f1",
        "unlabeled_prediction_rate",
        "prediction_rate",
        "skeleton_positive_recall",
        "skeleton_contamination",
        "skeleton_known_precision",
        "skeleton_known_f1",
    ]
    if labeled_vessels is not None:
        metric_families.extend(
            [
                "branch_positive_recall",
                "branch_contamination",
                "branch_known_precision",
                "branch_known_f1",
                "branch_unlabeled_prediction_rate",
                "branch_prediction_rate",
            ]
        )
    for family in metric_families:
        metrics[f"{family}_macro"] = _nanmean_or_nan(
            [metrics[f"{family}_{name}"] for name in class_names]
        )

    return metrics


def assign_clusters_to_correlation_stack(
    cluster_labels,
    X,
    labeled_vessels,
    negative=False,
):
    """Assign two correlation clusters and map rows to the actual branch IDs."""
    cluster_labels = np.asarray(cluster_labels)
    X = np.asarray(X)
    unique_clusters = np.unique(cluster_labels)
    if unique_clusters.size != 2:
        raise ValueError("correlation assignment requires exactly two clusters")
    if X.ndim != 2 or len(X) != len(cluster_labels):
        raise ValueError("X and cluster_labels must have matching sample dimensions")

    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if branch_ids.size != len(cluster_labels):
        raise ValueError("cluster_labels must contain one value per labeled branch")

    c0, c1 = unique_clusters
    index0 = np.flatnonzero(cluster_labels == c0)
    index1 = np.flatnonzero(cluster_labels == c1)
    correlation0 = np.median(X[index0], axis=0)
    correlation1 = np.median(X[index1], axis=0)
    if negative:
        correlation0 = -correlation0
        correlation1 = -correlation1
    artery_cluster = c0 if np.max(correlation0) > np.max(correlation1) else c1

    artery_rows = np.flatnonzero(cluster_labels == artery_cluster)
    vein_rows = np.flatnonzero(cluster_labels != artery_cluster)
    artery_mask = np.isin(labeled_vessels, branch_ids[artery_rows])
    vein_mask = np.isin(labeled_vessels, branch_ids[vein_rows])
    branch_labels = np.where(cluster_labels == artery_cluster, 1, 2)
    return artery_mask, vein_mask, branch_labels


def _round_metrics(metrics, decimals):
    if decimals is None:
        return metrics
    return {
        name: round(float(value), decimals) if np.isfinite(value) else float(value)
        for name, value in metrics.items()
    }


def _evaluate_annotated_branches(result, gt_branch_labels):
    gt_branch_labels = np.asarray(gt_branch_labels)
    pred_branch_labels = np.asarray(result.mask_labels)
    if gt_branch_labels.ndim != 1 or pred_branch_labels.shape != gt_branch_labels.shape:
        raise ValueError("ground truth and predicted branch labels must be matching vectors")
    annotated = gt_branch_labels > 0
    if not np.any(annotated):
        raise ValueError("at least one branch must have a positive ground-truth label")

    gt = gt_branch_labels[annotated]
    pred = pred_branch_labels[annotated]
    class_labels = np.unique(gt)
    metrics = {
        "annotated_branch_count": int(np.count_nonzero(annotated)),
        "ARI": adjusted_rand_score(gt, pred),
        "NMI": normalized_mutual_info_score(gt, pred),
        "accuracy": accuracy_score(gt, pred),
        "balanced_accuracy": balanced_accuracy_score(gt, pred),
        "precision": precision_score(
            gt, pred, labels=class_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            gt, pred, labels=class_labels, average="macro", zero_division=0
        ),
        "f1": f1_score(
            gt, pred, labels=class_labels, average="macro", zero_division=0
        ),
    }
    for class_label in class_labels:
        suffix = str(class_label)
        metrics[f"precision_class_{suffix}"] = precision_score(
            gt, pred, labels=[class_label], average="macro", zero_division=0
        )
        metrics[f"recall_class_{suffix}"] = recall_score(
            gt, pred, labels=[class_label], average="macro", zero_division=0
        )
        metrics[f"f1_class_{suffix}"] = f1_score(
            gt, pred, labels=[class_label], average="macro", zero_division=0
        )
    return metrics


def _evaluate_complete_masks(result, gt_artery_mask, gt_vein_mask):
    metrics = {}
    for name, pred_mask, gt_mask in (
        ("artery", result.artery_mask, gt_artery_mask),
        ("vein", result.vein_mask, gt_vein_mask),
    ):
        metrics[f"dice_{name}"] = dice_score(pred_mask, gt_mask)
        metrics[f"iou_{name}"] = iou_score(pred_mask, gt_mask)
        metrics[f"cldice_{name}"] = cldice_score(pred_mask, gt_mask)
        metrics[f"hd95_{name}"] = hd95_score(pred_mask, gt_mask)

    for metric_name in ("dice", "iou", "cldice", "hd95"):
        metrics[f"{metric_name}_mean"] = np.mean(
            [metrics[f"{metric_name}_artery"], metrics[f"{metric_name}_vein"]]
        )
    return metrics


def evaluate_experiment(
    result=None,
    gt_branch_labels=None,
    gt_artery_mask=None,
    gt_vein_mask=None,
    decimals=2,
    *,
    pu_predicted_masks=None,
    pu_known_positive_masks=None,
    pu_labeled_vessels=None,
    pu_evaluation_mask=None,
    pu_skeleton_tolerance=1,
    pu_branch_overlap_threshold=0.5,
    partial_branch_targets=None,
    partial_branch_weights=None,
    signal_videos=None,
    signal_predicted_masks=None,
    signal_reference_masks=None,
    signal_sampling_frequency=None,
    signal_beat_period=None,
    signal_frame_mask=None,
    signal_exclude_reference_pixels=True,
):
    """Evaluate an experiment using whichever evidence is available.

    Complete retinal masks can be supplied through ``gt_artery_mask`` and
    ``gt_vein_mask``. Incomplete choroidal annotations must instead be supplied
    through ``pu_known_positive_masks``; their metrics are prefixed with ``pu_``.
    This distinction prevents unlabeled choroid pixels from being silently treated
    as negatives. Three-class PU evaluation requires an explicit
    ``pu_predicted_masks`` mapping.
    """
    metrics = {}
    if result is not None:
        X = np.asarray(result.X)
        cluster_labels = np.asarray(result.cluster_labels)
        if X.ndim != 2 or len(X) != len(cluster_labels):
            raise ValueError("result.X and cluster_labels must have matching rows")

        assigned = cluster_labels >= 0
        assigned_labels = cluster_labels[assigned]
        assigned_X = X[assigned]
        cluster_count = np.unique(assigned_labels).size
        noise_count = int(np.count_nonzero(~assigned))
        metrics.update(
            {
                "cluster_count": int(cluster_count),
                "noise_count": noise_count,
                "noise_fraction": noise_count / len(cluster_labels),
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
            }
        )
        if 1 < cluster_count < len(assigned_labels):
            metrics.update(
                {
                    "silhouette": silhouette_score(assigned_X, assigned_labels),
                    "davies_bouldin": davies_bouldin_score(
                        assigned_X, assigned_labels
                    ),
                    "calinski_harabasz": calinski_harabasz_score(
                        assigned_X, assigned_labels
                    ),
                }
            )

    if gt_branch_labels is not None:
        if result is None:
            raise ValueError("result is required for branch-label evaluation")
        metrics.update(_evaluate_annotated_branches(result, gt_branch_labels))

    complete_masks_supplied = (
        gt_artery_mask is not None,
        gt_vein_mask is not None,
    )
    if any(complete_masks_supplied) and not all(complete_masks_supplied):
        raise ValueError("gt_artery_mask and gt_vein_mask must be supplied together")
    if all(complete_masks_supplied):
        if result is None:
            raise ValueError("result is required for complete-mask evaluation")
        metrics.update(_evaluate_complete_masks(result, gt_artery_mask, gt_vein_mask))

    if pu_predicted_masks is not None and pu_known_positive_masks is None:
        raise ValueError(
            "pu_known_positive_masks is required when pu_predicted_masks is supplied"
        )
    if pu_known_positive_masks is not None:
        if pu_predicted_masks is None:
            if result is None:
                raise ValueError(
                    "result or pu_predicted_masks is required for PU evaluation"
                )
            available_masks = {
                "artery": getattr(result, "artery_mask", None),
                "vein": getattr(result, "vein_mask", None),
            }
            if not set(pu_known_positive_masks).issubset(available_masks) or any(
                available_masks[name] is None for name in pu_known_positive_masks
            ):
                raise ValueError(
                    "pu_predicted_masks is required for classes not exposed by result"
                )
            pu_predicted_masks = {
                name: available_masks[name] for name in pu_known_positive_masks
            }
        pu_metrics = evaluate_positive_unlabeled_masks(
            pu_predicted_masks,
            pu_known_positive_masks,
            labeled_vessels=pu_labeled_vessels,
            evaluation_mask=pu_evaluation_mask,
            skeleton_tolerance=pu_skeleton_tolerance,
            branch_overlap_threshold=pu_branch_overlap_threshold,
        )
        metrics.update({f"pu_{name}": value for name, value in pu_metrics.items()})

    if partial_branch_targets is not None:
        if result is None:
            raise ValueError("result is required for partial branch evaluation")
        metrics.update(
            evaluate_partial_branch_clustering(
                result.cluster_labels,
                partial_branch_targets,
                branch_ids=getattr(result, "branch_ids", None),
                sample_weight=partial_branch_weights,
            )
        )

    signal_inputs = (
        signal_videos,
        signal_predicted_masks,
        signal_reference_masks,
        signal_sampling_frequency,
        signal_beat_period,
    )
    if any(value is not None for value in signal_inputs):
        if not all(value is not None for value in signal_inputs):
            raise ValueError(
                "signal videos, masks, sampling frequency, and beat period "
                "must be supplied together"
            )
        metrics.update(
            evaluate_mask_signal_similarity(
                signal_videos,
                signal_predicted_masks,
                signal_reference_masks,
                sampling_frequency=signal_sampling_frequency,
                beat_period=signal_beat_period,
                frame_mask=signal_frame_mask,
                exclude_reference_pixels=signal_exclude_reference_pixels,
            )
        )

    if not metrics:
        raise ValueError("no evaluation inputs were supplied")
    return _round_metrics(metrics, decimals)


def save_experiment_h5(h5_path, experiment_name, result, metadata, metrics=None):
    """Persist one clustering result, replacing an experiment of the same name."""
    with h5py.File(h5_path, "a") as file:
        if experiment_name in file:
            del file[experiment_name]
        group = file.create_group(experiment_name)
        group.create_dataset("embedding_matrix", data=result.X, compression="gzip")
        group.create_dataset("cluster_labels", data=result.cluster_labels)
        group.create_dataset("mask_labels", data=result.mask_labels)
        group.create_dataset("artery_mask", data=np.asarray(result.artery_mask, dtype=np.uint8))
        group.create_dataset("vein_mask", data=np.asarray(result.vein_mask, dtype=np.uint8))
        for key, value in metadata.items():
            if np.ndim(value) == 0 or isinstance(value, str):
                group.attrs[key] = value
            else:
                group.create_dataset(key, data=value)
        if metrics is not None:
            metrics_group = group.create_group("metrics")
            for key, value in metrics.items():
                metrics_group.create_dataset(key, data=value)
