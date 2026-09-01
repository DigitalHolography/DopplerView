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
    return {name: mask & unambiguous for name, mask in masks.items()}


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
    skeleton_tolerance=1,
    branch_overlap_threshold=0.5,
):
    """Evaluate incomplete multiclass masks without treating unlabeled pixels as negatives.

    For every class, annotated-positive recall measures coverage of that class's
    exclusive annotations. Contamination measures coverage of exclusive annotations
    belonging to all other classes. Undefined metrics have value ``numpy.nan``.
    """
    if set(predicted_masks) != set(known_positive_masks):
        raise ValueError("predicted and known-positive masks must use the same classes")
    if not isinstance(skeleton_tolerance, (int, np.integer)) or skeleton_tolerance < 0:
        raise ValueError("skeleton_tolerance must be a non-negative integer")
    if not 0 < branch_overlap_threshold <= 1:
        raise ValueError("branch_overlap_threshold must lie in (0, 1]")

    known = exclusive_known_masks(known_positive_masks)
    predicted = {
        name: _as_bool_mask(mask, f"predicted_masks[{name!r}]")
        for name, mask in predicted_masks.items()
    }
    expected_shape = next(iter(known.values())).shape
    if any(mask.shape != expected_shape for mask in predicted.values()):
        raise ValueError("predicted and known-positive masks must have the same shape")

    metrics = {}
    for name, pred in predicted.items():
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

        for name, pred_branches in predicted_branches.items():
            positive_branches = known_branches[name]
            negative_branches = set().union(
                *(branches for other, branches in known_branches.items() if other != name)
            )
            metrics[f"branch_positive_recall_{name}"] = _safe_fraction(
                len(pred_branches & positive_branches),
                len(positive_branches),
            )
            metrics[f"branch_contamination_{name}"] = _safe_fraction(
                len(pred_branches & negative_branches),
                len(negative_branches),
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


def evaluate_experiment(
    result,
    gt_branch_labels,
    gt_artery_mask,
    gt_vein_mask,
    decimals=2,
):
    """Evaluate clustering, annotated branches, and complete retinal masks."""
    X = np.asarray(result.X)
    cluster_labels = np.asarray(result.cluster_labels)
    if X.ndim != 2 or len(X) != len(cluster_labels):
        raise ValueError("result.X and cluster_labels must have matching rows")

    metrics = {
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    cluster_count = np.unique(cluster_labels).size
    if 1 < cluster_count < len(cluster_labels):
        metrics.update(
            {
                "silhouette": silhouette_score(X, cluster_labels),
                "davies_bouldin": davies_bouldin_score(X, cluster_labels),
                "calinski_harabasz": calinski_harabasz_score(X, cluster_labels),
            }
        )

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

    metrics.update(
        {
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
    )
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
