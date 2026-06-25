from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import dopplerview.segmentation.pulse_analysis as pa

def assign_clusters_to_av(
    cluster_labels,
    signals,
    periods,
    labeled_vessels,
):
    """
    Assign artery/vein labels from cluster labels.

    Assumes two clusters.
    """

    unique_clusters = np.unique(cluster_labels)

    if len(unique_clusters) != 2:
        raise ValueError(
            "Current artery/vein assignment "
            "requires exactly 2 clusters."
        )

    c0, c1 = unique_clusters

    idx0 = np.where(cluster_labels == c0)[0]
    idx1 = np.where(cluster_labels == c1)[0]

    signal0 = np.median(
        signals[idx0],
        axis=0,
    )

    signal1 = np.median(
        signals[idx1],
        axis=0,
    )

    period0 = int(
        np.median(periods[idx0])
    )

    period1 = int(
        np.median(periods[idx1])
    )

    peaks0 = pa.get_nb_of_positive_peaks(
        signal0,
        period0,
    )

    peaks1 = pa.get_nb_of_positive_peaks(
        signal1,
        period1,
    )

    mask0 = np.isin(
        labeled_vessels,
        idx0 + 1,
    )

    mask1 = np.isin(
        labeled_vessels,
        idx1 + 1,
    )

    if peaks0 > peaks1:

        artery_mask = mask0
        vein_mask = mask1

        mask_labels = np.where(
            cluster_labels == c0,
            0,
            1,
        )

    else:

        artery_mask = mask1
        vein_mask = mask0

        mask_labels = np.where(
            cluster_labels == c0,
            1,
            0,
        )

    return (
        artery_mask,
        vein_mask,
        mask_labels,
    )

def dice_score(
    pred,
    gt,
):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt,
    ).sum()

    denom = pred.sum() + gt.sum()

    return (
        2 * inter
        / (denom + 1e-8)
    )

def iou_score(
    pred,
    gt,
):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt,
    ).sum()

    union = np.logical_or(
        pred,
        gt,
    ).sum()

    return inter / (union + 1e-8)


def evaluate_experiment(
    result,
    gt_branch_labels,
    gt_artery_mask,
    gt_vein_mask,
):
    metrics = {}

    X = result.X
    cluster_labels = result.cluster_labels

    # ------------------
    # clustering metrics
    # ------------------

    if len(np.unique(cluster_labels)) > 1:

        metrics["silhouette"] = (
            silhouette_score(
                X,
                cluster_labels,
            )
        )

        metrics["davies_bouldin"] = (
            davies_bouldin_score(
                X,
                cluster_labels,
            )
        )

        metrics["calinski_harabasz"] = (
            calinski_harabasz_score(
                X,
                cluster_labels,
            )
        )

    # ------------------
    # branch metrics
    # ------------------

    valid = gt_branch_labels > 0

    gt = gt_branch_labels[valid]
    pred = result.mask_labels[valid]

    metrics["ARI"] = adjusted_rand_score(
        gt,
        pred,
    )

    metrics["NMI"] = (
        normalized_mutual_info_score(
            gt,
            pred,
        )
    )

    metrics["accuracy"] = (
        accuracy_score(
            gt,
            pred,
        )
    )

    metrics["balanced_accuracy"] = (
        balanced_accuracy_score(
            gt,
            pred,
        )
    )

    metrics["precision"] = (
        precision_score(
            gt,
            pred,
            average="binary",
            zero_division=0,
        )
    )

    metrics["recall"] = (
        recall_score(
            gt,
            pred,
            average="binary",
            zero_division=0,
        )
    )

    metrics["f1"] = (
        f1_score(
            gt,
            pred,
            average="binary",
            zero_division=0,
        )
    )

    # ------------------
    # segmentation
    # ------------------

    metrics["dice_artery"] = (
        dice_score(
            result.artery_mask,
            gt_artery_mask,
        )
    )

    metrics["dice_vein"] = (
        dice_score(
            result.vein_mask,
            gt_vein_mask,
        )
    )

    metrics["iou_artery"] = (
        iou_score(
            result.artery_mask,
            gt_artery_mask,
        )
    )

    metrics["iou_vein"] = (
        iou_score(
            result.vein_mask,
            gt_vein_mask,
        )
    )

    metrics["dice_mean"] = (
        metrics["dice_artery"]
        + metrics["dice_vein"]
    ) / 2

    metrics["iou_mean"] = (
        metrics["iou_artery"]
        + metrics["iou_vein"]
    ) / 2

    return metrics