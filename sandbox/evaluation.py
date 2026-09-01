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
import logging
import numpy as np
import dopplerview.segmentation.pulse_analysis as pa
import dopplerview.segmentation.signal_processing as sp

logger = logging.getLogger(__name__)

def assign_clusters_to_correlation_stack(
    cluster_labels,
    X,
    labeled_vessels,
    negative=False
):
    """
    Assign artery/vein labels from cluster labels.

    Assumes two clusters.
    """

    # Assign artery and vein based on correlation
    cluster0 = np.where(cluster_labels == 0)[0]
    cluster1 = np.where(cluster_labels == 1)[0]
    correlation0 = np.median(X[cluster0], axis=0)
    correlation1 = np.median(X[cluster1], axis=0)
    
    if negative:
        correlation0 = -correlation0
        correlation1 = -correlation1

    if np.max(correlation0) > np.max(correlation1):
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)
        artery_mask[np.isin(labeled_vessels, cluster0 + 1)] = True
        vein_mask[np.isin(labeled_vessels, cluster1 + 1)] = True
    else:
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)
        artery_mask[np.isin(labeled_vessels, cluster1 + 1)] = True
        vein_mask[np.isin(labeled_vessels, cluster0 + 1)] = True

    mask_labels = np.zeros_like(labeled_vessels, dtype=int)
    mask_labels[artery_mask] = 1
    mask_labels[vein_mask] = 2

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
    decimals=2
):
    metrics = {}

    X = result.X
    cluster_labels = result.cluster_labels

    # ------------------
    # clustering metrics
    # ------------------

    if len(np.unique(cluster_labels)) > 1:

        metrics["silhouette"] = round((
            silhouette_score(
                X,
                cluster_labels,
            )
        ), decimals)

        metrics["davies_bouldin"] = round((
            davies_bouldin_score(
                X,
                cluster_labels,
            )
        ), decimals)

        metrics["calinski_harabasz"] = round((
            calinski_harabasz_score(
                X,
                cluster_labels,
            )
        ), decimals)

    # ------------------
    # branch metrics
    # ------------------

    # valid = gt_branch_labels > 0

    gt = gt_branch_labels
    pred = result.mask_labels

    metrics["ARI"] = round(adjusted_rand_score(
        gt,
        pred,
    ), decimals)

    metrics["NMI"] = round((
        normalized_mutual_info_score(
            gt,
            pred,
        )
    ), decimals)

    metrics["accuracy"] = round((
        accuracy_score(
            gt,
            pred,
        )
    ), decimals)

    metrics["balanced_accuracy"] = round((
        balanced_accuracy_score(
            gt,
            pred,
        )
    ), decimals)

    metrics["precision"] = round((
        precision_score(
            gt,
            pred,
            average="micro",
            zero_division=0,
        )
    ), decimals)

    metrics["recall"] = round((
        recall_score(
            gt,
            pred,
            average="micro",
            zero_division=0,
        )
    ), decimals)

    metrics["f1"] = round((
        f1_score(
            gt,
            pred,
            average="micro",
            zero_division=0,
        )
    ), decimals)

    # ------------------
    # segmentation
    # ------------------

    metrics["dice_artery"] = (
        dice_score(
            result.artery_mask,
            gt_artery_mask,
        ).round(decimals).item()
    )

    metrics["dice_vein"] = (
        dice_score(
            result.vein_mask,
            gt_vein_mask,
        ).round(decimals).item()
    )

    metrics["iou_artery"] = (
        iou_score(
            result.artery_mask,
            gt_artery_mask,
        ).round(decimals).item()
    )

    metrics["iou_vein"] = (
        iou_score(
            result.vein_mask,
            gt_vein_mask,
        ).round(decimals).item()
    )

    metrics["dice_mean"] = round((
        metrics["dice_artery"]
        + metrics["dice_vein"]
    ) / 2, decimals)

    metrics["iou_mean"] = round((
        metrics["iou_artery"]
        + metrics["iou_vein"]
    ) / 2, decimals)

    return metrics

def save_experiment_h5(
    h5_path,
    experiment_name,
    result,
    metadata,
):
    """
    Save one experiment.
    """

    with h5py.File(
        h5_path,
        "a",
    ) as f:

        if experiment_name in f:
            del f[experiment_name]

        g = f.create_group(
            experiment_name
        )

        g.create_dataset(
            "embedding_matrix",
            data=result.X,
            compression="gzip",
        )

        g.create_dataset(
            "cluster_labels",
            data=result.cluster_labels,
        )

        g.create_dataset(
            "mask_labels",
            data=result.mask_labels,
        )

        g.create_dataset(
            "artery_mask",
            data=result.artery_mask.astype(
                np.uint8
            ),
        )

        g.create_dataset(
            "vein_mask",
            data=result.vein_mask.astype(
                np.uint8
            ),
        )

        for k, v in metadata.items():
            g.attrs[k] = v
