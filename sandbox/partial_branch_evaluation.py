"""Evaluation of clustering against incomplete branch annotations."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    recall_score,
    v_measure_score,
)


UNLABELED = -1
AMBIGUOUS = -2


@dataclass(frozen=True)
class PartialBranchTargets:
    """Branch-aligned evidence derived from incomplete positive masks."""

    branch_ids: np.ndarray
    branch_sizes: np.ndarray
    class_names: tuple[str, ...]
    positive_pixels: np.ndarray
    positive_fractions: np.ndarray
    ambiguous_pixels: np.ndarray
    annotated_pixels: np.ndarray
    annotated_fraction: np.ndarray
    dominant_fraction: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray

    @property
    def labeled_mask(self):
        return self.labels >= 0

    @property
    def ambiguous_mask(self):
        return self.labels == AMBIGUOUS

    def records(self):
        """Return flat records suitable for ``pandas.DataFrame``."""
        records = []
        for row, branch_id in enumerate(self.branch_ids):
            label = int(self.labels[row])
            record = {
                "branch_id": int(branch_id),
                "branch_size": int(self.branch_sizes[row]),
                "annotated_pixels": int(self.annotated_pixels[row]),
                "ambiguous_pixels": int(self.ambiguous_pixels[row]),
                "annotated_fraction": float(self.annotated_fraction[row]),
                "dominant_fraction": float(self.dominant_fraction[row]),
                "target_label": label,
                "target_class": self.class_names[label] if label >= 0 else None,
                "target_status": (
                    "labeled"
                    if label >= 0
                    else "ambiguous"
                    if label == AMBIGUOUS
                    else "unlabeled"
                ),
                "confidence": float(self.confidence[row]),
            }
            for column, class_name in enumerate(self.class_names):
                record[f"{class_name}_pixels"] = int(
                    self.positive_pixels[row, column]
                )
                record[f"{class_name}_branch_fraction"] = float(
                    self.positive_fractions[row, column]
                )
            records.append(record)
        return records


def _validate_masks(labeled_vessels, known_positive_masks):
    labeled_vessels = np.asarray(labeled_vessels)
    if labeled_vessels.ndim != 2 or not np.issubdtype(
        labeled_vessels.dtype, np.integer
    ):
        raise ValueError("labeled_vessels must be a two-dimensional integer image")
    if not known_positive_masks:
        raise ValueError("known_positive_masks cannot be empty")
    masks = {
        str(name): np.asarray(mask, dtype=bool)
        for name, mask in known_positive_masks.items()
    }
    if any(mask.shape != labeled_vessels.shape for mask in masks.values()):
        raise ValueError("all masks must have the same shape as labeled_vessels")
    branch_ids, branch_sizes = np.unique(labeled_vessels, return_counts=True)
    positive = branch_ids > 0
    if not np.any(positive):
        raise ValueError("labeled_vessels does not contain a positive branch ID")
    return labeled_vessels, masks, branch_ids[positive], branch_sizes[positive]


def build_partial_branch_targets(
    labeled_vessels,
    known_positive_masks,
    *,
    min_annotated_pixels=3,
    min_dominance=0.8,
    min_annotated_fraction=0.0,
    evidence_saturation_pixels=10,
):
    """Create branch targets without requiring masks to cover whole branches.

    Pixels annotated as several classes are counted as ambiguous and excluded from
    class evidence. A branch receives its dominant class only when it has enough
    exclusive pixels and that class reaches ``min_dominance`` among those pixels.
    """
    if not isinstance(min_annotated_pixels, (int, np.integer)) or min_annotated_pixels < 1:
        raise ValueError("min_annotated_pixels must be a positive integer")
    if not 0.5 <= min_dominance <= 1:
        raise ValueError("min_dominance must lie in [0.5, 1]")
    if not 0 <= min_annotated_fraction <= 1:
        raise ValueError("min_annotated_fraction must lie in [0, 1]")
    if evidence_saturation_pixels <= 0:
        raise ValueError("evidence_saturation_pixels must be positive")

    labeled_vessels, masks, branch_ids, branch_sizes = _validate_masks(
        labeled_vessels, known_positive_masks
    )
    class_names = tuple(masks)
    mask_stack = np.stack([masks[name] for name in class_names])
    annotation_multiplicity = np.sum(mask_stack, axis=0)
    exclusive_stack = mask_stack & (annotation_multiplicity[None] == 1)

    positive_pixels = np.zeros((len(branch_ids), len(class_names)), dtype=int)
    ambiguous_pixels = np.zeros(len(branch_ids), dtype=int)
    for row, branch_id in enumerate(branch_ids):
        branch = labeled_vessels == branch_id
        positive_pixels[row] = np.count_nonzero(exclusive_stack & branch, axis=(1, 2))
        ambiguous_pixels[row] = np.count_nonzero(
            branch & (annotation_multiplicity > 1)
        )

    annotated_pixels = np.sum(positive_pixels, axis=1)
    positive_fractions = positive_pixels / branch_sizes[:, None]
    annotated_fraction = annotated_pixels / branch_sizes
    dominant_index = np.argmax(positive_pixels, axis=1)
    dominant_pixels = positive_pixels[np.arange(len(branch_ids)), dominant_index]
    dominant_fraction = np.divide(
        dominant_pixels,
        annotated_pixels,
        out=np.zeros(len(branch_ids), dtype=float),
        where=annotated_pixels > 0,
    )

    enough_evidence = (
        (dominant_pixels >= min_annotated_pixels)
        & (annotated_fraction >= min_annotated_fraction)
    )
    labeled = enough_evidence & (dominant_fraction >= min_dominance)
    conflicting = enough_evidence & ~labeled
    ambiguous_only = (annotated_pixels == 0) & (ambiguous_pixels > 0)

    labels = np.full(len(branch_ids), UNLABELED, dtype=int)
    labels[labeled] = dominant_index[labeled]
    labels[conflicting | ambiguous_only] = AMBIGUOUS
    confidence = np.zeros(len(branch_ids), dtype=float)
    confidence[labeled] = dominant_fraction[labeled] * np.minimum(
        1.0,
        dominant_pixels[labeled] / float(evidence_saturation_pixels),
    )

    return PartialBranchTargets(
        branch_ids=branch_ids,
        branch_sizes=branch_sizes,
        class_names=class_names,
        positive_pixels=positive_pixels,
        positive_fractions=positive_fractions,
        ambiguous_pixels=ambiguous_pixels,
        annotated_pixels=annotated_pixels,
        annotated_fraction=annotated_fraction,
        dominant_fraction=dominant_fraction,
        labels=labels,
        confidence=confidence,
    )


def _validate_clustering_inputs(cluster_labels, targets, branch_ids, sample_weight):
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.shape != targets.labels.shape:
        raise ValueError("cluster_labels must contain one value per target branch")
    if not np.issubdtype(cluster_labels.dtype, np.integer):
        raise ValueError("cluster_labels must contain integers")
    if branch_ids is not None and not np.array_equal(branch_ids, targets.branch_ids):
        raise ValueError("branch_ids are not aligned with partial branch targets")
    if sample_weight is None:
        sample_weight = np.ones(len(cluster_labels), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != cluster_labels.shape:
            raise ValueError("sample_weight must contain one value per branch")
        if np.any(sample_weight <= 0) or not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight must be finite and strictly positive")
    return cluster_labels, sample_weight


def _pair_mass(labels, weights):
    total = 0.0
    for value in np.unique(labels):
        group = weights[labels == value]
        total += 0.5 * (np.sum(group) ** 2 - np.sum(group**2))
    return total


def weighted_adjusted_rand_score(labels_true, labels_pred, sample_weight):
    """Continuous pair-mass analogue of adjusted Rand for branch weights."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    weights = np.asarray(sample_weight, dtype=float)
    if labels_true.shape != labels_pred.shape or weights.shape != labels_true.shape:
        raise ValueError("labels and sample_weight must have matching shapes")
    total_pairs = 0.5 * (np.sum(weights) ** 2 - np.sum(weights**2))
    if total_pairs <= 0:
        return np.nan
    true_pairs = _pair_mass(labels_true, weights)
    pred_pairs = _pair_mass(labels_pred, weights)
    joint_pairs = 0.0
    for true_label in np.unique(labels_true):
        for pred_label in np.unique(labels_pred):
            group = weights[(labels_true == true_label) & (labels_pred == pred_label)]
            joint_pairs += 0.5 * (np.sum(group) ** 2 - np.sum(group**2))
    expected = true_pairs * pred_pairs / total_pairs
    maximum = 0.5 * (true_pairs + pred_pairs)
    denominator = maximum - expected
    return 1.0 if denominator == 0 and joint_pairs == maximum else (
        np.nan if denominator == 0 else float((joint_pairs - expected) / denominator)
    )


def _weighted_information_scores(labels_true, labels_pred, weights):
    true_values, true_inverse = np.unique(labels_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=float)
    np.add.at(contingency, (true_inverse, pred_inverse), weights)
    contingency /= np.sum(contingency)
    true_probability = np.sum(contingency, axis=1)
    pred_probability = np.sum(contingency, axis=0)

    def entropy(probability):
        positive = probability > 0
        return float(-np.sum(probability[positive] * np.log(probability[positive])))

    true_entropy = entropy(true_probability)
    pred_entropy = entropy(pred_probability)
    expected = true_probability[:, None] * pred_probability[None, :]
    positive = contingency > 0
    mutual_information = float(
        np.sum(contingency[positive] * np.log(contingency[positive] / expected[positive]))
    )
    homogeneity = 1.0 if true_entropy == 0 else mutual_information / true_entropy
    completeness = 1.0 if pred_entropy == 0 else mutual_information / pred_entropy
    v_measure = (
        0.0
        if homogeneity + completeness == 0
        else 2 * homogeneity * completeness / (homogeneity + completeness)
    )
    mean_entropy = 0.5 * (true_entropy + pred_entropy)
    nmi = 1.0 if mean_entropy == 0 else mutual_information / mean_entropy
    return homogeneity, completeness, v_measure, nmi


def _cluster_class_mapping(labels_true, labels_pred, weights):
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred[labels_pred >= 0])
    if not len(clusters):
        return {}
    scores = np.zeros((len(clusters), len(classes)), dtype=float)
    for row, cluster in enumerate(clusters):
        for column, class_label in enumerate(classes):
            scores[row, column] = np.sum(
                weights[(labels_pred == cluster) & (labels_true == class_label)]
            )
    rows, columns = linear_sum_assignment(-scores)
    return {int(clusters[row]): int(classes[column]) for row, column in zip(rows, columns)}


def _external_scores(labels_true, labels_pred, weights, prefix):
    scores = {
        f"{prefix}ARI": adjusted_rand_score(labels_true, labels_pred),
        f"{prefix}NMI": normalized_mutual_info_score(labels_true, labels_pred),
        f"{prefix}homogeneity": homogeneity_score(labels_true, labels_pred),
        f"{prefix}completeness": completeness_score(labels_true, labels_pred),
        f"{prefix}v_measure": v_measure_score(labels_true, labels_pred),
        f"{prefix}weighted_ARI": weighted_adjusted_rand_score(
            labels_true, labels_pred, weights
        ),
    }
    weighted = _weighted_information_scores(labels_true, labels_pred, weights)
    for name, value in zip(
        ("weighted_homogeneity", "weighted_completeness", "weighted_v_measure", "weighted_NMI"),
        weighted,
    ):
        scores[f"{prefix}{name}"] = value
    return scores


def evaluate_partial_branch_clustering(
    cluster_labels,
    targets,
    *,
    branch_ids=None,
    sample_weight=None,
):
    """Evaluate arbitrary cluster IDs on the exclusively annotated branches."""
    cluster_labels, sample_weight = _validate_clustering_inputs(
        cluster_labels, targets, branch_ids, sample_weight
    )
    labeled = targets.labeled_mask
    if np.count_nonzero(labeled) < 2:
        raise ValueError("at least two partially labeled branches are required")
    true = targets.labels[labeled]
    pred = cluster_labels[labeled]
    weights = sample_weight[labeled] * targets.confidence[labeled]
    if not np.any(weights > 0):
        raise ValueError("partially labeled branches have zero total confidence")
    assigned = pred >= 0

    metrics = {
        "partial_branch_count": int(len(cluster_labels)),
        "partial_labeled_branch_count": int(np.count_nonzero(labeled)),
        "partial_ambiguous_branch_count": int(np.count_nonzero(targets.ambiguous_mask)),
        "partial_unlabeled_branch_count": int(np.count_nonzero(targets.labels == UNLABELED)),
        "partial_cluster_count": int(np.unique(cluster_labels[cluster_labels >= 0]).size),
        "partial_noise_branch_count": int(np.count_nonzero(cluster_labels < 0)),
        "partial_labeled_coverage": float(np.mean(assigned)),
        "partial_overall_coverage": float(np.mean(cluster_labels >= 0)),
        "partial_effective_label_weight_sum": float(np.sum(weights)),
    }
    metrics.update(_external_scores(true, pred, weights, "partial_"))
    if np.count_nonzero(assigned) >= 2:
        metrics.update(
            _external_scores(
                true[assigned], pred[assigned], weights[assigned], "partial_assigned_"
            )
        )
    else:
        for name in (
            "ARI", "NMI", "homogeneity", "completeness", "v_measure",
            "weighted_ARI", "weighted_homogeneity", "weighted_completeness",
            "weighted_v_measure", "weighted_NMI",
        ):
            metrics[f"partial_assigned_{name}"] = np.nan

    mapping = _cluster_class_mapping(true, pred, weights)
    mapped = np.array([mapping.get(int(label), UNLABELED) for label in pred])
    classes = np.arange(len(targets.class_names))
    metrics.update(
        {
            "partial_mapped_accuracy_resubstitution": accuracy_score(true, mapped),
            "partial_mapped_balanced_accuracy_resubstitution": recall_score(
                true, mapped, labels=classes, average="macro", zero_division=0
            ),
            "partial_mapped_macro_f1_resubstitution": f1_score(
                true, mapped, labels=classes, average="macro", zero_division=0
            ),
            "partial_weighted_mapped_accuracy_resubstitution": accuracy_score(
                true, mapped, sample_weight=weights
            ),
            "partial_weighted_mapped_macro_f1_resubstitution": f1_score(
                true,
                mapped,
                labels=classes,
                average="macro",
                sample_weight=weights,
                zero_division=0,
            ),
        }
    )
    for class_label, class_name in enumerate(targets.class_names):
        metrics[f"partial_recall_{class_name}_resubstitution"] = recall_score(
            true,
            mapped,
            labels=[class_label],
            average="macro",
            zero_division=0,
        )
    return metrics
