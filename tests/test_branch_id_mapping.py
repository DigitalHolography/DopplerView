import numpy as np

from benchmark import evaluation
from dopplerview.segmentation import process_masks, pulse_analysis


def test_branch_signal_rows_follow_present_noncontiguous_ids():
    video = np.zeros((10, 2, 2), dtype=float)
    video[:, 0, 0] = 2
    video[:, 1, 1] = 5
    labeled_vessels = np.array([[2, 0], [0, 5]])

    signals = pulse_analysis.get_filtered_branch_signals(
        video,
        labeled_vessels,
        sampling_frequency=40,
    )

    assert signals.shape == (2, 10)
    assert np.array_equal(signals[0], np.full(10, 2.0))
    assert np.array_equal(signals[1], np.full(10, 5.0))


def test_gradient_masks_map_signal_rows_to_actual_branch_ids(monkeypatch):
    labeled_vessels = np.array([[2, 2, 0], [0, 5, 5]])
    signals = np.ones((2, 20))
    monkeypatch.setattr(pulse_analysis, "compute_period", lambda *_args: 5)
    monkeypatch.setattr(
        pulse_analysis,
        "select_regular_peaks",
        lambda *_args: (np.array([1, 0]), []),
    )
    monkeypatch.setattr(pulse_analysis, "check_validity", lambda *_args: True)

    artery, vein = pulse_analysis.compute_pre_masks_by_systolic_gradient(
        signals,
        labeled_vessels,
        sampling_frequency=40,
    )

    assert np.array_equal(artery, labeled_vessels == 2)
    assert np.array_equal(vein, labeled_vessels == 5)


def test_clustering_masks_map_cluster_rows_to_actual_branch_ids(monkeypatch):
    class FakeKMeans:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, _features):
            return np.array([0, 1])

    labeled_vessels = np.array([[2, 2, 0], [0, 5, 5]])
    signals = np.vstack([np.full(20, 2.0), np.full(20, 5.0)])
    monkeypatch.setattr(pulse_analysis, "KMeans", FakeKMeans)
    monkeypatch.setattr(
        pulse_analysis,
        "get_cycle_template",
        lambda branch, *_args, **_kwargs: (branch, 5),
    )
    monkeypatch.setattr(
        pulse_analysis,
        "compute_z",
        lambda template: complex(template[0], 0),
    )
    monkeypatch.setattr(
        pulse_analysis,
        "get_nb_of_positive_peaks",
        lambda signal, _period: 2 if signal[0] == 2 else 1,
    )

    artery, vein, labels, _ = pulse_analysis.compute_pre_masks_by_clustering(
        signals,
        labeled_vessels,
        sampling_frequency=40,
    )

    assert np.array_equal(artery, labeled_vessels == 2)
    assert np.array_equal(vein, labeled_vessels == 5)
    assert np.array_equal(labels, [0, 1])


def test_branch_overlaps_exclude_missing_label_numbers():
    labeled_vessels = np.array([[2, 2, 0], [0, 5, 5]])
    artery_mask = labeled_vessels == 2
    vein_mask = labeled_vessels == 5

    overlaps = process_masks.compute_branch_overlaps(
        labeled_vessels,
        artery_mask,
        vein_mask,
    )

    assert np.array_equal(overlaps["branch_ids"], [2, 5])
    assert np.array_equal(overlaps["size"], [2, 2])
    assert np.array_equal(overlaps["artery_ratio"], [1.0, 0.0])
    assert np.array_equal(overlaps["vein_ratio"], [0.0, 1.0])


def test_branch_differences_use_compact_label_order():
    labeled_vessels = np.array([[2, 2, 0], [0, 5, 5]])

    differences = process_masks.get_branch_differences(
        pred_labels=np.array([1, 2]),
        gt_labels=np.array([1, 1]),
        labeled_vessels=labeled_vessels,
    )

    assert not np.any(differences[labeled_vessels == 2])
    assert np.all(differences[labeled_vessels == 5] == 5)


def test_benchmark_assignment_maps_clusters_to_actual_branch_ids(monkeypatch):
    labeled_vessels = np.array([[2, 2, 0], [0, 5, 5]])
    signals = np.vstack([np.full(20, 2.0), np.full(20, 5.0)])
    monkeypatch.setattr(
        evaluation.pa,
        "get_nb_of_positive_peaks",
        lambda signal, _period: 2 if signal[0] == 2 else 1,
    )

    artery, vein, labels = evaluation.assign_clusters_to_av(
        cluster_labels=np.array([0, 1]),
        signals=signals,
        periods=np.array([5, 5]),
        labeled_vessels=labeled_vessels,
    )

    assert np.array_equal(artery, labeled_vessels == 2)
    assert np.array_equal(vein, labeled_vessels == 5)
    assert np.array_equal(labels, [0, 1])
