from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dopplerview.segmentation import pulse_analysis
from sandbox import benchmark, evaluation


def test_run_benchmark_extracts_only_present_branch_ids(monkeypatch, tmp_path):
    labeled = np.array([[2, 2, 0], [0, 5, 5]])
    video = np.zeros((4, 2, 3), dtype=float)
    video[:, labeled == 2] = 2.0
    video[:, labeled == 5] = 5.0
    captured = {}

    def fake_pipeline(**kwargs):
        captured["signals"] = kwargs["signals"].copy()
        return SimpleNamespace(
            X=np.array([[0.0], [1.0]]),
            cluster_labels=np.array([0, 1]),
            mask_labels=np.array([1, 2]),
            artery_mask=labeled == 2,
            vein_mask=labeled == 5,
            templates=None,
            periods=np.array([4, 4]),
        )

    monkeypatch.setattr(benchmark, "run_clustering_pipeline", fake_pipeline)
    monkeypatch.setattr(
        evaluation,
        "evaluate_experiment",
        lambda *_args, **_kwargs: {"accuracy": 1.0},
    )

    results = benchmark.run_benchmark(
        videos={"M0": video},
        labeled_vessels=labeled,
        gt_branch_labels=np.array([1, 2]),
        gt_artery_mask=labeled == 2,
        gt_vein_mask=labeled == 5,
        embeddings={"identity": lambda values: values},
        clusterings={"test": lambda values: np.array([0, 1])},
        sampling_frequency=40,
        h5_path=tmp_path / "benchmark.h5",
        beat_period=4,
    )

    assert captured["signals"].shape == (2, 4)
    np.testing.assert_array_equal(captured["signals"][0], np.full(4, 2.0))
    np.testing.assert_array_equal(captured["signals"][1], np.full(4, 5.0))
    assert list(results["experiment_name"]) == ["experiment_0001"]

    with h5py.File(tmp_path / "benchmark.h5", "r") as file:
        experiment = file["experiment_0001"]
        np.testing.assert_array_equal(experiment["branch_ids"], [2, 5])
        assert experiment["metrics"]["accuracy"][()] == 1.0


def test_pipeline_rejects_signal_rows_not_matching_present_branches():
    labeled = np.array([[2, 0], [0, 5]])
    with pytest.raises(ValueError, match="one row per labeled branch"):
        clustering.run_clustering_pipeline(
            signals=np.ones((5, 8)),
            labeled_vessels=labeled,
            sampling_frequency=40,
            embedding_func=None,
            clustering_func=lambda values: np.zeros(len(values), dtype=int),
            video=np.ones((8, 2, 2)),
            assign_to_av=False,
        )


def test_pipeline_rejects_wrong_number_of_cluster_labels():
    labeled = np.array([[2, 0], [0, 5]])
    with pytest.raises(ValueError, match="one label per branch"):
        clustering.run_clustering_pipeline(
            signals=np.ones((2, 8)),
            labeled_vessels=labeled,
            sampling_frequency=40,
            embedding_func=None,
            clustering_func=lambda _values: np.array([0]),
            video=np.ones((8, 2, 2)),
            assign_to_av=False,
        )


def test_temporal_assignment_maps_rows_to_noncontiguous_branch_ids(monkeypatch):
    labeled = np.array([[2, 2, 0], [0, 5, 5]])
    video = np.zeros((10, 2, 3), dtype=float)
    video[:, labeled == 2] = 2.0
    video[:, labeled == 5] = 5.0

    monkeypatch.setattr(
        pulse_analysis.signal_processing,
        "get_filtered_pulse",
        lambda signal, sampling_frequency: signal,
    )
    monkeypatch.setattr(
        pulse_analysis,
        "get_nb_of_positive_peaks",
        lambda signal, _period: int(signal[0]),
    )

    artery, vein, labels = pulse_analysis.assign_clusters_to_av(
        cluster_labels=np.array([0, 1]),
        video=video,
        periods=np.array([4, 4]),
        labeled_vessels=labeled,
        sampling_freq=40,
    )

    assert np.array_equal(artery, labeled == 5)
    assert np.array_equal(vein, labeled == 2)
    assert np.array_equal(labels, [2, 1])


@pytest.mark.parametrize("argument", ["videos", "embeddings", "clusterings"])
def test_benchmark_rejects_empty_experiment_axes(argument, tmp_path):
    arguments = {
        "videos": {"M0": np.ones((4, 1, 1))},
        "labeled_vessels": np.ones((1, 1), dtype=int),
        "gt_branch_labels": np.array([1]),
        "gt_artery_mask": np.ones((1, 1), dtype=bool),
        "gt_vein_mask": np.zeros((1, 1), dtype=bool),
        "embeddings": {"identity": lambda values: values},
        "clusterings": {"one": lambda values: np.zeros(len(values), dtype=int)},
        "sampling_frequency": 40,
        "h5_path": tmp_path / "benchmark.h5",
    }
    arguments[argument] = {}
    with pytest.raises(ValueError, match=f"{argument} cannot be empty"):
        benchmark.run_benchmark(**arguments)
