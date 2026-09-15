from types import SimpleNamespace

import numpy as np
import pytest

from sandbox import evaluation
from sandbox.signal_evaluation import (
    alternating_cycle_split,
    cycle_templates_from_frame_mask,
    evaluate_mask_signal_similarity,
    soft_dtw_divergence,
)


def test_alternating_cycle_split_uses_only_complete_disjoint_cycles():
    train, held_out = alternating_cycle_split(43, 10)

    assert not np.any(train & held_out)
    assert np.all(train[:10])
    assert np.all(held_out[10:20])
    assert np.all(train[20:30])
    assert np.all(held_out[30:40])
    assert not np.any(train[40:] | held_out[40:])


def test_cycle_templates_from_frame_mask_averages_selected_complete_cycles():
    signals = np.arange(2 * 43, dtype=float).reshape(2, 43)
    train, _ = alternating_cycle_split(43, 10)

    templates = cycle_templates_from_frame_mask(signals, train, 10)

    expected = np.mean(signals[:, np.r_[0:10, 20:30]].reshape(2, 2, 10), axis=1)
    np.testing.assert_allclose(templates, expected)


def test_cycle_templates_from_frame_mask_rejects_stale_period_split():
    signals = np.zeros((3, 381))
    stale_mask = np.ones(381, dtype=bool)

    with pytest.raises(ValueError, match="recompute the cycle split"):
        cycle_templates_from_frame_mask(signals, stale_mask, 114)


def test_soft_dtw_divergence_is_zero_for_identical_signals():
    signal = np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))

    assert soft_dtw_divergence(signal, signal) == pytest.approx(0.0, abs=1e-12)
    assert soft_dtw_divergence(signal, -signal) > 0


def test_signal_similarity_excludes_annotation_pixels_and_recovers_waveforms():
    n_frames = 40
    beat_period = 10
    time = np.arange(n_frames)
    artery = np.sin(2 * np.pi * time / beat_period)
    vein = np.cos(2 * np.pi * time / beat_period)
    video = np.zeros((n_frames, 2, 2), dtype=float)
    video[:, 0, 0] = artery  # annotated artery
    video[:, 0, 1] = artery  # independently predicted artery
    video[:, 1, 0] = vein  # annotated vein
    video[:, 1, 1] = vein  # independently predicted vein
    references = {
        "artery": np.array([[1, 0], [0, 0]], dtype=bool),
        "vein": np.array([[0, 0], [1, 0]], dtype=bool),
    }
    predictions = {
        "artery": np.array([[1, 1], [0, 0]], dtype=bool),
        "vein": np.array([[0, 0], [1, 1]], dtype=bool),
    }
    _, held_out = alternating_cycle_split(n_frames, beat_period)

    metrics = evaluate_mask_signal_similarity(
        {"M0": video},
        predictions,
        references,
        sampling_frequency=100.0,
        beat_period=beat_period,
        frame_mask=held_out,
    )

    for class_name in references:
        prefix = f"signal_M0_{class_name}_"
        assert metrics[prefix + "predicted_pixel_count"] == 1
        assert metrics[prefix + "pearson"] == pytest.approx(1.0)
        assert metrics[prefix + "normalized_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert metrics[prefix + "soft_dtw_divergence"] == pytest.approx(0.0, abs=1e-12)


def test_signal_similarity_reports_undefined_when_no_independent_pixels_remain():
    time = np.arange(20)
    video = np.sin(2 * np.pi * time / 10)[:, None, None]
    mask = np.ones((1, 1), dtype=bool)

    metrics = evaluate_mask_signal_similarity(
        {"M0": video},
        {"artery": mask},
        {"artery": mask},
        sampling_frequency=100.0,
        beat_period=10,
    )

    assert metrics["signal_M0_artery_predicted_pixel_count"] == 0
    assert np.isnan(metrics["signal_M0_artery_pearson"])


def test_signal_similarity_interpolates_declared_artifact_frames_before_templates():
    time = np.arange(30)
    waveform = np.sin(2 * np.pi * time / 10)
    video = np.stack((waveform, waveform), axis=1).reshape(30, 1, 2)
    video[13, 0, 1] = 50
    reference = {"artery": np.array([[1, 0]], dtype=bool)}
    predicted = {"artery": np.array([[0, 1]], dtype=bool)}

    metrics = evaluate_mask_signal_similarity(
        {"M0": video},
        predicted,
        reference,
        sampling_frequency=100.0,
        beat_period=10,
        artifact_mask=np.arange(30) == 13,
    )

    assert metrics["signal_M0_artery_pearson"] > 0.99


def test_evaluate_experiment_accepts_signal_metrics_without_clustering_result():
    time = np.arange(20)
    waveform = np.sin(2 * np.pi * time / 10)
    video = np.stack((waveform, waveform), axis=1).reshape(20, 1, 2)
    predicted = {"artery": np.array([[0, 1]], dtype=bool)}
    reference = {"artery": np.array([[1, 0]], dtype=bool)}

    metrics = evaluation.evaluate_experiment(
        signal_videos={"M0": video},
        signal_predicted_masks=predicted,
        signal_reference_masks=reference,
        signal_sampling_frequency=100.0,
        signal_beat_period=10,
        decimals=None,
    )

    assert metrics["signal_M0_artery_pearson"] == pytest.approx(1.0)


def test_signal_similarity_rejects_mismatched_masks():
    with pytest.raises(ValueError, match="same classes"):
        evaluate_mask_signal_similarity(
            {"M0": np.zeros((20, 2, 2))},
            {"artery": np.zeros((2, 2), dtype=bool)},
            {"vein": np.zeros((2, 2), dtype=bool)},
            sampling_frequency=100.0,
            beat_period=10,
        )
