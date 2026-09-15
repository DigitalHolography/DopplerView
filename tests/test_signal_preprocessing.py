import numpy as np
import pytest

from sandbox.signal_preprocessing import (
    detect_short_impulse_artifacts,
    interpolate_artifact_samples,
    preprocess_cardiac_signal,
)


def _sawtooth_cycles(period=40, cycles=6):
    phase = np.arange(period) / period
    cycle = 2.0 * phase + 0.15 * np.sin(2 * np.pi * phase)
    return np.tile(cycle, cycles)


def test_impulse_detection_finds_reversible_spike_but_preserves_cardiac_edge():
    signal = _sawtooth_cycles()
    signal[93] -= 9

    artifacts = detect_short_impulse_artifacts(
        signal,
        sampling_frequency=100,
        derivative_z_threshold=5,
        max_duration_seconds=0.08,
        padding_frames=0,
    )

    assert artifacts[93]
    assert not artifacts[40]
    assert not artifacts[80]
    assert np.count_nonzero(artifacts) <= 3


def test_interpolation_repairs_multiple_signals_along_requested_axis():
    signals = np.vstack((np.arange(8.0), 2 * np.arange(8.0)))
    signals[:, 3] = -100

    repaired = interpolate_artifact_samples(
        signals,
        np.arange(8) == 3,
        axis=1,
    )

    assert repaired[:, 3] == pytest.approx([3, 6])
    assert np.array_equal(repaired[:, :3], signals[:, :3])


def test_preprocessing_uses_all_clean_cycles_and_removes_artifact_frames_from_fit():
    signal = _sawtooth_cycles(period=40, cycles=7)
    signal[93] -= 9

    result = preprocess_cardiac_signal(
        signal,
        sampling_frequency=100,
        beat_period=40,
        derivative_z_threshold=5,
        minimum_valid_cycles=2,
    )

    assert result.n_valid_cycles >= 5
    assert result.frame_artifact_mask[93]
    assert result.valid_cycle_frame_mask.sum() == result.n_valid_cycles * 40
    assert not result.fit_frame_mask[93]
    assert result.fit_frame_mask.sum() < result.valid_cycle_frame_mask.sum()
    assert result.summary()["valid_cycle_count"] == result.n_valid_cycles


def test_preprocessing_rejects_a_cycle_with_a_different_shape():
    signal = _sawtooth_cycles(period=40, cycles=7)
    phase = np.arange(40) / 40
    signal[3 * 40 : 4 * 40] = 1.5 * np.sin(4 * np.pi * phase)

    result = preprocess_cardiac_signal(
        signal,
        sampling_frequency=100,
        beat_period=40,
        minimum_cycle_correlation=0.6,
    )

    affected = [
        index
        for index, (start, stop) in enumerate(result.cycle_bounds)
        if start <= 3 * 40 < stop
    ]
    assert affected
    assert not result.cycle_valid[affected[0]]

