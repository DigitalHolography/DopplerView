import numpy as np
import pytest

from dopplerview.segmentation import pulse_analysis, signal_processing


def test_compute_period_returns_none_for_constant_signal():
    assert pulse_analysis.compute_period(np.ones(100), 40) is None


def test_compute_period_interpolates_isolated_nonfinite_samples():
    sampling_frequency = 40
    time = np.arange(400) / sampling_frequency
    signal = np.sin(2 * np.pi * time)
    signal[[25, 100, 275]] = np.nan

    period = pulse_analysis.compute_period(signal, sampling_frequency)

    assert period is not None
    assert 35 <= period <= 45


@pytest.mark.parametrize("sampling_frequency", [0, -1, np.nan])
def test_compute_period_rejects_invalid_sampling_frequency(sampling_frequency):
    with pytest.raises(ValueError, match="sampling_frequency"):
        pulse_analysis.compute_period(np.ones(100), sampling_frequency)


def test_get_pulse_from_mask_rejects_empty_mask():
    video = np.ones((10, 2, 3))

    with pytest.raises(ValueError, match="at least one selected pixel"):
        signal_processing.get_pulse_from_mask(video, np.zeros((2, 3), dtype=bool))


def test_get_pulse_from_mask_rejects_mismatched_shape():
    video = np.ones((10, 2, 3))

    with pytest.raises(ValueError, match="does not match"):
        signal_processing.get_pulse_from_mask(video, np.ones((3, 2), dtype=bool))


def test_get_pulse_from_mask_uses_finite_pixel_count_per_frame():
    video = np.array(
        [
            [[2.0, 4.0], [100.0, 100.0]],
            [[2.0, np.nan], [100.0, 100.0]],
        ]
    )
    mask = np.array([[True, True], [False, False]])

    pulse = signal_processing.get_pulse_from_mask(video, mask)

    assert np.array_equal(pulse, [3.0, 2.0])


def test_get_pulse_from_mask_preserves_fully_missing_frame_as_nan():
    video = np.array(
        [
            [[1.0, 3.0]],
            [[np.nan, np.nan]],
        ]
    )
    mask = np.ones((1, 2), dtype=bool)

    pulse = signal_processing.get_pulse_from_mask(video, mask)

    assert pulse[0] == 2.0
    assert np.isnan(pulse[1])


def test_get_pulse_from_mask_ignores_nonfinite_values_outside_mask():
    video = np.array([[[2.0, np.nan], [np.inf, 50.0]]])
    mask = np.array([[True, False], [False, False]])

    pulse = signal_processing.get_pulse_from_mask(video, mask)

    assert np.array_equal(pulse, [2.0])


def test_get_pulse_from_mask_excludes_infinite_selected_pixels():
    video = np.array([[[2.0, np.inf]]])
    mask = np.ones((1, 2), dtype=bool)

    pulse = signal_processing.get_pulse_from_mask(video, mask)

    assert np.array_equal(pulse, [2.0])


def test_short_pulse_bypasses_zero_phase_filter():
    pulse = np.arange(10, dtype=float)

    filtered = signal_processing.get_filtered_pulse(pulse, 40)

    assert filtered is not pulse
    assert np.array_equal(filtered, pulse)


def test_filter_rejects_cutoff_at_nyquist_frequency():
    with pytest.raises(ValueError, match="Nyquist"):
        signal_processing.get_filtered_pulse(np.ones(100), 30, cutoff=15)


def test_short_branch_signal_bypasses_zero_phase_filter():
    video = np.zeros((10, 2, 2), dtype=float)
    video[:, 0, 0] = 3
    labeled_vessels = np.zeros((2, 2), dtype=int)
    labeled_vessels[0, 0] = 1

    signals = pulse_analysis.get_filtered_branch_signals(
        video,
        labeled_vessels,
        sampling_frequency=40,
    )

    assert signals.shape == (1, 10)
    assert np.array_equal(signals[0], np.full(10, 3.0))


@pytest.mark.parametrize("sampling_frequency,stride", [(0, 1), (1, 0)])
def test_effective_sampling_frequency_rejects_nonpositive_inputs(
    sampling_frequency,
    stride,
):
    with pytest.raises(ValueError):
        pulse_analysis.get_effective_sampling_frequency(sampling_frequency, stride)
