import numpy as np

from dopplerview.segmentation import pulse_analysis


def test_peak_poor_signal_is_returned_unchanged():
    signal = np.zeros(20, dtype=float)

    cleaned, reference, correlations = pulse_analysis.remove_bad_beats(
        signal,
        beat_period=5,
    )

    assert np.array_equal(cleaned, signal)
    assert np.array_equal(reference, signal)
    assert correlations.size == 0


def test_peak_poor_video_is_returned_unchanged():
    signal = np.zeros(20, dtype=float)
    video = np.arange(80).reshape(20, 2, 2)

    cleaned_signal, cleaned_video, *_ = pulse_analysis.remove_bad_beats_on_video(
        signal,
        video,
        beat_period=5,
    )

    assert np.array_equal(cleaned_signal, signal)
    assert np.array_equal(cleaned_video, video)


def test_zero_samples_inside_good_beats_are_preserved(monkeypatch):
    signal = np.array([9.0, 0.0, 1.0, 0.0, 1.0, 8.0])
    monkeypatch.setattr(
        pulse_analysis,
        "get_peaks",
        lambda *_args, **_kwargs: np.array([1, 5]),
    )

    cleaned, _, _ = pulse_analysis.remove_bad_beats(
        signal,
        beat_period=4,
        threshold=-1,
    )

    assert np.array_equal(cleaned, signal[1:5])


def test_rejecting_every_detected_beat_falls_back_to_original(monkeypatch):
    signal = np.zeros(6, dtype=float)
    monkeypatch.setattr(
        pulse_analysis,
        "get_peaks",
        lambda *_args, **_kwargs: np.array([1, 5]),
    )

    cleaned, reference, correlations = pulse_analysis.remove_bad_beats(
        signal,
        beat_period=4,
    )

    assert np.array_equal(cleaned, signal)
    assert np.array_equal(reference, signal)
    assert np.array_equal(correlations, [0.0])


def test_cycle_based_video_removal_uses_a_final_cycle_endpoint():
    cycle = np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))
    signal = np.tile(cycle, 2)
    video = signal[:, None, None]

    cleaned_signal, cleaned_video, *_ = pulse_analysis.remove_bad_beats_on_video(
        signal,
        video,
        beat_period=10,
        threshold=-1,
        use_peaks=False,
    )

    assert np.array_equal(cleaned_signal, signal)
    assert np.array_equal(cleaned_video, video)
