import numpy as np
import pytest

from dopplerview.segmentation import signal_processing


def test_compute_correlation_is_pixelwise_pearson_over_time():
    signal = np.array([0.0, 1.0, 2.0, 3.0])
    video = np.stack(
        [
            signal,
            -signal,
            np.ones_like(signal),
            np.array([1.0, -1.0, -1.0, 1.0]),
        ],
        axis=1,
    ).reshape(4, 2, 2)

    correlation = signal_processing.compute_correlation(video, signal)

    assert correlation.shape == (2, 2)
    assert correlation[0, 0] == pytest.approx(1.0)
    assert correlation[0, 1] == pytest.approx(-1.0)
    assert np.isnan(correlation[1, 0])
    assert correlation[1, 1] == pytest.approx(0.0)


def test_compute_correlation_uses_pairwise_finite_samples_per_pixel():
    signal = np.array([0.0, 1.0, 2.0, 3.0])
    video = np.array([0.0, np.nan, 2.0, 3.0])[:, None, None]

    correlation = signal_processing.compute_correlation(video, signal)

    assert correlation[0, 0] == pytest.approx(1.0)


def test_compute_correlation_matches_numpy_for_each_pixel():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=50)
    video = rng.normal(size=(50, 3, 4))

    correlation = signal_processing.compute_correlation(video, signal)

    expected = np.empty((3, 4))
    for row in range(3):
        for column in range(4):
            expected[row, column] = np.corrcoef(
                video[:, row, column],
                signal,
            )[0, 1]
    assert np.allclose(correlation, expected)


@pytest.mark.parametrize(
    "video,signal,error",
    [
        (np.zeros((3, 2)), np.zeros(3), "video must have shape"),
        (np.zeros((3, 2, 2)), np.zeros((3, 1)), "one-dimensional"),
        (np.zeros((3, 2, 2)), np.zeros(2), "same temporal length"),
    ],
)
def test_compute_correlation_rejects_invalid_shapes(video, signal, error):
    with pytest.raises(ValueError, match=error):
        signal_processing.compute_correlation(video, signal)
