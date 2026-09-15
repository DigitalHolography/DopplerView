import numpy as np
import pytest

from dopplerview.segmentation.image_processing import bilateral_filter


def test_bilateral_filter_smooths_noise_without_crossing_strong_edge():
    rng = np.random.default_rng(123)
    image = np.zeros((40, 40), dtype=np.float32)
    image[:, 20:] = 1.0
    noisy = image + rng.normal(scale=0.08, size=image.shape).astype(np.float32)

    filtered = bilateral_filter(noisy, sigma_spatial=2, sigma_color=0.12)

    assert filtered.shape == noisy.shape
    assert filtered.dtype == np.float32
    assert filtered[:, :18].std() < noisy[:, :18].std()
    assert filtered[:, 22:].std() < noisy[:, 22:].std()
    assert filtered[:, 20:].mean() - filtered[:, :20].mean() > 0.9


def test_bilateral_filter_processes_video_frames_independently():
    frame = np.zeros((24, 24), dtype=np.float64)
    frame[:, 12:] = 1.0
    video = np.stack((frame, frame * 2), axis=0)

    filtered = bilateral_filter(
        video,
        frame_axis=0,
        sigma_spatial=2,
        sigma_color=0.1,
    )

    assert filtered.shape == video.shape
    np.testing.assert_allclose(filtered[0], frame, atol=1e-12)
    np.testing.assert_allclose(filtered[1], frame * 2, atol=1e-12)


def test_bilateral_filter_preserves_constant_integer_values_as_float():
    image = np.full((8, 8), 17, dtype=np.uint16)

    filtered = bilateral_filter(image)

    assert np.issubdtype(filtered.dtype, np.floating)
    np.testing.assert_array_equal(filtered, 17)


def test_bilateral_filter_requires_frame_axis_for_video():
    with pytest.raises(ValueError, match="frame_axis must be specified"):
        bilateral_filter(np.zeros((3, 8, 8)))


@pytest.mark.parametrize("sigma", [0, -1, np.nan])
def test_bilateral_filter_rejects_invalid_spatial_sigma(sigma):
    with pytest.raises(ValueError, match="sigma_spatial"):
        bilateral_filter(np.zeros((8, 8)), sigma_spatial=sigma)


def test_bilateral_filter_rejects_nonfinite_pixels():
    image = np.zeros((8, 8))
    image[2, 2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        bilateral_filter(image)
