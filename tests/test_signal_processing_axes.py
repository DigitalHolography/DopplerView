import numpy as np
import pytest

from dopplerview.segmentation import signal_processing


def test_remove_outliers_masks_complete_frames_on_time_axis():
    video = np.arange(24, dtype=float).reshape(3, 2, 4)

    cleaned = signal_processing.remove_outliers(
        video,
        np.array([False, True, False]),
    )

    assert np.isnan(cleaned[1]).all()
    assert np.array_equal(cleaned[0], video[0])
    assert np.array_equal(cleaned[2], video[2])


def test_remove_outliers_promotes_integer_video_for_nan_frames():
    video = np.arange(12).reshape(3, 2, 2)

    cleaned = signal_processing.remove_outliers(video, [False, True, False])

    assert np.issubdtype(cleaned.dtype, np.floating)
    assert np.isnan(cleaned[1]).all()


@pytest.mark.parametrize(
    "mask",
    [np.array([True, False]), np.zeros((3, 1), dtype=bool)],
)
def test_remove_outliers_rejects_invalid_temporal_mask(mask):
    with pytest.raises(ValueError, match="one value per frame"):
        signal_processing.remove_outliers(np.zeros((3, 2, 2)), mask)


def test_interpolate_outlier_frames_uses_temporal_neighbors():
    video = np.array([0, 10, 20], dtype=float)[:, None, None]

    interpolated = signal_processing.interpolate_outlier_frames(
        video,
        np.array([False, True, False]),
    )

    assert interpolated.shape == video.shape
    assert interpolated[1, 0, 0] == pytest.approx(10)


def test_interpolate_outlier_frames_rejects_all_outlier_frames():
    with pytest.raises(ValueError, match="at least one non-outlier frame"):
        signal_processing.interpolate_outlier_frames(
            np.zeros((3, 2, 2)),
            np.ones(3, dtype=bool),
        )


def test_interpolate_outlier_frames_rejects_invalid_temporal_mask():
    with pytest.raises(ValueError, match="one value per frame"):
        signal_processing.interpolate_outlier_frames(
            np.zeros((3, 2, 2)),
            np.zeros(2, dtype=bool),
        )
