"""Spatial image-processing helpers used by the segmentation notebooks."""

from __future__ import annotations

import operator

import numpy as np
from skimage.restoration import denoise_bilateral


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float = 3.0,
    sigma_color: float | None = None,
    *,
    frame_axis: int | None = None,
    win_size: int | None = None,
    bins: int = 10_000,
    mode: str = "reflect",
    cval: float = 0.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral smoothing to an image or video.

    A neighboring pixel contributes to the average only when it is both
    spatially close and similar in value.  This makes the filter useful for
    smoothing noise inside Doppler vessels without averaging as strongly
    across vessel boundaries.

    Parameters
    ----------
    image:
        A 2-D grayscale image, or a 3-D grayscale video when ``frame_axis``
        is specified.
    sigma_spatial:
        Spatial Gaussian standard deviation, in pixels. Larger values use a
        wider neighborhood.
    sigma_color:
        Gaussian standard deviation for differences in pixel value, expressed
        in the same units as ``image``. Smaller values preserve weaker edges.
        By default, the standard deviation of the complete image/video is
        used. A video therefore uses one consistent value for every frame.
    frame_axis:
        Axis containing time for a 3-D video. For a ``(T, H, W)`` Doppler
        video, use ``frame_axis=0``. Leave as ``None`` for a 2-D image.
    win_size:
        Odd spatial window size. By default scikit-image derives it from
        ``sigma_spatial``.
    bins:
        Number of samples in the lookup table used for value weights.
    mode, cval:
        Border handling passed to :func:`skimage.restoration.denoise_bilateral`.

    Returns
    -------
    numpy.ndarray
        A floating-point array with the same shape as ``image``. Filtering is
        frame-wise for videos and never mixes samples between time points.

    Examples
    --------
    Filter a time-averaged Doppler image::

        filtered = bilateral_filter(M0_ff_video.mean(axis=0),
                                    sigma_spatial=3,
                                    sigma_color=0.08)

    Filter every frame of a ``(T, H, W)`` video::

        filtered_video = bilateral_filter(M0_ff_video,
                                          frame_axis=0,
                                          sigma_spatial=3,
                                          sigma_color=0.08)

    Notes
    -----
    This function is intended for reconstructed Doppler maps. Applying a
    spatial filter before Doppler-spectrum estimation can alter the signal
    used to derive flow measurements.
    """
    values = np.asarray(image)
    if values.ndim not in (2, 3):
        raise ValueError("image must be a 2-D image or a 3-D grayscale video")
    if values.ndim == 2 and frame_axis is not None:
        raise ValueError("frame_axis is only valid for a 3-D video")
    if values.ndim == 3 and frame_axis is None:
        raise ValueError("frame_axis must be specified for a 3-D grayscale video")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("image must contain real numeric values")
    if values.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("image must contain only finite values")
    if not np.isfinite(sigma_spatial) or sigma_spatial <= 0:
        raise ValueError("sigma_spatial must be finite and greater than zero")
    if sigma_color is not None and (
        not np.isfinite(sigma_color) or sigma_color <= 0
    ):
        raise ValueError("sigma_color must be finite and greater than zero")
    if win_size is not None and (win_size < 1 or win_size % 2 == 0):
        raise ValueError("win_size must be a positive odd integer")
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer greater than or equal to 2")

    # Supplying a floating array prevents scikit-image from rescaling integer
    # inputs to [0, 1], so sigma_color always remains in the input's units.
    output_dtype = np.result_type(values.dtype, np.float32)
    float_values = values.astype(output_dtype, copy=False)

    if sigma_color is None:
        sigma_color = float(np.std(float_values, dtype=np.float64))
        if sigma_color == 0:
            return float_values.copy()

    def filter_frame(frame: np.ndarray) -> np.ndarray:
        return denoise_bilateral(
            frame,
            win_size=win_size,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            bins=int(bins),
            mode=mode,
            cval=cval,
            channel_axis=None,
        )

    if values.ndim == 2:
        return filter_frame(float_values).astype(output_dtype, copy=False)

    try:
        normalized_axis = operator.index(frame_axis)
    except TypeError as exc:
        raise TypeError("frame_axis must be an integer") from exc
    if not -values.ndim <= normalized_axis < values.ndim:
        raise ValueError(
            f"frame_axis {normalized_axis} is out of bounds for a {values.ndim}-D video"
        )
    normalized_axis %= values.ndim
    frames = np.moveaxis(float_values, normalized_axis, 0)
    filtered = np.empty_like(frames)
    for index, frame in enumerate(frames):
        filtered[index] = filter_frame(frame)
    return np.moveaxis(filtered, 0, normalized_axis)
