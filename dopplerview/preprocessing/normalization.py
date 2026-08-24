"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from functools import partial


def _flatfield(data, gw, offset=0.0, scale=1.0):
    if offset != 0.0 or scale != 1.0:
        normalized = np.empty_like(data)
        np.subtract(data, offset, out=normalized)
        np.multiply(normalized, scale, out=normalized)
    else:
        normalized = data

    blurred = gaussian_filter(
        normalized,
        sigma=gw,
        mode='reflect',
        truncate=2.0
    )
    np.add(blurred, np.asarray(1e-8, dtype=blurred.dtype), out=blurred)
    np.divide(normalized, blurred, out=blurred)
    return blurred


def flat_field_correction_3d(
    volume,
    gw=41,
    border_amount=0.15,
    n_jobs=-1,
    parallel=True,
    chunking=True,
    executor=None,
):
    """
    Parallel version of flat field correction.
    """

    volume = np.asarray(volume)
    if not np.issubdtype(volume.dtype, np.floating):
        volume = volume.astype(np.float32)
    elif volume.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        volume = volume.astype(np.float32)

    Im_min = volume.min()
    Im_max = volume.max()

    rescale = Im_min < 0 or Im_max > 1
    value_range = Im_max - Im_min
    constant = rescale and value_range == 0
    offset = float(Im_min) if rescale and not constant else 0.0
    scale = 1.0 / float(value_range) if rescale and not constant else 1.0

    T, H, W = volume.shape

    if border_amount == 0:
        a, b = 0, H
        c, d = 0, W
    else:
        a = int(np.ceil(H * border_amount))
        b = int(np.floor(H * (1 - border_amount)))
        c = int(np.ceil(W * border_amount))
        d = int(np.floor(W * (1 - border_amount)))

    roi = volume[:, a:b, c:d]
    if rescale and not constant:
        roi_elements = roi.size
        ms = (np.sum(roi, dtype=np.float64) - roi_elements * offset) * scale
    elif constant:
        ms = 0.0
    else:
        ms = np.sum(roi, dtype=np.float64)

    func = partial(_flatfield, gw=gw, offset=offset, scale=scale)

    if constant:
        volume_corr = np.zeros_like(volume)
    elif parallel:
        if executor is None:
            raise ValueError("Parallel flat-field correction requires an executor.")
        volume_corr = np.empty_like(volume)
        map_into = getattr(executor, "map_into", None)
        if map_into is not None:
            map_into(
                func,
                volume,
                volume_corr,
                n_jobs=n_jobs,
                task_name="flat-field correction",
            )
        else:
            volume_corr[...] = executor.map(
                func,
                volume,
                n_jobs=n_jobs,
                chunking=chunking,
                task_name="flat-field correction",
            )
    else:
        volume_corr = _flatfield(
            volume, (0, gw, gw), offset=offset, scale=scale
        )

    # Normalize globally -> breaks perfect parallelization but corrects for global intensity variations
    ms2 = np.sum(volume_corr[:, a:b, c:d], dtype=np.float64)
    if ms2 != 0:
        np.multiply(volume_corr, ms / ms2, out=volume_corr)

    if rescale and not constant:
        np.multiply(volume_corr, value_range, out=volume_corr)
        np.add(volume_corr, Im_min, out=volume_corr)

    return volume_corr
