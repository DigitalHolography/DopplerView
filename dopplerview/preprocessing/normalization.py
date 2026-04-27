"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter
from functools import partial

from dopplerview.utils.parallelization_utils import run_in_parallel


def _flatfield(data, gw):
    blurred = gaussian_filter(
        data,
        sigma=gw,
        mode='reflect',
        truncate=2.0
    )
    return data / (blurred + 1e-8)


def flat_field_correction_3d(
    volume,
    gw=41,
    border_amount=0.15,
    n_jobs=-1,
    parallel=True,
    chunking=True
):
    """
    Parallel version of flat field correction.
    """

    volume = volume.astype(np.float64)

    Im_min = volume.min()
    Im_max = volume.max()

    if Im_min < 0 or Im_max > 1:
        if Im_max > Im_min:
            volume = (volume - Im_min) / (Im_max - Im_min)
        else:
            volume = np.zeros_like(volume)
        flag = True
    else:
        flag = False

    T, H, W = volume.shape

    if border_amount == 0:
        a, b = 0, H
        c, d = 0, W
    else:
        a = int(np.ceil(H * border_amount))
        b = int(np.floor(H * (1 - border_amount)))
        c = int(np.ceil(W * border_amount))
        d = int(np.floor(W * (1 - border_amount)))

    ms = np.sum(volume[:, a:b, c:d])

    func = partial(_flatfield, gw=gw)

    if parallel:
        # Use parallel processing to apply flat field correction to each frame
        volume_corr = run_in_parallel(func, volume, n_jobs=n_jobs, chunking=chunking)
    else:
        # flat field correction without parallelization
        volume_corr = _flatfield(volume, (0,gw,gw))

    # Normalize globally -> breaks perfect parallelization but corrects for global intensity variations
    ms2 = np.sum(volume_corr[:, a:b, c:d])
    corrected = (ms / ms2) * volume_corr

    if flag:
        corrected = Im_min + (Im_max - Im_min) * corrected

    return corrected