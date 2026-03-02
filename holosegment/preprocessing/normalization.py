"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter
import math


def flat_field_correction_3d(volume, gw=41, border_amount=0.15):

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

    blurred = gaussian_filter(
        volume,
        sigma=(0, gw, gw),
        mode='reflect',
        truncate=2.0
    )

    volume_corr = volume / blurred

    ms2 = np.sum(volume_corr[:, a:b, c:d])
    corrected = (ms / ms2) * volume_corr

    if flag:
        corrected = Im_min + (Im_max - Im_min) * corrected

    return corrected


def normalize_video(frames, method='zscore'):
    """
    Normalize frame intensities
    
    Args:
        frames: numpy array of shape (num_frames, height, width)
        method: normalization method ('zscore', 'minmax', 'percentile')
    
    Returns:
        Normalized frames
    """
    normalized = frames.copy().astype(np.float32)
    
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(normalized, axis=(1, 2), keepdims=True)
        std = np.std(normalized, axis=(1, 2), keepdims=True)
        normalized = (normalized - mean) / (std + 1e-8)
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(normalized, axis=(1, 2), keepdims=True)
        max_val = np.max(normalized, axis=(1, 2), keepdims=True)
        normalized = (normalized - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'percentile':
        # Percentile-based normalization
        p_low = np.percentile(normalized, 1, axis=(1, 2), keepdims=True)
        p_high = np.percentile(normalized, 99, axis=(1, 2), keepdims=True)
        normalized = np.clip(normalized, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low + 1e-8)
        
    return normalized