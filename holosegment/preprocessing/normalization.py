"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter
import math


def flat_field_correction(image, gaussian_blur_ratio, border_amount=0.0):
    """
    FLAT_FIELD_CORRECTION Corrects an image for uneven illumination
    using Gaussian blur.

    Parameters
    ----------
    image : np.ndarray (2D)
        Input image.
    gaussian_blur_ratio : float
        Gaussian blur width (sigma).
    border_amount : float, optional
        Fraction of the image border to exclude (default = 0).

    Returns
    -------
    corrected_image : np.ndarray
        Flat-field corrected image.
    """

    image = image.astype(np.float32, copy=False)

    # --- Check normalization ---
    Im_min = np.min(image)
    Im_max = np.max(image)

    if Im_min < 0 or Im_max > 1:
        if Im_max > Im_min:
            image = (image - Im_min) / (Im_max - Im_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        flag = True
    else:
        flag = False

    h, w = image.shape

    # --- Define non-border region ---
    if border_amount == 0:
        a, b = 0, h
        c, d = 0, w
    else:
        a = int(math.ceil(h * border_amount))
        b = int(math.floor(h * (1 - border_amount)))
        c = int(math.ceil(w * border_amount))
        d = int(math.floor(w * (1 - border_amount)))

    # --- Sum of intensities in non-border region ---
    ms = np.sum(image[a:b, c:d])

    # --- Gaussian blur correction ---
    blurred = gaussian_filter(image, sigma=gaussian_blur_ratio)
    # avoid division by zero
    blurred[blurred == 0] = 1e-8

    image_corr = image / blurred

    # --- Rescale to maintain intensity ---
    ms2 = np.sum(image_corr[a:b, c:d])
    if ms2 != 0:
        corrected_image = (ms / ms2) * image_corr
    else:
        corrected_image = image_corr

    # --- Restore original range if normalized ---
    if flag:
        corrected_image = Im_min + (Im_max - Im_min) * corrected_image

    return corrected_image


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