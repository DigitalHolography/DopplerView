"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter
import math


def flat_field_correction(image, correction_params, border_amount=0):
    """
    Flat-field correction using Gaussian blur.

    Parameters:
        image (np.ndarray): 2D input image
        correction_params (float): Gaussian blur sigma (gw)
        border_amount (float): fraction of border to exclude (0–0.5)

    Returns:
        corrected_image (np.ndarray)
    """

    image = image.astype(np.float32)

    # --- Check normalization ---
    Im_min = np.min(image)
    Im_max = np.max(image)

    if Im_min < 0 or Im_max > 1:
        if Im_max > Im_min:
            image = (image - Im_min) / (Im_max - Im_min)
        else:
            image = np.zeros_like(image)
        flag = True
    else:
        flag = False

    H, W = image.shape

    # --- Define non-border region ---
    if border_amount == 0:
        a, b = 0, H
        c, d = 0, W
    else:
        a = int(np.ceil(H * border_amount))
        b = int(np.floor(H * (1 - border_amount)))
        c = int(np.ceil(W * border_amount))
        d = int(np.floor(W * (1 - border_amount)))

    # --- Sum of intensities in region ---
    ms = np.sum(image[a:b, c:d])

    # --- Gaussian blur correction ---
    gw = correction_params
    blurred = gaussian_filter(image, sigma=gw)

    # avoid division by zero
    eps = 1e-8
    image_corr = image / (blurred + eps)

    # --- Rescale to preserve energy ---
    ms2 = np.sum(image_corr[a:b, c:d])
    corrected_image = (ms / (ms2 + eps)) * image_corr

    # --- Restore original scale if needed ---
    if flag:
        corrected_image = Im_min + (Im_max - Im_min) * corrected_image

    return corrected_image

def compute_moment_ff(M0, gw_ratio=0.07, border=0.15):
    """
    Compute the flat-field corrected M0 (M0_ff).

    Parameters:
        M0 (np.ndarray): shape (numFrames, numX, numY)
        gw_ratio (float): Gaussian width ratio
        border (float): border fraction

    Returns:
        M0_ff (np.ndarray)
    """

    M0 = M0.astype(np.float32)
    numFrames, numX, numY  = M0.shape

    # --- flat-field correction ---
    gw = int(np.ceil(gw_ratio * numX))

    # Apply frame-wise flat field correction
    M0_ff = np.zeros_like(M0, dtype=np.float32)

    for i in range(numFrames):
        M0_ff[i, :, :] = flat_field_correction(
            M0[i, :, :],
            correction_params=gw,
            border_amount=border
        )

    # --- Global mean / std over full volume ---
    mu = np.mean(M0_ff)
    sigma = np.std(M0_ff)

    # Clip extreme values to ±5σ
    upper = mu + 5 * sigma
    lower = mu - 5 * sigma

    M0_ff = np.clip(M0_ff, lower, upper)

    return M0_ff


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