import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d, median_filter

import logging
logger = logging.getLogger(__name__)

def movmean(x, k):
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.empty(n)

    half = k // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        y[i] = np.mean(x[start:end])

    return y

def detect_global_drop(signal, drop_threshold=0.1):
    baseline = np.median(signal[:len(signal)//2])
    return signal < (1 - drop_threshold) * baseline

def interpolate_outliers_signal(signal, mask):
    x = np.arange(len(signal))
    valid = ~mask
    
    if np.sum(valid) < 2:
        return signal.copy()
    
    return np.interp(x, x[valid], signal[valid])

def detect_outliers_model_based(signal, window=31, poly=3, threshold=3):
    smooth = savgol_filter(signal, window, poly)
    
    residual = signal - smooth
    
    # robust scale
    mad = np.median(np.abs(residual))
    sigma = 1.4826 * mad if mad > 0 else np.std(residual)
    
    mask = np.abs(residual) > threshold * sigma
    
    return mask, smooth

def detect_outliers_derivative(signal, threshold=3):
    deriv = np.diff(signal, prepend=signal[0])
    
    mad = np.median(np.abs(deriv))
    sigma = 1.4826 * mad if mad > 0 else np.std(deriv)
    
    mask = np.abs(deriv) > threshold * sigma
    return mask

def hampel_filter(signal, window=11, n_sigmas=3):
    # Local median
    med = median_filter(signal, size=window, mode='nearest')
    
    # Local MAD
    abs_dev = np.abs(signal - med)
    mad = median_filter(abs_dev, size=window, mode='nearest')
    
    # Scale factor for Gaussian consistency
    sigma = 1.2 * mad
    
    # Avoid division issues
    sigma[sigma == 0] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    
    # Outlier mask
    outliers = abs_dev > n_sigmas * sigma
    
    return outliers, med

def post_smooth(signal, window=21, poly=3):
    return savgol_filter(signal, window, poly)  

# Detect outliers using a moving median and threshold
def detect_outliers_moving_median(signal, window=5, threshold_factor=2.0):
    padded = np.pad(signal, (window//2,), mode='edge')
    mov_median = uniform_filter1d(padded, size=window, mode='nearest')[window//2:-(window//2)]
    deviation = np.abs(signal - mov_median)
    mad = np.median(deviation)
    return deviation > threshold_factor * mad if mad != 0 else np.zeros_like(signal, dtype=bool)

def remove_outliers(video, outlier_mask):
    """Replace outlier frames with NaN in a ``(T, H, W)`` video."""
    video = np.asarray(video)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    if video.ndim != 3:
        raise ValueError("video must have shape (T, H, W)")
    if outlier_mask.ndim != 1 or outlier_mask.size != video.shape[0]:
        raise ValueError("outlier_mask must be one-dimensional with one value per frame")

    if np.issubdtype(video.dtype, np.inexact):
        cleaned_video = video.copy()
    else:
        cleaned_video = video.astype(float)
    cleaned_video[outlier_mask, :, :] = np.nan
    return cleaned_video

def interpolate_outlier_frames(video, outlier_frames_mask):
    """
    Interpolate outlier frames in a 3D video array.

    Parameters:
        video (np.ndarray): 3D array of shape (T, H, W)
        outlier_frames_mask (np.ndarray): 1D boolean array of length T

    Returns:
        video_cleaned (np.ndarray): 3D array with interpolated outlier frames
    """
    video = np.asarray(video)
    outlier_frames_mask = np.asarray(outlier_frames_mask, dtype=bool)
    if video.ndim != 3:
        raise ValueError("video must have shape (T, H, W)")
    if outlier_frames_mask.ndim != 1 or outlier_frames_mask.size != video.shape[0]:
        raise ValueError(
            "outlier_frames_mask must be one-dimensional with one value per frame"
        )
    if outlier_frames_mask.all() and outlier_frames_mask.size:
        raise ValueError("at least one non-outlier frame is required for interpolation")

    if np.issubdtype(video.dtype, np.inexact):
        video_cleaned = video.copy()
    else:
        video_cleaned = video.astype(float)
    outlier_indices = np.where(outlier_frames_mask)[0]

    for idx in outlier_indices:
        # Find previous and next non-outlier frames
        prev_candidates = np.where(~outlier_frames_mask[:idx])[0]
        next_candidates = np.where(~outlier_frames_mask[idx+1:])[0] + idx + 1

        prev_frame = prev_candidates[-1] if len(prev_candidates) > 0 else None
        next_frame = next_candidates[0] if len(next_candidates) > 0 else None

        # Handle edge cases
        if prev_frame is None:
            prev_frame = next_frame
        if next_frame is None:
            next_frame = prev_frame

        if prev_frame == next_frame:
            video_cleaned[idx, :, :] = video[prev_frame, :, :]
        elif next_frame - prev_frame <= 2:  # If frames are close, just take the average
            alpha = (idx - prev_frame) / (next_frame - prev_frame)
            video_cleaned[idx, :, :] = (
                (1 - alpha) * video[prev_frame, :, :] +
                alpha * video[next_frame, :, :]
            )
        else:
            video_cleaned[idx, :, :] = np.nan

    return video_cleaned

def local_percentile_outliers(signal, window=31, lower=5, upper=95, thresh=1.5):
    n = len(signal)
    mask = np.zeros(n, dtype=bool)
    
    half = window // 2
    
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        
        w = signal[start:end]
        
        p_low = np.percentile(w, lower)
        p_high = np.percentile(w, upper)
        
        # robust range
        r = p_high - p_low
        
        if r == 0:
            continue
        
        if signal[i] < p_low * (1/thresh) or signal[i] > p_high * thresh:
            mask[i] = True
            
    return mask

def interpolate_outliers(video, signal, artery_mask, sampling_frequency):
    outlier_frames_mask = detect_outliers_moving_median(signal, window=5, threshold_factor=2)
    logger.info(f"    - Detected {outlier_frames_mask.sum()} outlier frames based on arterial pulse signal.")
    video = interpolate_outlier_frames(video, outlier_frames_mask)
    # video = remove_outliers(video, outlier_frames_mask)
    signal = get_pulse_from_mask(video, artery_mask)
    signal_filtered = get_filtered_pulse(signal, sampling_frequency=sampling_frequency)
    return video, signal_filtered

def normalize(x, low=-1, high=1):
    x = np.asarray(x)
    xmin, xmax = x.min(), x.max()

    if xmax == xmin:
        return np.full_like(x, (low + high) / 2, dtype=float)

    return low + (x - xmin) * (high - low) / (xmax - xmin)

def compute_correlation(video, signal, normalization_interval = [-1, 1]):
    """
    Compute the zero-lag correlation between the video signal and the average signal in the mask.

    Parameters:
        video (np.ndarray): 3D array of shape (T, H, W)
        signal (np.ndarray): 1D temporal reference signal of length T

    Returns:
        R (np.ndarray): 2D correlation map of shape (H, W)
    """
    video = np.asarray(video)
    signal = np.asarray(signal)
    if video.ndim != 3:
        raise ValueError("video must have shape (T, H, W)")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if signal.size != video.shape[0]:
        raise ValueError("video and signal must have the same temporal length")
    if np.iscomplexobj(video) or np.iscomplexobj(signal):
        raise ValueError("video and signal must be real-valued")

    signal_valid = np.isfinite(signal)
    signal_work = signal.astype(float, copy=True)
    if signal_valid.any():
        signal_work[signal_valid] -= np.mean(signal_work[signal_valid])
    signal_work[~signal_valid] = 0

    work_dtype = video.dtype if np.issubdtype(video.dtype, np.floating) else float
    video_work = video.astype(work_dtype, copy=True)
    valid = np.isfinite(video_work) & signal_valid[:, np.newaxis, np.newaxis]
    video_work[~valid] = 0

    counts = np.sum(valid, axis=0, dtype=np.int64)
    safe_counts = np.maximum(counts, 1)
    video_means = np.sum(video_work, axis=0, dtype=float) / safe_counts
    signal_sums = np.einsum(
        "thw,t->hw",
        valid,
        signal_work,
        dtype=float,
        optimize=True,
    )
    signal_means = signal_sums / safe_counts

    # Reuse the video copy for centered pixel signals to keep peak memory near
    # one full video plus the finite-sample mask.
    np.subtract(video_work, video_means, out=video_work, casting="unsafe")
    video_work[~valid] = 0

    covariance = np.einsum(
        "thw,t->hw",
        video_work,
        signal_work,
        dtype=float,
        optimize=True,
    )
    video_sum_squares = np.einsum(
        "thw,thw->hw",
        video_work,
        video_work,
        dtype=float,
        optimize=True,
    )
    signal_sum_squares = np.einsum(
        "thw,t->hw",
        valid,
        signal_work ** 2,
        dtype=float,
        optimize=True,
    ) - counts * signal_means ** 2
    signal_sum_squares = np.maximum(signal_sum_squares, 0)
    denominator = np.sqrt(video_sum_squares * signal_sum_squares)

    correlation = np.full(video.shape[1:], np.nan, dtype=float)
    defined = (counts >= 2) & (denominator > np.finfo(float).eps)
    correlation[defined] = covariance[defined] / denominator[defined]
    correlation[defined] = np.clip(correlation[defined], -1, 1)
    if normalization_interval is not None:
        correlation = normalize(correlation, low=normalization_interval[0], high=normalization_interval[1])
    return correlation

def correlate_signal(signal, reference_signal):
    """
    Compute the zero-lag correlation between two signals.

    Parameters:
        signal (np.ndarray): 1D array of shape (T,)
        reference_signal (np.ndarray): 1D array of shape (T,)

    Returns:
        R (float): Correlation coefficient
    """
    signal_centered = signal - np.nanmean(signal)
    reference_signal_centered = reference_signal - np.nanmean(reference_signal)

    denominator = np.nanstd(signal_centered) * np.nanstd(reference_signal_centered)
    numerator = np.nanmean(signal_centered * reference_signal_centered)

    R = numerator / denominator

    return R

def correlate_signals(signals, reference_signal, include_std=False):
    """
    Compute the zero-lag correlations between multiple signals and a reference signal.

    Parameters:
        signals (np.ndarray): 2D array of shape (N, T)
        reference_signal (np.ndarray): 1D array of shape (T,)
        include_std (bool): If True, also compute the standard deviation of the correlations across signals.

    Returns:
        R (np.ndarray): 2D array of shape (N, T) containing the correlation coefficients
    """
    reference_signal_centered = reference_signal - np.nanmean(reference_signal)
    denominator = np.nanstd(reference_signal_centered)

    R = np.empty(signals.shape[0]) if not include_std else np.empty((signals.shape[0], 2))
    for i in range(signals.shape[0]):
        signal_centered = signals[i] - np.nanmean(signals[i])
        numerator = np.nanmean(signal_centered * reference_signal_centered)
        corr_avg = numerator / (np.nanstd(signal_centered) * denominator)

        if include_std:
            std_dev = np.nanstd(signal_centered * reference_signal_centered)
            R[i] = (corr_avg, std_dev)
        else:
            R[i] = corr_avg

    return R

def get_pulse_from_mask(video, mask):
    """
    Get the pulse signal from the video using the provided mask.

    Parameters:
        video (np.ndarray): 3D array of shape (T, H, W)
        mask (np.ndarray): 2D binary mask of shape (H, W)
    Returns:
        pulse (np.ndarray): 1D array of length T representing the pulse signal
    """
    video = np.asarray(video)
    mask = np.asarray(mask, dtype=bool)
    if video.ndim != 3:
        raise ValueError("video must have shape (T, H, W)")
    if mask.shape != video.shape[1:]:
        raise ValueError(
            f"mask shape {mask.shape} does not match video spatial shape {video.shape[1:]}"
        )
    num_mask_pixels = np.count_nonzero(mask)
    if num_mask_pixels == 0:
        raise ValueError("mask must contain at least one selected pixel")

    pulse = np.empty(video.shape[0], dtype=video.dtype)
    frame_scratch = np.empty(video.shape[1:], dtype=video.dtype)
    for index, frame in enumerate(video):
        # Preserve the former multiply-and-reduce ordering exactly while
        # reducing temporary storage from a full video to one frame.
        np.multiply(frame, mask, out=frame_scratch)
        pulse[index] = np.nansum(frame_scratch)
    pulse = pulse / num_mask_pixels
    return pulse

def get_filtered_pulse(pulse, sampling_frequency, cutoff=15, order=4):
    """
    Apply a low-pass Butterworth filter to the pulse signal.
    Parameters:
    pulse (np.ndarray): 1D array representing the pulse signal
    sampling_frequency (float): Sampling frequency of the pulse signal
    Returns:
    filtered_pulse (np.ndarray): 1D array representing the filtered pulse signal
    """
    pulse = np.asarray(pulse)
    if pulse.ndim != 1:
        raise ValueError("pulse must be one-dimensional")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= sampling_frequency / 2:
        raise ValueError("cutoff must lie strictly between 0 and the Nyquist frequency")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")

    b, a = butter(order, cutoff / (sampling_frequency / 2), btype="low")
    padlen = 3 * max(len(a), len(b))
    if pulse.size <= padlen:
        return pulse.copy()
    filtered_pulse = filtfilt(b, a, pulse)
    return filtered_pulse
