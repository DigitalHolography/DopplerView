import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d, median_filter

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

def interpolate_outlier_frames(video, outlier_frames_mask):
    """
    Interpolate outlier frames in a 3D video array.

    Parameters:
        video (np.ndarray): 3D array of shape (H, W, T)
        outlier_frames_mask (np.ndarray): 1D boolean array of length T

    Returns:
        video_cleaned (np.ndarray): 3D array with interpolated outlier frames
    """
    video_cleaned = video.copy()
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
        else:
            alpha = (idx - prev_frame) / (next_frame - prev_frame)
            video_cleaned[idx, :, :] = (
                (1 - alpha) * video[prev_frame, :, :] +
                alpha * video[next_frame, :, :]
            )

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
    print(f"    - Detected {outlier_frames_mask.sum()} outlier frames based on arterial pulse signal.")
    video = interpolate_outlier_frames(video, outlier_frames_mask)
    signal = get_pulse_from_mask(video, artery_mask)
    signal_filtered = get_filtered_pulse(signal, sampling_frequency=sampling_frequency)
    return video, signal_filtered

def compute_correlation(video, signal):
    """
    Compute the zero-lag correlation between the video signal and the average signal in the mask.

    Parameters:
        video (np.ndarray): 3D array of shape (H, W, T)
        mask (np.ndarray): 2D binary mask of shape (H, W)

    Returns:
        R (np.ndarray): 1D array of correlation values
    """
    # compute local-to-average signal wave zero-lag correlation
    signal_centered = signal - np.nanmean(signal)
    video_centered = video - np.nanmean(video)

    numerator = np.nanmean(video_centered * signal_centered[:, np.newaxis, np.newaxis], axis=0)
    denominator = np.nanstd(video_centered) * np.nanstd(signal_centered)
    
    R = numerator / denominator
    
    return R

def get_pulse_from_mask(video, mask):
    """
    Get the pulse signal from the video using the provided mask.

    Parameters:
        video (np.ndarray): 3D array of shape (H, W, T)
        mask (np.ndarray): 2D binary mask of shape (H, W)
    Returns:
        pulse (np.ndarray): 1D array of length T representing the pulse signal
    """
    pulse = np.nansum(video * mask[np.newaxis, :, :], axis=(1, 2))
    pulse = pulse / np.count_nonzero(mask)
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
    b, a = butter(order, cutoff / (sampling_frequency / 2), btype="low")
    filtered_pulse = filtfilt(b, a, pulse)
    return filtered_pulse