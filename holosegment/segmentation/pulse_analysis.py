"""
Pulse analysis module for analyzing temporal pulsatility in vessels
"""

from unittest import signals

import numpy as np
from scipy import fft
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d

from skimage.measure import label
from skimage import measure
from holosegment.segmentation import process_masks
from holosegment.utils import image_utils


# ================================ Pre-artery mask ================================ #

def moving_mean(sig, window):
    if window <= 1:
        return sig
    kernel = np.ones(window) / window
    return np.convolve(sig, kernel, mode="same")

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

def select_regular_peaks(signals_n, method, idx0, threshold=0.1, tolerance=0.3):
    gradient_n = np.gradient(signals_n, axis=1)

    if method == "minmax":
        return _select_minmax(signals_n, gradient_n, idx0)

    raise NotImplementedError

def _select_minmax(signals_n, gradient_n, idx0):
    """
    Python equivalent of MATLAB select_minmax
    """

    num_branches = signals_n.shape[0]

    s_idx = np.zeros(num_branches, dtype=int)
    locs_n = []

    for i in range(num_branches):
        # --- find peaks in |gradient| ---
        peaks, properties = find_peaks(
            np.abs(gradient_n[i]),
            distance=int(0.8 * idx0),
            prominence=1e-6   # helps avoid spurious peaks
        )

        locs = peaks  # MATLAB locs = indices

        # values of gradient at peak positions
        peaks_v = gradient_n[i, locs]

        # count positive peaks
        c = np.sum(peaks_v > 0)

        if c > len(locs) / 2:
            s_idx[i] = 1
        else:
            s_idx[i] = 0

        locs_n.append(locs)

    return s_idx, locs_n

def compute_idx0(signals_n, sampling_frequency):
    """""
    Find the index corresponding to the dominant frequency period in the cardiac pulse physiological range in the signals
    """
    num_frames = signals_n.shape[1]
    avg_signal = signals_n.mean(axis=0)

    # Compute FFT
    Y = np.fft.fft(avg_signal)
    P2 = np.abs(Y / num_frames)
    P1 = P2[:num_frames//2 + 1]
    P1[1:-1] *= 2

    # Frequency vector
    f = sampling_frequency * np.arange(len(P1)) / num_frames

    # Find dominant frequency in physiological range (e.g. 0.5 - 2 Hz)
    f_range = (f > 0.5) & (f < 2) # 30 - 120 bpm
    f_sel = f[f_range]
    P_sel = P1[f_range]

    f0 = f_sel[np.argmax(P_sel)]
    t0 = 1 / f0
    dt = 1 / sampling_frequency
    idx0 = int(round(t0 / dt))
    return idx0


def check_validity(signal, sampling_frequency):
    """
    CHECK_VALIDITY  Check if a temporal signal is periodic and not noise.

        is_valid = check_validity(signal, sampling_frequency)

        INPUTS:
            signal : 1 x N vector, normalized temporal signal of one branch
            sampling_frequency     : sampling frequency in Hz

        OUTPUT:
            is_valid : true if the signal is periodic,
                      false otherwise.

        The function uses the power spectrum of the signal to test:
            - if it has sufficient total energy,
            - if there is a strong dominant frequency,
            - if that frequency lies in a physiological range.
    """

    # ---------------- Parameters ----------------
    purity_threshold = 0.3   # required purity (tune as needed)
    freqRange = (0.5, 2.0)  # Hz, physiological range (30–120 bpm)

    # ---------------- Preprocessing ----------------
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)   # remove DC
    numFrames = signal.size

    if numFrames < 4:
        return False

    # ---------------- Power Spectrum ----------------
    Y = np.fft.fft(signal)
    P2 = np.abs(Y / numFrames) ** 2     # power spectrum

    half = numFrames // 2
    P1 = P2[:half + 1].copy()
    if P1.size > 2:
        P1[1:-1] *= 2

    f = sampling_frequency * np.arange(P1.size) / numFrames

    # Restrict to physiological range
    idxRange = (f >= freqRange[0]) & (f <= freqRange[1])
    if not np.any(idxRange):
        return False

    f_local = f[idxRange]
    P_local = P1[idxRange]

    total_power = np.sum(P_local)
    if total_power <= 0:
        return False

    P_local = P_local / total_power   # normalize for purity calc

    # ---- Dominant frequency ----
    idxMax = np.argmax(P_local)
    f_branch = f_local[idxMax]

    # ---- Spectral Purity Metrics ----
    # 1. Energy concentration near dominant frequency
    band = (f_local > f_branch - 0.2) & (f_local < f_branch + 0.2)
    energyConcentration = np.sum(P_local[band])

    # 2. Spectral entropy
    eps = np.finfo(float).eps
    spectralEntropy = -np.sum(P_local * np.log(P_local + eps)) / np.log(P_local.size)
    purityEntropy = 1.0 - spectralEntropy   # invert so 1 = pure, 0 = noisy

    # 3. Combine into final purity score (weighted average)
    purity = 0.7 * energyConcentration + 0.3 * purityEntropy

    is_valid = purity > purity_threshold

    return bool(is_valid)

def get_filtered_branch_signals(video, labeled_vessels, sampling_frequency):
    """
    Get mean temporal signal for each branch in the labeled vessel mask.
    """
    num_frames = video.shape[0]
    num_branches = labeled_vessels.max()
    signals = np.zeros((num_branches, num_frames))
    b, a = butter(4, 15 / (sampling_frequency / 2), btype='low')
    moving_window = round(sampling_frequency * 0.1)

    for i in range(1, num_branches + 1):
        branch_mask = (labeled_vessels == i)
        branch_pixels = video[:, branch_mask]
        branch_mean = np.mean(branch_pixels, axis=1)

        signals[i - 1, :] = filtfilt(b, a, branch_mean)

        if moving_window > 1:
            signals[i - 1, :] = movmean(signals[i - 1, :], moving_window)

    return signals


def compute_pre_masks(signals, labeled_vessels, sampling_frequency):
    """
    Compute a preliminary artery mask based on pulse analysis of the video frames within the vessel mask
    """

    idx0 = compute_idx0(signals, sampling_frequency)
    s_idx, _ = select_regular_peaks(signals, "minmax", idx0)

    is_pure = np.array([check_validity(sig, sampling_frequency) for sig in signals])
    if not is_pure.any():
        is_pure[:] = True

    # Step 4: Combine into artery / vein masks
    pre_mask_artery = np.zeros_like(labeled_vessels, bool)
    pre_mask_vein = np.zeros_like(labeled_vessels, bool)

    num_branches = labeled_vessels.max()

    for i in range(1, num_branches+1):
        if not is_pure[i-1]:
            continue
        if s_idx[i-1] == 1:
            pre_mask_artery |= labeled_vessels == i
        else:
            pre_mask_vein |= labeled_vessels == i

    return pre_mask_artery, pre_mask_vein

# ================================ Correlation ============================================== #

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


def compute_correlation(video, signal):
    """
    Compute the zero-lag correlation between the video signal and the average signal in the mask.

    Parameters:
        video (np.ndarray): 3D array of shape (H, W, T)
        mask (np.ndarray): 2D binary mask of shape (H, W)

    Returns:
        R (np.ndarray): 1D array of correlation values
    """
    
    # --- 1) Compute first correlation ---
    # # compute signal in 3 dimensions for correlation in the mask
    # signal = np.nansum(video * mask[np.newaxis, :, :], axis=(1, 2))
    # signal = signal / np.count_nonzero(mask)

    # # Detect outliers using a moving median and threshold
    # def detect_outliers_moving_median(x, window=5, threshold_factor=2.0):
    #     padded = np.pad(x, (window//2,), mode='edge')
    #     mov_median = uniform_filter1d(padded, size=window, mode='nearest')[window//2:-(window//2)]
    #     deviation = np.abs(x - mov_median)
    #     mad = np.median(deviation)
    #     return deviation > threshold_factor * mad if mad != 0 else np.zeros_like(x, dtype=bool)

    # outlier_frames_mask = detect_outliers_moving_median(signal, window=5, threshold_factor=2)
    # video = interpolate_outlier_frames(video, outlier_frames_mask)  # Needs to be defined

    # # Recompute signal after outlier interpolation
    # signal = np.nansum(video * mask[np.newaxis, :, :], axis=(1, 2))
    # signal = signal / np.count_nonzero(mask)

    # compute local-to-average signal wave zero-lag correlation
    signal_centered = signal - np.nanmean(signal)
    video_centered = video - np.nanmean(video)

    numerator = np.nanmean(video_centered * signal_centered[:, np.newaxis, np.newaxis], axis=0)
    denominator = np.nanstd(video_centered) * np.nanstd(signal_centered)
    
    R = numerator / denominator
    
    return R

# ================================ Diastole/Systole Analysis ================================ #

def validate_peaks(sys_idx_list, min_distance):
    """
    Validate Peaks (Removes peaks that are too close)
    Equivalent to MATLAB validate_peaks.
    """

    sys_idx_list = list(sys_idx_list)
    i = 0

    while i < len(sys_idx_list) - 1:
        if sys_idx_list[i + 1] - sys_idx_list[i] < min_distance:
            # remove next peak
            sys_idx_list.pop(i + 1)
        else:
            i += 1

    return sys_idx_list

def get_effective_sampling_freqency(sampling_freq, stride):
    return sampling_freq / stride * 1000.0

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


def find_systole_index(
    pulse_artery,
    sampling_freq,
    pulse_vein=None,
    lowpass_freq=15,
):
    """
    FIND_SYSTOLE_INDEX Identifies systole peaks in the pulse signal.

    Inputs:
        pulse_artery : 1D numpy array
        pulse_vein    : optional 1D numpy array
        savepng      : bool
        lowpass_freq : float

    Outputs:
        sys_idx_list
        sys_max_list
        sys_min_list
    """

    dt = 1.0 / sampling_freq

    flagVein = pulse_vein is not None and len(pulse_vein) > 0

    # ---------------- Step 1: Compute derivative ----------------
    diff_artery_signal = np.gradient(pulse_artery)

    if flagVein:
        diff_vein_signal = np.gradient(pulse_vein)

    # ---------------- Step 2: Detect peaks ----------------
    min_duration = 0.5  # seconds
    min_peak_height = np.percentile(diff_artery_signal, 95)
    min_peak_distance = int(np.floor(min_duration / dt))

    peaks, _ = find_peaks(
        diff_artery_signal,
        height=min_peak_height,
        distance=min_peak_distance
    )

    sys_idx_list = peaks.tolist()

    # ---------------- Step 3: Validate peaks ----------------
    sys_idx_list = validate_peaks(sys_idx_list, 10)

    # ---------------- Step 4: Find local maxima and minima ----------------
    num_peaks = len(sys_idx_list)

    if num_peaks == 0:
        raise RuntimeError(
            "No systole peaks detected. Check signal quality or adjust parameters."
        )

    sys_max_list = np.zeros(num_peaks, dtype=int)
    sys_min_list = np.zeros(num_peaks, dtype=int)

    # main cycles
    for i in range(num_peaks - 1):
        L = sys_idx_list[i + 1] - sys_idx_list[i]
        D = int(round(L / 2))

        # --- max in first half ---
        start = sys_idx_list[i]
        end = start + D + 1
        local = pulse_artery[start:end]
        amax = np.argmax(local)
        sys_max_list[i] = start + amax

        # --- min in second half ---
        start2 = sys_idx_list[i] + D
        end2 = sys_idx_list[i + 1]
        local2 = pulse_artery[start2:end2]
        amin = np.argmin(local2)
        sys_min_list[i + 1] = start2 + amin

    # --- minimum before first cycle ---
    first_peak = sys_idx_list[0]
    amin = np.argmin(pulse_artery[:first_peak + 1])
    sys_min_list[0] = amin

    # --- maximum after last cycle ---
    last_peak = sys_idx_list[-1]
    amax = np.argmax(pulse_artery[last_peak:])
    sys_max_list[-1] = last_peak + amax

    # MATLAB transposes → ensure row-like arrays
    sys_max_list = sys_max_list.tolist()
    sys_min_list = sys_min_list.tolist()

    return (
        sys_idx_list,
        sys_max_list,
        sys_min_list,
    )

def compute_diasys(video, pulse_artery, sampling_frequency, pulse_vein=None):
    numFrames = video.shape[0]

    # --- Filter pulse_artery to remove high frequency noise ---

    sys_index_list, _, _ = find_systole_index(
        pulse_artery, sampling_frequency, pulse_vein
    )

    # --- Empty systole case ---
    if sys_index_list is None or len(sys_index_list) == 0:
        print("Warning: sys_index_list is empty. Skipping systole/diastole.")

        amin = np.argmin(video, axis=2)
        amax = np.argmax(video, axis=2)

        # approximate MATLAB behavior
        M0_Systole_img = np.take_along_axis(video, amax[..., None], axis=0)[..., 0]
        M0_Diastole_img = np.take_along_axis(video, amin[..., None], axis=0)[..., 0]

        return M0_Systole_img, M0_Diastole_img, 

    numSys = len(sys_index_list)
    fpCycle = int(round(numFrames / numSys))

    sysindexes = []
    diasindexes = []

    # ---------------- Diastole ranges ----------------
    for idx in range(numSys):
        try:
            start_idx = max(sys_index_list[idx] + int(round(fpCycle * 0.60)), 0)
            search_end = min(sys_index_list[idx] + int(round(fpCycle * 0.95)), numFrames - 1)

            local = pulse_artery[start_idx:search_end + 1]
            if len(local) == 0:
                continue

            end_rel = np.argmin(local)
            end_idx = start_idx + end_rel

            dias_range = list(range(start_idx, min(end_idx + 1, numFrames)))
            diasindexes.extend(dias_range)

        except Exception:
            pass

    # ---------------- Systole ranges ----------------
    for idx in range(numSys):
        try:
            start_idx = sys_index_list[idx]
            search_end = min(start_idx + int(round(fpCycle * 0.35)), numFrames - 1)

            local = pulse_artery[start_idx:search_end + 1]
            if len(local) == 0:
                continue

            end_rel = np.argmax(local)
            end_idx = start_idx + end_rel

            sys_range = list(range(start_idx, min(end_idx + 1, numFrames)))
            sysindexes.extend(sys_range)

        except Exception:
            pass

    # --- Bounds / uniqueness ---
    sysindexes = sorted(set(i for i in sysindexes if 0 <= i < numFrames))
    diasindexes = sorted(set(i for i in diasindexes if 0 <= i < numFrames))

    print(f"    - Identified {len(sysindexes)} systole frames and {len(diasindexes)} diastole frames.")

    if len(sysindexes) == 0:
        sysindexes = [0]
    if len(diasindexes) == 0:
        diasindexes = [0]

    # --- Mean images ---
    M0_Systole_img, M0_Diastole_img = np.nanmean(video[sysindexes], axis=0), np.nanmean(video[diasindexes], axis=0), 

    return M0_Systole_img, M0_Diastole_img, sysindexes, diasindexes

def compute_diasys_image(video, pulse_artery, sampling_frequency, pulse_vein=None):
    M0_Systole_img, M0_Diastole_img, _, _, = compute_diasys(video, pulse_artery, sampling_frequency=sampling_frequency, pulse_vein=pulse_vein)

    sys = image_utils.normalize_image(M0_Systole_img)
    dias = image_utils.normalize_image(M0_Diastole_img)
    diasys_image = image_utils.normalize_image(sys - dias)
    return diasys_image, M0_Systole_img, M0_Diastole_img
 