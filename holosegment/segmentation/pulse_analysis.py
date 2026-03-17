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


def select_regular_peaks(signals_n, method, threshold=0.1, tolerance=0.3):
    """
    Select signals with regular or periodic derivative peaks.

    Parameters
    ----------
    signals_n : np.ndarray
        Normalized signals (num_branches x num_frames)
    method : str
        One of {'regular', 'minmax', 'kmeans_cosine'}
    params : dict
        Dictionary with optional fields:
            'threshold': float (used for 'regular' method)
            'tolerance': float in (0, 1) (used for 'regular' method)

    Returns
    -------
    s_idx : np.ndarray
        1D array of length num_branches, with 1 = artery, 0 = vein
    """

    stride = 512
    sampling_freq = 37.037
    dt = stride / sampling_freq / 1000.0  # seconds per frame
    sampling_freq = 1.0 / dt
    gradient_n = np.gradient(signals_n, axis=1)

    if method == "minmax":
        return _select_minmax(signals_n, gradient_n, sampling_freq, dt)
    # elif method == "regular":
    #     return _select_regular(gradient_n, threshold, tolerance)
    # elif method == "kmeans_cosine":
    #     return _select_kmeans(signals_n, "cosine", sampling_freq, dt)
    # else:
    #     raise ValueError(f"Unknown method: {method}")


# === Subfunctions ===

def _select_minmax(signals_n, gradient_n, sampling_freq, dt):
    num_branches, num_frames = signals_n.shape

    # Average normalized signal across all branches
    avg_signal = np.mean(signals_n, axis=0)

    # --- FFT analysis ---
    Y = fft.fft(avg_signal)
    P2 = np.abs(Y / num_frames)

    half = num_frames // 2
    P1 = P2[:half + 1]
    if len(P1) > 2:
        P1[1:-1] *= 2

    f = sampling_freq * np.arange(len(P1)) / num_frames

    f_range = (f > 0.5) & (f < 5)  # 30–300 bpm
    if not np.any(f_range):
        return np.zeros(num_branches, dtype=int)

    P1_sel = P1[f_range]
    f_sel = f[f_range]
    f0 = f_sel[np.argmax(P1_sel)]
    idx0 = int(round(f0 / dt))

    # --- Peak-based classification ---
    s_idx = np.zeros(num_branches, dtype=int)

    for i in range(num_branches):
        locs, _ = find_peaks(np.abs(gradient_n[i, :]), distance=0.6 * idx0)
        peaks = np.abs(gradient_n[i, locs])
        print(peaks, locs)
        if len(peaks) == 0 or len(locs) == 0:
            continue

        # Ensure both arrays are numpy arrays
        locs = np.asarray(locs)
        peaks = np.asarray(peaks)

        loc_ind = int(np.argmax(peaks))  # safest conversion
        loc = int(locs[loc_ind])         # ensure Python int

        print(f"Branch {i+1}: f0 = {f0:.2f} Hz, peak at index {loc} with value {gradient_n[i, loc]:.4f}")

        s_idx[i] = 1 if gradient_n[i, loc] > 0 else 0

    return s_idx

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

    print(f"FFT computed. P1 length: {len(P1)}")

    # Frequency vector
    f = sampling_frequency * np.arange(len(P1)) / num_frames

    # Find dominant frequency in physiological range (e.g. 0.5 - 2 Hz)
    f_range = (f > 0.5) & (f < 2) # 30 - 120 bpm
    f_sel = f[f_range]
    P_sel = P1[f_range]

    print(f"Selected frequencies in range: {f_sel}")
    print(f"Corresponding power values: {P_sel}")

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

def compute_pre_artery_mask(video, vessel_mask, optic_disc_center, sampling_frequency, output_manager):
    """
    Compute a preliminary artery mask based on pulse analysis of the video frames within the vessel mask
    """
    # Step 1: Separate mask into branches
    labeled_vessels, _ = process_masks.get_labeled_vesselness(vessel_mask, *optic_disc_center)

    # image_utils.save_array_as_image(labeled_vessels, "all_20_label_Vesselness.png", foldername=step_mask_folder)
    num_branches = labeled_vessels.max()
    num_frames = video.shape[0]

    # Step 2: Compute mean temporal signal for each branch
    signals = np.zeros((num_branches, num_frames))

    # Design low-pass Butterworth filter
    b, a = butter(4, 15 / (sampling_frequency / 2), btype='low')

    for i in range(1, num_branches + 1):
        branch_mask = (labeled_vessels == i)
        # Extract pixels for this branch over time
        branch_pixels = video[:, branch_mask]
        branch_mean = np.mean(branch_pixels, axis=1)
        # Apply zero-phase filtering
        signals[i - 1, :] = filtfilt(b, a, branch_mean)
        # output_manager.debug("pulse_analysis", f"branch_{i}_signal", signals[i - 1, :], title=f"Branch {i} Temporal Signal")

    signals_n = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)

    # Step 3: Select regular peaks to classify arteries vs veins
    # idx0 = compute_idx0(signals_n, sampling_frequency)
    s_idx  = select_regular_peaks(signals_n, "minmax")

    is_pure = np.array([check_validity(sig, sampling_frequency) for sig in signals_n])
    if not is_pure.any():
        is_pure[:] = True

    # Step 4: Combine into artery / vein masks
    pre_mask_artery = np.zeros_like(vessel_mask, bool)
    pre_mask_vein = np.zeros_like(vessel_mask, bool)

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


def compute_correlation(video, mask):
    """
    Compute the zero-lag correlation between the video signal and the average signal in the mask.

    Parameters:
        video (np.ndarray): 3D array of shape (H, W, T)
        mask (np.ndarray): 2D binary mask of shape (H, W)

    Returns:
        R (np.ndarray): 1D array of correlation values
    """
    
    # --- 1) Compute first correlation ---
    # compute signal in 3 dimensions for correlation in the mask
    signal = np.nansum(video * mask[np.newaxis, :, :], axis=(1, 2))
    signal = signal / np.count_nonzero(mask)

    # # Detect outliers using a moving median and threshold
    # def detect_outliers_moving_median(x, window=5, threshold_factor=2.0):
    #     padded = np.pad(x, (window//2,), mode='edge')
    #     mov_median = uniform_filter1d(padded, size=window, mode='nearest')[window//2:-(window//2)]
    #     deviation = np.abs(x - mov_median)
    #     mad = np.median(deviation)
    #     return deviation > threshold_factor * mad if mad != 0 else np.zeros_like(x, dtype=bool)

    # outlier_frames_mask = detect_outliers_moving_median(signal, window=5, threshold_factor=2)
    # video = interpolate_outlier_frames(video, outlier_frames_mask)  # Needs to be defined

    # Recompute signal after outlier interpolation
    signal = np.nansum(video * mask[np.newaxis, :, :], axis=(1, 2))
    signal = signal / np.count_nonzero(mask)

    # compute local-to-average signal wave zero-lag correlation
    signal_centered = signal - np.nanmean(signal)
    video_centered = video - np.nanmean(video)

    numerator = np.nanmean(video_centered * signal_centered[:, np.newaxis, np.newaxis], axis=0)
    denominator = np.nanstd(video_centered) * np.nanstd(signal_centered)
    
    R = numerator / denominator
    
    return R, signal_centered

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




def get_filtered_pulse():
    pass


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
        pulse_artery_filtered
        sys_max_list
        sys_min_list
    """

    dt = 1.0 / sampling_freq

    flagVein = pulse_vein is not None and len(pulse_vein) > 0

    # ---------------- Step 1: Extract pulse signal ----------------
    b, a = butter(4, lowpass_freq / (sampling_freq / 2), btype="low")
    pulse_artery_filtered = filtfilt(b, a, pulse_artery)

    if flagVein:
        pulse_vein_filtered = filtfilt(b, a, pulse_vein)

    # ---------------- Step 2: Compute derivative ----------------
    diff_artery_signal = np.gradient(pulse_artery_filtered)

    if flagVein:
        diff_vein_signal = np.gradient(pulse_vein_filtered)

    # ---------------- Step 3: Detect peaks ----------------
    min_duration = 0.5  # seconds
    min_peak_height = np.percentile(diff_artery_signal, 95)
    min_peak_distance = int(np.floor(min_duration / dt))

    peaks, _ = find_peaks(
        diff_artery_signal,
        height=min_peak_height,
        distance=min_peak_distance
    )

    sys_idx_list = peaks.tolist()

    # ---------------- Step 4: Validate peaks ----------------
    sys_idx_list = validate_peaks(sys_idx_list, 10)

    # ---------------- Step 5: Find local maxima and minima ----------------
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
        local = pulse_artery_filtered[start:end]
        amax = np.argmax(local)
        sys_max_list[i] = start + amax

        # --- min in second half ---
        start2 = sys_idx_list[i] + D
        end2 = sys_idx_list[i + 1]
        local2 = pulse_artery_filtered[start2:end2]
        amin = np.argmin(local2)
        sys_min_list[i + 1] = start2 + amin

    # --- minimum before first cycle ---
    first_peak = sys_idx_list[0]
    amin = np.argmin(pulse_artery_filtered[:first_peak + 1])
    sys_min_list[0] = amin

    # --- maximum after last cycle ---
    last_peak = sys_idx_list[-1]
    amax = np.argmax(pulse_artery_filtered[last_peak:])
    sys_max_list[-1] = last_peak + amax

    # MATLAB transposes → ensure row-like arrays
    sys_max_list = sys_max_list.tolist()
    sys_min_list = sys_min_list.tolist()

    return (
        sys_idx_list,
        pulse_artery_filtered,
        sys_max_list,
        sys_min_list,
    )

def compute_diasys(video, mask, stride=512, sampling_frequency=37.037):
    numFrames, H, W  = video.shape

    # --- Pulse artery signal ---
    # sum over H,W for each frame, normalized by mask area
    mask_nnz = np.count_nonzero(mask)
    pulse_artery = np.nansum(video[:, mask.astype(bool)], axis=(1)) / max(mask_nnz, 1)

    # --- Filter pulse_artery to remove high frequency noise ---
    sampling_freq = sampling_frequency * 1000 / stride  # Hz
    b, a = butter(4, 15 / (sampling_freq / 2), btype='low')
    pulse_artery = filtfilt(b, a, pulse_artery)

    sys_index_list, fullPulse, _, _ = find_systole_index(
        pulse_artery, sampling_freq=sampling_freq
    )

    fullPulse = np.asarray(fullPulse).ravel()

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

            local = fullPulse[start_idx:search_end + 1]
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

            local = fullPulse[start_idx:search_end + 1]
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

    print(f"Identified {len(sysindexes)} systole frames and {len(diasindexes)} diastole frames.")

    if len(sysindexes) == 0:
        sysindexes = [0]
    if len(diasindexes) == 0:
        diasindexes = [0]

    # --- Mean images ---
    M0_Systole_img, M0_Diastole_img = np.nanmean(video[sysindexes], axis=0), np.nanmean(video[diasindexes], axis=0), 

    return M0_Systole_img, M0_Diastole_img, sysindexes, diasindexes, fullPulse

def compute_diasys_image(video, mask, stride=512, sampling_frequency=37.037):
    M0_Systole_img, M0_Diastole_img, _, _, fullPulse = compute_diasys(video, mask, stride=stride, sampling_frequency=sampling_frequency)

    sys = image_utils.normalize_image(M0_Systole_img)
    dias = image_utils.normalize_image(M0_Diastole_img)
    diasys_image = image_utils.normalize_image(sys - dias)
    return diasys_image, M0_Systole_img, M0_Diastole_img, fullPulse
 