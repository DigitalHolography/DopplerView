"""
Pulse analysis module for analyzing temporal pulsatility in vessels
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter,detrend
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

from skimage.measure import label
from skimage import measure
from sklearn.cluster import KMeans

from dopplerview.segmentation import signal_processing
from dopplerview.utils import image_utils

import warnings

import logging
logger = logging.getLogger(__name__)

# ================================ Pre-artery mask ================================ #

def get_beats(signal, sys_idx, target_len=None):
    """
    Parameters
    ----------
    signal : 1D array
    sys_idx : array-like
        Indices of systolic peaks (length = n_beats + 1 ideally)
    target_len : int or None
        Length to interpolate each beat to. If None, uses max beat length.

    Returns
    -------
    beats : (n_beats, target_len)
    """
    
    signal = np.asarray(signal)
    sys_idx = np.asarray(sys_idx, dtype=int).reshape(-1)
    if sys_idx.size < 2:
        empty_length = 0 if target_len is None else int(target_len)
        return np.empty((0, max(0, empty_length)), dtype=float)
    if np.any(np.diff(sys_idx) <= 0):
        raise ValueError("sys_idx must contain strictly increasing peak indices")
    if sys_idx[0] < 0 or sys_idx[-1] > signal.size:
        raise ValueError("peak indices must lie within the signal")
    n_beats = len(sys_idx) - 1

    # --- Determine target length ---
    lengths = np.diff(sys_idx)
    if target_len is None:
        target_len = int(np.max(lengths))
    target_len = int(target_len)
    if target_len < 1:
        raise ValueError("target_len must be positive")

    beats = np.zeros((n_beats, target_len))

    # --- Process each beat ---
    for i in range(n_beats):
        start, end = sys_idx[i], sys_idx[i + 1]
        beat = signal[start:end]

        if len(beat) < 2:
            beats[i, :] = np.nan
            continue

        # Normalize time axis
        x_old = np.linspace(0, 1, len(beat))
        x_new = np.linspace(0, 1, target_len)

        f = interp1d(x_old, beat, kind='linear', fill_value="extrapolate")
        beats[i, :] = f(x_new)

    # --- Median beat (robust template) ---
    return beats

def fill_with_beat(beat, start_offset, end_offset, length):
    """
    Fill the start and end of the signal with the average beat to create a pseudo beat.
    """
    beat = np.asarray(beat)
    l = len(beat)
    if l == 0:
        raise ValueError("beat must not be empty")
    pseudo_signal = np.zeros(length)
    i = start_offset

    # If the beat is shorter than the offsets, we need to repeat it
    for i in range(start_offset // l):
        index = i * l + start_offset % l
        pseudo_signal[index:index + l] = beat
    for i in range(end_offset // l):
        index = length - end_offset + i * l
        pseudo_signal[index:index + l] = beat

    # Fill the remaining part with the appropriate segment of the beat
    start_remainder = start_offset % l
    if start_remainder > 0:
        pseudo_signal[:start_remainder] = beat[-start_remainder:]
    if end_offset % l > 0:
        pseudo_signal[-(end_offset % l):] = beat[:(end_offset % l)]
    return pseudo_signal

def get_pseudo_signal(beat, peaks, length):
    """
    Create a pseudo beat by repeating the average beat and aligning it with the peaks.
    """
    beat = np.asarray(beat)
    if beat.size == 0:
        raise ValueError("beat must not be empty")
    x_old = np.linspace(0, 1, len(beat))
    if peaks is None or len(peaks) == 0:
        pseudo_signal = np.tile(beat, (length // len(beat) + 1))[:length]
        return pseudo_signal

    pseudo_signal = fill_with_beat(beat, int(peaks[0]), length - int(peaks[-1]), length)
    f = interp1d(x_old, beat, kind='linear', fill_value="extrapolate")
    for i in range(len(peaks) - 1):
        pseudo_signal[peaks[i]:peaks[i + 1]] = f(np.linspace(0, 1, peaks[i + 1] - peaks[i]))
        
    return pseudo_signal

def correct_signal(signal, pseudo_signal, k=2):
    """
    Correct the signal using a robust method based on the pseudo signal.
    
    Parameters
    ----------
    signal : 1D array
        The original signal to be corrected.
    pseudo_signal : 1D array
        The pseudo signal used as a reference.
    k : float
        Tuning parameter for the soft rejection (default: 1.5).
    
    Returns
    -------
    signal_clean : 1D array
        The corrected signal.
    """

    signal = np.asarray(signal)
    pseudo_signal = np.asarray(pseudo_signal)
    if signal.shape != pseudo_signal.shape:
        raise ValueError("signal and pseudo_signal must have the same shape")
    if k <= 0:
        raise ValueError("k must be positive")

    residual = signal - pseudo_signal

    # robust scale
    sigma = 1.4826 * np.median(np.abs(residual))

    # weight (soft rejection)
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        return signal.copy()
    alpha = np.clip(1 - (np.abs(residual) / (k * sigma)), 0, 1)

    signal_clean = pseudo_signal + alpha * residual
    return signal_clean

def get_peaks(signal, beat_period, height_percentile=80, distance_tolerance=0.8):
    """
    Get peaks of the signal using a robust method based on the beat period.
    """
    signal = np.asarray(signal).reshape(-1)
    if beat_period is None or not np.isfinite(beat_period) or beat_period <= 0:
        raise ValueError("beat_period must be a finite positive number")
    if signal.size < 3:
        return np.empty(0, dtype=int)
    diff_artery_signal = np.gradient(signal)
    min_peak_height = np.percentile(diff_artery_signal, height_percentile)
    min_peak_distance = max(1, int(beat_period * distance_tolerance))  # Allow some variability in heart rate

    peaks, _ = find_peaks(
        diff_artery_signal,
        height=min_peak_height,
        distance=min_peak_distance
    )

    return peaks

def correct_signal_with_heartbeat(signal, beat_period, sampling_freq=None, k=2, distance_tolerance=0.8, use_peaks=True):
    """Correct the signal using heartbeat-based correction.
    Parameters
    ----------
    signal : 1D array
        The original signal to be corrected.
    beat_period : int
        The estimated period of the heartbeat in samples.
    k : float
        Tuning parameter for the soft rejection (default: 2).
    distance_tolerance : float
        Tolerance for peak distance (default: 0.8).
    use_peaks : bool
        Whether to use peaks for correction (default: True).

    Returns
    -------
    corrected_signal : 1D array
        The corrected signal.
    """
    signal = np.asarray(signal)
    signal_length = len(signal)
    if use_peaks:
        peaks = get_peaks(signal, beat_period, distance_tolerance=distance_tolerance)
        if peaks.size < 2:
            return signal.copy()
        beats = get_beats(signal, peaks, target_len=signal_length)
        average_beat = np.nanmedian(beats, axis=0)
    else:
        peaks = None
        average_beat, beat_period = get_cycle_template(signal, sampling_freq=sampling_freq, beat_period=beat_period, return_period=True)

    pseudo_signal = get_pseudo_signal(average_beat, peaks=peaks, length=signal_length)
    corrected_signal = correct_signal(signal, pseudo_signal, k=k)
    return corrected_signal


def _safe_beat_correlation(beat, reference):
    """Return zero when a beat pair cannot define a Pearson correlation."""
    beat = np.asarray(beat, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(beat) & np.isfinite(reference)
    if np.count_nonzero(valid) < 2:
        return 0.0
    beat_centered = beat[valid] - np.mean(beat[valid])
    reference_centered = reference[valid] - np.mean(reference[valid])
    denominator = np.linalg.norm(beat_centered) * np.linalg.norm(reference_centered)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(beat_centered, reference_centered) / denominator)


def remove_bad_beats(signal, beat_period, threshold=0.5):
    """Remove bad beats from the signal based on correlation with the average beat."""
    signal = np.asarray(signal)
    peaks = get_peaks(signal, beat_period)
    if peaks.size < 2:
        return signal.copy(), signal.copy(), np.empty(0, dtype=float)
    beats = get_beats(signal, peaks)
    median_beat = np.nanmedian(beats, axis=0)

    # Compute correlation of each beat with the average beat
    correlations = np.array([
        _safe_beat_correlation(beat, median_beat) for beat in beats
    ])
    good_beats_mask = correlations > threshold
    if not np.any(good_beats_mask):
        return signal.copy(), signal.copy(), correlations

    # Create a cleaned signal by keeping only good beats
    beat_signal = get_pseudo_signal(median_beat, peaks, len(signal))
    keep_samples = np.zeros(signal.shape, dtype=bool)
    for i, is_good in enumerate(good_beats_mask):
        if is_good:
            start, end = peaks[i], peaks[i + 1]
            keep_samples[start:end] = True
        else:
            start, end = peaks[i], peaks[i + 1]
            beat_signal[start:end] = 0

    return signal[keep_samples], beat_signal[keep_samples], correlations

def remove_bad_beats_on_video(signal, video, beat_period, threshold=0.5, distance_tolerance=0.8, use_peaks=True):
    """Remove frames on the video corresponding to bad beats from the signal based on correlation with the average beat."""
    signal = np.asarray(signal)
    video = np.asarray(video)
    if video.shape[0] != signal.size:
        raise ValueError("video and signal must have the same temporal length")
    if use_peaks:
        signal_length = len(signal)
        peaks = get_peaks(signal, beat_period, distance_tolerance=distance_tolerance)
        if peaks.size < 2:
            return (
                signal.copy(),
                video.copy(),
                signal.copy(),
                np.empty(0, dtype=float),
                peaks,
            )
        beats = get_beats(signal, peaks, target_len=signal_length)
        median_beat = np.nanmedian(beats, axis=0)
    else:
        beats = get_cycle_templates(signal, beat_period)
        if beats.shape[0] == 0:
            return (
                signal.copy(),
                video.copy(),
                signal.copy(),
                np.empty(0, dtype=float),
                np.empty(0, dtype=int),
            )
        peaks = np.arange(beats.shape[0] + 1, dtype=int) * int(beat_period)
        median_beat = np.nanmedian(beats, axis=0)

    # Compute correlation of each beat with the average beat
    correlations = np.array([
        _safe_beat_correlation(beat, median_beat) for beat in beats
    ])
    good_beats_mask = correlations > threshold
    if not np.any(good_beats_mask):
        return signal.copy(), video.copy(), signal.copy(), median_beat, peaks

    # Create a cleaned signal by keeping only good beats
    beat_signal = get_pseudo_signal(median_beat, peaks, len(signal))
    mask_signal = np.zeros(signal.shape, dtype=bool)
    for i, is_good in enumerate(good_beats_mask):
        if is_good:
            start, end = max(peaks[i]-10,0), peaks[i + 1] # keep the beat and a small window before the peak to capture the systolic upstroke

            mask_signal[start:end] = True

    cleaned_video = video[mask_signal]
    cleaned_signal = signal[mask_signal]

    return cleaned_signal, cleaned_video, beat_signal, median_beat, peaks


# Backward compatibility for notebooks that used the accidental duplicated
# suffix before the public function name was corrected.
remove_bad_beats_on_video_on_video = remove_bad_beats_on_video


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

        locs = peaks

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

# def compute_idx0(signals_n, sampling_frequency,
#                  fmin=0.5, fmax=2.0):
#     """
#     Robust estimation of cardiac period (idx0)
#     """

#     num_frames = signals_n.shape[1]

#     # --- Robust average ---
#     avg_signal = np.median(signals_n, axis=0)

#     # --- Detrend ---
#     avg_signal = detrend(avg_signal, type='linear')

#     # --- Windowing ---
#     window = np.hanning(num_frames)
#     avg_signal = avg_signal * window

#     # --- FFT ---
#     Y = np.fft.rfft(avg_signal)
#     P = np.abs(Y)**2

#     # --- Frequency vector ---
#     f = np.fft.rfftfreq(num_frames, d=1/sampling_frequency)

#     # --- Physiological band ---
#     mask = (f > fmin) & (f < fmax)
#     f_sel = f[mask]
#     P_sel = P[mask]

#     if len(P_sel) == 0 or np.sum(P_sel) == 0:
#         return None

#     # --- Smooth spectrum ---
#     P_sel = gaussian_filter1d(P_sel, sigma=2)

#     # --- Weighted frequency (robust) ---
#     f0 = np.sum(f_sel * P_sel) / np.sum(P_sel)

#     # --- Convert to index ---
#     t0 = 1 / f0
#     idx0 = int(round(t0 * sampling_frequency))

#     return idx0

def compute_period(signal, sampling_frequency,
                 fmin=0.5, fmax=2.0):
    """
    Robust estimation of cardiac period in number of frames
    """
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(fmin) or not np.isfinite(fmax) or not 0 < fmin < fmax:
        raise ValueError("fmin and fmax must be finite and satisfy 0 < fmin < fmax")

    signal = np.squeeze(np.asarray(signal, dtype=float))
    if signal.ndim == 2:
        signal = np.nanmedian(signal, axis=0)
    elif signal.ndim != 1:
        raise ValueError("signal must be one-dimensional or a 2D collection of signals")

    num_frames = len(signal)
    if num_frames < 4:
        return None

    finite = np.isfinite(signal)
    if np.count_nonzero(finite) < 4:
        return None
    if not finite.all():
        sample_indices = np.arange(num_frames)
        signal = np.interp(sample_indices, sample_indices[finite], signal[finite])
    scale = max(1.0, np.max(np.abs(signal)))
    if np.ptp(signal) <= np.finfo(float).eps * scale:
        return None

    # --- Detrend ---
    signal = detrend(signal, type='linear')

    # --- Windowing ---
    window = np.hanning(num_frames)
    signal = signal * window

    # --- FFT ---
    Y = np.fft.rfft(signal)
    P = np.abs(Y)**2

    # --- Frequency vector ---
    f = np.fft.rfftfreq(num_frames, d=1/sampling_frequency)

    # --- Physiological band ---
    mask = (f > fmin) & (f < fmax)
    f_sel = f[mask]
    P_sel = P[mask]

    selected_power = np.sum(P_sel)
    if len(P_sel) == 0 or not np.isfinite(selected_power) or selected_power <= 0:
        return None

    # --- Smooth spectrum ---
    P_sel = gaussian_filter1d(P_sel, sigma=2)

    # --- Weighted frequency (robust) ---
    smoothed_power = np.sum(P_sel)
    if not np.isfinite(smoothed_power) or smoothed_power <= 0:
        return None
    f0 = np.sum(f_sel * P_sel) / smoothed_power
    if not np.isfinite(f0) or f0 <= 0:
        return None

    # --- Convert to index ---
    t0 = 1 / f0
    period_frames = int(round(t0 * sampling_frequency))

    return period_frames


def check_validity(signal, sampling_frequency, beat_period=None):
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
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size < 4 or not np.all(np.isfinite(signal)):
        return False
    signal = signal - np.mean(signal)   # remove DC
    numFrames = signal.size

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
    if P_local.size == 1:
        spectralEntropy = 0.0
    else:
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
    video = np.asarray(video)
    labeled_vessels = np.asarray(labeled_vessels)
    if video.ndim != 3:
        raise ValueError("video must have shape (T, H, W)")
    if labeled_vessels.shape != video.shape[1:]:
        raise ValueError("labeled_vessels must match the video's spatial shape")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 30:
        raise ValueError("sampling_frequency must be greater than 30 Hz for a 15 Hz low-pass filter")

    num_frames = video.shape[0]
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    signals = np.zeros((branch_ids.size, num_frames))
    b, a = butter(4, 15 / (sampling_frequency / 2), btype='low')
    padlen = 3 * max(len(a), len(b))
    moving_window = round(sampling_frequency * 0.1)

    flat_labels = labeled_vessels.ravel()
    vessel_positions = np.flatnonzero(flat_labels > 0)
    vessel_labels = flat_labels[vessel_positions]
    order = np.argsort(vessel_labels, kind="stable")
    vessel_positions = vessel_positions[order]
    vessel_labels = vessel_labels[order]
    vessel_pixels = video.reshape(num_frames, -1)[:, vessel_positions]

    for signal_index, branch_id in enumerate(branch_ids):
        start = np.searchsorted(vessel_labels, branch_id, side="left")
        stop = np.searchsorted(vessel_labels, branch_id, side="right")
        branch_pixels = vessel_pixels[:, start:stop]
        branch_mean = np.nanmean(branch_pixels, axis=1)

        if branch_mean.size <= padlen:
            signals[signal_index, :] = branch_mean
        else:
            signals[signal_index, :] = filtfilt(b, a, branch_mean)

        if moving_window > 1:
            signals[signal_index, :] = signal_processing.movmean(
                signals[signal_index, :], moving_window
            )

    return signals

def get_cycle_templates(signal, beat_period):
    if beat_period is None or not np.isfinite(beat_period) or beat_period <= 0:
        raise ValueError("beat_period must be a finite positive number")
    beat_period = int(beat_period)
    if beat_period < 1:
        raise ValueError("beat_period must be at least one sample")
    n_cycles = len(signal)//beat_period

    cycles = signal[:n_cycles*beat_period].reshape(
        n_cycles,
        beat_period
    )
    return cycles

def get_cycle_template(signal, sampling_freq=None, beat_period=None, return_period=False):
    if beat_period is None:
        if sampling_freq is None:
            raise ValueError("Either sampling_freq or beat_period must be provided.")
        beat_period = compute_period(signal, sampling_freq)
    cycles = get_cycle_templates(signal, beat_period)
    if cycles.shape[0] == 0:
        raise ValueError("signal does not contain a complete cardiac cycle")
    result = np.average(cycles, axis=0)
    if return_period:
        return result, beat_period
    return result

def compute_z(template):
    template = template - np.mean(template)

    H1 = np.fft.rfft(template)[1]

    if np.abs(H1) < 1e-12:
        return np.nan + 1j*np.nan

    return H1 / np.abs(H1)

def get_nb_of_positive_peaks(signal, beat_period):
    gradient = np.gradient(signal)
    # --- find peaks in |gradient| ---
    peaks, properties = find_peaks(
        np.abs(gradient),
        distance=int(0.8 * beat_period),
        prominence=1e-6   # helps avoid spurious peaks
    )

    # values of gradient at peak positions
    peaks_v = gradient[peaks]

    # count positive peaks
    return np.sum(peaks_v > 0)

def compute_pre_masks_by_systolic_gradient(signals, labeled_vessels, sampling_frequency):
    """
    Compute a preliminary artery mask based on pulse analysis of the video frames within the vessel mask.
    """

    idx0 = compute_period(signals, sampling_frequency)
    s_idx, _ = select_regular_peaks(signals, "minmax", idx0)

    is_pure = np.array([check_validity(sig, sampling_frequency) for sig in signals])
    if not is_pure.any():
        is_pure[:] = True

    # Step 4: Combine into artery / vein masks
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if branch_ids.size != len(signals):
        raise ValueError("signals must contain exactly one row per labeled vessel branch")
    valid_arteries = branch_ids[is_pure & (s_idx == 1)]
    valid_veins = branch_ids[is_pure & (s_idx != 1)]
    pre_mask_artery = np.isin(labeled_vessels, valid_arteries)
    pre_mask_vein = np.isin(labeled_vessels, valid_veins)

    return pre_mask_artery, pre_mask_vein


def canonicalize_binary_cluster_labels(labels, features):
    """Make binary cluster IDs independent of K-means' arbitrary label order."""
    labels = np.asarray(labels)
    features = np.asarray(features)
    if labels.ndim != 1 or features.ndim != 2 or len(labels) != len(features):
        raise ValueError("labels and features must have matching sample dimensions")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary cluster canonicalization requires labels 0 and 1")
    centers = np.array([features[labels == cluster].mean(axis=0) for cluster in (0, 1)])

    def sort_key(cluster):
        center = centers[cluster]
        if center.size == 2:
            return (np.arctan2(center[1], center[0]), *center)
        return tuple(center)

    order = sorted(
        (0, 1),
        key=sort_key,
    )
    mapping = {original: canonical for canonical, original in enumerate(order)}
    return np.array([mapping[label] for label in labels], dtype=int)


def compute_pre_masks_by_clustering(
    signals,
    labeled_vessels,
    sampling_frequency,
    ambiguity_policy="warn",
):
    """
    Compute a preliminary artery mask based on the clustering of the complex first Fourier harmonic of all signals. 
    The 2 clusters are then classified as artery or vein based on the number of positive peaks in their median signal's gradient.
    """

    signals = np.asarray(signals)
    if signals.ndim != 2 or signals.shape[0] < 2:
        raise ValueError("clustering requires at least two branch signals")
    if ambiguity_policy not in {"warn", "raise"}:
        raise ValueError("ambiguity_policy must be either 'warn' or 'raise'")

    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if branch_ids.size != len(signals):
        raise ValueError("signals must contain exactly one row per labeled vessel branch")

    # --- Compute median heartbeat template for each branch ---
    cycle_templates = [get_cycle_template(branch, sampling_frequency, return_period=True) for branch in signals]
    templates, periods = zip(*cycle_templates)
    z = np.array([compute_z(template) for template in templates])
    X = np.column_stack([
        np.real(z),
        np.imag(z)
    ])
    if not np.all(np.isfinite(X)):
        raise ValueError("clustering features contain a non-finite first harmonic")
    if np.unique(X, axis=0).shape[0] < 2:
        raise ValueError("clustering requires at least two distinct first-harmonic features")

    # --- Cluster branches based on heartbeat template shape ---
    labels = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=20,
        random_state=0,
        algorithm="lloyd",
    ).fit_predict(X)
    if np.unique(labels).size != 2:
        raise ValueError("K-means did not produce two non-empty clusters")
    labels = canonicalize_binary_cluster_labels(labels, X)

    cluster0 = np.where(labels == 0)[0]
    cluster1 = np.where(labels == 1)[0]

    mask0 = np.isin(labeled_vessels, branch_ids[cluster0])
    mask1 = ~mask0 & (labeled_vessels > 0)

    cluster0_period = np.median(np.array(periods)[cluster0], axis=0)
    cluster1_period = np.median(np.array(periods)[cluster1], axis=0)

    cluster0_signal = np.median(signals[cluster0], axis=0)
    cluster1_signal = np.median(signals[cluster1], axis=0)

    # --- Classify based on number of positive peaks in the gradient of the median heartbeat ---
    cluster0_peaks = get_nb_of_positive_peaks(cluster0_signal, cluster0_period)
    cluster1_peaks = get_nb_of_positive_peaks(cluster1_signal, cluster1_period)

    if cluster0_peaks != cluster1_peaks:
        artery_cluster = 0 if cluster0_peaks > cluster1_peaks else 1
    else:
        # Peak counts are discrete and can tie. A sharper positive upstroke is
        # the continuous physiological tie-breaker; normalized branch signals
        # make the two cluster scores comparable.
        upstroke0 = np.max(np.gradient(cluster0_signal))
        upstroke1 = np.max(np.gradient(cluster1_signal))
        if not np.isclose(upstroke0, upstroke1, rtol=1e-6, atol=1e-12):
            artery_cluster = 0 if upstroke0 > upstroke1 else 1
            logger.warning(
                "Equal positive-peak counts for both clusters; artery/vein "
                "assignment was resolved using systolic upstroke strength."
            )
        else:
            message = (
                "Ambiguous artery/vein cluster assignment: positive-peak "
                "counts and systolic upstroke strengths are equal."
            )
            if ambiguity_policy == "raise":
                raise ValueError(message)
            logger.warning(
                "%s Using canonical centroid-phase order as a deterministic fallback.",
                message,
            )
            artery_cluster = 0

    mask_artery = mask0 if artery_cluster == 0 else mask1
    mask_vein = mask1 if artery_cluster == 0 else mask0
    labels = np.where(labels == artery_cluster, 0, 1)

    return mask_artery, mask_vein, labels, z


# ================================ Correlation ============================================== #

def clean_cardiac_signal(sig, fs=250):
    sig = sig.copy().astype(float)

    # Step 1: Detect sudden large jumps (lead-off, disconnection)
    diff = np.abs(np.diff(sig))
    threshold = 3 * np.std(diff)
    bad_idx = np.where(diff > threshold)[0] + 1  # indices after the jump

    # Expand bad regions by a small window around each jump
    mask = np.zeros(len(sig), dtype=bool)
    for i in bad_idx:
        mask[max(0, i-3):min(len(sig), i+10)] = True

    # Step 2: Hampel identifier for isolated spikes
    window = 11
    for i in range(window//2, len(sig) - window//2):
        if mask[i]:
            continue
        local = sig[i - window//2 : i + window//2 + 1]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if np.abs(sig[i] - med) > 3 * 1.4826 * mad:
            mask[i] = True

    # Step 3: Cubic spline interpolation over bad regions
    good = ~mask
    x_good = np.where(good)[0]
    cs = CubicSpline(x_good, sig[good])
    sig[mask] = cs(np.where(mask)[0])

    # Step 4: Savitzky-Golay smoothing (preserves peaks)
    sig = savgol_filter(sig, window_length=9, polyorder=3)

    return sig, mask

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

def get_effective_sampling_frequency(sampling_freq, stride):
    if not np.isfinite(sampling_freq) or sampling_freq <= 0:
        raise ValueError("sampling_freq must be finite and positive")
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError("stride must be finite and positive")
    return sampling_freq / stride


def find_systole_index(
    pulse_artery,
    sampling_freq,
    pulse_vein=None,
    thresh=95,
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
    min_peak_height = np.percentile(diff_artery_signal, thresh)
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
        logger.warning("Warning: sys_index_list is empty. Skipping systole/diastole.")

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

    logger.info(f"    - Identified {len(sysindexes)} systole frames and {len(diasindexes)} diastole frames.")

    if len(sysindexes) == 0:
        sysindexes = [0]
    if len(diasindexes) == 0:
        diasindexes = [0]

    # --- Mean images ---
    M0_Systole_img, M0_Diastole_img = np.nanmean(video[sysindexes], axis=0), np.nanmean(video[diasindexes], axis=0)

    return M0_Systole_img, M0_Diastole_img, sysindexes, diasindexes, sys_index_list

def compute_diasys_image(video, pulse_artery, sampling_frequency, pulse_vein=None):
    M0_Systole_img, M0_Diastole_img, sysindexes, diasindexes, sys_index_list = compute_diasys(video, pulse_artery, sampling_frequency=sampling_frequency, pulse_vein=pulse_vein)

    diasys_image = M0_Systole_img - M0_Diastole_img
    return diasys_image, sysindexes, diasindexes, M0_Systole_img, M0_Diastole_img, sys_index_list

def assign_clusters_to_av(
    cluster_labels,
    video,
    periods,
    labeled_vessels,
    sampling_freq
):
    """
    Assign artery/vein labels from cluster labels.

    Assumes two clusters.
    """

    unique_clusters = np.unique(cluster_labels)

    if len(unique_clusters) != 2:
        raise ValueError(
            "Current artery/vein assignment "
            "requires exactly 2 clusters."
        )

    c0, c1 = unique_clusters

    idx0 = np.where(cluster_labels == c0)[0]
    idx1 = np.where(cluster_labels == c1)[0]

    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if branch_ids.size != len(cluster_labels):
        raise ValueError(
            "cluster_labels must contain exactly one value per labeled branch"
        )

    mask0 = np.isin(
        labeled_vessels,
        branch_ids[idx0],
    )

    mask1 = np.isin(
        labeled_vessels,
        branch_ids[idx1],
    )

    signal0 = signal_processing.get_pulse_from_mask(
        video,
        mask0,
    )

    signal1 = signal_processing.get_pulse_from_mask(
        video,
        mask1,
    )

    signal0 = signal_processing.get_filtered_pulse(
        signal0,
        sampling_frequency=sampling_freq,
    )

    signal1 = signal_processing.get_filtered_pulse(
        signal1,
        sampling_frequency=sampling_freq,
    )

    period0 = int(
        np.median(periods[idx0])
    )

    period1 = int(
        np.median(periods[idx1])
    )

    peaks0 = get_nb_of_positive_peaks(
        signal0,
        period0,
    )

    peaks1 = get_nb_of_positive_peaks(
        signal1,
        period1,
    )

    if peaks0 == peaks1:
        upstroke0 = np.max(np.gradient(signal0))
        upstroke1 = np.max(np.gradient(signal1))
        if np.isclose(upstroke0, upstroke1, rtol=1e-6, atol=1e-12):
            logger.warning(
                "Ambiguous benchmark artery/vein assignment; using canonical "
                "cluster order as a deterministic fallback."
            )
            artery_cluster = c0
        else:
            artery_cluster = c0 if upstroke0 > upstroke1 else c1
    else:
        artery_cluster = c0 if peaks0 > peaks1 else c1

    if artery_cluster == c0:

        artery_mask = mask0
        vein_mask = mask1

        mask_labels = np.where(
            cluster_labels == c0,
            0,
            1,
        )

    else:

        artery_mask = mask1
        vein_mask = mask0

        mask_labels = np.where(
            cluster_labels == c0,
            1,
            0,
        )

    mask_labels += 1

    return (
        artery_mask,
        vein_mask,
        mask_labels,
    )

def assign_corr_stack_to_av(corr_stack, cluster_labels, labeled_vessels, negative=False):
    """
    Assign clusters to artery and vein based on correlation stack.

    Parameters
    ----------
    corr_stack : ndarray, shape (n_samples, n_features)
        Correlation stack features.
    cluster_labels : ndarray, shape (n_samples,)
        Cluster labels for each sample.
    labeled_vessels : ndarray, shape (H, W)
        Labeled vessel mask.
    negative : bool
        If True, assign the cluster with lower correlation to artery.

    Returns
    -------
    artery_mask : ndarray, shape (H, W)
        Binary mask for arteries.
    vein_mask : ndarray, shape (H, W)
        Binary mask for veins.
    mask_labels : ndarray, shape (H, W)
        labels for each labeled vessel (1 for artery, 2 for vein).
    """

    # Assign artery and vein based on correlation
    if negative:
        corr_stack = -corr_stack

    c0, c1 = np.unique(cluster_labels)

    cluster0 = np.where(cluster_labels == c0)[0]
    cluster1 = np.where(cluster_labels == c1)[0]
    correlation0 = np.median(corr_stack[cluster0], axis=0)
    correlation1 = np.median(corr_stack[cluster1], axis=0)

    if np.mean(correlation0) > np.mean(correlation1):
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)
        artery_mask[np.isin(labeled_vessels, cluster0 + 1)] = True
        vein_mask[np.isin(labeled_vessels, cluster1 + 1)] = True
        mask_labels = np.where(
            cluster_labels == c0,
            1,
            2,
        )
    else:
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)
        artery_mask[np.isin(labeled_vessels, cluster1 + 1)] = True
        vein_mask[np.isin(labeled_vessels, cluster0 + 1)] = True
        mask_labels = np.where(
            cluster_labels == c0,
            2,
            1,
        )

    return artery_mask, vein_mask, mask_labels