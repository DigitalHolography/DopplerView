"""Detect arterial pulse landmarks in a one-dimensional intensity signal.

Pipeline: flag artifacts -> smooth -> estimate period -> find candidates ->
select a plausible sequence -> refine landmarks -> report quality warnings.

Two landmarks are returned for each beat:
* peaks: intensity maxima, used to reset Noise2Time's phase counter.
* upstrokes: preceding maximum positive slopes, as used by DopplerView.

This is an offline detector: symmetric filters use past AND future samples.
The input is never modified; missing beats are not filled with regularly spaced
guesses. Thresholds are the heuristics evaluated in ARTERIAL_PEAKS.md. Quality
warnings are review aids, not calibrated confidence probabilities.
"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d, label, median_filter
from scipy.signal import correlate, detrend, find_peaks


def _mad(values):
    """Robust spread: median absolute deviation, scaled like Gaussian noise SD."""
    center = np.median(values)
    return 1.4826 * np.median(np.abs(values - center))


def _keep_short_runs(mask, max_length):
    """Keep only contiguous True runs no longer than max_length samples."""
    short_runs = mask.copy()
    groups, group_count = label(mask)
    for group_id in range(1, group_count + 1):
        indices = np.flatnonzero(groups == group_id)
        if len(indices) > max_length:
            short_runs[indices] = False
    return short_runs


def _find_artifacts(raw, fps, signal_range):
    """Flag brief dropouts/spikes without mistaking normal diastole for noise."""
    # Differencing suppresses slow variation. Independent sample noise has
    # sqrt(2) times its original SD after differencing. This remains a heuristic
    # scale estimate because real arterial samples need not be independent.
    noise = _mad(np.diff(raw)) / np.sqrt(2)
    window = max(3, int(round(0.18 * fps)))
    if window % 2 == 0:
        window += 1  # An odd 180 ms window is centered on the current sample.
    local_median = median_filter(raw, size=window, mode="nearest")

    # Require both noise-relative and amplitude-relative thresholds. Positive
    # spikes use a stricter threshold: a normal arterial peak is also positive.
    negative_threshold = max(6 * noise, 0.3 * signal_range)
    positive_threshold = max(8 * noise, 0.7 * signal_range)
    dropouts = raw < local_median - negative_threshold
    spikes = raw > local_median + positive_threshold
    spikes = _keep_short_runs(spikes, max_length=0.06 * fps)

    # Long disturbances cannot safely be reconstructed by this local heuristic.
    artifacts = _keep_short_runs(dropouts | spikes, max_length=0.18 * fps)
    if artifacts.any():
        margin = max(1, int(round(0.01 * fps)))  # Include shoulders: about 10 ms.
        artifacts = binary_dilation(artifacts, iterations=margin)
    return artifacts, noise


def _repair_for_detection(raw, artifacts, repair):
    """Interpolate flagged samples in a copy, never in the source signal/video."""
    working = raw.copy()
    if repair and artifacts.any():
        valid_indices = np.flatnonzero(~artifacts)
        if len(valid_indices) < 2:
            raise ValueError("Too little signal outside artifacts")
        # At recording boundaries, np.interp uses the nearest valid endpoint.
        # These interpolated values are not recovered physical measurements.
        working[artifacts] = np.interp(
            np.flatnonzero(artifacts), valid_indices, raw[valid_indices]
        )
    return working


def _estimate_period(pulse, fps, min_hz, max_hz):
    """Find a strong recurrence, preferring one beat over a two-beat period."""
    centered = detrend(pulse)
    frame_count = len(pulse)
    shortest_period = max(2, int(np.ceil(fps / max_hz)))
    longest_period = min(int(np.floor(fps / min_hz)), frame_count // 2)
    if longest_period <= shortest_period:
        raise ValueError("Insufficient duration for period estimation")

    # At lag k, compare centered[k:] with centered[:-k]. Normalize by the
    # energies of those overlapping segments, not a single global energy.
    # Cumulative sums calculate all segment energies without a loop over lags.
    autocorrelation = correlate(centered, centered, mode="full", method="fft")
    autocorrelation = autocorrelation[frame_count - 1:]
    energy = np.r_[0.0, np.cumsum(centered**2)]
    lags = np.arange(frame_count)
    right_energy = energy[-1] - energy[lags]
    left_energy = energy[frame_count - lags]
    denominator = np.sqrt(right_energy * left_energy)
    autocorrelation = np.divide(
        autocorrelation, denominator,
        out=np.zeros_like(autocorrelation), where=denominator > 0,
    )

    search = autocorrelation[shortest_period:longest_period + 1]
    candidates, _ = find_peaks(search)
    candidates = candidates + shortest_period
    if len(candidates) == 0:
        # Retain a fallback estimate; weak support is flagged later.
        candidates = np.array([shortest_period + np.argmax(search)])

    best_correlation = float(autocorrelation[candidates].max())
    strong_enough = max(0.25, 0.7 * best_correlation)
    strong_candidates = candidates[autocorrelation[candidates] >= strong_enough]
    if len(strong_candidates):
        period = int(strong_candidates[0])  # Increasing lag order.
    else:
        period = int(candidates[np.argmax(autocorrelation[candidates])])
    return period, autocorrelation, candidates


def _find_candidates(pulse, artifacts, noise, fps, period):
    """Generate permissive maxima and give each a bounded evidence score."""
    amplitude = float(np.percentile(pulse, 95) - np.percentile(pulse, 5))
    # Prominence measures how much a peak rises above its surrounding valleys.
    # Keep weak candidates here: sequence selection resolves competing humps.
    noise_floor = 3 * noise / np.sqrt(max(1, 0.04 * fps))
    minimum_prominence = max(0.12 * amplitude, noise_floor)
    candidates, properties = find_peaks(
        pulse,
        distance=max(1, int(0.2 * period)),
        prominence=minimum_prominence,
    )
    # Bound support to [1,2] before penalizing artifacts: one huge motion spike
    # should not outweigh several plausible beats merely because it is taller.
    strengths = 1 + np.minimum(properties["prominences"] / amplitude, 1.0)
    for index, peak in enumerate(candidates):
        start = max(0, peak - int(0.1 * period))
        if artifacts[start:peak + 1].any():
            strengths[index] -= 0.6
    return candidates, strengths


def _select_sequence(candidates, strengths, period):
    """Return indices into candidates for the highest-scoring peak sequence.

    Dynamic programming asks, for each candidate: "What is the best sequence
    ending here?" Start a new sequence or extend an earlier one. This avoids
    a greedy decision where one tall artifact excludes a weaker real peak.
    """
    if len(candidates) == 0:
        return np.array([], int)
    scores = strengths.copy()  # Initial option: start here.
    previous = np.full(len(candidates), -1, int)
    beat_counts = np.arange(1, 4)

    for current in range(len(candidates)):
        for earlier in range(current - 1, -1, -1):
            gap = (candidates[current] - candidates[earlier]) / period
            if gap > 3.5:
                break  # All remaining earlier candidates are farther away.
            if gap < 0.45:
                continue

            # Consider gaps spanning 1, 2 or 3 beats. Missing beats cost 2 points
            # each but never cause a peak to be inserted. The logarithmic term
            # penalizes proportional departures from the expected interval.
            interval_cost = 8 * np.log(gap / beat_counts)**2
            missing_beat_cost = 2 * (beat_counts - 1)
            penalty = np.min(interval_cost + missing_beat_cost)
            extended_score = scores[earlier] + strengths[current] - penalty
            if extended_score > scores[current]:
                scores[current] = extended_score
                previous[current] = earlier

    # Follow predecessors backward, then restore chronological order.
    selected = []
    current = int(np.argmax(scores))
    while current >= 0:
        selected.append(current)
        current = previous[current]
    return np.array(selected[::-1], int)


def _refine_landmarks(candidates, smoothed, period):
    """Locate maxima on intensity, then find the preceding steepest upstroke."""
    refined = []
    radius = int(0.08 * period)
    for candidate in candidates:
        start = max(0, candidate - radius)
        stop = min(len(smoothed), candidate + radius + 1)
        refined.append(start + int(np.argmax(smoothed[start:stop])))
    peaks = np.unique(refined).astype(int)

    derivative = np.gradient(smoothed)
    upstrokes = []
    for peak in peaks:
        start = max(0, peak - int(0.4 * period))
        upstroke = start + int(np.argmax(derivative[start:peak + 1]))
        upstrokes.append(upstroke)
    return peaks, np.array(upstrokes, int)


def _quality_flags(peaks, artifacts, period, periodicity):
    """Describe uncertainty without deleting peaks or changing their timing."""
    intervals = np.diff(peaks)
    bad_intervals = (intervals < 0.65 * period) | (intervals > 1.45 * period)
    near_artifact = []
    radius = int(0.15 * period)
    for peak in peaks:
        start = max(0, peak - radius)
        stop = min(len(artifacts), peak + radius + 1)
        near_artifact.append(artifacts[start:stop].any())
    near_artifact = np.array(near_artifact)

    warnings = []
    if periodicity < 0.45:
        warnings.append("Weak periodicity: inspect waveform and candidates")
    if len(peaks) < 3:
        warnings.append("Fewer than three detected peaks: period estimate has limited support")
    if bad_intervals.any():
        warnings.append("Irregular intervals or missed/extra peaks: inspect flagged intervals")
    if near_artifact.any():
        warnings.append("Some peaks are near detected artifacts; timing is uncertain")
    if artifacts.mean() > 0.05:
        warnings.append("More than 5% of frames flagged as artifacts")
    return intervals, bad_intervals, near_artifact, warnings


def detect_arterial_peaks(signal, fps, min_hz=0.5, max_hz=2.5, repair=True, sequence=True):
    """Detect arterial landmarks and return the arrays needed to inspect them.

    Parameters
    ----------
    signal : one-dimensional array
        Mean retinal artery intensity, one value per video frame.
    fps : float
        Effective video rate (camera rate / batch stride).
    min_hz, max_hz : float
        Frequency bounds for period estimation, not a band-pass filter.
    repair : bool
        Interpolate artifacts in the detection copy. False is for experiments;
        artifacts are still flagged and candidates penalized.
    sequence : bool
        Apply sequence selection. False returns all prominence candidates after
        refinement, exposing what sequence selection removes.

    Returns
    -------
    dict
        peaks and upstrokes are zero-based original-frame indices.
        period_frames is the recurrence estimate; period_hz is fps/period.
        candidates and rejected_candidates refer to positions BEFORE refinement.
        smoothed, detection_signal and repaired are diagnostic traces, not
        corrected measurements. artifact_mask flags samples; peak_near_artifact
        flags peaks; flagged_intervals flags the intervals BETWEEN peaks.
        warnings describes cases requiring inspection. acf contains normalized
        autocorrelation; period_candidates contains candidate lags in frames.
        periodicity is acf at the chosen period; noise_scale is the robust noise
        estimate. Neither is a calibrated confidence probability.

    Raises ValueError for nonfinite, very short, flat or otherwise invalid input.
    """
    raw = np.asarray(signal, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("Arterial signal must be a finite one-dimensional array")
    if not np.isfinite(fps) or not 0 < min_hz < max_hz < fps / 2:
        raise ValueError("Need 0 < min_hz < max_hz < fps/2")
    if len(raw) < max(16, int(2 * fps / max_hz)):
        raise ValueError("Recording too short for two beats in the search band")
    signal_range = float(np.percentile(raw, 95) - np.percentile(raw, 5))
    flat_tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(raw))))
    if signal_range <= flat_tolerance:
        raise ValueError("Arterial signal has no usable variation")

    # 1. Flag artifacts and make a separate detection copy.
    artifacts, noise = _find_artifacts(raw, fps, signal_range)
    working = _repair_for_detection(raw, artifacts, repair)

    # 2. Suppress frame noise (20 ms sigma), then remove slow drift (600 ms sigma).
    # Keep smoothed intensity for final localization; only pulse loses its baseline.
    smoothed = gaussian_filter1d(working, max(0.5, 0.02 * fps), mode="reflect")
    baseline = gaussian_filter1d(smoothed, max(1.0, 0.6 * fps), mode="reflect")
    pulse = smoothed - baseline

    # 3. Estimate recurrence, then find possible intensity maxima.
    period, autocorrelation, period_candidates = _estimate_period(pulse, fps, min_hz, max_hz)
    candidates, strengths = _find_candidates(pulse, artifacts, noise, fps, period)

    # 4. Resolve competing candidates using evidence from the whole sequence.
    if sequence:
        selected = _select_sequence(candidates, strengths, period)
    else:
        selected = np.arange(len(candidates))

    # 5. Localize both landmarks and report uncertainty without hiding it.
    peaks, upstrokes = _refine_landmarks(candidates[selected], smoothed, period)
    periodicity = float(autocorrelation[period])
    intervals, bad_intervals, near_artifact, warnings = _quality_flags(
        peaks, artifacts, period, periodicity
    )
    return {
        "peaks": peaks,
        "upstrokes": upstrokes,
        "smoothed": smoothed,
        "detection_signal": pulse,
        "repaired": working,
        "artifact_mask": artifacts,
        "period_frames": period,
        "period_hz": fps / period,
        "acf": autocorrelation,
        "period_candidates": period_candidates,
        "periodicity": periodicity,
        "intervals": intervals,
        "flagged_intervals": bad_intervals,
        "peak_near_artifact": near_artifact,
        "candidates": candidates,
        "rejected_candidates": np.delete(candidates, selected),
        "warnings": warnings,
        "noise_scale": float(noise),
    }
