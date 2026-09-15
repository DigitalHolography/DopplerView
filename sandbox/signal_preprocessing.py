"""Conservative temporal artifact cleaning for cardiac clustering experiments.

The functions in this module deliberately keep detection separate from use:

* short, reversible impulses are detected on a global reference signal;
* interpolation is only applied to small temporal signals, never silently to the
  source videos;
* complete, phase-aligned cycles are scored and accepted/rejected;
* clustering can use every accepted cycle instead of a 50/50 temporal hold-out.

This is research preprocessing, so :class:`SignalCleaningResult` exposes every
decision needed to audit a measure rather than returning only a cleaned signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


_EPSILON = np.finfo(float).eps


@dataclass(frozen=True)
class SignalCleaningResult:
    """Diagnostics and masks produced by :func:`preprocess_cardiac_signal`."""

    raw_signal: np.ndarray
    cleaned_signal: np.ndarray
    initial_beat_period: int
    beat_period: int
    phase_offset: int
    anchor_polarity: str
    anchors: np.ndarray
    frame_artifact_mask: np.ndarray
    cycle_bounds: np.ndarray
    cycle_valid: np.ndarray
    cycle_correlations: np.ndarray
    cycle_amplitudes: np.ndarray
    cycle_amplitude_robust_z: np.ndarray
    cycle_artifact_fractions: np.ndarray
    cycle_quality_scores: np.ndarray
    valid_cycle_frame_mask: np.ndarray
    fit_frame_mask: np.ndarray
    fallback_used: bool

    @property
    def n_cycles(self):
        return int(len(self.cycle_bounds))

    @property
    def n_valid_cycles(self):
        return int(np.count_nonzero(self.cycle_valid))

    def summary(self):
        """Return a JSON/CSV-friendly summary of the cleaning decisions."""
        return {
            "n_frames": int(len(self.raw_signal)),
            "initial_beat_period": int(self.initial_beat_period),
            "beat_period": int(self.beat_period),
            "phase_offset": int(self.phase_offset),
            "anchor_polarity": self.anchor_polarity,
            "anchor_count": int(len(self.anchors)),
            "artifact_frame_count": int(np.count_nonzero(self.frame_artifact_mask)),
            "artifact_frame_fraction": float(np.mean(self.frame_artifact_mask)),
            "cycle_count": self.n_cycles,
            "valid_cycle_count": self.n_valid_cycles,
            "valid_cycle_fraction": float(np.mean(self.cycle_valid)),
            "fit_frame_count": int(np.count_nonzero(self.fit_frame_mask)),
            "fallback_used": bool(self.fallback_used),
        }

    def cycle_rows(self):
        """Return one plain dictionary per complete cycle for progressive CSV output."""
        rows = []
        for index, (start, stop) in enumerate(self.cycle_bounds):
            rows.append(
                {
                    "cycle": index,
                    "start_frame": int(start),
                    "stop_frame_exclusive": int(stop),
                    "valid": bool(self.cycle_valid[index]),
                    "shape_correlation": float(self.cycle_correlations[index]),
                    "amplitude": float(self.cycle_amplitudes[index]),
                    "amplitude_robust_z": float(self.cycle_amplitude_robust_z[index]),
                    "artifact_fraction": float(self.cycle_artifact_fractions[index]),
                    "quality_score": float(self.cycle_quality_scores[index]),
                }
            )
        return rows


def _as_finite_signal(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 8:
        raise ValueError("signal must be one-dimensional with at least eight samples")
    finite = np.isfinite(signal)
    if np.count_nonzero(finite) < 4:
        raise ValueError("signal must contain at least four finite samples")
    repaired = signal.copy()
    if not finite.all():
        indices = np.arange(len(signal))
        repaired[~finite] = np.interp(indices[~finite], indices[finite], signal[finite])
    return signal, repaired, ~finite


def _robust_scale(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    centered = values[finite] - np.median(values[finite])
    scale = 1.4826 * np.median(np.abs(centered))
    if not np.isfinite(scale) or scale <= _EPSILON:
        scale = np.std(centered)
    return float(scale) if np.isfinite(scale) else 0.0


def _robust_z(values):
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    scale = _robust_scale(values)
    if scale <= _EPSILON:
        return np.zeros_like(values)
    return (values - center) / scale


def detect_short_impulse_artifacts(
    signal,
    sampling_frequency,
    *,
    derivative_z_threshold=6.0,
    max_duration_seconds=0.12,
    return_tolerance=0.35,
    padding_frames=1,
):
    """Detect short spike/drop-and-recovery artifacts without flagging long steps.

    A candidate starts at an unusually large temporal derivative and must be
    followed shortly by a derivative in the opposite direction.  The signal
    after the pair must return near its pre-event level.  This last condition is
    what protects a genuine steep cardiac edge followed by a slow recovery.
    """
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(derivative_z_threshold) or derivative_z_threshold <= 0:
        raise ValueError("derivative_z_threshold must be finite and positive")
    if not np.isfinite(max_duration_seconds) or max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be finite and positive")
    if not 0 <= return_tolerance <= 1:
        raise ValueError("return_tolerance must lie in [0, 1]")
    if not isinstance(padding_frames, (int, np.integer)) or padding_frames < 0:
        raise ValueError("padding_frames must be a non-negative integer")

    raw, work, nonfinite = _as_finite_signal(signal)
    differences = np.diff(work)
    difference_center = np.median(differences)
    scale = _robust_scale(differences)
    mask = nonfinite.copy()
    if scale <= _EPSILON:
        return mask

    unusual = np.abs(differences - difference_center) >= derivative_z_threshold * scale
    event_indices = np.flatnonzero(unusual)
    maximum_gap = max(1, int(round(max_duration_seconds * sampling_frequency)))
    consumed = set()
    for position, entry in enumerate(event_indices):
        if int(entry) in consumed:
            continue
        entry_step = differences[entry] - difference_center
        for exit_index in event_indices[position + 1 :]:
            if exit_index - entry > maximum_gap:
                break
            exit_step = differences[exit_index] - difference_center
            if entry_step * exit_step >= 0:
                continue
            excursion = max(abs(entry_step), abs(exit_step))
            return_error = abs(work[exit_index + 1] - work[entry])
            if return_error <= return_tolerance * excursion + 2 * scale:
                first = max(0, entry + 1 - padding_frames)
                last = min(len(work), exit_index + 1 + padding_frames)
                mask[first:last] = True
                consumed.add(int(entry))
                consumed.add(int(exit_index))
                break
    return mask


def interpolate_artifact_samples(signals, artifact_mask, *, axis=-1):
    """Linearly interpolate marked time samples in one or many signals."""
    signals = np.asarray(signals)
    artifact_mask = np.asarray(artifact_mask, dtype=bool)
    if artifact_mask.ndim != 1:
        raise ValueError("artifact_mask must be one-dimensional")
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    axis = int(axis)
    if axis < 0:
        axis += signals.ndim
    if not 0 <= axis < signals.ndim:
        raise np.AxisError(axis, signals.ndim)
    if signals.shape[axis] != artifact_mask.size:
        raise ValueError("artifact_mask must contain one value per temporal sample")
    if artifact_mask.all():
        raise ValueError("at least one non-artifact sample is required")
    if not artifact_mask.any():
        return signals.astype(float, copy=True)

    moved = np.moveaxis(signals.astype(float, copy=True), axis, -1)
    flattened = moved.reshape(-1, moved.shape[-1])
    indices = np.arange(artifact_mask.size)
    valid = ~artifact_mask
    for row in flattened:
        row[artifact_mask] = np.interp(
            indices[artifact_mask], indices[valid], row[valid]
        )
    return np.moveaxis(flattened.reshape(moved.shape), -1, axis)


def _odd_window(target, maximum):
    target = max(3, min(int(target), int(maximum)))
    if target % 2 == 0:
        target -= 1
    return max(3, target)


def _candidate_anchors(signal, beat_period, polarity):
    window = _odd_window(round(0.05 * beat_period), len(signal) - 1)
    smooth = savgol_filter(signal, window_length=window, polyorder=min(2, window - 1))
    gradient = np.gradient(smooth)
    oriented = gradient if polarity == "positive" else -gradient
    scale = _robust_scale(oriented)
    prominence = max(2.5 * scale, _EPSILON * max(1.0, np.max(np.abs(oriented))))
    peaks, properties = find_peaks(
        oriented,
        distance=max(2, int(round(0.6 * beat_period))),
        prominence=prominence,
    )
    strengths = properties.get("prominences", np.empty(0, dtype=float))
    return peaks.astype(int), strengths


def _anchor_score(anchors, strengths, beat_period):
    if len(anchors) < 2:
        return -np.inf
    intervals = np.diff(anchors)
    regularity = np.median(np.abs(intervals - beat_period)) / beat_period
    coverage = min(len(intervals), 6) / 6
    strength_score = np.log1p(np.median(strengths) / max(_robust_scale(strengths), _EPSILON))
    return float(coverage + 0.15 * strength_score - regularity)


def _phase_aligned_bounds(n_frames, beat_period, anchors):
    if len(anchors):
        phases = np.mod(anchors, beat_period)
        angles = 2 * np.pi * phases / beat_period
        phase = int(round(np.angle(np.mean(np.exp(1j * angles))) * beat_period / (2 * np.pi)))
        phase %= beat_period
    else:
        phase = 0
    starts = np.arange(phase, n_frames - beat_period + 1, beat_period, dtype=int)
    if phase:
        earlier = np.arange(phase - beat_period, -1, -beat_period, dtype=int)
        starts = np.concatenate((earlier[::-1], starts))
    starts = starts[(starts >= 0) & (starts + beat_period <= n_frames)]
    return phase, np.column_stack((starts, starts + beat_period))


def _autocorrelation_period(signal, initial_period):
    """Return a nearby dominant autocorrelation peak and its coefficient."""
    indices = np.arange(len(signal), dtype=float)
    slope, intercept = np.polyfit(indices, signal, 1)
    centered = signal - (slope * indices + intercept)
    scale = np.linalg.norm(centered)
    if scale <= _EPSILON:
        return int(initial_period), 0.0
    full = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    overlap = np.arange(len(centered), 0, -1)
    autocorrelation = full / overlap
    autocorrelation /= autocorrelation[0]
    lower = max(4, int(round(0.7 * initial_period)))
    upper = min(len(signal) - 2, int(round(1.4 * initial_period)))
    if upper <= lower:
        return int(initial_period), 0.0
    local = autocorrelation[lower : upper + 1]
    peaks, properties = find_peaks(local, prominence=0.05)
    if not len(peaks):
        return int(initial_period), 0.0
    prominences = properties["prominences"]
    # Prominence prevents a slowly decaying autocorrelation shoulder from
    # winning simply because it occurs at the shortest admissible lag.
    best = int(np.argmax(prominences))
    period = int(peaks[best] + lower)
    return period, float(autocorrelation[period])


def _best_phase_bounds(signal, beat_period):
    """Choose a fixed-period grid that maximizes agreement between cycles."""
    best = None
    for phase in range(beat_period):
        starts = np.arange(phase, len(signal) - beat_period + 1, beat_period, dtype=int)
        if len(starts) < 2:
            continue
        bounds = np.column_stack((starts, starts + beat_period))
        cycles = np.asarray([signal[start:stop] for start, stop in bounds])
        standardized, nonconstant = _standardize_cycles(cycles)
        standardized = standardized[nonconstant]
        if len(standardized) < 2:
            continue
        template = np.median(standardized, axis=0)
        template -= np.mean(template)
        template_norm = np.linalg.norm(template)
        if template_norm <= _EPSILON:
            continue
        correlations = standardized @ template / (
            np.linalg.norm(standardized, axis=1) * template_norm
        )
        # Cycle count is the first criterion; within that maximum, select the
        # most self-consistent phase rather than the sharpest individual edge.
        candidate = (len(bounds), float(np.median(correlations)), -phase, bounds)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return 0, np.empty((0, 2), dtype=int)
    return int(-best[2]), best[3]


def _standardize_cycles(cycles):
    centered = cycles - np.mean(cycles, axis=1, keepdims=True)
    scales = np.std(centered, axis=1, keepdims=True)
    standardized = np.zeros_like(centered)
    valid = scales[:, 0] > _EPSILON
    standardized[valid] = centered[valid] / scales[valid]
    return standardized, valid


def preprocess_cardiac_signal(
    signal,
    sampling_frequency,
    beat_period,
    *,
    derivative_z_threshold=6.0,
    max_artifact_duration_seconds=0.12,
    minimum_cycle_correlation=0.5,
    cycle_outlier_z=3.5,
    maximum_cycle_artifact_fraction=0.15,
    minimum_valid_cycles=2,
    refine_beat_period=True,
):
    """Clean a reference and select all robust, complete cardiac cycles.

    The returned ``valid_cycle_frame_mask`` includes complete cycles and is
    appropriate for cycle templates. ``fit_frame_mask`` additionally excludes
    the individual artifact frames and is appropriate for correlations computed
    directly from raw videos.
    """
    if not isinstance(beat_period, (int, np.integer)) or beat_period < 4:
        raise ValueError("beat_period must be an integer of at least four samples")
    if not isinstance(minimum_valid_cycles, (int, np.integer)) or minimum_valid_cycles < 1:
        raise ValueError("minimum_valid_cycles must be a positive integer")
    if not -1 <= minimum_cycle_correlation <= 1:
        raise ValueError("minimum_cycle_correlation must lie in [-1, 1]")
    if not np.isfinite(cycle_outlier_z) or cycle_outlier_z <= 0:
        raise ValueError("cycle_outlier_z must be finite and positive")
    if not 0 <= maximum_cycle_artifact_fraction <= 1:
        raise ValueError("maximum_cycle_artifact_fraction must lie in [0, 1]")

    initial_beat_period = int(beat_period)
    raw, initially_repaired, nonfinite = _as_finite_signal(signal)
    artifacts = detect_short_impulse_artifacts(
        initially_repaired,
        sampling_frequency,
        derivative_z_threshold=derivative_z_threshold,
        max_duration_seconds=max_artifact_duration_seconds,
    ) | nonfinite
    cleaned = interpolate_artifact_samples(initially_repaired, artifacts)

    candidates = {}
    for polarity in ("positive", "negative"):
        anchors, strengths = _candidate_anchors(cleaned, beat_period, polarity)
        candidates[polarity] = (anchors, strengths, _anchor_score(anchors, strengths, beat_period))
    polarity = max(candidates, key=lambda name: candidates[name][2])
    anchors = candidates[polarity][0]
    if not np.isfinite(candidates[polarity][2]):
        polarity = "fixed-grid"
        anchors = np.empty(0, dtype=int)

    # FFT-based period estimates can be biased by short records, slow trends,
    # and harmonics. A clear nearby autocorrelation peak is a more direct test
    # of waveform repetition; transition intervals provide a second fallback.
    if refine_beat_period:
        autocorrelation_period, autocorrelation_value = _autocorrelation_period(
            cleaned, initial_beat_period
        )
        refined_period = initial_beat_period
        refinement_supported = autocorrelation_value >= 0.25
        if refinement_supported:
            refined_period = autocorrelation_period
        elif len(anchors) >= 4:
            intervals = np.diff(anchors)
            transition_period = int(round(np.median(intervals)))
            interval_dispersion = _robust_scale(intervals) / max(transition_period, 1)
            if interval_dispersion <= 0.15:
                refined_period = transition_period
                refinement_supported = True
        refinement_ratio = refined_period / initial_beat_period
        refined_frequency = sampling_frequency / refined_period
        if (
            refinement_supported
            and 0.7 <= refinement_ratio <= 1.4
            and 0.5 <= refined_frequency <= 3.0
        ):
            beat_period = int(refined_period)
            candidates = {}
            for candidate_polarity in ("positive", "negative"):
                candidate_anchors, strengths = _candidate_anchors(
                    cleaned, beat_period, candidate_polarity
                )
                candidates[candidate_polarity] = (
                    candidate_anchors,
                    strengths,
                    _anchor_score(candidate_anchors, strengths, beat_period),
                )
            polarity = max(candidates, key=lambda name: candidates[name][2])
            anchors = candidates[polarity][0]

    phase, bounds = _best_phase_bounds(cleaned, int(beat_period))
    if len(bounds) < minimum_valid_cycles:
        phase, bounds = _phase_aligned_bounds(len(raw), int(beat_period), np.empty(0, dtype=int))
    if len(bounds) < minimum_valid_cycles:
        raise ValueError(
            f"only {len(bounds)} complete cycle(s); at least {minimum_valid_cycles} are required"
        )

    cycles = np.asarray([cleaned[start:stop] for start, stop in bounds])
    standardized, nonconstant = _standardize_cycles(cycles)
    template = np.median(standardized[nonconstant], axis=0) if np.any(nonconstant) else np.zeros(beat_period)
    template_centered = template - np.mean(template)
    template_norm = np.linalg.norm(template_centered)
    correlations = np.zeros(len(cycles), dtype=float)
    if template_norm > _EPSILON:
        correlations[nonconstant] = (
            standardized[nonconstant] @ template_centered
            / (np.linalg.norm(standardized[nonconstant], axis=1) * template_norm)
        )
    amplitudes = np.percentile(cycles, 95, axis=1) - np.percentile(cycles, 5, axis=1)
    amplitude_z = _robust_z(np.log(np.maximum(amplitudes, _EPSILON)))
    artifact_fractions = np.asarray(
        [np.mean(artifacts[start:stop]) for start, stop in bounds]
    )
    correlation_floor = max(
        minimum_cycle_correlation,
        float(np.median(correlations) - cycle_outlier_z * _robust_scale(correlations)),
    )
    valid = (
        nonconstant
        & (correlations >= correlation_floor)
        & (np.abs(amplitude_z) <= cycle_outlier_z)
        & (artifact_fractions <= maximum_cycle_artifact_fraction)
    )
    quality = (
        correlations
        - 0.08 * np.abs(amplitude_z)
        - 0.5 * artifact_fractions
    )

    fallback_used = False
    if np.count_nonzero(valid) < minimum_valid_cycles:
        fallback_used = True
        eligible = nonconstant & (artifact_fractions <= maximum_cycle_artifact_fraction)
        ranked = np.flatnonzero(eligible)
        if len(ranked) < minimum_valid_cycles:
            ranked = np.flatnonzero(nonconstant)
        if len(ranked) < minimum_valid_cycles:
            raise ValueError("too few non-constant cycles remain after preprocessing")
        keep = ranked[np.argsort(quality[ranked])[-minimum_valid_cycles:]]
        valid = np.zeros(len(cycles), dtype=bool)
        valid[keep] = True

    valid_cycle_frames = np.zeros(len(raw), dtype=bool)
    for is_valid, (start, stop) in zip(valid, bounds):
        if is_valid:
            valid_cycle_frames[start:stop] = True
    fit_frames = valid_cycle_frames & ~artifacts
    if np.count_nonzero(fit_frames) < 4:
        raise ValueError("fewer than four artifact-free samples remain for fitting")

    return SignalCleaningResult(
        raw_signal=raw.copy(),
        cleaned_signal=cleaned,
        initial_beat_period=initial_beat_period,
        beat_period=int(beat_period),
        phase_offset=int(phase),
        anchor_polarity=polarity,
        anchors=anchors,
        frame_artifact_mask=artifacts,
        cycle_bounds=bounds,
        cycle_valid=valid,
        cycle_correlations=correlations,
        cycle_amplitudes=amplitudes,
        cycle_amplitude_robust_z=amplitude_z,
        cycle_artifact_fractions=artifact_fractions,
        cycle_quality_scores=quality,
        valid_cycle_frame_mask=valid_cycle_frames,
        fit_frame_mask=fit_frames,
        fallback_used=fallback_used,
    )
