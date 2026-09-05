"""Held-out physiological signal metrics for incomplete vessel annotations."""

import numpy as np
from scipy.signal import coherence
from scipy.stats import spearmanr


def alternating_cycle_split(n_frames, beat_period, *, first_cycle=0):
    """Return disjoint boolean masks containing alternating complete cycles."""
    if not isinstance(n_frames, (int, np.integer)) or n_frames < 1:
        raise ValueError("n_frames must be a positive integer")
    if not isinstance(beat_period, (int, np.integer)) or beat_period < 2:
        raise ValueError("beat_period must be an integer of at least two samples")
    n_cycles = n_frames // beat_period
    if n_cycles < 2:
        raise ValueError("at least two complete cardiac cycles are required")
    train = np.zeros(n_frames, dtype=bool)
    evaluation = np.zeros(n_frames, dtype=bool)
    for cycle in range(n_cycles):
        start = cycle * beat_period
        stop = start + beat_period
        destination = train if (cycle - first_cycle) % 2 == 0 else evaluation
        destination[start:stop] = True
    if not np.any(train) or not np.any(evaluation):
        raise ValueError("alternating split produced an empty partition")
    return train, evaluation


def _mask_signal(video, mask, frame_mask):
    video = np.asarray(video, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if video.ndim != 3 or mask.shape != video.shape[1:]:
        raise ValueError("video must be time-by-height-by-width and match the mask")
    if not np.any(mask):
        return None
    signal = np.mean(video[:, mask], axis=1)
    if frame_mask is not None:
        if frame_mask.shape != (len(video),):
            raise ValueError("frame_mask must contain one value per video frame")
        signal = signal[frame_mask]
    return signal


def _cycle_template(signal, beat_period):
    n_cycles = len(signal) // beat_period
    if n_cycles < 1:
        raise ValueError("signal does not contain one complete cycle")
    cycles = signal[: n_cycles * beat_period].reshape(n_cycles, beat_period)
    return np.median(cycles, axis=0)


def _standardize(signal):
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return None
    return centered / scale


def _soft_dtw_cost(x, y, gamma, window):
    n, m = len(x), len(y)
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0.0
    for i in range(1, n + 1):
        lower = max(1, i - window)
        upper = min(m, i + window)
        for j in range(lower, upper + 1):
            left = accumulated[i, j - 1]
            up = accumulated[i - 1, j]
            diagonal = accumulated[i - 1, j - 1]
            minimum = min(left, up, diagonal)
            # Scalar stable soft-min avoids allocating an array and calling
            # scipy.special.logsumexp once per dynamic-programming cell.
            soft_min = minimum - gamma * np.log(
                np.exp((minimum - left) / gamma)
                + np.exp((minimum - up) / gamma)
                + np.exp((minimum - diagonal) / gamma)
            )
            accumulated[i, j] = (x[i - 1] - y[j - 1]) ** 2 + soft_min
    return float(accumulated[n, m])


def soft_dtw_divergence(x, y, *, gamma=0.1, window=None):
    """Non-negative self-corrected Soft-DTW divergence between 1-D signals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError("x and y must be non-empty one-dimensional signals")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("signals must contain only finite values")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if window is None:
        window = max(len(x), len(y))
    if not isinstance(window, (int, np.integer)) or window < abs(len(x) - len(y)):
        raise ValueError("window must be an integer covering the length difference")
    cross = _soft_dtw_cost(x, y, gamma, window)
    self_x = _soft_dtw_cost(x, x, gamma, window)
    self_y = _soft_dtw_cost(y, y, gamma, window)
    return float(max(0.0, cross - 0.5 * (self_x + self_y)))


def _signal_pair_metrics(predicted, reference, sampling_frequency, beat_period, max_lag):
    predicted_template = _standardize(_cycle_template(predicted, beat_period))
    reference_template = _standardize(_cycle_template(reference, beat_period))
    names = (
        "pearson",
        "spearman",
        "max_correlation",
        "max_correlation_lag_samples",
        "phase_error_radians",
        "cardiac_coherence",
        "normalized_rmse",
        "soft_dtw_divergence",
    )
    if predicted_template is None or reference_template is None:
        return {name: np.nan for name in names}

    pearson = float(np.mean(predicted_template * reference_template))
    spearman = float(spearmanr(predicted_template, reference_template).statistic)
    full = np.correlate(predicted_template, reference_template, mode="full") / beat_period
    lags = np.arange(-beat_period + 1, beat_period)
    keep = np.abs(lags) <= max_lag
    best_index = np.argmax(full[keep])
    kept_lags = lags[keep]
    kept_correlations = full[keep]

    predicted_h1 = np.fft.rfft(predicted_template)[1]
    reference_h1 = np.fft.rfft(reference_template)[1]
    phase_error = abs(np.angle(predicted_h1 * np.conj(reference_h1)))

    nperseg = min(len(predicted), max(beat_period * 2, 8))
    frequencies, coherence_values = coherence(
        predicted,
        reference,
        fs=sampling_frequency,
        nperseg=nperseg,
    )
    cardiac_frequency = sampling_frequency / beat_period
    cardiac_index = int(np.argmin(np.abs(frequencies - cardiac_frequency)))
    normalized_rmse = float(
        np.sqrt(np.mean((predicted_template - reference_template) ** 2))
    )
    window = max(1, round(0.1 * beat_period))
    return {
        "pearson": pearson,
        "spearman": spearman,
        "max_correlation": float(kept_correlations[best_index]),
        "max_correlation_lag_samples": int(kept_lags[best_index]),
        "phase_error_radians": float(phase_error),
        "cardiac_coherence": float(coherence_values[cardiac_index]),
        "normalized_rmse": normalized_rmse,
        "soft_dtw_divergence": soft_dtw_divergence(
            predicted_template,
            reference_template,
            gamma=0.1,
            window=window,
        ),
    }


def evaluate_mask_signal_similarity(
    videos,
    predicted_masks,
    reference_masks,
    *,
    sampling_frequency,
    beat_period,
    frame_mask=None,
    exclude_reference_pixels=True,
    max_lag_fraction=0.25,
):
    """Compare predicted-class and manual-reference signals on selected frames."""
    if set(predicted_masks) != set(reference_masks):
        raise ValueError("predicted_masks and reference_masks must use the same classes")
    if not videos:
        raise ValueError("videos cannot be empty")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not isinstance(beat_period, (int, np.integer)) or beat_period < 2:
        raise ValueError("beat_period must be an integer of at least two samples")
    if not 0 <= max_lag_fraction <= 0.5:
        raise ValueError("max_lag_fraction must lie in [0, 0.5]")

    first_video = np.asarray(next(iter(videos.values())))
    spatial_shape = first_video.shape[1:]
    references = {name: np.asarray(mask, dtype=bool) for name, mask in reference_masks.items()}
    predictions = {name: np.asarray(mask, dtype=bool) for name, mask in predicted_masks.items()}
    if any(mask.shape != spatial_shape for mask in (*references.values(), *predictions.values())):
        raise ValueError("all masks must match the videos' spatial shape")
    reference_union = np.any(np.stack(list(references.values())), axis=0)
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=bool)

    metrics = {}
    metric_groups = {}
    max_lag = int(round(max_lag_fraction * beat_period))
    for video_name, video in videos.items():
        video = np.asarray(video, dtype=float)
        if video.shape[1:] != spatial_shape or video.shape[0] != first_video.shape[0]:
            raise ValueError("all videos must have matching temporal and spatial shapes")
        for class_name in predictions:
            predicted_mask = predictions[class_name]
            if exclude_reference_pixels:
                predicted_mask = predicted_mask & ~reference_union
            reference_mask = references[class_name]
            predicted_signal = _mask_signal(video, predicted_mask, frame_mask)
            reference_signal = _mask_signal(video, reference_mask, frame_mask)
            prefix = f"signal_{video_name}_{class_name}_"
            metrics[f"{prefix}predicted_pixel_count"] = int(np.count_nonzero(predicted_mask))
            metrics[f"{prefix}reference_pixel_count"] = int(np.count_nonzero(reference_mask))
            if predicted_signal is None or reference_signal is None:
                pair_metrics = {
                    name: np.nan
                    for name in (
                        "pearson", "spearman", "max_correlation",
                        "max_correlation_lag_samples", "phase_error_radians",
                        "cardiac_coherence", "normalized_rmse", "soft_dtw_divergence",
                    )
                }
            else:
                pair_metrics = _signal_pair_metrics(
                    predicted_signal,
                    reference_signal,
                    sampling_frequency,
                    beat_period,
                    max_lag,
                )
            for name, value in pair_metrics.items():
                metrics[f"{prefix}{name}"] = value
                if name != "max_correlation_lag_samples":
                    metric_groups.setdefault((video_name, name), []).append(value)

    for (video_name, name), values in metric_groups.items():
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        metrics[f"signal_{video_name}_{name}_macro"] = (
            float(np.mean(values[finite])) if np.any(finite) else np.nan
        )
    return metrics
