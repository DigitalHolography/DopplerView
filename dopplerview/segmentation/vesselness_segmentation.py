"""Configurable Frangi-based binary vessel segmentation for experiments."""

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from skimage.filters import apply_hysteresis_threshold, frangi, threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk, opening


ThresholdMethod = Literal["absolute", "quantile", "otsu"]
RidgePolarity = Literal["bright", "dark", "both"]


@dataclass(frozen=True)
class VesselnessSegmentationConfig:
    """All choices affecting a Frangi segmentation candidate.

    ``threshold`` is a vesselness value for ``absolute`` and a quantile in
    [0, 1] for ``quantile``. It is ignored for ``otsu``. When hysteresis is
    enabled, ``low_threshold`` follows the same convention. For ``otsu``, the
    low threshold is ``otsu_threshold * hysteresis_low_ratio``.
    """

    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    ridge_polarity: RidgePolarity = "bright"
    threshold_method: ThresholdMethod = "quantile"
    threshold: float = 0.95
    use_hysteresis: bool = False
    low_threshold: float | None = None
    hysteresis_low_ratio: float = 0.5
    intensity_percentiles: tuple[float, float] = (1.0, 99.0)
    closing_radius: int = 0
    opening_radius: int = 0
    min_object_size: int = 0
    min_hole_size: int = 0


@dataclass(frozen=True)
class VesselnessSegmentationResult:
    vesselness: np.ndarray
    raw_mask: np.ndarray
    mask: np.ndarray
    high_threshold: float
    low_threshold: float | None
    config: VesselnessSegmentationConfig


def _validate_inputs(image, roi_mask, config):
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be two-dimensional, got shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or infinite values")

    if not config.sigmas or any(sigma <= 0 for sigma in config.sigmas):
        raise ValueError("sigmas must contain only positive values")
    if config.ridge_polarity not in {"bright", "dark", "both"}:
        raise ValueError(f"unknown ridge polarity: {config.ridge_polarity}")
    if config.threshold_method not in {"absolute", "quantile", "otsu"}:
        raise ValueError(f"unknown threshold method: {config.threshold_method}")

    low_percentile, high_percentile = config.intensity_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("intensity_percentiles must be increasing values in [0, 100]")
    if config.threshold_method == "quantile":
        values = [config.threshold]
        if config.use_hysteresis:
            if config.low_threshold is None:
                raise ValueError("quantile hysteresis requires low_threshold")
            values.append(config.low_threshold)
        if any(value is None or not 0 <= value <= 1 for value in values):
            raise ValueError("quantile thresholds must be in [0, 1]")
    elif config.threshold_method == "absolute":
        if config.threshold < 0:
            raise ValueError("absolute threshold cannot be negative")
        if config.use_hysteresis:
            if config.low_threshold is None:
                raise ValueError("absolute hysteresis requires low_threshold")
            if config.low_threshold < 0:
                raise ValueError("absolute low_threshold cannot be negative")
    if not 0 < config.hysteresis_low_ratio < 1:
        raise ValueError("hysteresis_low_ratio must be strictly between 0 and 1")
    for name in ("closing_radius", "opening_radius", "min_object_size", "min_hole_size"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} cannot be negative")

    if roi_mask is None:
        roi_mask = np.ones(image.shape, dtype=bool)
    else:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        if roi_mask.shape != image.shape:
            raise ValueError("roi_mask must have the same shape as image")
        if not np.any(roi_mask):
            raise ValueError("roi_mask cannot be empty")
    return image, roi_mask


def _normalize_image(image, roi_mask, percentiles):
    low, high = np.percentile(image[roi_mask], percentiles)
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _frangi_response(image, config):
    kwargs = {"sigmas": config.sigmas}
    if config.ridge_polarity == "bright":
        return frangi(image, black_ridges=False, **kwargs)
    if config.ridge_polarity == "dark":
        return frangi(image, black_ridges=True, **kwargs)
    bright = frangi(image, black_ridges=False, **kwargs)
    dark = frangi(image, black_ridges=True, **kwargs)
    return np.maximum(bright, dark)


def _resolve_thresholds(vesselness, roi_mask, config):
    values = vesselness[roi_mask]
    if config.threshold_method == "absolute":
        high = float(config.threshold)
        low = float(config.low_threshold) if config.use_hysteresis else None
    elif config.threshold_method == "quantile":
        high = float(np.quantile(values, config.threshold))
        low = (
            float(np.quantile(values, config.low_threshold))
            if config.use_hysteresis
            else None
        )
    else:
        high = float(threshold_otsu(values)) if np.ptp(values) > 0 else float(values[0])
        low = high * config.hysteresis_low_ratio if config.use_hysteresis else None
    if low is not None and low > high:
        raise ValueError("the resolved low threshold cannot exceed the high threshold")
    return low, high


def _clean_mask(mask, config):
    if config.closing_radius:
        mask = closing(mask, footprint=disk(config.closing_radius))
    if config.opening_radius:
        mask = opening(mask, footprint=disk(config.opening_radius))
    if config.min_object_size:
        components = label(mask, connectivity=mask.ndim)
        sizes = np.bincount(components.ravel())
        small = sizes < config.min_object_size
        small[0] = False
        mask = mask & ~small[components]
    if config.min_hole_size:
        holes = label(~mask, connectivity=mask.ndim)
        sizes = np.bincount(holes.ravel())
        border_labels = np.unique(
            np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
        )
        fill = sizes < config.min_hole_size
        fill[border_labels] = False
        mask = mask | fill[holes]
    return np.asarray(mask, dtype=bool)


def segment_vessels_by_vesselness(
    image: np.ndarray,
    config: VesselnessSegmentationConfig,
    *,
    roi_mask: np.ndarray | None = None,
) -> VesselnessSegmentationResult:
    """Generate one transparent, reproducible binary segmentation candidate."""

    image, roi_mask = _validate_inputs(image, roi_mask, config)
    normalized = _normalize_image(image, roi_mask, config.intensity_percentiles)
    vesselness = _frangi_response(normalized, config)
    vesselness = np.where(roi_mask, vesselness, 0.0)
    low, high = _resolve_thresholds(vesselness, roi_mask, config)

    if not np.any(vesselness[roi_mask] > 0):
        raw_mask = np.zeros_like(roi_mask)
    elif config.use_hysteresis:
        raw_mask = apply_hysteresis_threshold(vesselness, low, high)
    else:
        raw_mask = vesselness > high
    raw_mask = np.asarray(raw_mask & roi_mask, dtype=bool)
    mask = _clean_mask(raw_mask, config) & roi_mask

    return VesselnessSegmentationResult(
        vesselness=vesselness,
        raw_mask=raw_mask,
        mask=mask,
        high_threshold=high,
        low_threshold=low,
        config=config,
    )


def compare_vesselness_segmentations(
    image: np.ndarray,
    candidates: Mapping[str, VesselnessSegmentationConfig],
    *,
    roi_mask: np.ndarray | None = None,
) -> dict[str, VesselnessSegmentationResult]:
    """Evaluate named candidates without selecting or changing a default mask."""

    if not candidates:
        raise ValueError("at least one candidate configuration is required")
    return {
        name: segment_vessels_by_vesselness(image, config, roi_mask=roi_mask)
        for name, config in candidates.items()
    }
