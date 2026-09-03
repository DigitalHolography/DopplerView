import numpy as np
import pytest

from dopplerview.segmentation.vesselness_segmentation import (
    VesselnessSegmentationConfig,
    compare_vesselness_segmentations,
    segment_vessels_by_vesselness,
)


def _bright_ridge_image():
    image = np.zeros((64, 64), dtype=float)
    image[8:56, 30:34] = 1.0
    return image


def test_bright_ridge_candidate_detects_the_ridge_without_filling_background():
    result = segment_vessels_by_vesselness(
        _bright_ridge_image(),
        VesselnessSegmentationConfig(
            sigmas=(1.0, 2.0, 3.0),
            ridge_polarity="bright",
            threshold_method="quantile",
            threshold=0.95,
        ),
    )

    assert result.mask.dtype == bool
    assert result.mask[:, 30:34].sum() > 0
    assert 0 < result.mask.sum() < result.mask.size // 2
    assert result.vesselness[:, 30:34].max() > result.vesselness[:, :10].max()


def test_roi_controls_threshold_estimation_and_final_mask_extent():
    roi = np.zeros((64, 64), dtype=bool)
    roi[4:60, 16:48] = True
    result = segment_vessels_by_vesselness(
        _bright_ridge_image(),
        VesselnessSegmentationConfig(threshold_method="quantile", threshold=0.9),
        roi_mask=roi,
    )

    assert not np.any(result.mask[~roi])
    assert not np.any(result.vesselness[~roi])


def test_named_comparison_keeps_candidates_independent():
    configs = {
        "bright": VesselnessSegmentationConfig(ridge_polarity="bright"),
        "dark": VesselnessSegmentationConfig(ridge_polarity="dark"),
        "both": VesselnessSegmentationConfig(ridge_polarity="both"),
    }
    results = compare_vesselness_segmentations(_bright_ridge_image(), configs)

    assert list(results) == list(configs)
    np.testing.assert_allclose(
        results["both"].vesselness,
        np.maximum(results["bright"].vesselness, results["dark"].vesselness),
    )


def test_hysteresis_requires_an_explicit_low_quantile():
    config = VesselnessSegmentationConfig(
        threshold_method="quantile",
        threshold=0.95,
        use_hysteresis=True,
    )
    with pytest.raises(ValueError, match="requires low_threshold"):
        segment_vessels_by_vesselness(_bright_ridge_image(), config)


def test_hysteresis_and_morphology_are_reported_and_produce_boolean_masks():
    result = segment_vessels_by_vesselness(
        _bright_ridge_image(),
        VesselnessSegmentationConfig(
            threshold_method="quantile",
            threshold=0.97,
            use_hysteresis=True,
            low_threshold=0.85,
            closing_radius=1,
            min_object_size=8,
            min_hole_size=8,
        ),
    )

    assert result.low_threshold is not None
    assert result.low_threshold <= result.high_threshold
    assert result.raw_mask.dtype == bool
    assert result.mask.dtype == bool
    assert np.any(result.mask)


def test_invalid_roi_shape_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        segment_vessels_by_vesselness(
            _bright_ridge_image(),
            VesselnessSegmentationConfig(),
            roi_mask=np.ones((32, 32), dtype=bool),
        )


def test_constant_image_produces_an_empty_mask():
    result = segment_vessels_by_vesselness(
        np.ones((32, 32)), VesselnessSegmentationConfig()
    )
    assert not np.any(result.mask)
