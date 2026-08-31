from types import SimpleNamespace

import numpy as np

from dopplerview.pipeline.steps.preprocess import PreprocessStep
from dopplerview.pipeline.steps.pulse_analysis import (
    ComputeTemporalCuesStep,
    PreArteryMaskStep,
    PulseAnalysisStep,
)
import dopplerview.pipeline.steps.pulse_analysis as pulse_step_module


class DummyOutputManager:
    def output(self, *_args, **_kwargs):
        pass

    def save_overlay(self, *_args, **_kwargs):
        pass

    def save_clusterization(self, *_args, **_kwargs):
        pass


class DummyContext:
    def __init__(self, values):
        self.values = dict(values)
        self.dopplerview_config = {"Mask": {}}
        self.holodoppler_config = {"sampling_freq": 40, "batch_stride": 1}
        self.parallel = SimpleNamespace(max_workers=1)
        self.output_manager = DummyOutputManager()

    def require(self, key):
        if key not in self.values:
            raise RuntimeError(f"Missing required context key: {key}")
        return self.values[key]

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def has(self, key):
        return key in self.values


def test_pulse_step_declares_every_context_input_it_reads():
    assert PreArteryMaskStep.requires >= {"M0_ff_image"}
    assert ComputeTemporalCuesStep.requires >= {
        "pre_vein_mask",
        "pre_artery_mask_gradient",
        "LF_M0_ff",
        "HF_M0_ff",
        "band_ratio_ff",
    }

    nested_step = PulseAnalysisStep()
    assert "pre_vein_mask" not in nested_step.requires
    assert {
        "M0_ff_image",
        "LF_M0_ff",
        "HF_M0_ff",
        "band_ratio_ff",
    } <= nested_step.requires


def test_preprocess_sets_all_declared_outputs_when_frequency_bands_are_absent(
    monkeypatch,
):
    moment = np.arange(12, dtype=float).reshape(3, 2, 2)
    ctx = DummyContext(
        {
            "moment0": moment,
            "moment1": moment + 1,
            "moment2": moment + 2,
            "HF_M0": None,
            "LF_M0": None,
        }
    )
    step = PreprocessStep()
    monkeypatch.setattr(
        step,
        "normalize",
        lambda *_args: (moment, moment + 2, None, None),
    )

    step.run(ctx)

    assert all(ctx.has(key) for key in step.produces)
    assert ctx.get("M1_video") is not None
    assert ctx.get("HF_M0_ff") is None
    assert ctx.get("LF_M0_ff") is None
    assert ctx.get("band_ratio_ff") is None


def test_temporal_cues_sets_conditional_outputs_on_short_recording(monkeypatch):
    video = np.arange(40, dtype=float).reshape(10, 2, 2)
    artery_mask = np.array([[True, False], [False, False]])
    ctx = DummyContext(
        {
            "M0_ff_video": video,
            "pre_artery_mask": artery_mask,
            "pre_vein_mask": np.zeros((2, 2), dtype=bool),
            "choroidal_vessel_mask": np.zeros((2, 2), dtype=bool),
            "LF_M0_ff": None,
            "HF_M0_ff": None,
            "band_ratio_ff": None,
        }
    )
    monkeypatch.setattr(
        pulse_step_module.signal_processing,
        "get_filtered_pulse",
        lambda pulse, _sampling_frequency: np.asarray(pulse),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_period",
        lambda *_args: 5,
    )
    monkeypatch.setattr(
        pulse_step_module.signal_processing,
        "compute_correlation",
        lambda input_video, _pulse: np.zeros(input_video.shape[1:]),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_diasys_image",
        lambda *_args: (
            np.zeros((2, 2)),
            np.array([0]),
            np.array([1]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            [0],
        ),
    )
    monkeypatch.setattr(
        pulse_step_module.image_utils,
        "normalize_to_uint8",
        lambda image: np.zeros(image.shape, dtype=np.uint8),
    )

    step = ComputeTemporalCuesStep()
    step.run(ctx)

    assert all(ctx.has(key) for key in step.produces)
    assert np.array_equal(
        ctx.get("pre_arterial_pulse_cleaned"),
        ctx.get("pre_arterial_pulse_filtered"),
    )
    assert ctx.get("correlation_LF_M0_ff") is None
    assert ctx.get("correlation_HF_M0_ff") is None
    assert ctx.get("correlation_band_ratio_ff") is None


def test_both_pre_mask_method_keeps_clustering_as_canonical(monkeypatch):
    video = np.zeros((10, 2, 2), dtype=float)
    labeled_vessels = np.array([[1, 0], [0, 2]])
    clustering_artery = labeled_vessels == 1
    clustering_vein = labeled_vessels == 2
    gradient_artery = labeled_vessels == 2
    gradient_vein = labeled_vessels == 1
    ctx = DummyContext(
        {
            "M0_ff_video": video,
            "M0_ff_image": np.zeros((2, 2), dtype=float),
            "retinal_vessel_mask": labeled_vessels > 0,
            "optic_disc_center": None,
        }
    )
    ctx.dopplerview_config["Mask"] = {
        "PreMaskMethod": "both",
        "CorrectBranchSignals": False,
    }

    monkeypatch.setattr(
        pulse_step_module.process_masks,
        "get_labeled_vessels",
        lambda *_args, **_kwargs: (labeled_vessels, None),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "get_filtered_branch_signals",
        lambda *_args, **_kwargs: np.vstack(
            [np.linspace(0, 1, 10), np.linspace(1, 0, 10)]
        ),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_pre_masks_by_clustering",
        lambda *_args, **_kwargs: (
            clustering_artery,
            clustering_vein,
            np.array([0, 1]),
            np.array([1 + 0j, -1 + 0j]),
        ),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_pre_masks_by_systolic_gradient",
        lambda *_args, **_kwargs: (gradient_artery, gradient_vein),
    )

    step = PreArteryMaskStep()
    step.run(ctx)

    assert np.array_equal(ctx.get("pre_artery_mask"), clustering_artery)
    assert np.array_equal(ctx.get("pre_vein_mask"), clustering_vein)
    assert np.array_equal(
        ctx.get("pre_artery_mask_clustering"), clustering_artery
    )
    assert np.array_equal(ctx.get("pre_artery_mask_gradient"), gradient_artery)


def test_both_method_outputs_clustering_and_gradient_temporal_cues(monkeypatch):
    video = np.arange(40, dtype=float).reshape(10, 2, 2)
    clustering_mask = np.array([[True, False], [False, False]])
    gradient_mask = np.array([[False, True], [False, False]])
    ctx = DummyContext(
        {
            "M0_ff_video": video,
            "pre_artery_mask": clustering_mask,
            "pre_vein_mask": np.zeros((2, 2), dtype=bool),
            "pre_artery_mask_gradient": gradient_mask,
            "choroidal_vessel_mask": np.zeros((2, 2), dtype=bool),
            "LF_M0_ff": None,
            "HF_M0_ff": None,
            "band_ratio_ff": None,
        }
    )
    ctx.dopplerview_config["Mask"]["PreMaskMethod"] = "both"
    monkeypatch.setattr(
        pulse_step_module.signal_processing,
        "get_filtered_pulse",
        lambda pulse, _sampling_frequency: np.asarray(pulse),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_period",
        lambda *_args: 5,
    )
    monkeypatch.setattr(
        pulse_step_module.signal_processing,
        "compute_correlation",
        lambda input_video, pulse: np.full(input_video.shape[1:], pulse[0]),
    )
    monkeypatch.setattr(
        pulse_step_module.pulse_analysis,
        "compute_diasys_image",
        lambda _video, pulse, _sampling_frequency: (
            np.full((2, 2), pulse[0]),
            np.array([0]),
            np.array([1]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            [0],
        ),
    )
    monkeypatch.setattr(
        pulse_step_module.image_utils,
        "normalize_to_uint8",
        lambda image: np.zeros(image.shape, dtype=np.uint8),
    )

    ComputeTemporalCuesStep().run(ctx)

    assert np.array_equal(
        ctx.get("correlation_M0_clustering"), ctx.get("correlation_M0")
    )
    assert np.array_equal(
        ctx.get("diasys_image_clustering"), ctx.get("diasys_image")
    )
    assert np.all(ctx.get("correlation_M0_gradient") == 1)
    assert np.all(ctx.get("diasys_image_gradient") == 1)
