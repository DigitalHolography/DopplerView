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
