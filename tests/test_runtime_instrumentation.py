from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dopplerview.models.wrapper import BaseModelWrapper
from dopplerview.pipeline.dag import DAGEngine
from dopplerview.utils.parallelization_utils import run_in_parallel
from dopplerview.utils.runtime_metrics import RuntimeMetrics

from test_dag_engine import make_step


def metric_records(caplog, kind):
    return [
        record.message
        for record in caplog.records
        if "[Metrics]" in record.message and f"kind={kind}" in record.message
    ]


def test_step_records_timing_memory_threads_and_status(fake_context_factory, caplog):
    ctx = fake_context_factory({"input": "source"})
    ctx.runtime_metrics = RuntimeMetrics()
    step = make_step("measured", {"input"}, {"output"})

    with caplog.at_level("DEBUG"):
        DAGEngine([step]).run(ctx)

    metric = next(
        record for record in ctx.runtime_metrics.snapshot() if record["kind"] == "step"
    )
    assert metric["step"] == "measured"
    assert metric["status"] == "success"
    assert metric["duration_s"] >= 0
    assert metric["process_peak_rss_mb"] >= metric["process_rss_start_mb"]
    assert metric["process_threads_start"] >= 1
    assert metric_records(caplog, "step")
    wave_metric = next(
        record
        for record in ctx.runtime_metrics.snapshot()
        if record["kind"] == "dag_wave"
    )
    assert wave_metric["ready_steps"] == 1
    assert wave_metric["effective_workers"] == 1


def test_failed_step_still_records_measurement(fake_context_factory):
    ctx = fake_context_factory({"input": "source"})
    ctx.runtime_metrics = RuntimeMetrics()

    def fail(_ctx):
        raise RuntimeError("failure")

    step = make_step("failing", {"input"}, {"output"}, fail)

    try:
        DAGEngine([step]).run(ctx)
    except RuntimeError:
        pass

    metric = next(
        record for record in ctx.runtime_metrics.snapshot() if record["kind"] == "step"
    )
    assert metric["status"] == "failed"


def test_parallel_operation_logs_requested_and_effective_workers(caplog):
    with caplog.at_level("DEBUG"):
        result = run_in_parallel(
            lambda value: value * 2,
            np.arange(4),
            n_jobs=0.5,
            chunking=False,
            task_name="doubling",
        )

    np.testing.assert_array_equal(result, [0, 2, 4, 6])
    messages = metric_records(caplog, "parallel_operation")
    assert len(messages) == 1
    assert "task=doubling" in messages[0]
    assert "requested_workers=0.500" in messages[0]
    assert "effective_workers=" in messages[0]


class FakeModelWrapper(BaseModelWrapper):
    def _forward(self, x):
        return np.asarray(x)[None, None, :, :]

    def provider_name(self):
        return "TestProvider"


def test_model_prediction_logs_backend_provider_and_phase_timings(caplog):
    spec = SimpleNamespace(
        name="fake_model",
        input_norm="none",
        output_activation="none",
    )
    model = FakeModelWrapper(spec, "unused")

    with caplog.at_level("DEBUG"):
        result = model.predict(np.ones((2, 2), dtype=np.uint8))

    assert result.shape == (1, 1, 2, 2)
    messages = metric_records(caplog, "inference")
    assert len(messages) == 1
    assert "model=fake_model" in messages[0]
    assert "provider=TestProvider" in messages[0]
    assert "preprocess_s=" in messages[0]
    assert "inference_s=" in messages[0]
    assert "postprocess_s=" in messages[0]


def test_output_queue_logs_depth(bare_output_manager_factory, caplog):
    manager = bare_output_manager_factory()
    manager.running = True
    ctx = SimpleNamespace(get=lambda key: "value", has=lambda key: True)

    with caplog.at_level("DEBUG"):
        manager.save_async("step", "artifact", ctx)

    messages = metric_records(caplog, "output_queue")
    assert len(messages) == 1
    assert "queue_depth=1" in messages[0]
    manager.close_workers()
