from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from dopplerview.pipeline.pipeline import Pipeline


class BatchContext:
    def __init__(self, inputs):
        self.input_list = list(inputs)
        self.loaded = []
        self.current = None

    def load_input_folder(self, input_path):
        self.current = input_path
        self.loaded.append(input_path)


def make_batch_pipeline(inputs, failing_inputs=()):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.ctx = BatchContext(inputs)
    failing_inputs = set(failing_inputs)

    def run(self, targets=None, callback=None):
        if self.ctx.current in failing_inputs:
            raise RuntimeError(f"failed {self.ctx.current}")
        return self.ctx

    pipeline.run = MethodType(run, pipeline)
    return pipeline


def test_batch_processes_inputs_in_order_and_reports_success():
    pipeline = make_batch_pipeline(["one.holo", "two.holo"])
    events = []

    results = pipeline.run_batch(
        targets=["segment"],
        callback=lambda event, *args: events.append((event, args)),
    )

    assert pipeline.ctx.loaded == ["one.holo", "two.holo"]
    assert [result["status"] for result in results] == ["success", "success"]
    assert events[0] == ("batch_start", (2,))
    assert events[-1][0] == "batch_done"


def test_batch_continues_after_one_input_fails():
    pipeline = make_batch_pipeline(
        ["one.holo", "broken.holo", "three.holo"],
        failing_inputs={"broken.holo"},
    )
    events = []

    results = pipeline.run_batch(
        callback=lambda event, *args: events.append((event, args))
    )

    assert pipeline.ctx.loaded == ["one.holo", "broken.holo", "three.holo"]
    assert [result["status"] for result in results] == [
        "success",
        "failed",
        "success",
    ]
    assert "failed broken.holo" in results[1]["error"]
    assert any(event == "pipeline_failed" for event, _ in events)
    assert events[-1][0] == "batch_done"


def test_pipeline_stops_output_manager_when_engine_fails():
    calls = []

    class FailingEngine:
        max_workers = 1

        def run(self, ctx, targets, callback=None):
            raise RuntimeError("step failed")

    ctx = SimpleNamespace(
        dopplerview_config={"configured": True},
        execution_profile=SimpleNamespace(value="default"),
        has=lambda key: key == "input_file",
        get_number_of_workers=lambda: 1,
        ensure_config=lambda: calls.append("ensure_config"),
        create_output_folder=lambda: calls.append("create_output_folder"),
        start_output_manager=lambda: calls.append("start"),
        stop_output_manager=lambda: calls.append("stop"),
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.ctx = ctx
    pipeline.engine = FailingEngine()

    with pytest.raises(RuntimeError, match="step failed"):
        pipeline.run()

    assert calls[-1] == "stop"


def test_pipeline_preserves_execution_error_when_cleanup_also_fails():
    class FailingEngine:
        max_workers = 1

        def run(self, ctx, targets, callback=None):
            raise RuntimeError("step failed")

    def fail_cleanup():
        raise OSError("cleanup failed")

    ctx = SimpleNamespace(
        dopplerview_config={"configured": True},
        execution_profile=SimpleNamespace(value="default"),
        has=lambda key: key == "input_file",
        get_number_of_workers=lambda: 1,
        ensure_config=lambda: None,
        create_output_folder=lambda: None,
        start_output_manager=lambda: None,
        stop_output_manager=fail_cleanup,
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.ctx = ctx
    pipeline.engine = FailingEngine()

    with pytest.raises(RuntimeError, match="step failed"):
        pipeline.run()
