from __future__ import annotations

import threading
import time

import pytest

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.step import BaseStep


def make_step(name, requires=(), produces=(), action=None, relevant_config=None):
    def run(self, ctx):
        if action is not None:
            action(ctx)
        for key in self.produces:
            if not ctx.has(key):
                ctx.set(key, f"{self.name}:{key}")

    def _relevant_config(self, ctx):
        if relevant_config is None:
            return ctx.dopplerview_config
        return relevant_config(ctx)

    return type(
        f"{name.title().replace('_', '')}Step",
        (BaseStep,),
        {
            "name": name,
            "requires": set(requires),
            "produces": set(produces),
            "run": run,
            "_relevant_config": _relevant_config,
        },
    )()


def test_builds_dependencies_and_resolves_minimal_target_subgraph():
    read = make_step("read", {"input"}, {"raw"})
    preprocess = make_step("preprocess", {"raw"}, {"image"})
    segment = make_step("segment", {"image"}, {"mask"})
    unrelated = make_step("unrelated", {"input"}, {"report"})

    engine = DAGEngine([read, preprocess, segment, unrelated])

    assert engine.execution_order.index("read") < engine.execution_order.index("preprocess")
    assert engine.execution_order.index("preprocess") < engine.execution_order.index("segment")
    assert engine._resolve_required_steps(["segment"]) == ["read", "preprocess", "segment"]


def test_rejects_duplicate_output_producers():
    first = make_step("first", produces={"shared"})
    second = make_step("second", produces={"shared"})

    with pytest.raises(ValueError, match="Multiple steps produce the same key"):
        DAGEngine([first, second])


def test_rejects_duplicate_step_names():
    first = make_step("duplicate", produces={"first"})
    second = make_step("duplicate", produces={"second"})

    with pytest.raises(ValueError, match="Duplicate step names"):
        DAGEngine([first, second])


def test_registration_order_breaks_dependency_ties_deterministically():
    first_root = make_step("first_root", {"input"}, {"first"})
    second_root = make_step("second_root", {"input"}, {"second"})
    second_child = make_step("second_child", {"second"}, {"second_result"})
    first_child = make_step("first_child", {"first"}, {"first_result"})

    engine = DAGEngine([first_root, second_root, second_child, first_child])

    assert engine.execution_order == [
        "first_root",
        "second_root",
        "second_child",
        "first_child",
    ]


def test_default_dag_concurrency_is_one():
    step = make_step("only", produces={"output"})

    assert DAGEngine([step]).max_workers == 1


def test_independent_steps_in_a_wave_overlap(fake_context_factory):
    barrier = threading.Barrier(2, timeout=2)
    intervals = {}
    interval_lock = threading.Lock()

    def action(name):
        def execute(_ctx):
            started = time.perf_counter()
            barrier.wait()
            time.sleep(0.02)
            finished = time.perf_counter()
            with interval_lock:
                intervals[name] = (started, finished)

        return execute

    left = make_step("left", {"input"}, {"left_value"}, action("left"))
    right = make_step("right", {"input"}, {"right_value"}, action("right"))
    engine = DAGEngine([left, right], max_workers=2)
    ctx = fake_context_factory({"input": "source"})

    engine.run(ctx)

    left_start, left_end = intervals["left"]
    right_start, right_end = intervals["right"]
    assert max(left_start, right_start) < min(left_end, right_end)


def test_failure_stops_dependent_execution(fake_context_factory):
    events = []

    def fail(_ctx):
        raise ValueError("deliberate failure")

    failing = make_step("failing", {"input"}, {"intermediate"}, fail)
    downstream = make_step("downstream", {"intermediate"}, {"result"})
    engine = DAGEngine([failing, downstream])
    ctx = fake_context_factory({"input": "source"})

    with pytest.raises(ValueError, match="deliberate failure"):
        engine.run(ctx, callback=lambda event, *args: events.append((event, args)))

    assert not ctx.has("result")
    assert any(event == "step_start" and args[0] == "failing" for event, args in events)
    assert not any(args and args[0] == "downstream" for _, args in events)
