from __future__ import annotations

import threading
import time

import pytest

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.execution_profile import ExecutionProfile
from dopplerview.pipeline.pipeline import Context, Pipeline

from test_dag_engine import make_step


def test_sequential_profile_can_be_selected_by_name_and_alias():
    assert (
        ExecutionProfile.resolve("sequential_reference")
        is ExecutionProfile.SEQUENTIAL_REFERENCE
    )
    assert ExecutionProfile.resolve("sequential") is ExecutionProfile.SEQUENTIAL_REFERENCE
    assert ExecutionProfile.resolve("reference") is ExecutionProfile.SEQUENTIAL_REFERENCE


def test_unknown_execution_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown execution profile"):
        ExecutionProfile.resolve("fastest")


def test_profile_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("DOPPLERVIEW_EXECUTION_PROFILE", "sequential_reference")

    assert ExecutionProfile.resolve() is ExecutionProfile.SEQUENTIAL_REFERENCE


def test_sequential_profile_overrides_workers_without_mutating_config():
    ctx = Context(output_manager=None, execution_profile="sequential_reference")
    ctx.dopplerview_config = {"NumberOfWorkers": 0.75}
    ctx.configure_execution_policy()

    assert ctx.get_number_of_workers() == 1
    assert ctx.dopplerview_config["NumberOfWorkers"] == 0.75


def test_default_profile_preserves_configured_worker_value():
    ctx = Context(output_manager=None, execution_profile="default")
    ctx.dopplerview_config = {"NumberOfWorkers": -1}
    ctx.configure_execution_policy()

    assert ctx.get_number_of_workers() == ctx.execution_policy.available_cpus


def test_switching_pipeline_profile_updates_dag_limit():
    pipeline = Pipeline(output_manager=None, execution_profile="default")
    assert pipeline.engine.max_workers == 1

    pipeline.set_execution_profile("sequential_reference")

    assert pipeline.execution_profile is ExecutionProfile.SEQUENTIAL_REFERENCE
    assert pipeline.engine.max_workers == 1


def test_single_worker_dag_never_overlaps_independent_steps(fake_context_factory):
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def action(_ctx):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1

    left = make_step("left", {"input"}, {"left_value"}, action)
    right = make_step("right", {"input"}, {"right_value"}, action)
    engine = DAGEngine([left, right], max_workers=1)

    engine.run(fake_context_factory({"input": "source"}))

    assert maximum_active == 1
