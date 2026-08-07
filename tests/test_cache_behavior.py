from __future__ import annotations

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.step import BaseStep

from test_dag_engine import make_step


def test_missing_output_causes_execution(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory({"input": "source"}, config={"threshold": 1})

    assert engine._should_run(step, ctx) is True


def test_matching_configuration_hash_is_a_cache_hit(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory(
        {"input": "source", "output": "cached"},
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"][step.name] = step.config_fingerprint(ctx)

    assert engine._should_run(step, ctx) is False


def test_changed_configuration_invalidates_step_and_downstream(fake_context_factory):
    upstream = make_step("upstream", {"input"}, {"middle"})
    downstream = make_step("downstream", {"middle"}, {"output"})
    engine = DAGEngine([upstream, downstream])
    ctx = fake_context_factory(
        {"input": "source", "middle": "cached", "output": "cached"},
        config={"threshold": 2},
    )
    ctx.metadata["step_hashes"] = {
        "upstream": "old-configuration-hash",
        "downstream": downstream.config_fingerprint(ctx),
    }

    assert engine._should_run(upstream, ctx) is True
    assert engine._should_run(downstream, ctx) is True


def test_current_cache_validation_ignores_required_input_content(fake_context_factory):
    """Characterize the current config-only cache check before it is redesigned."""
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory(
        {"input": "original", "output": "cached"},
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"][step.name] = step.config_fingerprint(ctx)
    ctx.set("input", "changed")

    assert step.fingerprint(ctx) != step.config_fingerprint(ctx)
    assert engine._should_run(step, ctx) is False


def test_debug_export_queues_cache_with_configuration_hash(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    ctx = fake_context_factory(
        {"input": "source", "output": "value"},
        config={"threshold": 1},
    )

    step.export(ctx, debug_mode=True)

    assert ctx.output_manager.saved == [("compute", "output")]
    assert ctx.output_manager.cached == [
        ("compute", step.config_fingerprint(ctx))
    ]


def test_cached_upstream_step_exports_outputs_when_skipped(fake_context_factory):
    source = make_step("source", {"input"}, {"cached_image"})
    target = make_step("target", {"cached_image"}, {"result"})
    engine = DAGEngine([source, target])
    ctx = fake_context_factory(
        {
            "input": "source",
            "cached_image": "cached pixels",
            "result": "old result",
        },
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"] = {
        source.name: source.config_fingerprint(ctx),
        target.name: target.config_fingerprint(ctx),
    }
    events = []

    # The target is deliberately re-run by current target semantics, while its
    # valid cached dependency must still pass through the output manager.
    engine.run(
        ctx,
        targets=["target"],
        callback=lambda event, *args: events.append((event, args)),
    )

    assert ("step_skipped", ("source",)) in events
    assert ("source", "cached_image") in ctx.output_manager.saved


def test_execution_policy_does_not_change_default_step_fingerprint(
    fake_context_factory,
):
    class DefaultConfigStep(BaseStep):
        name = "default_config"
        requires = set()
        produces = set()

    step = DefaultConfigStep()
    ctx = fake_context_factory(config={"Threshold": 3, "Execution": {"NumberOfWorkers": 1}})
    first = step.config_fingerprint(ctx)
    ctx.dopplerview_config["Execution"] = {
        "NumberOfWorkers": -1,
        "DagConcurrency": 4,
    }

    assert step.config_fingerprint(ctx) == first
