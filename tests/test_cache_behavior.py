from __future__ import annotations

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.step import BaseStep

from test_dag_engine import make_step


def test_missing_output_causes_execution(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory({"input": "source"}, config={"threshold": 1})

    assert engine._should_run(step, ctx) is True


def test_matching_computation_fingerprint_is_a_cache_hit(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory(
        {"input": "source", "output": "cached"},
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"][step.name] = step.fingerprint(ctx)

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
        "downstream": downstream.fingerprint(ctx),
    }

    assert engine._should_run(upstream, ctx) is True
    assert engine._should_run(downstream, ctx) is True


def test_changed_required_input_content_invalidates_cache(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory(
        {"input": "original", "output": "cached"},
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"][step.name] = step.fingerprint(ctx)
    ctx.set("input", "changed")

    assert engine._should_run(step, ctx) is True


def test_legacy_configuration_only_hash_is_invalidated(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    engine = DAGEngine([step])
    ctx = fake_context_factory(
        {"input": "source", "output": "cached"},
        config={"threshold": 1},
    )
    ctx.metadata["step_hashes"][step.name] = step.config_fingerprint(ctx)

    assert engine._should_run(step, ctx) is True


def test_upstream_artifact_identity_participates_in_downstream_fingerprint(
    fake_context_factory,
):
    step = make_step("downstream", {"middle"}, {"output"})
    ctx = fake_context_factory({"middle": "same value", "output": "cached"})
    ctx.set_artifact_fingerprints("upstream", {"middle"}, "first-run")
    first = step.fingerprint(ctx)

    ctx.set_artifact_fingerprints("upstream", {"middle"}, "second-run")

    assert step.fingerprint(ctx) != first


def test_model_revision_participates_in_fingerprint(fake_context_factory):
    step = make_step("model_task", {"input"}, {"output"})
    ctx = fake_context_factory({"input": "source", "output": "cached"})
    model = {"name": "segmenter", "revision": "revision-a"}
    ctx.get_current_model_name_for_task = lambda task: "segmenter"
    ctx.get_model_identity = lambda name: dict(model)
    first = step.fingerprint(ctx)

    model["revision"] = "revision-b"

    assert step.fingerprint(ctx) != first


def test_source_file_identity_participates_in_fingerprint(
    fake_context_factory,
    tmp_path,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"first")
    step = make_step("read", {"input_file"}, {"output"})
    ctx = fake_context_factory({"input_file": source, "output": "cached"})
    first = step.fingerprint(ctx)

    source.write_bytes(b"second and larger")

    assert step.fingerprint(ctx) != first


def test_debug_export_queues_cache_with_computation_fingerprint(fake_context_factory):
    step = make_step("compute", {"input"}, {"output"})
    ctx = fake_context_factory(
        {"input": "source", "output": "value"},
        config={"threshold": 1},
    )

    step.export(ctx, debug_mode=True)

    assert ctx.output_manager.saved == [("compute", "output")]
    assert ctx.output_manager.cached == [
        ("compute", step.fingerprint(ctx))
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
        source.name: source.fingerprint(ctx),
        target.name: target.fingerprint(ctx),
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
    first_computation = step.fingerprint(ctx)
    ctx.dopplerview_config["Execution"] = {
        "NumberOfWorkers": -1,
        "DagConcurrency": 4,
    }

    assert step.config_fingerprint(ctx) == first
    assert step.fingerprint(ctx) == first_computation
