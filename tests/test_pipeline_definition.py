from __future__ import annotations

import pytest

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.definition import PipelineDefinition
from dopplerview.pipeline.step import BaseStep


def make_step(name, requires=(), produces=()):
    return type(
        name.title().replace("_", ""),
        (BaseStep,),
        {
            "name": name,
            "requires": set(requires),
            "produces": set(produces),
            "run": lambda self, ctx: None,
        },
    )()


def test_definition_exposes_deterministic_graph_without_runtime_context():
    first = make_step("first", produces={"a"})
    independent = make_step("independent", produces={"b"})
    last = make_step("last", requires={"a"}, produces={"c"})

    definition = PipelineDefinition([first, independent, last])

    assert definition.execution_order == ["first", "independent", "last"]
    assert definition.resolve_execution_graph(["last"]) == ["first", "last"]
    assert definition.get_downstream_steps("first") == {"last"}


def test_definition_metadata_cannot_be_mutated_through_public_views():
    definition = PipelineDefinition([make_step("only")])

    with pytest.raises(TypeError):
        definition.steps_by_name["other"] = make_step("other")

    graph = definition.graph
    graph["only"].add("invented")
    assert definition.graph["only"] == set()


def test_engine_uses_definition_graph_and_step_order():
    first = make_step("first", produces={"a"})
    last = make_step("last", requires={"a"})
    definition = PipelineDefinition([first, last])

    engine = DAGEngine(definition)

    assert engine.definition is definition
    assert engine.execution_order == definition.execution_order
    assert list(engine.steps) == ["first", "last"]


def test_definition_rejects_unknown_targets():
    definition = PipelineDefinition([make_step("only")])

    with pytest.raises(ValueError, match="Unknown target step"):
        definition.resolve_execution_graph(["missing"])
