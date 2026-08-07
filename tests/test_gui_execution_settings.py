import json

import pytest

from dopplerview.pipeline.pipeline import Context
from dopplerview.ui.main_window import MainWindow


@pytest.mark.parametrize(
    ("raw", "allow_auto", "expected"),
    [
        ("Use config", False, None),
        ("auto", True, "auto"),
        ("0.5", False, 0.5),
        ("2", True, 2),
        ("-1", False, -1),
    ],
)
def test_gui_execution_setting_parser(raw, allow_auto, expected):
    assert (
        MainWindow._parse_execution_setting("Setting", raw, allow_auto)
        == expected
    )


@pytest.mark.parametrize("raw", ["zero", "0", "1.5", "-1.5"])
def test_gui_execution_setting_parser_rejects_ambiguous_values(raw):
    with pytest.raises(ValueError):
        MainWindow._parse_execution_setting("Setting", raw, allow_auto=False)


@pytest.mark.parametrize("raw", [1, "4", 8.0])
def test_gui_exact_worker_count_accepts_integers_within_capacity(raw):
    assert MainWindow._parse_exact_worker_count(raw, maximum=8) == int(raw)


@pytest.mark.parametrize("raw", ["Use config", "0.5", "-1", 0, 9])
def test_gui_exact_worker_count_rejects_relative_or_out_of_range_values(raw):
    with pytest.raises(ValueError):
        MainWindow._parse_exact_worker_count(raw, maximum=8)


def test_execution_overrides_are_reapplied_and_can_return_to_config(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )
    config_path = tmp_path / "params.json"
    config_path.write_text(
        json.dumps(
            {
                "Execution": {
                    "NumberOfWorkers": 0.25,
                    "DagConcurrency": "auto",
                }
            }
        )
    )
    ctx = Context(output_manager=None)
    try:
        ctx.load_dopplerview_config(config_path)
        ctx.set_execution_overrides(
            {"NumberOfWorkers": -1, "DagConcurrency": 1}
        )

        assert ctx.execution_policy.cpu_workers == 8
        assert ctx.execution_policy.dag_concurrency == 1

        ctx.set_execution_overrides({})

        assert ctx.dopplerview_config["Execution"]["NumberOfWorkers"] == 0.25
        assert ctx.dopplerview_config["Execution"]["DagConcurrency"] == "auto"
        assert ctx.execution_policy.cpu_workers == 2
        assert ctx.execution_policy.dag_concurrency == 2
    finally:
        ctx.close()


def test_unknown_execution_override_is_rejected():
    ctx = Context(output_manager=None)
    try:
        with pytest.raises(ValueError, match="Unknown execution setting"):
            ctx.set_execution_overrides({"Unexpected": 2})
    finally:
        ctx.close()
