from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import warnings

import imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dopplerview.input_output.output_renderer import (
    LabeledMaskRenderer,
    MaskRenderer,
    SignalRenderer,
)
from dopplerview.utils.matplotlib_backend import new_agg_figure


class DictionaryContext:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


def test_dopplerview_imports_preserve_the_selected_matplotlib_backend():
    """App imports must not replace a backend selected by a notebook or UI."""
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repository_root), environment.get("PYTHONPATH")])
    )
    code = """
import matplotlib

matplotlib.use("svg", force=True)
selected_backend = matplotlib.get_backend()

import dopplerview.ui.app
import dopplerview.cli
from dopplerview.input_output.output_renderer import SignalRenderer

assert matplotlib.get_backend() == selected_backend
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_pipeline_figures_use_an_explicit_agg_canvas():
    assert isinstance(new_agg_figure().canvas, FigureCanvasAgg)


def test_mask_renderer_expands_uint8_binary_mask_to_full_range(tmp_path):
    context = DictionaryContext(
        {"mask": np.array([[0, 1], [1, 0]], dtype=np.uint8)}
    )
    path = tmp_path / "mask.png"

    MaskRenderer().render("mask", context, path)

    rendered = imageio.imread(path)
    assert rendered.dtype == np.uint8
    assert np.array_equal(rendered, np.array([[0, 255], [255, 0]], dtype=np.uint8))


def test_matplotlib_renderers_work_from_background_threads(tmp_path):
    signal_context = DictionaryContext({"signal": np.sin(np.linspace(0, 1, 20))})
    mask_context = DictionaryContext(
        {"mask": np.array([[0, 1, 1], [0, 0, 2], [2, 2, 2]], dtype=np.int32)}
    )
    signal_path = tmp_path / "signal.png"
    mask_path = tmp_path / "mask.png"

    with warnings.catch_warnings(record=True) as caught:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    SignalRenderer().render,
                    "signal",
                    signal_context,
                    signal_path,
                ),
                executor.submit(
                    LabeledMaskRenderer().render,
                    "mask",
                    mask_context,
                    mask_path,
                ),
            ]
            for future in futures:
                future.result()

    assert signal_path.exists() and signal_path.stat().st_size > 0
    assert mask_path.exists() and mask_path.stat().st_size > 0
    assert not any("GUI outside of the main thread" in str(item.message) for item in caught)
