from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import warnings

import matplotlib
import numpy as np

from dopplerview.input_output.output_renderer import (
    LabeledMaskRenderer,
    SignalRenderer,
)


class DictionaryContext:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


def test_pipeline_rendering_uses_non_gui_backend():
    assert matplotlib.get_backend().lower() == "agg"


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
