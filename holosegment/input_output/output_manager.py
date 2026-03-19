import h5py
import numpy as np
from pathlib import Path
import os
from holosegment.utils.image_utils import normalize_to_uint8
import imageio
import holosegment.input_output.debug_renderer as debug_renderer


class OutputManager:
    def __init__(
        self,
        output_folder,
        h5_path,
        schema,
        debug_config=None,
    ):
        self.h5 = h5py.File(h5_path, "a")
        self.schema = self._flatten_schema(schema)

        self.debug_dir = Path(os.path.join(output_folder, "debug"))
        self.debug_dir.mkdir(exist_ok=True)
        self.debug_config = debug_config or {}

        self.renderers = {
            "image": debug_renderer.ImageRenderer(),
            "mask": debug_renderer.ImageRenderer(),
            "signal": debug_renderer.SignalRenderer(),
            "video": debug_renderer.VideoRenderer(),
            "optic_disc": debug_renderer.OpticDiscRenderer(),
            "labeled_mask": debug_renderer.LabeledMaskRenderer()
        }


    def _flatten_schema(self, schema):
        flat = {}

        def walk(node, path=""):
            for k, v in node.items():
                new_path = os.path.join(path, k) if path else k
                if isinstance(v, dict):
                    walk(v, new_path)
                else:
                    flat[v] = new_path

        walk(schema)
        return flat

    def save_h5(self, key, cache):
        if key not in self.schema:
            return

        path = self.schema[key]

        path = path.replace("\\", "/")  # Ensure consistent path format

        if path in self.h5:
            del self.h5[path]

        value = cache.get(key)
        self.h5.create_dataset(path, data=value)

    def debug_cache(self, step_name, key, cache, type=None):
        if key not in self.debug_config or key not in cache:
            return
        
        if type is None:
            type = self.debug_config[key]

        step_dir = self.debug_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{key}.png"
        renderer = self.renderers.get(type)

        if renderer:
            renderer.render(key, cache, path)

    def debug(self, step_name, filename, value, type=None):
        if type is None:
            Warning(f"No debug type specified for key '{step_name}', skipping debug output.")
            return

        step_dir = self.debug_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{filename}.png"
        renderer = self.renderers.get(type)

        if renderer:
            renderer.render("value", {"value": value}, path)

    def save(self, step_name, key, cache):
        self.save_h5(key, cache)
        self.debug_cache(step_name, key, cache)

    def close(self):
        self.h5.close()