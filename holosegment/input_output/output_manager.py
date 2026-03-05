import h5py
import numpy as np
from pathlib import Path
import os
from typing import overload
from holosegment.utils.image_utils import normalize_to_uint8
import imageio
import holosegment.input_output.debug_renderer as debug_renderer


class OutputManager:
    def __init__(
        self,
        output_folder,
        schema,
        debug_config=None,
    ):
        self.h5_path = os.path.join(output_folder, "h5")
        self.h5 = h5py.File(self.h5_path, "a")
        self.schema = self._flatten_schema(schema)

        self.debug_dir = os.path.join(output_folder, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_config = debug_config or {}
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.renderers = {
            "image": debug_renderer.ImageRenderer(),
            "mask": debug_renderer.ImageRenderer(),
            "signal": debug_renderer.SignalRenderer(),
            "video": debug_renderer.VideoRenderer(),
            "optic_disc": debug_renderer.OpticDiscRenderer()
        }


    def _flatten_schema(self, schema):

        flat = {}

        def walk(node):
            for k, v in node.items():
                if isinstance(v, dict):
                    walk(v)
                else:
                    flat[k] = v

        walk(schema)
        return flat

    def save_h5(self, key, cache):
        if key not in self.schema:
            return

        path = self.schema[key]

        if path in self.h5:
            del self.h5[path]

        value = cache.get(key)
        self.h5.create_dataset(path, data=value)

    @overload
    def debug(self, step_name, key, cache):
        if key not in self.debug_config:
            return
        
        type = self.debug_config[key]
        self.debug(step_name, key, cache, type)

    @overload
    def debug(self, step_name, key, cache, type):
        step_dir = self.debug_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{key}.png"
        renderer = self.renderers.get(type)

        if renderer:
            renderer.render(key, cache, self.debug_dir, path)

    def save(self, step_name, key, cache):
        self.save_h5(key, cache)
        self.debug(step_name, key, cache)

    def close(self):
        self.h5.close()