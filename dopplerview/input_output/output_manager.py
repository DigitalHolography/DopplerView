import h5py
import numpy as np
from pathlib import Path
import os
from dopplerview.utils.image_utils import normalize_to_uint8
import dopplerview.utils.json_utils as json_utils
import imageio
import dopplerview.input_output.output_renderer as output_renderer
import re


class OutputManager:
    def __init__(
        self,
        output_folder,
        h5_path,
        schema,
        cache_folder,
        output_config=None
    ):
        h5_path = Path(h5_path)
        filename = h5_path.stem
        parent = h5_path.parent
        print(f"{filename=}, {parent=}")
        template = re.compile(r"^(.*)_(output|raw)$")
        match = template.match(filename)
        if match:
            self.h5_path = parent / f"{match.group(1)}_DV.h5"
        else:
            self.h5_path = parent / f"{filename}_DV.h5"
        with h5py.File(self.h5_path, "w") as h5:
            pass  # Just create an empty H5 file or overwrite if it exists

        self.schema = json_utils.flatten_schema(schema)

        self.output_dir = Path(output_folder)
        self.output_dir.mkdir(exist_ok=True)
        self.output_config = output_config or {}

        self.cache_dir = Path(cache_folder)
        self.cache_dir.mkdir(exist_ok=True)
        self.renderers = {
            "image": output_renderer.ImageRenderer(),
            "mask": output_renderer.ImageRenderer(),
            "signal": output_renderer.SignalRenderer(),
            "video": output_renderer.VideoRenderer(),
            "optic_disc": output_renderer.OpticDiscRenderer(),
            "labeled_mask": output_renderer.LabeledMaskRenderer()
        }

    def save_cache(self, cache):
        """Saves the entire cache to the H5 file"""
        with h5py.File(self.cache_dir / "cache.h5", "w") as h5_cache:
            for key in cache:
                if key in h5_cache:
                    del h5_cache[key]
                h5_cache.create_dataset(key, data=cache[key])

    def save_h5(self, key, cache):
        """Saves a value from the cache to the H5 file based on the provided schema."""
        if key not in self.schema:
            return

        path = self.schema[key]

        path = path.replace("\\", "/")  # Ensure consistent path format

        with h5py.File(self.h5_path, "a") as h5:
            if path in h5:
                del h5[path]

            value = cache.get(key)
            h5.create_dataset(path, data=value)

    def output_cache(self, step_name, key, cache, type=None):
        """Outputs a value from the cache for debugging purposes based on the provided output configuration."""
        if key not in self.output_config or key not in cache:
            return
        
        if type is None:
            type = self.output_config[key]

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{key}.png"
        renderer = self.renderers.get(type)

        if renderer:
            renderer.render(key, cache, path)

    def output(self, step_name, filename, value, type=None, options=None):
        """Outputs a value manually for debugging purposes based on the provided output configuration."""
        if type is None:
            Warning(f"No output type specified for key '{step_name}', skipping debug output.")
            return

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{filename}.png"
        renderer = self.renderers.get(type)

        if renderer:
            renderer.render("value", {"value": value}, path, options=options)

    def save(self, step_name, key, cache):
        self.save_h5(key, cache)
        self.output_cache(step_name, key, cache)