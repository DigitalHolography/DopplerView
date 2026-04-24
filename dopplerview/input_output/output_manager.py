import json

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
        dopplerview_folder,
        schema,
        dopplerview_config,
        output_config=None
    ):
        self.h5_path = dopplerview_folder.get_h5_path()
        # Create an empty H5 file if it doesn't exist, and overwrite it if it does (to ensure a clean slate for each run)
        with h5py.File(self.h5_path, "w") as h5:
            pass

        self.schema = json_utils.flatten_schema(schema)

        self.dopplerview_folder = dopplerview_folder
        self.output_dir = dopplerview_folder.get_output_folder()[0] # Get the output folder if it exists, otherwise it will be created when needed
        self.output_config = output_config or {}

        self.dopplerview_config = dopplerview_config

        self.cache_dir = dopplerview_folder.get_cache_folder()
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

    def write_dopplerview_config(self):
        if self.output_dir is None:
            raise ValueError("Output directory is not set. Cannot write DopplerView configuration.")
        
        config_path = self.output_dir / self.dopplerview_folder.config_name
        with open(config_path, "w") as f:
            json.dump(self.dopplerview_config, f)

    def ensure_output_folder(self):
        if self.output_dir is None:
            self.output_dir = self.dopplerview_folder.create_output_folder()
            self.write_dopplerview_config()

    def output_cache(self, step_name, key, cache, type=None):
        """Outputs a value from the cache for debugging purposes based on the provided output configuration."""
        if key not in self.output_config or key not in cache:
            return
        
        if type is None:
            type = self.output_config[key]
        renderer = self.renderers.get(type)

        if renderer is None:
            Warning(f"No renderer found for output type '{type}' of key '{key}', skipping output.")
            return
        
        # Lasily create the output folder when we actually need to output something, to avoid creating empty output folders for runs that don't produce any outputs
        self.ensure_output_folder()

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{key}.png"

        renderer.render(key, cache, path)

    def output(self, step_name, filename, value, type=None, options=None):
        """Outputs a value manually for debugging purposes based on the provided output configuration."""
        if type is None:
            Warning(f"No output type specified for key '{step_name}', skipping debug output.")
            return
        
        renderer = self.renderers.get(type)
        if renderer is None:
            Warning(f"No renderer found for output type '{type}' of key '{step_name}', skipping output.")
            return

        # Lasily create the output folder when we actually need to output something, to avoid creating empty output folders for runs that don't produce any outputs
        self.ensure_output_folder()

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{filename}.png"

        renderer.render("value", {"value": value}, path, options=options)

    def save(self, step_name, key, cache):
        self.save_h5(key, cache)
        self.output_cache(step_name, key, cache)