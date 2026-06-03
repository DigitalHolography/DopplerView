import json
import os
from pathlib import Path

import cv2
import h5py
from dopplerview.input_output import h5_file
from dopplerview._version import __version__
import dopplerview.utils.json_utils as json_utils
import dopplerview.input_output.output_renderer as output_renderer
import matplotlib.pyplot as plt
import numpy as np

import queue
import threading

import logging
logger = logging.getLogger(__name__)

class OutputManager:
    def __init__(
        self,
        schema_path,
        output_config_path
    ):
        self.schema_path = schema_path
        self.schema = self.load_h5_schema(schema_path)

        self.output_dir = None # It will be created when needed
        self.output_config_path = output_config_path
        self.output_config = self.load_output_config(output_config_path)

        self.renderers = {
            "image": output_renderer.ImageRenderer(),
            "mask": output_renderer.ImageRenderer(),
            "signal": output_renderer.SignalRenderer(),
            "video": output_renderer.VideoRenderer(),
            "optic_disc": output_renderer.OpticDiscRenderer(),
            "labeled_mask": output_renderer.LabeledMaskRenderer()
        }

        self.running = False

        self.output_queue = queue.Queue()
        self.output_worker = None

        self.cache_queue = queue.Queue()
        self.cache_worker = None

    def __del__(self):
        self.close_workers()

    def start(self):
        if self.running:
            return

        self.running = True

        self.output_worker = threading.Thread(
            target=self._output_worker,
            daemon=True
        )
        self.output_worker.start()

    def close_workers(self):
        if not self.running:
            return

        self.output_queue.put((None, None, None))
        self.cache_queue.put((None, None, None))

        if self.output_worker is not None:
            self.output_worker.join()

        if self.cache_worker is not None:
            self.cache_worker.join()

        self.running = False

    def _output_worker(self):
        while self.running:
            step_name, key, ctx = self.output_queue.get()
            if step_name is None and key is None and ctx is None:
                break
            try:
                self.save(step_name, key, ctx)
            except Exception as e:
                logger.exception(f"Error saving output for step '{step_name}' and key '{key}': {e}")
            self.output_queue.task_done()

    def _cache_worker(self):
        """Worker thread that saves the cache to disk asynchronously. It listens on the cache_queue for contexts to save, and calls save_cache on them.
        """
        while self.running:
            ctx, step_fingerprint, step_name = self.cache_queue.get()
            if ctx is None:
                break
            try:
                # We use the step fingerprint to determine if we need to overwrite the existing cache values for this step or not. If the fingerprint is the same as the one already saved in the h5 file, it means that the configuration and the input for this step haven't changed since the last run, so we can keep the existing cache values. If the fingerprint is different, it means that something has changed since the last run, so we need to overwrite the existing cache values with the new ones.
                overwrite = False
                # Save the fingerprint of the step, to re-run the step if the configuration and/or the input change in the future
                with h5py.File(self.cache_path, "a") as h5:
                    if "metadata" not in h5:
                        h5.create_group("metadata")
                        overwrite = True
                    if "step_hashes" not in h5["metadata"]:
                        h5["metadata"].create_group("step_hashes")
                        overwrite = True

                    # If the step already has a saved hash, only update it if it's different from the new hash. This way we can keep track of which steps have changed since the last run, and which haven't.
                    if step_name in h5["metadata"]["step_hashes"]:
                        if h5["metadata"]["step_hashes"][step_name][()] != step_fingerprint:
                            del h5["metadata"]["step_hashes"][step_name]
                            overwrite = True
                    else:
                        overwrite = True

                    if overwrite:    
                        h5["metadata"]["step_hashes"].create_dataset(step_name, data=step_fingerprint, dtype=h5py.string_dtype())
                # Get the produced values from the context and save them to the h5 file.
                produced_cache = ctx.get_produced_values()
                h5_file.write_dict_to_h5(produced_cache, self.cache_path, overwrite=overwrite)
                ctx.cache_values(produced_cache.keys())

            except Exception as e:
                logger.exception(f"Error saving cache: {e}")
            self.cache_queue.task_done()

    def load_h5_schema(self, schema_path):
        schema = json.load(open(schema_path))
        logger.info(f"[OutputManager] Loading H5 schema from {schema_path}")
        return json_utils.flatten_schema(schema)
    
    def load_output_config(self, output_config_path):
        output_config = json.load(open(output_config_path))
        logger.info(f"[OutputManager] Loading output configuration from {output_config_path}")
        return output_config

    def save_h5(self, key, ctx):
        """Saves a value from the cache to the H5 file based on the provided schema."""
        if key not in self.schema:
            return

        path = self.schema[key]

        path = path.replace("\\", "/")  # Ensure consistent path format

        with h5py.File(self.h5_path, "a") as h5:
            if path in h5:
                del h5[path]

            value = ctx.get(key)
            h5.create_dataset(path, data=value)

    def write_dopplerview_config(self):
        if self.output_dir is None:
            raise ValueError("Output directory is not set. Cannot write DopplerView configuration.")
        
        config_path = self.output_dir / self.dopplerview_folder.config_name
        with open(config_path, "w") as f:
            json.dump(self.dopplerview_config, f)

    def write_version_file(self):
        self.dopplerview_folder.version_file.write_text(__version__)
        if self.output_dir is None:
            raise ValueError("Output directory is not set. Cannot write version file.")
        
        output_version_path = self.output_dir / "version.txt"
        output_version_path.write_text(__version__)

    def set_dopplerview_config(self, config):
        self.dopplerview_config = config

    def set_DV_folder(self, DV_folder):
        self.dopplerview_folder = DV_folder
        self.h5_path = self.dopplerview_folder.get_h5_path()
        # Create an empty H5 file if it doesn't exist, and overwrite it if it does (to ensure a clean slate for each run)
        with h5py.File(self.h5_path, "w") as h5:
            pass
        cache_dir = Path(os.path.expanduser("~/.cache/dopplerview/cache")) / self.dopplerview_folder.measure_name
        self.cache_path = cache_dir / "cache.h5"

    def unset_DV_folder(self):
        self.close_output_folder()
        self.dopplerview_folder = None
        self.h5_path = None
        self.cache_path = None

    def ensure_output_folder(self):
        if self.dopplerview_folder is None:
            raise ValueError("DopplerView folder is not set. Cannot ensure output folder.")
        if self.output_dir is None:
            self.output_dir = self.dopplerview_folder.create_output_folder()
            self.write_dopplerview_config()
            self.write_version_file()
            logger.info(f"[OutputManager] Created output folder: {self.output_dir}")


    def close_output_folder(self):
        self.output_dir = None

    def output_cache(self, step_name, key, ctx, type=None):
        """Outputs a value from the cache for debugging purposes based on the provided output configuration."""
        if key not in self.output_config or not ctx.has(key):
            return
        
        if type is None:
            type = self.output_config[key]
        renderer = self.renderers.get(type)

        if renderer is None:
            logger.warning(f"No renderer found for output type '{type}' of key '{key}', skipping output.")
            return
        
        # Lasily create the output folder when we actually need to output something, to avoid creating empty output folders for runs that don't produce any outputs
        self.ensure_output_folder()

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        path = step_dir / f"{key}.png"

        renderer.render(key, ctx, path)
    
    def ensure_step_dir(self, step_name):
            # Lasily create the output folder when we actually need to output something, to avoid creating empty output folders for runs that don't produce any outputs
        self.ensure_output_folder()

        step_dir = self.output_dir / step_name
        step_dir.mkdir(exist_ok=True)

        return step_dir

    def output(self, step_name, filename, value, type=None, options=None):
        """Outputs a value manually for debugging purposes based on the provided output configuration."""
        if type is None:
            logger.warning(f"No output type specified for key '{step_name}', skipping debug output.")
            return
        
        renderer = self.renderers.get(type)
        if renderer is None:
            logger.warning(f"No renderer found for output type '{type}' of key '{step_name}', skipping output.")
            return

        step_dir = self.ensure_step_dir(step_name)

        path = step_dir / f"{filename}.png"

        renderer.render("value", {"value": value}, path, options=options)

    def save_async(self, step_name, key, ctx):
        self.output_queue.put((step_name, key, ctx)) 

    def cache_async(self, ctx, step_fingerprint, step_name):
        """Save the 'produced' cache values to disk, using a worker thread. The h5 file is lazily created when the first value is saved."""
        if self.cache_worker is None:
            self.cache_worker = threading.Thread(
                target=self._cache_worker,
                daemon=True
            )
            self.cache_worker.start()
            os.makedirs(self.cache_path.parent, exist_ok=True)
            logger.info(f"[OutputManager] Saving cache to {self.cache_path}")
        self.cache_queue.put((ctx, step_fingerprint, step_name))

    def save(self, step_name, key, ctx):
        self.save_h5(key, ctx)
        self.output_cache(step_name, key, ctx)

    def save_overlay(self, step_name, filename, image, artery_mask, vein_mask=None):
        step_dir = self.ensure_step_dir(step_name)
        path = step_dir / f"{filename}.png"

        img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if artery_mask is not None:
            if vein_mask is not None:
                img[artery_mask > 0] = [0, 0, 255]
            else:
                img[artery_mask > 0] = [255, 250, 250]

        if vein_mask is not None:
            img[vein_mask > 0] = [255, 0, 0]

        cv2.imwrite(str(path), img)

    def save_clusterization(self, step_name, filename, labels, z):
        plt.figure(figsize=(6,6))
        theta = np.linspace(0, 2*np.pi, 500)

        for lab in np.unique(labels):
            idx = labels == lab

            plt.scatter(
                np.real(z[idx]),
                np.imag(z[idx]),
                label=f'cluster {lab}'
            )

        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

        plt.axis('equal')
        plt.legend()
        plt.savefig(self.ensure_step_dir(step_name) / f"{filename}.png")
        plt.close()