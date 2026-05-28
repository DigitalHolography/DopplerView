import json

import cv2
import h5py
import dopplerview.utils.json_utils as json_utils
import dopplerview.input_output.output_renderer as output_renderer
import matplotlib.pyplot as plt
import numpy as np

import queue
import threading

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
        self.output_dir = None # It will be created when needed
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

        self.running = True

        self.output_queue = queue.Queue()
        self.output_worker = threading.Thread(target=self._output_worker, daemon=True)
        self.output_worker.start()

        self.cache_queue = queue.Queue()
        self.cache_worker = threading.Thread(target=self._cache_worker, daemon=True)



    def _output_worker(self):
        while self.running:
            step_name, key, ctx = self.output_queue.get()
            try:
                self.save(step_name, key, ctx)
            except Exception as e:
                print(f"Error saving output for step '{step_name}' and key '{key}': {e}")
            self.output_queue.task_done()

    def _cache_worker(self):
        while self.running:
            ctx = self.cache_queue.get()
            try:
                self.save_cache(ctx)
            except Exception as e:
                print(f"Error saving cache: {e}")
            self.cache_queue.task_done()

    def close(self):
        self.running = False
        self.output_queue.put((None, None, None))  # Unblock the output worker
        self.output_worker.join()
        if self.cache_worker.is_alive():
            self.cache_queue.put(None)  # Unblock the cache worker
            self.cache_worker.join()

    def save_cache(self, ctx):
        """Saves the entire cache to the H5 file"""
        filepath = self.cache_dir / "cache.h5"
        ctx.export_cache(filepath)

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

    def ensure_output_folder(self):
        if self.output_dir is None:
            self.output_dir = self.dopplerview_folder.create_output_folder()
            self.write_dopplerview_config()

    def output_cache(self, step_name, key, ctx, type=None):
        """Outputs a value from the cache for debugging purposes based on the provided output configuration."""
        if key not in self.output_config or not ctx.has(key):
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
            Warning(f"No output type specified for key '{step_name}', skipping debug output.")
            return
        
        renderer = self.renderers.get(type)
        if renderer is None:
            Warning(f"No renderer found for output type '{type}' of key '{step_name}', skipping output.")
            return

        step_dir = self.ensure_step_dir(step_name)

        path = step_dir / f"{filename}.png"

        renderer.render("value", {"value": value}, path, options=options)

    def save_async(self, step_name, key, ctx):
        self.output_queue.put((step_name, key, ctx))

    def cache_async(self, ctx):
        self.cache_queue.put(ctx)

    def save(self, step_name, key, ctx):
        self.save_h5(key, ctx)
        self.output_cache(step_name, key, ctx)

    def save_overlay(self, step_name, filename, image, artery_mask, vein_mask):
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