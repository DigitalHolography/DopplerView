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
from dopplerview.utils.runtime_metrics import emit_metric
logger = logging.getLogger(__name__)

class OutputManager:
    def __init__(
        self,
        schema_path,
        output_config_path,
        output_enabled=True,
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

        self.output_enabled = output_enabled

    def __del__(self):
        try:
            self.close_workers()
        except Exception:
            # Destructors must not surface errors during interpreter shutdown.
            pass

    def start(self):
        if self.running and self.output_worker is not None and self.output_worker.is_alive():
            return

        self.running = True
        self.output_worker = threading.Thread(
            target=self._output_worker,
            name="dopplerview-output-writer",
            daemon=True
        )
        self.output_worker.start()
        emit_metric(
            "output_workers",
            action="start",
            output_queue_depth=self.output_queue.qsize(),
            cache_queue_depth=self.cache_queue.qsize(),
        )

    def close_workers(self):
        output_worker = getattr(self, "output_worker", None)
        cache_worker = getattr(self, "cache_worker", None)

        # Stop messages are queued behind pending work, so join() also flushes
        # every item accepted before shutdown. Never poison an unstarted queue.
        if output_worker is not None and output_worker.is_alive():
            self.output_queue.put((None, None, None))
        if cache_worker is not None and cache_worker.is_alive():
            self.cache_queue.put((None, None, None))

        if output_worker is not None and output_worker.is_alive():
            output_worker.join()
        if cache_worker is not None and cache_worker.is_alive():
            cache_worker.join()

        self.output_worker = None
        self.cache_worker = None
        self.running = False
        emit_metric(
            "output_workers",
            action="stop",
            output_queue_depth=self.output_queue.qsize(),
            cache_queue_depth=self.cache_queue.qsize(),
        )

    def _output_worker(self):
        while True:
            step_name, key, ctx = self.output_queue.get()
            if step_name is None and key is None and ctx is None:
                self.output_queue.task_done()
                break
            try:
                self.save(step_name, key, ctx)
            except Exception as e:
                logger.exception(f"Error saving output for step '{step_name}' and key '{key}': {e}")
            self.output_queue.task_done()

    def _cache_worker(self):
        """Worker thread that saves the cache to disk asynchronously. It listens on the cache_queue for contexts to save, and calls save_cache on them.
        """
        while True:
            ctx, step_fingerprint, step_name = self.cache_queue.get()
            if ctx is None:
                self.cache_queue.task_done()
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
                logger.exception(f"Error saving cache: {e} for step '{step_name}' with fingerprint '{step_fingerprint}'")
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

    def write_app_versions(self, input_h5_path):
        """Write the Holodoppler and DopplerView versions to the output H5 file."""
        with h5py.File(input_h5_path, "r") as input_h5:
            if "HD_version" not in input_h5:
                raise KeyError(f"HD_version not found in Holodoppler H5 file: {input_h5_path}")

            hd_version = input_h5["HD_version"][()]
            if isinstance(hd_version, bytes):
                hd_version = hd_version.decode("utf-8")
            else:
                hd_version = str(hd_version)

        with h5py.File(self.h5_path, "a") as h5:
            if "app_versions" in h5:
                del h5["app_versions"]

            app_versions = json.dumps({
                "HD_version": hd_version,
                "DV_version": __version__,
            })
            string_dtype = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("app_versions", data=app_versions, dtype=string_dtype)

        logger.info(
            "[OutputManager] Wrote app versions to %s: HD=%s, DV=%s",
            self.h5_path,
            hd_version,
            __version__,
        )

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
        if not self.output_enabled:
            return
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
        if not self.output_enabled:
            return
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
        if not self.running:
            raise RuntimeError("Output manager is not running.")
        self.output_queue.put((step_name, key, ctx))
        emit_metric(
            "output_queue",
            action="enqueue",
            step=step_name,
            key=key,
            queue_depth=self.output_queue.qsize(),
        )

    def cache_async(self, ctx, step_fingerprint, step_name):
        """Save the 'produced' cache values to disk, using a worker thread. The h5 file is lazily created when the first value is saved."""
        if not self.running:
            raise RuntimeError("Output manager is not running.")
        if self.cache_worker is None or not self.cache_worker.is_alive():
            self.cache_worker = threading.Thread(
                target=self._cache_worker,
                name="dopplerview-cache-writer",
                daemon=True
            )
            self.cache_worker.start()
            os.makedirs(self.cache_path.parent, exist_ok=True)
            logger.info(f"[OutputManager] Saving cache to {self.cache_path}")
        self.cache_queue.put((ctx, step_fingerprint, step_name))
        emit_metric(
            "cache_queue",
            action="enqueue",
            step=step_name,
            queue_depth=self.cache_queue.qsize(),
        )

    def save(self, step_name, key, ctx):
        if ctx.get(key) is None:
            logger.warning(f"Value for key '{key}' is None, skipping save.")
            return
        self.save_h5(key, ctx)
        if self.output_enabled:
            self.output_cache(step_name, key, ctx)

    def enable_output(self):
        self.output_enabled = True
    
    def disable_output(self):
        self.output_enabled = False

    def save_overlay(self, step_name, filename, image, masks, colors=[(0, 0, 255), (255, 0, 0)], artery_mask=None, vein_mask=None):
        if not self.output_enabled:
            return
        step_dir = self.ensure_step_dir(step_name)
        path = step_dir / f"{filename}.png"

        img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            img[mask > 0] = color

        cv2.imwrite(str(path), img)

    def save_clusterization(self, step_name, filename, labels, z):
        if not self.output_enabled:
            return
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


    def save_optic_disc_detections(self, step_name, filename, boxes, scale_x, scale_y, ctx):
        if not self.output_enabled:
            return
        # Image on which to draw
        img = ctx.get("M0_ff_image")

        # Convert grayscale to RGB for visualization
        if img.ndim == 2:
            vis = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            vis = img.copy()

        candidates = boxes[0, :4, :].T.tolist()
        scores = boxes[0, 4, :].tolist()
        nb_boxes = min(len(candidates), 10)  # Limit to first 20 boxes for visualization
        sorted_scores = sorted(scores, reverse=True)
        score_treshold = sorted_scores[nb_boxes]
        indices = cv2.dnn.NMSBoxes(
            bboxes=candidates,
            scores=scores,
            score_threshold=score_treshold,
            nms_threshold=0.4,
        )

        for j in indices:
            x, y, w_box, h_box = candidates[j]
            score = scores[j]

            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + w_box) * scale_x))
            y2 = int(round((y + h_box) * scale_y))

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{score:.3f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Highlight the selected box
        best = boxes[:, :, np.argmax(boxes[:, 4, :])].flatten()
        xc = best[0] * scale_x
        yc = best[1] * scale_y
        bw = best[2] * scale_x
        bh = best[3] * scale_y

        cv2.rectangle(
            vis,
            (int(xc), int(yc)),
            (int(xc + bw), int(yc + bh)),
            (255, 255, 0),   # cyan
            3,
        )

        cv2.imwrite(self.ensure_step_dir(step_name) / f"{filename}.png", vis)
