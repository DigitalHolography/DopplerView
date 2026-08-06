from pathlib import Path
import os
import threading
import traceback
import time
from dopplerview.input_output import user_config, read_folder, h5_file
from dopplerview.models.registry import ModelRegistryConfig
import json
from typing import Any, Dict

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.pipeline.execution_profile import ExecutionProfile
from dopplerview.models.manager import ModelManager
from dopplerview.input_output.output_manager import OutputManager
from dopplerview.utils import json_utils
from dopplerview.utils.runtime_metrics import (
    RuntimeMetrics,
    process_snapshot,
    record_for_context,
)

from dopplerview.pipeline.steps.read_moments import ReadMomentsStep
from dopplerview.pipeline.steps.preprocess import PreprocessStep
from dopplerview.pipeline.steps.optic_disc import OpticDiscSegmentationStep
from dopplerview.pipeline.steps.eye_laterality_classification import EyeLateralityClassificationStep
from dopplerview.pipeline.steps.vessel_segmentation import RetinalVesselSegmentationStep, ChoroidalVesselSegmentationStep
from dopplerview.pipeline.steps.pulse_analysis import PulseAnalysisStep
from dopplerview.pipeline.steps.av_segmentation import ChoroidalAVSegmentationStep, RetinalAVSegmentationStep
from dopplerview.input_output.read_folder import DopplerViewFolder, HolodopplerFolder
from dopplerview.pipeline.steps.vessel_velocity_estimator import VesselVelocityEstimatorStep
from dopplerview.pipeline.steps.arterial_waveform_analysis import ArterialWaveformAnalysisStep

import logging
logger = logging.getLogger(__name__)
class Context:
    """
    Execution context shared across all steps.

    Holds:
        - runtime data (intermediate results)
        - configuration
        - services (models, output, etc.)
    """

    def __init__(self, output_manager, debug_mode=False, execution_profile=None):
        self.model_registry_path = None
        self.model_manager = None
        self.model_instances = {}
        self.metadata = {
            "step_hashes": {}
        }
        self.runtime_metrics = RuntimeMetrics()
        self.input_list = []

        self.measure_folder = None  # The measure folder containing the HD folder and the DV folder, set when loading input
        self.HD_folder = None       # The Holodoppler folder containing the raw input data, set when loading input
        self.DV_folder = None       # The DopplerView folder containing the output and cache, set when running the pipeline
        self.output_manager = output_manager
        self.debug_mode = debug_mode
        self.execution_profile = ExecutionProfile.resolve(execution_profile)
        self.dopplerview_config = None
        self.dopplerview_config_path = None
        self.DV_config_mode = "local"
        self.holodoppler_config = None

        # Runtime data storage
        self.__cache: Dict[str, Any] = {}

        self.lock = threading.RLock()
        self.cache_lock = threading.Lock()

    def _init_cache(self, initial_data: Dict[str, Any] = None):
        with self.lock:
            for key, value in (initial_data or {}).items():
                self.__cache[key] = (value, 'cached')

    def load_default_manager(self):
        models_config = user_config.ensure_config_file("models.yaml")
        logger.info(f"[Pipeline] Loading model registry from {models_config}")
        registry = ModelRegistryConfig(models_config)
        self.model_manager = ModelManager(registry, cache_dir="~/.cache/dopplerview/models")

    def load_manager(self, config_path):
        logger.info(f"[Pipeline] Loading model registry from {config_path}")
        self.model_registry_path = config_path
        registry = ModelRegistryConfig(config_path)
        self.model_manager = ModelManager(registry, cache_dir="~/.cache/dopplerview/models")

    def ensure_config(self):
        if self.model_manager is None:
            self.load_default_manager()

    def load_config(self, config_path):
        config = json.load(open(config_path))
        return json_utils.remove_spaces_from_keys(config)
    
    def set_config_mode(self, mode):
        if mode not in ["local", "default"]:
            raise ValueError(f"Invalid config mode: {mode}. Supported modes are 'local' and 'default'.")
        self.DV_config_mode = mode
    
    def load_dopplerview_config(self, config_path):
        self.dopplerview_config_path = config_path
        self.dopplerview_config = self.load_config(config_path)
        logger.info(f"[Pipeline] Using DopplerView config file: {config_path}")
    
    def load_holodoppler_config(self, config_path):
        self.holodoppler_config = self.load_config(config_path)
        logger.info(f"[Pipeline] Using Holodoppler config file: {config_path}")

    def _read_h5_into_cache(self):
        if self.DV_folder is None:
            raise RuntimeError("DopplerView folder not initialized. Cannot read from H5 cache.")
        h5_cache_path = self.output_manager.cache_path

        if not h5_cache_path.exists():
            logger.info(f"[Pipeline] No cache file found at {h5_cache_path}. Skipping cache loading.")
            return
        
        logger.info(f"[Pipeline] Reading cache from {h5_cache_path}")

        cache, metadata = h5_file.read_h5_to_dict(h5_cache_path)
        self._init_cache(cache)
        self.metadata.update(metadata)

    def ensure_directory(self, path):
        path = Path(path)
        if not os.path.isdir(path):
            extension = path.suffix
            if extension == ".holo":
                self.measure_name = path.stem
                return path.parent / self.measure_name
            raise NotADirectoryError(f"Expected a directory or .holo file, but got: {path}")
        return path

    def load_input_folder(self, folder_path):
        measure_folder = self.ensure_directory(folder_path)
        if measure_folder == self.measure_folder:
            logger.info(f"[Pipeline] Input folder already loaded: {measure_folder}")
            return
        self.clear()  # Clear cache before loading new input
        self.measure_folder = measure_folder

        self.output_manager.unset_DV_folder()  # Unset previous DV folder to avoid accidentally writing outputs to the wrong place if the new input doesn't have a DV folder
        self.load_DV_folder()
        self.output_manager.set_DV_folder(self.DV_folder)

        if self.debug_mode:
            self._read_h5_into_cache()

        self.HD_folder = HolodopplerFolder(self.measure_folder)
        logger.info(f"[Pipeline] Loading Holodoppler folder: {self.HD_folder.directory}")
        self.set("input_file", self.HD_folder.input_file)
        self.load_holodoppler_config(self.HD_folder.holodoppler_config)

    def load_DV_folder(self):
        if not self.measure_folder:
            raise RuntimeError("Measure folder not set. Cannot load DopplerView folder.")
        self.DV_folder = DopplerViewFolder(self.measure_folder)
        
        if self.DV_config_mode == "local":
            # Load configs from folder
            self.load_dopplerview_config(self.DV_folder.dopplerview_config)

    def extend_input_list(self, input_list):
        with self.lock:
            self.input_list.extend(input_list)

    def load_input_list_from_file(self, input_list):
        """ 
        Loads a list of input folders for batch processing. 
        The list can be provided as a text file (one folder path per line) or as a directory containing subdirectories for each input folder.
        """
        if not os.path.exists(input_list):
            raise FileNotFoundError(f"Folder list file not found: {input_list}")
        
        with open(input_list, "r") as f:
            for line in f:
                line = line.strip()
                if os.path.isdir(line) or (line.endswith(".holo") and os.path.isfile(line)):
                    self.input_list.append(line)
    
    def change_model_for_task(self, task_name: str, model_name: str):
        self.model_manager.change_task_model(task_name, model_name)

    def get_model(self, model_name):
        if model_name not in self.model_instances:
            spec, path = self.model_manager.resolve(model_name)
            model = ModelManager.build_model_wrapper(spec, path)
            self.model_instances[model_name] = model

        return self.model_instances[model_name]
    
    def get_current_model_name_for_task(self, task_name):
        return self.model_manager.get_current_model_name_for_task(task_name)
    
    def get_current_model_for_task(self, task_name):
        model_name = self.model_manager.get_current_model_name_for_task(task_name)
        return self.get_model(model_name)
    
    def create_output_folder(self):
        if self.DV_folder is None:
            self.load_DV_folder()

        self.output_manager.set_DV_folder(self.DV_folder)
        self.output_manager.set_dopplerview_config(self.dopplerview_config)
        self.output_manager.ensure_output_folder()  # Lazily create the output folder when we actually need to output something, to avoid creating empty output folders for runs that don't produce any outputs
        self.output_manager.write_app_versions(self.HD_folder.input_file)
    
    def start_output_manager(self):
        self.output_manager.start()

    def stop_output_manager(self):
        self.output_manager.close_workers()

    def set(self, key: str, value: Any):
        with self.lock:
            self.__cache[key] = (value, 'produced')

    def get(self, key: str):
        with self.lock:
            return self.__cache.get(key)[0] if key in self.__cache else None

    def has(self, key: str) -> bool:
        with self.lock:
            return key in self.__cache

    def require(self, key: str):
        with self.lock:
            if key not in self.__cache:
                raise RuntimeError(f"Missing required context key: '{key}'")
            return self.get(key)
        
    def clear_input_list(self):
        with self.lock:
            self.input_list = []

    def clear(self):
        with self.lock:
            self.__cache.clear()

    def is_empty(self):
        return self.__cache == {}
    
    def cache_value(self, key):
        with self.lock:
            self.__cache[key] = (self.__cache[key][0], 'cached')
    
    def cache_values(self, keys):
        for key in keys:
            if key in self.__cache and self.__cache[key][1] != 'cached':
                self.cache_value(key)

    def get_produced_values(self):
        with self.lock:
            return dict([(key, value) for key, (value, status) in self.__cache.items() if status == 'produced'])

    def get_number_of_workers(self, default=0.5):
        """Resolve an operation's workers without mutating scientific config."""
        configured = self.dopplerview_config.get("NumberOfWorkers", default)
        return self.execution_profile.operation_workers(configured)
    
    # def export_cache(self, filepath):
    #     cache = {}
    #     with self.cache_lock:
    #         for key, (value, status) in self.__cache.items():
    #             if status == 'produced':  # Only export values that were produced and not yet cached
    #                 cache[key] = value
    #                 self.cache_value(key)  # Mark as cached after exporting
    #     h5_file.write_dict_to_h5(cache, filepath, overwrite=False)

class Pipeline:
    def __init__(self, output_manager, debug_mode=False, execution_profile=None):
        """
        Initializes the pipeline with the given model registry and configuration.
        Args:
            model_registry: Configuration for available models.
            h5_schema: Schema defining how to store outputs in HDF5.
            dopplerview_config: DopplerView configuration dictionary (optional). If None, the dopplerview configuration found in the dopplerview folder will be used.
            debug_mode: If True, steps outputs are read from the .h5, and only targeted steps are re-run. This is useful for debugging specific steps without having to re-run the entire pipeline.
        """
        self.ctx = Context(
            output_manager=output_manager,
            debug_mode=debug_mode,
            execution_profile=execution_profile,
        )

        # Register steps
        self.steps = {
            ReadMomentsStep(),
            PreprocessStep(),
            EyeLateralityClassificationStep(),
            OpticDiscSegmentationStep(),
            RetinalVesselSegmentationStep(),
            ChoroidalVesselSegmentationStep(),
            PulseAnalysisStep(),
            RetinalAVSegmentationStep(),
            ChoroidalAVSegmentationStep(),
            VesselVelocityEstimatorStep(),
            ArterialWaveformAnalysisStep(),
        }

        self.engine = DAGEngine(
            self.steps,
            debug_mode=debug_mode,
            max_workers=self.ctx.execution_profile.dag_max_workers,
        )

    @property
    def execution_profile(self):
        return self.ctx.execution_profile

    def set_execution_profile(self, profile):
        resolved = ExecutionProfile.resolve(profile)
        self.ctx.execution_profile = resolved
        self.engine.max_workers = resolved.dag_max_workers

    def get_step_names(self):
        return self.engine.execution_order
    
    def is_cached(self, step_name):
        if self.ctx.is_empty():
            return False  # Cache not loaded, treat as not cached
        step = self.engine.steps[step_name]
        return self.engine._should_run(step, self.ctx) == False
    
    def resolve_execution_graph(self, targets=None):
        if targets == []:
            return []

        if targets is None:
            return self.engine.execution_order

        required_steps = self.engine._resolve_required_steps(targets)
        return required_steps
    
    def get_downstream_steps(self, step_name):
        return self.engine._collect_downstream(step_name)

    def load_dopplerview_config(self, config_path):
        self.ctx.load_dopplerview_config(config_path)

    def load_input(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path not found: {input_path}")
        if os.path.isdir(input_path):
            self.load_batch_folder(input_path)
        elif input_path.suffix == ".txt":
            self.ctx.load_input_list_from_file(input_path)
        elif os.path.isfile(input_path) and input_path.suffix == ".holo":
            self.ctx.extend_input_list([input_path])

    def load_input_list_from_file(self, folder_list_path):
        self.ctx.load_input_list_from_file(folder_list_path)

    def load_input_list_from_list(self, input_list):
        for input in input_list:
            self.load_input(input)

    def load_batch_folder(self, folder_path):
        holo_files = read_folder.search_holo_files(folder_path)
        logger.info(f"[Pipeline] Found {len(holo_files)} .holo files in {folder_path} for batch processing: {holo_files}")
        if len(holo_files) == 0:
            raise FileNotFoundError(f"No .holo file found in {folder_path}")
        self.ctx.extend_input_list(holo_files)

    def load_model_registry(self, config_path):
        self.ctx.load_manager(config_path)
    
    def load_h5_schema(self, config_path):
        self.ctx.load_h5_schema(config_path)

    def load_output_config(self, config_path):
        self.ctx.load_output_config(config_path)

    def set_targets(self, targets):
        self.engine.set_targets(targets)

    def set_config_mode(self, mode):
        self.ctx.set_config_mode(mode)

    def run(self, targets=None, callback=None):
        if not self.ctx.has("input_file"):
            raise RuntimeError("Input path not set. Please load input folder before running the pipeline.")
        if self.ctx.dopplerview_config is None:
            raise RuntimeError("Configuration not loaded. Please load a configuration file before running the pipeline.")
        
        self.ctx.ensure_config()

        logger.info(
            "[Pipeline] Execution profile: %s (DAG workers: %s, operation workers: %s)",
            self.execution_profile.value,
            self.engine.max_workers if self.engine.max_workers is not None else "automatic",
            self.ctx.get_number_of_workers(),
        )

        self.ctx.create_output_folder()
        self.ctx.start_output_manager()

        start_time = time.perf_counter()
        run_start = process_snapshot()
        status = "failed"
        try:
            self.engine.run(self.ctx, targets, callback=callback)
            status = "success"
        finally:
            elapsed = time.perf_counter() - start_time
            run_end = process_snapshot()
            record_for_context(
                self.ctx,
                "pipeline",
                status=status,
                duration_s=elapsed,
                process_rss_start_mb=run_start["rss_mb"],
                process_rss_end_mb=run_end["rss_mb"],
                process_rss_delta_mb=run_end["rss_mb"] - run_start["rss_mb"],
                process_threads_start=run_start["process_threads"],
                process_threads_end=run_end["process_threads"],
            )
        logger.info(f"[Pipeline] Finished execution in {elapsed:.2f}s")

        # # If in debug mode, save the entire cache to the H5 file after execution
        # if self.ctx.debug_mode:
        #     logger.info(f"[Pipeline] Saving cache to H5 file.")
        #     self.ctx.output_manager.save_cache(self.ctx)
        self.ctx.stop_output_manager()
        return self.ctx

    def run_batch(self, targets=None, callback=None):
        batch_started = time.perf_counter()
        if callback:
            callback("batch_start", len(self.ctx.input_list))

        results = []
        total = len(self.ctx.input_list)

        for i, input_path in enumerate(self.ctx.input_list):
            logger.info("[Run Batch] Processing file: %s", input_path)

            if callback:
                callback("pipeline_start", i, total)

            try:
                self.ctx.load_input_folder(input_path)
                self.run(targets=targets, callback=callback)

            except Exception:
                error_text = traceback.format_exc()

                logger.error(
                    "[Run Batch] Failed processing file %s:\n%s",
                    input_path,
                    error_text,
                )

                results.append({
                    "input": str(input_path),
                    "status": "failed",
                    "error": error_text,
                })

                if callback:
                    callback("pipeline_failed", i, total, str(input_path), error_text)

                continue

            results.append({
                "input": str(input_path),
                "status": "success",
                "error": None,
            })

            if callback:
                callback("pipeline_done", i, total)

        if callback:
            callback("batch_done", results)

        succeeded = sum(result["status"] == "success" for result in results)
        failed = len(results) - succeeded
        metrics = getattr(self.ctx, "runtime_metrics", None)
        if metrics is not None:
            metrics.record(
                "batch",
                duration_s=time.perf_counter() - batch_started,
                inputs=total,
                succeeded=succeeded,
                failed=failed,
            )

        return results
