from pathlib import Path
import os
import time
from dopplerview.input_output import user_config
from dopplerview.models.registry import ModelRegistryConfig
import h5py
import json
from typing import Any, Dict

from dopplerview.pipeline.dag import DAGEngine
from dopplerview.models.manager import ModelManager
from dopplerview.input_output.output_manager import OutputManager
from dopplerview.utils import json_utils

from dopplerview.pipeline.steps.read_moments import ReadMomentsStep
from dopplerview.pipeline.steps.preprocess import PreprocessStep
from dopplerview.pipeline.steps.optic_disc import OpticDiscDetectionStep
from dopplerview.pipeline.steps.vessel_segmentation import RetinalVesselSegmentationStep, ChoroidalVesselSegmentationStep
from dopplerview.pipeline.steps.pulse_analysis import PulseAnalysisStep
from dopplerview.pipeline.steps.av_segmentation import AVSegmentationStep
from dopplerview.input_output.read_folder import DopplerViewFolder, HolodopplerFolder
from dopplerview.pipeline.steps.vessel_velocity_estimator import VesselVelocityEstimatorStep
from dopplerview.pipeline.steps.arterial_waveform_analysis import ArterialWaveformAnalysisStep

class Context:
    """
    Execution context shared across all steps.

    Holds:
        - runtime data (intermediate results)
        - configuration
        - services (models, output, etc.)
    """

    def __init__(self, debug_mode=False):
        self.model_manager = None
        self.model_instances = {}
        self.metadata = {
            "step_hashes": {}
        }
        self.input_folder_list = []

        self.measure_folder = None  # The measure folder containing the HD folder and the DV folder, set when loading input
        self.HD_folder = None       # The Holodoppler folder containing the raw input data, set when loading input
        self.DV_folder = None       # The DopplerView folder containing the output and cache, set when running the pipeline
        self.output_manager = None
        self.h5_schema = None
        self.output_config = None
        self.debug_mode = debug_mode
        self.dopplerview_config = None
        self.holodoppler_config = None

        # Runtime data storage
        self.cache: Dict[str, Any] = {}

    def load_default_manager(self):
        models_config = user_config.ensure_config_file("models.yaml")
        print(f"[Pipeline] Loading model registry from {models_config}")
        registry = ModelRegistryConfig(models_config)
        self.model_manager = ModelManager(registry, cache_dir="~/.cache/dopplerview/models")

    def load_manager(self, config_path):
        print(f"[Pipeline] Loading model registry from {config_path}")
        registry = ModelRegistryConfig(config_path)
        self.model_manager = ModelManager(registry, cache_dir="~/.cache/dopplerview/models")
        
    def load_default_h5_schema(self):
        h5_schema_config = user_config.ensure_config_file("h5_schema.json")
        print(f"[Pipeline] Loading default H5 schema from {h5_schema_config}")
        self.h5_schema = json.load(open(h5_schema_config))

    def load_h5_schema(self, config_path):
        print(f"[Pipeline] Loading H5 schema from {config_path}")
        self.h5_schema = json.load(open(config_path))

    def load_default_output_config(self):
        output_config = user_config.ensure_config_file("output_config.json")
        print(f"[Pipeline] Loading default output config from {output_config}")
        self.output_config = json.load(open(output_config))

    def load_output_config(self, config_path):
        print(f"[Pipeline] Loading output config from {config_path}")
        self.output_config = json.load(open(config_path))

    def ensure_config(self):
        if self.model_manager is None:
            self.load_default_manager()
        if self.h5_schema is None:
            self.load_default_h5_schema()
        if self.output_config is None:
            self.load_default_output_config()

    def load_config(self, config_path):
        config = json.load(open(config_path))
        return json_utils.remove_spaces_from_keys(config)
    
    def load_dopplerview_config(self, config_path):
        self.dopplerview_config = self.load_config(config_path)
        print(f"[Pipeline] Using DopplerView config file: {config_path}")
    
    def load_holodoppler_config(self, config_path):
        self.holodoppler_config = self.load_config(config_path)
        print(f"[Pipeline] Using Holodoppler config file: {config_path}")

    def _read_h5_into_cache(self):
        if self.DV_folder is None:
            raise RuntimeError("DopplerView folder not initialized. Cannot read from H5 cache.")
        cache_folder = self.DV_folder.cache_folder
        h5_cache_path = cache_folder / "cache.h5"

        if not h5_cache_path.exists():
            print(f"[Pipeline] No cache file found at {h5_cache_path}. Skipping cache loading.")
            return
        
        print(f"[Pipeline] Reading cache from {h5_cache_path}")
        with h5py.File(h5_cache_path, "r") as input_file:
            for key in input_file.keys():
                self.cache[key] = input_file[key][()]

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
        self.clear()  # Clear cache before loading new input
        self.measure_folder = self.ensure_directory(folder_path)

        self.HD_folder = HolodopplerFolder(self.measure_folder)
        self.cache["input_file"] = self.HD_folder.input_file
        self.load_holodoppler_config(self.HD_folder.holodoppler_config)

        self.load_DV_folder()

        if self.debug_mode:
            self._read_h5_into_cache()

    def load_DV_folder(self):
        if not self.measure_folder:
            raise RuntimeError("Measure folder not set. Cannot load DopplerView folder.")
        self.DV_folder = DopplerViewFolder(self.measure_folder)
        
        if self.dopplerview_config is None:
            # Load configs from folder if not already loaded
            self.load_dopplerview_config(self.DV_folder.dopplerview_config)

    def load_folder_list(self, folder_list_path):
        if not os.path.exists(folder_list_path):
            raise FileNotFoundError(f"Folder list file not found: {folder_list_path}")
        
        if os.path.isfile(folder_list_path):
            with open(folder_list_path, "r") as f:
                self.input_folder_list = [line.strip() for line in f.readlines()]
        elif os.path.isdir(folder_list_path):
            # Load all subdirectories as folders
            subdirs = [d for d in os.listdir(folder_list_path) if os.path.isdir(os.path.join(folder_list_path, d))]
            self.input_folder_list = [Path(os.path.join(folder_list_path, d)) for d in subdirs]

    def get(self, key: str):
        return self.cache.get(key)
    
    def change_model_for_task(self, task_name: str, model_name: str):
        self.model_manager.change_task_model(task_name, model_name)

    def get_model(self, model_name):
        if model_name not in self.model_instances:
            spec, path = self.model_manager.resolve(model_name)
            model = ModelManager.build_model_wrapper(spec, path)
            self.model_instances[model_name] = model

        return self.model_instances[model_name]
    
    def get_current_model_for_task(self, task_name):
        model_name = self.model_manager.get_current_model_name_for_task(task_name)
        return self.get_model(model_name)
    
    def create_output_folder(self):
        if self.DV_folder is None:
            self.load_DV_folder()

        # Create the output manager. It will lazily create the output folder when needed, to avoid creating empty output folders for runs that don't produce any outputs
        self.output_manager = OutputManager(dopplerview_folder=self.DV_folder, schema=self.h5_schema, dopplerview_config=self.dopplerview_config, output_config=self.output_config)


    def set(self, key: str, value: Any):
        self.cache[key] = value

    def has(self, key: str) -> bool:
        return key in self.cache

    def require(self, key: str):
        if key not in self.cache:
            raise RuntimeError(f"Missing required context key: '{key}'")
        return self.cache[key]

    def clear(self):
        self.cache.clear()

class Pipeline:
    def __init__(self, debug_mode=False):
        """
        Initializes the pipeline with the given model registry and configuration.
        Args:
            model_registry: Configuration for available models.
            h5_schema: Schema defining how to store outputs in HDF5.
            output_config: Configuration for debug outputs (optional). If None, outputs are manually saved.
            dopplerview_config: DopplerView configuration dictionary (optional). If None, the dopplerview configuration found in the dopplerview folder will be used.
            debug_mode: If True, steps outputs are read from the .h5, and only targeted steps are re-run. This is useful for debugging specific steps without having to re-run the entire pipeline.
        """
        self.ctx = Context(
            debug_mode=debug_mode
        )

        # Register steps
        self.steps = {
            ReadMomentsStep(),
            PreprocessStep(),
            OpticDiscDetectionStep(),
            RetinalVesselSegmentationStep(),
            ChoroidalVesselSegmentationStep(),
            PulseAnalysisStep(),
            AVSegmentationStep(),
            VesselVelocityEstimatorStep(),
            ArterialWaveformAnalysisStep(),
        }

        self.engine = DAGEngine(self.steps, debug_mode=debug_mode)

    def get_step_names(self):
        return self.engine.execution_order
    
    def is_cached(self, step_name):
        if self.ctx.cache == {}:
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
        self.ctx.load_input_folder(input_path)

    def load_folder_list(self, folder_list_path):
        self.ctx.load_folder_list(folder_list_path)

    def load_model_registry(self, config_path):
        self.ctx.load_manager(config_path)
    
    def load_h5_schema(self, config_path):
        self.ctx.load_h5_schema(config_path)

    def load_output_config(self, config_path):
        self.ctx.load_output_config(config_path)

    def set_targets(self, targets):
        self.engine.set_targets(targets)

    def run(self, targets=None, callback=None):
        if not self.ctx.has("input_file"):
            raise RuntimeError("Input path not set. Please load input folder before running the pipeline.")
        if self.ctx.dopplerview_config is None:
            raise RuntimeError("Configuration not loaded. Please load a configuration file before running the pipeline.")
        
        self.ctx.ensure_config()

        self.ctx.create_output_folder()
        print(f"[Pipeline] Created output folder: {self.ctx.output_manager.output_dir}")

        start_time = time.time()
        self.engine.run(self.ctx, targets, callback=callback)
        elapsed = time.time() - start_time
        print(f"[Pipeline] Finished execution in {elapsed:.2f}s")

        # If in debug mode, save the entire cache to the H5 file after execution
        if self.ctx.debug_mode:
            print(f"[Pipeline] Saving cache to H5 file.")
            self.ctx.output_manager.save_cache(self.ctx.cache)
        return self.ctx.cache

    def run_batch(self, targets=None, callback=None):
        for folder in self.ctx.input_folder_list:
            try:
                print(f"[Run Batch] Processing folder: {folder}")
                self.load_input(folder)
                self.run(targets=targets, callback=callback)
            except Exception as e:
                print(f"[Run Batch] Error processing folder {folder}: {e}")