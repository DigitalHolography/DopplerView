from holosegment.pipeline import step
from holosegment.pipeline.dag import DAGEngine
from holosegment.models.manager import ModelManager
from holosegment.input_output.output_manager import OutputManager
from typing import Any, Dict
import json
from holosegment.utils import json_utils
from pathlib import Path
import os


from holosegment.pipeline.steps.load_moments import LoadMomentsStep
from holosegment.pipeline.steps.preprocess import PreprocessStep
from holosegment.pipeline.steps.optic_disc import OpticDiscDetectionStep
from holosegment.pipeline.steps.vessel_segmentation import RetinalVesselSegmentationStep, ChoroidalVesselSegmentationStep
from holosegment.pipeline.steps.pulse_analysis import PulseAnalysisStep
from holosegment.pipeline.steps.av_segmentation import AVSegmentationStep
from holosegment.input_output.read_folder import HolodopplerFolder

class Context:
    """
    Execution context shared across all steps.

    Holds:
        - runtime data (intermediate results)
        - configuration
        - services (models, output, etc.)
    """

    def __init__(self, eyeflow_config, model_manager, h5_schema, debug_config=None):
        self.eyeflow_config = eyeflow_config
        self.model_manager = model_manager
        self.model_instances = {}
        self.metadata = {
            "step_hashes": {}
        }
        self.input_folder_list = []
        self.folder = None
        self.output_manager = None
        self.h5_schema = h5_schema
        self.debug_config = debug_config or {}

        # Runtime data storage
        self.cache: Dict[str, Any] = {}

    def load_eyeflow_config(self, config_path):
        eyeflow_config = json.load(open(config_path))
        self.eyeflow_config = json_utils.remove_spaces_from_keys(eyeflow_config) 
        print(f"Using Eyeflow config file: {config_path}")

    def load_input_folder(self, folder_path):
        self.folder = HolodopplerFolder(folder_path)
        self.cache["h5_file"] = self.folder.h5_file
        self.holodoppler_config = json.load(open(self.folder.holodoppler_config))
        print(f"Using Holodoppler config file: {self.folder.holodoppler_config}")

        if self.eyeflow_config is None:
            # Load configs from folder if not already loaded
            self.load_eyeflow_config(self.folder.eyeflow_config)

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
    
    def create_output_folder(self, debug=True):
        if self.folder is None:
            raise RuntimeError("Input folder not loaded. Cannot determine output folder.")
        # Create a new output folder with an incremented index
        self.output_manager = OutputManager(output_folder=self.folder.create_output_folder(), h5_path=self.folder.h5_file, schema=self.h5_schema, debug_config=self.debug_config)

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
    def __init__(self, model_registry, h5_schema, debug_config=None, eyeflow_config=None):
        """
        Initializes the pipeline with the given model registry and configuration.
        Args:
            model_registry: Configuration for available models.
            h5_schema: Schema defining how to store outputs in HDF5.
            debug_config: Configuration for debug outputs (optional). If None, outputs are manually saved.
            eyeflow_config: Eyeflow configuration dictionary (optional) If None, the eyeflow configuration found in the input folder will be used.
        """
        self.ctx = Context(
            eyeflow_config=eyeflow_config,
            model_manager=ModelManager(model_registry),
            h5_schema=h5_schema,
            debug_config=debug_config
        )

        # Register steps
        self.steps = {
            LoadMomentsStep(),
            PreprocessStep(),
            OpticDiscDetectionStep(),
            RetinalVesselSegmentationStep(),
            ChoroidalVesselSegmentationStep(),
            PulseAnalysisStep(),
            AVSegmentationStep(),
        }

        self.engine = DAGEngine(self.steps)

    def get_step_names(self):
        return self.engine.execution_order
    
    def resolve_execution_graph(self, targets=None):
        if targets == []:
            return []

        if targets is None:
            return self.engine.execution_order

        required_steps = self.engine._resolve_required_steps(targets)
        return required_steps
    
    def get_downstream_steps(self, step_name):
        return self.engine._collect_downstream(step_name)

    def load_eyeflow_config(self, config_path):
        self.ctx.load_eyeflow_config(config_path)

    def load_input(self, input_path):
        self.ctx.load_input_folder(input_path)

    def load_folder_list(self, folder_list_path):
        self.ctx.load_folder_list(folder_list_path)

    def run(self, targets=None, debug=True):
        if not self.ctx.has("h5_file"):
            raise RuntimeError("Input path not set. Please load input folder before running the pipeline.")
        if self.ctx.eyeflow_config is None:
            raise RuntimeError("Configuration not loaded. Please load a configuration file before running the pipeline.")
        self.ctx.create_output_folder(debug=debug)
        self.engine.run(self.ctx, targets)
        return self.ctx.cache
    
    def run_batch(self, targets=None, debug=True):
        for folder in self.ctx.input_folder_list:
            try:
                print(f"Processing folder: {folder}")
                self.load_input(folder)
                self.run(targets=targets, debug=debug)
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
