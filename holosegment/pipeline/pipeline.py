from holosegment.pipeline.dag import DAGEngine
from holosegment.models.manager import ModelManager
from holosegment.input_output.output_manager import OutputManager
from typing import Any, Dict
import json
from holosegment.utils.json_utils import ordered
from pathlib import Path


from holosegment.pipeline.steps.load_moments import LoadMomentsStep
from holosegment.pipeline.steps.preprocess import PreprocessStep
from holosegment.pipeline.steps.optic_disc import OpticDiscDetectionStep
from holosegment.pipeline.steps.vessel_segmentation import VesselSegmentation
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

    def __init__(self, config, model_manager):
        self.eyeflow_config = config
        self.model_manager = model_manager
        self.model_instances = {}
        self.metadata = {
            "step_hashes": {}
        }
        self.folder = None
        self.output_manager = None

        # Runtime data storage
        self.cache: Dict[str, Any] = {}

    def load_eyeflow_config(self, config_path):
        self.eyeflow_config = json.load(open(config_path))
        print(f"Using Eyeflow config file: {config_path}")


    def load_input_folder(self, folder_path):
        self.folder = HolodopplerFolder(folder_path)
        self.cache["input_path"] = folder_path
        self.holodoppler_config = json.load(open(self.folder.holodoppler_config))
        print(f"Using Holodoppler config file: {self.folder.holodoppler_config}")

        if self.eyeflow_config is None:
            # Load configs from folder if not already loaded
            self.load_eyeflow_config(self.folder.eyeflow_config)

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
        self.output_manager = OutputManager(output_dir=self.folder.create_output_folder(), enabled=debug)

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
    def __init__(self, model_registry, config=None):
        self.ctx = Context(
            config=config,
            model_manager=ModelManager(model_registry),
        )

        # Register steps
        self.steps = {
            LoadMomentsStep(),
            PreprocessStep(),
            OpticDiscDetectionStep(),
            VesselSegmentation(),
            PulseAnalysisStep(),
            AVSegmentationStep(),
        }

        self.engine = DAGEngine(self.steps)

    def load_eyeflow_config(self, config_path):
        self.ctx.load_eyeflow_config(config_path)

    def load_input(self, input_path):
        self.ctx.load_input_folder(input_path)

    def run(self, targets=None, debug=True):
        if not self.ctx.has("input_path"):
            raise RuntimeError("Input path not set. Please load input folder before running the pipeline.")
        if self.ctx.eyeflow_config is None:
            raise RuntimeError("Configuration not loaded. Please load a configuration file before running the pipeline.")
        self.ctx.create_output_folder(debug=debug)
        self.engine.run(self.ctx, targets)
        return self.ctx.cache
