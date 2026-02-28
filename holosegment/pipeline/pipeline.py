from holosegment.pipeline.dag import DAGEngine
from holosegment.models.manager import ModelManager
from holosegment.models.builder import build_model_wrapper
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

class Context:
    """
    Execution context shared across all steps.

    Holds:
        - runtime data (intermediate results)
        - configuration
        - services (models, output, etc.)
    """


    def __init__(self, config, model_manager, output_manager):
        self.config = config
        self.output_manager = output_manager
        self.model_manager = model_manager
        self.model_instances = {}
        self.metadata = {
            "step_hashes": {}
        }

        # Runtime data storage
        self.cache: Dict[str, Any] = {}

    def load_config(self, config_path):
        self.config = json.load(open(config_path))

    def get(self, key: str):
        return self.cache.get(key)
    
    def change_model_for_task(self, task_name: str, model_name: str):
        self.model_manager.change_task_model(task_name, model_name)

    def get_model(self, model_name):
        if model_name not in self.model_instances:
            spec, path = self.model_manager.resolve(model_name)
            model = build_model_wrapper(spec, path)
            self.model_instances[model_name] = model

        return self.model_instances[model_name]
    
    def get_current_model_for_task(self, task_name):
        model_name = self.model_manager.get_current_model_name_for_task(task_name)
        return self.get_model(model_name)

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
    def __init__(self, model_registry, config=None, output_dir=None, debug=False):
        self.ctx = Context(
            config=config,
            model_manager=ModelManager(model_registry),
            output_manager=OutputManager(output_dir=output_dir, enabled=debug)
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

    

    def load_config(self, config_path):
        self.ctx.load_config(config_path)

    def load_input(self, input_path):
        self.ctx.set("input_path", input_path)

    def run(self, targets=None):
        if not self.ctx.has("input_path"):
            raise RuntimeError("Input path not set. Please load input folder before running the pipeline.")
        if self.ctx.config is None:
            raise RuntimeError("Configuration not loaded. Please load a configuration file before running the pipeline.")
        self.engine.run(self.ctx, targets)
        return self.ctx.cache

    # def run_all(self, input_path):
    #     self.ctx.cache["input_path"] = input_path

    #     for name in self.steps:
    #         self.run_step(name)

    #     return (
    #         self.ctx.cache.get("artery_mask"),
    #         self.ctx.cache.get("vein_mask"),
    #     )

    # def run_step(self, step_name):
    #     step = self.steps[step_name]

    #     # Check dependencies
    #     for dep in getattr(step, "requires", []):
    #         if dep not in self.ctx.cache:
    #             raise RuntimeError(
    #                 f"Step '{step_name}' requires '{dep}' but it is missing."
    #             )

    #     print(f"Running step: {step_name}")
    #     step.run(self.ctx)

    # def run_from(self, step_name):
    #     run = False
    #     for name in self.steps:
    #         if name == step_name:
    #             run = True
    #         if run:
    #             self.run_step(name)
