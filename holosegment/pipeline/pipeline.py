from holosegment.pipeline.dag import DAGEngine
from holosegment.models.manager import ModelManager
from holosegment.models.builder import build_model_wrapper
from holosegment.pipeline.output_manager import OutputManager
from typing import Any, Dict

from holosegment.steps.load_moments import LoadMomentsStep
from holosegment.steps.preprocess import PreprocessStep
from holosegment.steps.optic_disc import OpticDiscDetectionStep
from holosegment.steps.binary_segmentation import BinarySegmentationStep
from holosegment.steps.pulse_analysis import PulseAnalysisStep
from holosegment.steps.av_segmentation import AVSegmentationStep

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

        # Runtime data storage
        self.cache: Dict[str, Any] = {}

    def get(self, key: str):
        return self.cache.get(key)

    def get_model(self, model_name):
        if model_name not in self.model_instances:
            spec, path = self.model_manager.resolve(model_name)
            model = build_model_wrapper(spec, path)
            self.model_instances[model_name] = model

        return self.model_instances[model_name]

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
    def __init__(self, config, model_registry, output_dir=None, debug=False):
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
            BinarySegmentationStep(),
            PulseAnalysisStep(),
            AVSegmentationStep(),
        }

        self.engine = DAGEngine(self.steps)

    def run(self, input_path, targets=None):
        self.ctx.cache["input_path"] = input_path
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
