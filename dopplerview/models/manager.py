import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from dopplerview.models.wrapper import ONNXModelWrapper, TorchModelWrapper

import logging
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, registry, cache_dir):
        self.registry = registry
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.model_tasks = {task: models[0] for task, models in registry._tasks.items()}

    def resolve(self, model_name: str, force_update: bool = False):
        spec = self.registry.get(model_name)

        local_path = hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            revision=spec.revision,
            cache_dir=self.cache_dir,
            force_download=force_update,
        )

        return spec, Path(local_path)

    def get_identity(self, model_name: str):
        """Return stable scientific identity without consulting Hugging Face.

        Local cache paths, snapshot links, timestamps and download metadata are
        deliberately excluded: they can change while the selected model is
        scientifically identical and would turn every debug run into a miss.
        """
        spec = self.registry.get(model_name)
        return {
            "name": spec.name,
            "task": spec.task,
            "repository": spec.hf_repo,
            "filename": spec.filename,
            "format": spec.format,
            "revision": spec.revision,
            "input_norm": spec.input_norm,
            "output_activation": spec.output_activation,
            "input_channels": spec.input_channels,
            "input_shape": spec.input_shape,
            "output_shape": spec.output_shape,
        }
    
    def change_task_model(self, task_name: str, model_name: str):
        if task_name not in self.model_tasks:
            raise ValueError(f"Unknown task '{task_name}'")
        if model_name not in self.get_model_name_list_for_task(task_name):
            raise ValueError(f"Unknown model '{model_name}' for task '{task_name}'")
        if self.registry.get(model_name).task != task_name:
            raise ValueError(f"Model '{model_name}' is not compatible with task '{task_name}'")
        if model_name != self.model_tasks[task_name]:
            self.model_tasks[task_name] = model_name
            logger.info(f"Changed model for task '{task_name}' to '{model_name}'")

    def get_model_name_list_for_task(self, task_name: str):
        return self.registry.list_models_for_task(task_name)
    
    def get_current_model_name_for_task(self, task_name: str):
        model_name = self.model_tasks.get(task_name)
        if not model_name:
            raise ValueError(f"No model registered for task '{task_name}'")
        return model_name
    
    @staticmethod
    def build_model_wrapper(spec, local_path, execution_policy=None):
        if spec.format == "pt":
            return TorchModelWrapper(spec, local_path, execution_policy=execution_policy)

        if spec.format == "onnx":
            return ONNXModelWrapper(spec, local_path, execution_policy=execution_policy)

        raise ValueError(f"Unsupported model format: {spec.format}")
