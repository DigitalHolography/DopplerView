import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from dopplerview.models.wrapper import TorchModelWrapper, ONNXModelWrapper


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
    
    def change_task_model(self, task_name: str, model_name: str):
        if task_name not in self.model_tasks:
            raise ValueError(f"Unknown task '{task_name}'")
        if model_name not in self.get_model_name_list_for_task(task_name):
            raise ValueError(f"Unknown model '{model_name}' for task '{task_name}'")
        if self.registry.get(model_name).task != task_name:
            raise ValueError(f"Model '{model_name}' is not compatible with task '{task_name}'")
        if model_name != self.model_tasks[task_name]:
            self.model_tasks[task_name] = model_name
            print(f"Changed model for task '{task_name}' to '{model_name}'")

    def get_model_name_list_for_task(self, task_name: str):
        return self.registry.list_models_for_task(task_name)
    
    def get_current_model_name_for_task(self, task_name: str):
        model_name = self.model_tasks.get(task_name)
        if not model_name:
            raise ValueError(f"No model registered for task '{task_name}'")
        return model_name
    
    @staticmethod
    def build_model_wrapper(spec, local_path):
        if spec.format == "pt":
            return TorchModelWrapper(spec, local_path)

        if spec.format == "onnx":
            return ONNXModelWrapper(spec, local_path)

        raise ValueError(f"Unsupported model format: {spec.format}")