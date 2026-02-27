from typing import Any, Dict

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
        self.output = output_manager
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
