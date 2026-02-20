

class ModelRegistry:
    _models = {}

    @classmethod
    def get(cls, name, path):
        if name not in cls._models:
            cls._models[name] = load_model(path)
        return cls._models[name]
