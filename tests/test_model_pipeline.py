import numpy as np
from pathlib import Path

from registry import ModelRegistryConfig
from manager import ModelManager


def main():
    print("Loading registry...")
    registry = ModelRegistryConfig(Path("models.yaml"))

    print("Available models:")
    for m in registry.list_models():
        print(" -", m)

    model_name = registry.list_models()[0]
    print(f"\nResolving model: {model_name}")

    manager = ModelManager(registry)
    spec, model_path = manager.resolve(model_name)

    print("Local model path:", model_path)

    print("Building wrapper...")
    model = ModelManager.build_model_wrapper(spec, model_path)

    print("Running dummy inference...")

    dummy = np.random.rand(512, 512).astype(np.float32)
    output = model.predict(dummy)

    print("Output shape:", output.shape)
    print("Done.")


if __name__ == "__main__":
    main()