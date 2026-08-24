from types import SimpleNamespace

from dopplerview.models.manager import ModelManager
from dopplerview.models.spec import ModelSpec


def test_model_identity_uses_only_stable_registry_coordinates():
    spec = ModelSpec(
        name="model",
        task="segmentation",
        hf_repo="owner/repository",
        filename="model.onnx",
        format="onnx",
        input_norm="minmax",
        output_activation="sigmoid",
        input_channels=["image"],
        input_shape=[32, 32],
        output_shape=[32, 32],
        revision="configured-revision",
    )
    manager = ModelManager.__new__(ModelManager)
    manager.registry = SimpleNamespace(get=lambda name: spec)
    identity = manager.get_identity("model")

    assert identity["revision"] == "configured-revision"
    assert identity["repository"] == "owner/repository"
    assert identity["filename"] == "model.onnx"
    assert "artifact" not in identity
