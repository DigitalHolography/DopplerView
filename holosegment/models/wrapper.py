"""
Load pre-trained models and handle inference.
Supports .pt (state_dict recommended) and .onnx.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import onnxruntime as ort


class BaseModelWrapper(ABC):
    def __init__(self, spec, model_path):
        self.spec = spec
        self.model_path = str(model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        x = self._preprocess(image)
        y = self._forward(x)
        y = self._postprocess(y)
        return y
    
    def _preprocess_channel(self, channel):
        if self.spec.input_norm == "zscore":
            return (channel - channel.mean()) / (channel.std() + 1e-8)

        if self.spec.input_norm == "minmax":
            return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

        if self.spec.input_norm == "rescale":
            return channel / 255.0

    def _preprocess(self, image):
        if image.ndim == 2:
            return self._preprocess_channel(image)
        if image.ndim == 3:
            return np.stack([self._preprocess_channel(image[i]) for i in range(image.shape[0])], axis=0)

    def _postprocess(self, output):
        act = self.spec.output_activation

        if act == "sigmoid":
            return 1 / (1 + np.exp(-output))

        if act == "softmax":
            exp = np.exp(output - np.max(output, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        
        if act == "argmax":
            return np.argmax(output, axis=1)

        return output

    @abstractmethod
    def _forward(self, x):
        pass


class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, spec, model_path, device=None):
        super().__init__(spec, model_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SAFER: assume state_dict unless you explicitly allow full model loading
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # You must define your architecture elsewhere
        # For now assume full model was saved (less safe)
        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            raise RuntimeError(
                "Torch model loading requires architecture definition "
                "or full model object."
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        y = self.model(x)
        return y.cpu().numpy()


class ONNXModelWrapper(BaseModelWrapper):
    def __init__(self, spec, model_path):
        super().__init__(spec, model_path)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _forward(self, x):
        x = x.astype(np.float32)[None, None, :, :]
        outputs = self.session.run(None, {self.input_name: x})
        return outputs[0]