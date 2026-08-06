"""
Load pre-trained models and handle inference.
Supports .pt (state_dict recommended) and .onnx.
"""

from abc import ABC, abstractmethod
import logging
import time
import numpy as np
from dopplerview.utils.image_utils import normalize_to_uint8
from dopplerview.utils.runtime_metrics import emit_metric, process_snapshot
import cv2


logger = logging.getLogger(__name__)

class BaseModelWrapper(ABC):
    def __init__(self, spec, model_path):
        self.spec = spec
        self.model_path = str(model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        total_started = time.perf_counter()
        memory_started = process_snapshot()
        preprocess_started = time.perf_counter()
        x = self._preprocess(image)
        preprocess_s = time.perf_counter() - preprocess_started
        inference_started = time.perf_counter()
        y = self._forward(x)
        inference_s = time.perf_counter() - inference_started
        postprocess_started = time.perf_counter()
        y = self._postprocess(y)
        postprocess_s = time.perf_counter() - postprocess_started
        memory_finished = process_snapshot()
        emit_metric(
            "inference",
            model=getattr(self.spec, "name", "unknown"),
            backend=self.backend_name(),
            provider=self.provider_name(),
            input_shape="x".join(str(size) for size in image.shape),
            preprocess_s=preprocess_s,
            inference_s=inference_s,
            postprocess_s=postprocess_s,
            total_s=time.perf_counter() - total_started,
            process_rss_delta_mb=(
                memory_finished["rss_mb"] - memory_started["rss_mb"]
            ),
            process_threads=memory_finished["process_threads"],
        )
        logger.info(
            "[Model] %s inference on %s: %.3fs (total %.3fs)",
            getattr(self.spec, "name", "unknown"),
            self.provider_name(),
            inference_s,
            time.perf_counter() - total_started,
        )
        return y

    def backend_name(self):
        return type(self).__name__

    def provider_name(self):
        return "unknown"
    
    def prepare_input(self, ctx):
        channels = []
        for ch_name in self.spec.input_channels:
            ch = ctx.require(ch_name)
            if ch.shape != tuple(self.spec.input_shape):
                ch = cv2.resize(ch, tuple(self.spec.input_shape))
            channels.append(ch)
        return np.stack(channels, axis=0)
    
    def _preprocess_channel(self, channel):
        if self.spec.input_norm == "zscore":
            return (channel - channel.mean()) / (channel.std() + 1e-8)

        if self.spec.input_norm == "minmax":
            return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

        if self.spec.input_norm == "rescale":
            return channel / 255.0

        if self.spec.input_norm == "none":
            return normalize_to_uint8(channel)

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
        import torch
        super().__init__(spec, model_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SAFER: assume state_dict unless you explicitly allow full model loading
        checkpoint = torch.jit.load(self.model_path, map_location=self.device)

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

        emit_metric(
            "model_loaded",
            model=getattr(self.spec, "name", "unknown"),
            backend=self.backend_name(),
            provider=self.provider_name(),
        )

    def provider_name(self):
        return str(self.device)

    def _forward(self, x):
        import torch
        with torch.no_grad():
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            y = self.model(x)
            return y.cpu().numpy()


class ONNXModelWrapper(BaseModelWrapper):
    def __init__(self, spec, model_path):
        import onnxruntime as ort
        super().__init__(spec, model_path)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        emit_metric(
            "model_loaded",
            model=getattr(self.spec, "name", "unknown"),
            backend=self.backend_name(),
            provider=self.provider_name(),
        )

    def provider_name(self):
        return ",".join(self.session.get_providers())

    def _forward(self, x):
        b, c, h, w = self.session.get_inputs()[0].shape

        # Handle single image input and ensure it matches expected shape
        if x.ndim == 2:
            if c > 1:
                x = np.stack([x] * c, axis=0)
            else:
                x = x[None, :, :]
        if h != x.shape[1] or w != x.shape[2]:
            raise ValueError(f"Input shape {x.shape} does not match model expected shape (C, H, W) = ({c}, {h}, {w})")

        x = x.astype(np.float32)[None, :, :, :]
        outputs = self.session.run(None, {self.input_name: x})

        return outputs[0]
