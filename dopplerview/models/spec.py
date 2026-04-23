from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    name: str
    task: str
    hf_repo: str
    filename: str
    format: str              # "pt" or "onnx"
    input_norm: str
    output_activation: str
    input_channels: Optional[list]
    revision: Optional[str] = "main"