"""
Load pre-trained models for optic disc detection and vessel segmentation tasks. Handle .pt and .onnx formats.
"""

import torch
import onnxruntime

def load_pt_model(model_name, device='cuda'):
    """
    Load a PyTorch model from a .pt file
    
    Args:
        model_name: name of the model to load (e.g., 'AVCombinedSegmentationNet')   
    Returns:
        Loaded PyTorch model ready for inference
    """
    model_path = f'models/{model_name}.pt'
    model = torch.load(model_path, map_location=torch.device(device))
    return model

def load_onnx_model(model_name):
    """
    Load an ONNX model from a .onnx file
    
    Args:
        model_name: name of the model to load (e.g., 'AVCombinedSegmentationNet')
    Returns:
        ONNX Runtime session ready for inference
    """
    model_path = f'models/{model_name}.onnx'
    session = onnxruntime.InferenceSession(model_path)
    return session