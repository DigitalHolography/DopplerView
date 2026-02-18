"""
Utility functions for deep learning models
"""

import torch

def preprocess_for_model(input_data):
    """
    Preprocess input data for model inference
    
    Args:
        input_data: numpy array of shape (height, width, channels)
    Returns:
        Preprocessed input data suitable for model inference
    """
    # Example preprocessing: normalize to [0, 1]
    preprocessed_data = input_data / 255.0
    return preprocessed_data


def postprocess_model_output(output):
    """
    Post-process model output to get segmentation mask
    
    Args:
        output: raw output from the model (e.g., logits or probabilities)
    
    Returns:
        Segmentation mask of shape (height, width) with class labels
    """
    # Example post-processing: take argmax to get class labels
    segmentation_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return segmentation_mask


def run_model(input_data, model):
    """
    Run a segmentation model on the input data
    
    Args:
        input_data: numpy array of shape (height, width, channels)
        model: pre-trained segmentation model (e.g., PyTorch or TensorFlow)
    
    Returns:
        Segmentation mask of shape (height, width) with class labels
    """
    # Preprocess input data for model
    # This may include normalization, resizing, etc. depending on the model requirements
    preprocessed_input = preprocess_for_model(input_data)
    
    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(preprocessed_input).unsqueeze(0).float()  # Add batch dimension
        output = model(input_tensor)
    
    # Post-process output to get segmentation mask
    segmentation_mask = postprocess_model_output(output)
    
    return segmentation_mask  