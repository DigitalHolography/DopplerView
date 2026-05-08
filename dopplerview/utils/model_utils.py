"""
Utility functions for deep learning models
"""

import torch
import numpy as np
import cv2

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_yolo_mask(dets, protos, target_size):
    """
    Decode YOLOv11 output into a binary mask.

    Parameters
    ----------
    dets : np.ndarray
        Detection head of shape (1, 37, 21504) or similar.
    protos : np.ndarray
        Prototype head of shape (256, 256, 32, 1) or similar.
    target_size : tuple
        Final mask size as (height, width), e.g. (1024, 1024)

    Returns
    -------
    binary_mask : np.ndarray
        Boolean mask of shape target_size.
    """

    # 1. Decode detections
    dets = np.squeeze(dets)

    # Find best anchor
    scores = dets[4, :]  # MATLAB index 5 -> Python index 4
    idx = np.argmax(scores)
    max_score = scores[idx]

    if max_score < 0.25:
        return np.zeros(target_size, dtype=bool)

    mask_coeffs = dets[5:37, idx]  # MATLAB 6:37 -> Python 5:37
    box = dets[0:4, idx]           # [cx, cy, w, h]

    # 2. Decode prototypes & assemble
    protos = np.squeeze(protos)

    protos = np.transpose(protos, (1, 2, 0))  # (256, 256, 32)

    pw, ph, pc = protos.shape

    # Flatten
    protos_flat = protos.reshape(pw * ph, pc)

    # Matrix multiplication
    mask_raw = protos_flat @ mask_coeffs

    # Reshape back
    mask_img = mask_raw.reshape(pw, ph)

    # Transpose back
    mask_img = mask_img.T

    # Sigmoid activation
    mask_img = sigmoid(mask_img)

    # 3. Resize
    target_h, target_w = target_size
    mask_full = cv2.resize(
        mask_img,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Crop to bounding box
    bx, by, bw, bh = box

    x1 = max(0, round(bx - bw / 2))
    y1 = max(0, round(by - bh / 2))
    x2 = min(target_w - 1, round(bx + bw / 2))
    y2 = min(target_h - 1, round(by + bh / 2))

    box_mask = np.zeros(target_size, dtype=bool)
    box_mask[y1:y2 + 1, x1:x2 + 1] = True

    # Final binary mask
    binary_mask = (mask_full > 0.5) & box_mask

    return binary_mask