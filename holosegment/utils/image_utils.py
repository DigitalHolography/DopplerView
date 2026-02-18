"""
Utils for handling images, such as loading, saving, and preprocessing.
"""

import numpy as np
from PIL import Image

def load_image_as_array(image_path):
    """
    Load an image from the specified path and convert it to a numpy array
    
    Args:
        image_path: path to the image file (e.g., .png, .jpg)   
    Returns:
        Numpy array representation of the image (height, width, channels)
    """
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    return np.array(image)

def save_array_as_image(array, filename, foldername):
    """
    Save a numpy array as an image to the specified path
    
    Args:
        array: numpy array representation of the image (height, width, channels)
        save_path: path to save the image file (e.g., .png, .jpg)   
    """
    image = Image.fromarray((array * 255).astype(np.uint8))  # Convert back to uint8 format
    image.save(f"{foldername}/{filename}")

def normalize_image(image_array):
    """
    Normalize a numpy array image to the range [0, 1]
    
    Args:
        image_array: numpy array representation of the image (height, width, channels)
    
    Returns:
        Normalized image array with values in the range [0, 1]
    """
    return (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)