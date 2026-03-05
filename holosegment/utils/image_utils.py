"""
Utils for handling images, such as loading, saving, and preprocessing.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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

def normalize_to_uint8(arr):
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint8:
        return arr

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return (norm * 255).astype(np.uint8)

def save_bounding_box(image, x_center, y_center, diameter_x, diameter_y, output_path):
    plt.figure(figsize=(6, 6))
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        
    plt.imshow(image, cmap='gray')

    a = diameter_x / 2
    b = diameter_y / 2

    # Generate ellipse points
    angle = np.linspace(0, 2 * np.pi, 360)
    x_ellipsis = x_center + a * np.cos(angle)
    y_ellipsis = y_center + b * np.sin(angle)
    plt.plot(x_ellipsis, y_ellipsis, "r", linewidth=2, label="Ellipse")

    # Bounding box coordinates
    x_min = x_center - a
    y_min = y_center - b

    # Create a rectangle patch
    plt.gca().add_patch(
        plt.Rectangle((x_min, y_min), diameter_x, diameter_y, 
                  fill=False, edgecolor="lime", linewidth=2, label="Box"))

    # Add the rectangle to the Axes

    plt.legend()
    plt.savefig(output_path)
    plt.close()

def save_numpy_as_avi(video: np.ndarray, filename: str, fps: int = 30):
    """
    Saves a NumPy video array to an AVI file using OpenCV.

    Parameters:
        video (np.ndarray): Shape (T, H, W) for grayscale, or (T, H, W, 3) for RGB.
        filename (str): Path to output .avi file.
        fps (int): Frame rate.
    """
    T = video.shape[0]
    is_color = video.ndim == 4

    H, W = video.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)

    for t in range(T):
        frame = video[t]
        
        # Normalize and convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = normalize_to_uint8(frame)
        
        # Convert grayscale to BGR
        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

    out.release()
    print(f"Saved video to {filename}")