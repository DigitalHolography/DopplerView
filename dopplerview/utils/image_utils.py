"""
Utils for handling images, such as loading, saving, and preprocessing.
"""

import numpy as np
from PIL import Image
from skimage.measure import regionprops
from dopplerview.utils.matplotlib_backend import serialized_render
import matplotlib.pyplot as plt
import cv2
from skimage.color import lab2rgb
from skimage.restoration import inpaint
from dopplerview.utils.parallelization_utils import run_in_parallel

import logging
logger = logging.getLogger(__name__)

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

def normalize_image(image_array, min_val=0, max_val=1):
    """
    Normalize a numpy array image to the range [min_val, max_val]

    Args:
        image_array: numpy array representation of the image (height, width, channels)
        min_val: Minimum value for the normalized range
        max_val: Maximum value for the normalized range
    
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

@serialized_render
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

@serialized_render
def save_labeled_branches(label_mask, output_path):
    """
    Display a labeled mask with the label ID written on each branch.

    Parameters
    ----------
    label_mask : ndarray (H, W)
        Image where each branch has a unique integer label.
        Background must be 0.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # show mask
    ax.imshow(label_mask, cmap="nipy_spectral")

    # compute region properties
    props = regionprops(label_mask)

    for region in props:
        y, x = region.centroid
        label = region.label

        ax.text(
            x,
            y,
            str(label),
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round")
        )

    ax.set_axis_off()
    plt.tight_layout()
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

def lab_duo_image(image_1, image_2, h=45):
    """
    Parameters
    ----------
    image_1 : ndarray
        First image (controls L channel).
    image_2 : ndarray
        Second image (controls a/b channels).
    h : float, optional
        Hue angle in degrees.

    Returns
    -------
    rgb : ndarray
        RGB image in range [0, 1].
    """
    h = np.mod(h, 360)

    cos_h = np.cos(np.deg2rad(h))
    sin_h = np.sin(np.deg2rad(h))

    if abs(h) <= 45 or abs(h - 180) <= 45:
        rx = np.sign(cos_h)
        ry = np.sign(sin_h) * (1 / (cos_h**2) - 1)
    else:
        ry = np.sign(sin_h)
        rx = np.sign(cos_h) * (1 / (sin_h**2) - 1)

    v = np.array([rx, ry], dtype=float)
    v /= np.linalg.norm(v)
    rx, ry = v

    image_1 = image_1.astype(np.float32)
    image_2 = image_2.astype(np.float32)

    # Normalize to [-1, 1]
    denom1 = max(np.max(np.abs(image_1)), np.abs(np.min(image_1)))
    denom2 = max(np.max(np.abs(image_2)), np.abs(np.min(image_2)))

    if denom1 > 0:
        image_1 = image_1 / denom1
    if denom2 > 0:
        image_2 = image_2 / denom2

    L = 100.0 * image_1
    chroma = image_2 / (np.max(image_2) + 1e-8)

    a = 80 * chroma * rx
    b = 80 * chroma * ry

    lab = np.stack([L, a, b], axis=-1)

    rgb = lab2rgb(lab)

    return rgb

def inpaint_frame(frame, mask):
    """
    Inpaint a single frame using biharmonic inpainting.

    Parameters
    ----------
    frame : ndarray
        2D array representing the image to be inpainted.
    mask : ndarray
        2D boolean array where True indicates the pixels to be inpainted.

    Returns
    -------
    inpainted_frame : ndarray
        The inpainted image.
    """

    # Ensure the frame is in float format for inpainting
    frame_float = frame.astype(np.float32)

    # Inpaint the frame using the provided mask
    inpainted_frame = inpaint.inpaint_biharmonic(frame_float, mask)

    return inpainted_frame

def inpaint_stack(stack, mask, n_jobs=-1, dilation_radius=0):
    """
    Inpaint a stack of frames using biharmonic inpainting.

    Parameters
    ----------
    stack : ndarray
        3D array representing the stack of images to be inpainted (T, H, W).
    mask : ndarray
        2D boolean array where True indicates the pixels to be inpainted.
    n_jobs : int
        Number of parallel jobs to run.
    dilation_radius : int
        Radius for dilating the mask before inpainting.

    Returns
    -------
    inpainted_stack : ndarray
        The inpainted stack of images.
    """

    if dilation_radius > 0:
        from skimage.morphology import dilation, disk
        mask = dilation(mask, disk(dilation_radius))
        
    def _inpaint_frame(frame):
        return inpaint_frame(frame, mask)

    if n_jobs == 1:
        inpainted_stack = np.array([_inpaint_frame(frame) for frame in stack])
    else:
        inpainted_stack = np.array(run_in_parallel(_inpaint_frame, stack, n_jobs=n_jobs))

    return inpainted_stack
