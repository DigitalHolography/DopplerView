import numpy as np
from pathlib import Path
import imageio
import cv2
import matplotlib.pyplot as plt
from holosegment.utils.image_utils import normalize_to_uint8

class OutputManager:
    def __init__(self, output_dir, enabled=True, formats=("npy",)):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.formats = formats
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step_name, key, value, format):
        if not self.enabled:
            return

        filename = f"{step_name}_{key}"

        if format == "npy":
            np.save(self.output_dir / f"{filename}.npy", value)

        if format == "png":
            imageio.imwrite(self.output_dir / f"{filename}.png", normalize_to_uint8(value))

        if format == "avi":
            save_numpy_as_avi(value, self.output_dir / f"{filename}.avi")

    def save_plot(self, step_name, key, value, title=None):
        if not self.enabled:
            return

        plt.plot(value)
        if title:
            plt.title(title)
        
        plt.savefig(self.output_dir / f"{step_name}_{key}.png", bbox_inches='tight')
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


def save_bounding_box(image, x_center, y_center, diameter_x, diameter_y, output_path):
    plt.figure(figsize=(6, 6))
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