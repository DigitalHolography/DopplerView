import numpy as np
from pathlib import Path
import imageio
import cv2
import matplotlib.pyplot as plt

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

def normalize_to_uint8(arr):
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint8:
        return arr

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return (norm * 255).astype(np.uint8)

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

#  def _normalize_uint8(self, array):
#         array = array.astype(float)
#         array -= array.min()
#         if array.max() > 0:
#             array /= array.max()
#         return (array * 255).astype("uint8")


# class OutputManager:
#     def __init__(self, output_dir=None, enabled=False):
#         self.enabled = enabled
#         if enabled:
#             self.output_dir = Path(output_dir)
#             self.output_dir.mkdir(parents=True, exist_ok=True)

#     def save(self, step_name, outputs):
#         if not self.enabled:
#             return

#         for key, value in outputs.items():
#             filename = self.output_dir / f"{step_name}_{key}"

#             if isinstance(value, np.ndarray):
#                 np.save(str(filename) + ".npy", value)

#                 # Save 2D arrays as PNG for visualization
#                 if value.ndim == 2:
#                     imageio.imwrite(
#                         str(filename) + ".png",
#                         self._normalize_uint8(value)
#                     )

   