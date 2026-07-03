import cv2
import numpy as np
from PIL import Image, ImageTk


def np_to_tk(img: np.ndarray):
    """Convert numpy image to Tkinter-compatible PhotoImage."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.uint8)
    pil_img = Image.fromarray(img)
    return ImageTk.PhotoImage(pil_img)


def overlay_preview(image, artery_mask=None, vein_mask=None):
    """Build a lightweight RGB preview image inside the worker process."""
    if image is None:
        return None

    img = np.asarray(image).copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if artery_mask is not None:
        if vein_mask is not None:
            img[np.asarray(artery_mask) > 0] = [255, 0, 0]
        else:
            img[np.asarray(artery_mask) > 0] = [255, 250, 250]

    if vein_mask is not None:
        img[np.asarray(vein_mask) > 0] = [0, 0, 255]

    return img.astype(np.uint8, copy=False)


def resize_preview_for_queue(img, max_side=900):
    """Avoid sending very large arrays through multiprocessing.Queue."""
    if img is None:
        return None

    img = np.asarray(img)
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return img.astype(np.uint8, copy=False)

    scale = max_side / largest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


def build_step_preview(pipeline, step_name):
    """Extract the same previews as the Tk app, but from the child pipeline context."""
    ctx = pipeline.ctx

    if step_name == "preprocess":
        return resize_preview_for_queue(ctx.get("M0_ff_image"))

    if step_name == "retinal_vessel_segmentation":
        img = ctx.get("M0_ff_image")
        vessel = ctx.get("retinal_vessel_mask")
        return resize_preview_for_queue(overlay_preview(img, vessel, None))

    if step_name == "retinal_artery_vein_segmentation":
        img = ctx.get("M0_ff_image")
        art = ctx.get("retinal_artery_mask")
        vein = ctx.get("retinal_vein_mask")
        return resize_preview_for_queue(overlay_preview(img, art, vein))

    return None
