import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import json

from holosegment.pipeline.pipeline import Pipeline
from holosegment.models.registry import ModelRegistryConfig

def load_eyeflow_config():
    config_path = select_file()
    if config_path is None or not Path(config_path).exists():
        st.warning("Please select a valid configuration file.")
        return None
    if config_path.suffix != ".json":
        st.warning("Please select a JSON configuration file.")
        return None
    return config_path
    
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(master=root)
    root.destroy()
    return Path(file_path)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return Path(folder_path) 

def overlay_masks(image, artery_mask=None, vein_mask=None):
    img = image.copy()

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if artery_mask is not None:
        img[artery_mask > 0] = [255, 0, 0]  # red

    if vein_mask is not None:
        img[vein_mask > 0] = [0, 0, 255]  # blue

    return img
    

def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Pipeline(model_registry=ModelRegistryConfig(Path("models.yaml")))

    if "input_folder" not in st.session_state:
        st.session_state.input_folder = None

    if "image" not in st.session_state:
        st.session_state.image = None

    if "artery_mask" not in st.session_state:
        st.session_state.artery_mask = None

    if "vein_mask" not in st.session_state:
        st.session_state.vein_mask = None


init_session()

binary_seg_model = st.selectbox("Selected binary segmentation model", options=st.session_state.pipeline.ctx.model_manager.get_model_name_list_for_task("vessel_segmentation"))
st.session_state.pipeline.ctx.change_model_for_task("vessel_segmentation", binary_seg_model)
av_seg_model = st.selectbox("Selected artery/vein segmentation model", options=st.session_state.pipeline.ctx.model_manager.get_model_name_list_for_task("artery_vein_segmentation"))
st.session_state.pipeline.ctx.change_model_for_task("artery_vein_segmentation", av_seg_model)
optic_disc_model = st.selectbox("Selected optic disk detection model", options=st.session_state.pipeline.ctx.model_manager.get_model_name_list_for_task("optic_disc_detection"))
st.session_state.pipeline.ctx.change_model_for_task("optic_disc_detection", optic_disc_model)

if st.button("Load config"):
    config_path = load_eyeflow_config()
    if config_path is not None:
        st.session_state.pipeline.load_eyeflow_config(config_path)
        st.success("Config loaded.")

# 1. BROWSER BUTTON
if st.button("Browse Folder"):
    st.session_state.image = None
    st.session_state.artery_mask = None
    st.session_state.vein_mask = None

    selected_path = select_folder()
        
    if not selected_path:
        st.warning("Please select a folder path.")
    else:
        st.session_state.pipeline.load_input(selected_path)
        st.success("Folder loaded.")


if st.button("Run Full Pipeline"):

    pipeline = st.session_state.pipeline

    if pipeline is None:
        st.warning("Load a folder first.")
    else:
        pipeline.run(
            targets=None  # full pipeline
        )

        st.session_state.image = pipeline.ctx.get("M0_ff_image")
        st.session_state.artery_mask = pipeline.ctx.get("artery_mask")
        st.session_state.vein_mask = pipeline.ctx.get("vein_mask")

        st.success("Pipeline completed.")


if st.session_state.image is not None:
    display_img = overlay_masks(
        st.session_state.image,
        st.session_state.artery_mask,
        st.session_state.vein_mask,
    )

    st.image(display_img)