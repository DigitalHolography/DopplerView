from dopplerview.input_output import user_config
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import json

from dopplerview.pipeline.pipeline import Pipeline

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
        st.session_state.pipeline = Pipeline()

        models_config = user_config.ensure_config_file("models.yaml")
        st.session_state.pipeline.load_model_registry(models_config)
        
        h5_schema_config = user_config.ensure_config_file("h5_schema.json")
        output_config = user_config.ensure_config_file("output_config.json")
        st.session_state.pipeline.load_h5_schema(h5_schema_config)
        st.session_state.pipeline.load_output_config(output_config)

    if "input_folder" not in st.session_state:
        st.session_state.input_folder = None

    if "image" not in st.session_state:
        st.session_state.image = None

    if "retinal_artery_mask" not in st.session_state:
        st.session_state.artery_mask = None

    if "retinal_vein_mask" not in st.session_state:
        st.session_state.vein_mask = None


init_session()

binary_seg_model = st.selectbox("Selected binary segmentation model", options=st.session_state.pipeline.ctx.model_manager.get_model_name_list_for_task("retinal_vessel_segmentation"))
st.session_state.pipeline.ctx.change_model_for_task("retinal_vessel_segmentation", binary_seg_model)
av_seg_model = st.selectbox("Selected artery/vein segmentation model", options=st.session_state.pipeline.ctx.model_manager.get_model_name_list_for_task("retinal_artery_vein_segmentation"))
st.session_state.pipeline.ctx.change_model_for_task("retinal_artery_vein_segmentation", av_seg_model)
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
        st.session_state.input_folder = selected_path
        st.success("Folder loaded.")


def on_step_toggle(step):
    pipeline = st.session_state.pipeline

    if not st.session_state[f"ui_{step}"]:
        downstream = pipeline.get_downstream_steps(step)
        for s in downstream:
            st.session_state[f"ui_{s}"] = False

    selected = [
        s for s in pipeline.get_step_names()
        if st.session_state[f"ui_{s}"]
    ]

    pipeline.set_targets(selected)
    steps_to_run = pipeline.engine.steps_to_run
    for s in pipeline.get_step_names():
        st.session_state[f"ui_{s}"] = s in steps_to_run

if st.session_state.input_folder is not None:
    st.subheader("Pipeline Steps")

    pipeline = st.session_state.pipeline
    all_steps = pipeline.get_step_names()

    if "ui_steps_initialized" not in st.session_state:
        for step in all_steps:
            st.session_state[f"ui_{step}"] = True
        st.session_state.selected_targets = list(all_steps)
        st.session_state.ui_steps_initialized = True

    for step in all_steps:
        # Color code steps based on cache status: green if cached, yellow if not cached
        # The last checked step is targeted for execution, so it is yellow.
        if pipeline.is_cached(step):
            label = f"🟢 {step}"
        else:
            label = f"🟡 {step}"
        st.checkbox(
            label,
            key=f"ui_{step}",
            on_change=on_step_toggle,
            args=(step,)
        )


    if st.button("Run Pipeline"):

        pipeline = st.session_state.pipeline

        if pipeline is None:
            st.warning("Load a folder first.")
        else:
            pipeline.run(
                targets=st.session_state.selected_targets
            )

            st.session_state.image = pipeline.ctx.get("M0_ff_image")

            print("Pipeline execution completed.")

            st.success("Pipeline completed.")

            st.rerun()


if st.session_state.image is not None:
    display_img = overlay_masks(
        st.session_state.image,
        pipeline.ctx.get("retinal_artery_mask"),
        pipeline.ctx.get("retinal_vein_mask")
    )

    st.image(display_img)