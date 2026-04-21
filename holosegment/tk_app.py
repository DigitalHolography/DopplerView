import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import json

import numpy as np
import cv2
from PIL import Image, ImageTk

from holosegment.pipeline.pipeline import Pipeline
from holosegment.models.registry import ModelRegistryConfig

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None
    print("Warning: tkinterdnd2 not found, drag-and-drop functionality will be disabled.")

def np_to_tk(img: np.ndarray):
    """Convert numpy image to Tkinter-compatible PhotoImage"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = (img).astype(np.uint8)
    pil_img = Image.fromarray(img)
    return ImageTk.PhotoImage(pil_img)


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Holosegment")

        # --- pipeline init ---
        config_path = Path("config")
        registry = ModelRegistryConfig(config_path / "models.yaml")
        h5_schema = json.load(open(config_path / "h5_schema.json"))
        output_config = json.load(open(config_path / "output_config.json"))

        self.pipeline = Pipeline(
            model_registry=registry,
            h5_schema=h5_schema,
            output_config=output_config
        )

        self.image_tk = None  # keep reference (IMPORTANT)

        # --- UI layout ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        self.minimal_frame = tk.Frame(main_frame)
        self.advanced_frame = tk.Frame(main_frame)

        self.minimal_frame.pack(fill="both", expand=True)
        self.advanced_frame.pack_forget()

        self.drop_label = tk.Label(self.minimal_frame, text="Drop folder here", relief="ridge", height=5)
        self.drop_label.pack(fill="x", padx=10, pady=10)

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)

        # Buttons
        self.btn_load = tk.Button(self.advanced_frame, text="Load Folder", command=self.load_folder)
        self.btn_load.pack(pady=5)

        self.btn_run = tk.Button(self.advanced_frame, text="Run Pipeline", command=self.run_pipeline)
        self.btn_run.pack(pady=5)

        # Step list (checkboxes)
        self.step_vars = {}
        self.step_checkboxes = {}

        self.steps_frame = tk.LabelFrame(self.advanced_frame, text="Pipeline Steps")
        self.steps_frame.pack(fill="x", padx=5, pady=5)

        for step in self.pipeline.get_step_names():
            var = tk.BooleanVar(value=True)

            cb = tk.Checkbutton(
                self.steps_frame,
                text=step,
                variable=var,
                command=lambda s=step: self.on_step_toggle(s)
            )
            cb.pack(anchor="w")

            self.step_vars[step] = var
            self.step_checkboxes[step] = cb
    
        # Image display
        self.image_label = tk.Label(self.advanced_frame)
        self.image_label.pack(pady=10)

        self.mode = tk.StringVar(value="minimal")

        mode_frame = tk.Frame(self.advanced_frame)
        mode_frame.pack(pady=5)

        tk.Label(mode_frame, text="Mode:").pack(side="left")

        tk.Radiobutton(mode_frame, text="Minimal", variable=self.mode, value="minimal", command=self.update_mode).pack(side="left")
        tk.Radiobutton(mode_frame, text="Advanced", variable=self.mode, value="advanced", command=self.update_mode).pack(side="left")

    # -------------------
    # Actions
    # -------------------

    def on_step_toggle(self, step):
        pipeline = self.pipeline

        selected = self.get_selected_steps()

        if self.step_vars[step].get():
            # ADD step → recompute full dependency closure
            resolved = pipeline.resolve_execution_graph(selected)

            for s in pipeline.get_step_names():
                self.step_vars[s].set(s in resolved)

        else:
            # REMOVE step + downstream
            downstream = pipeline.get_downstream_steps(step)

            for s in downstream:
                self.step_vars[s].set(False)

            self.step_vars[step].set(False)

        self.update_step_display()

    def update_mode(self):
        if self.mode.get() == "minimal":
            self.advanced_frame.pack_forget()
            self.minimal_frame.pack(fill="both", expand=True)
        else:
            self.minimal_frame.pack_forget()
            self.advanced_frame.pack(fill="both", expand=True)

    def update_step_display(self):
        pipeline = self.pipeline

        selected = self.get_selected_steps()
        resolved = pipeline.resolve_execution_graph(selected)

        # Steps that will actually run
        pipeline.set_targets(selected)
        steps_to_run = set(pipeline.engine.steps_to_run)

        for step, cb in self.step_checkboxes.items():
            is_checked = self.step_vars[step].get()
            is_required = step in resolved
            is_cached = pipeline.is_cached(step)

            # -------- label logic --------
            if is_checked:
                if is_cached:
                    label = f"🟢 {step}"  # already done
                else:
                    label = f"🟡 {step}"  # will run
            else:
                if is_required:
                    label = f"🔵 {step}"  # required but not explicitly selected
                else:
                    label = f"⚪ {step}"  # ignored

            cb.config(text=label)


    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.pipeline.load_input(Path(folder))

    def get_selected_steps(self):
        return [step for step, var in self.step_vars.items() if var.get()]

    def load_and_run_minimal(self):
        folder = filedialog.askdirectory()
        if folder:
            self.run_full_pipeline(Path(folder))

    def on_drop(self, event):
        path = event.data.strip("{}")  # windows fix
        self.pipeline.load_input(Path(path))
        self.run_full_pipeline()

    def run_full_pipeline(self):
        # full pipeline
        self.pipeline.run(targets=None)

        img = self.pipeline.ctx.get("M0_ff_image")
        art = self.pipeline.ctx.get("retinal_artery_mask")
        vein = self.pipeline.ctx.get("retinal_vein_mask")

        if img is not None:
            overlay = self.overlay(img, art, vein)
            self.display_image(overlay)

    def run_pipeline(self):
        steps = self.get_selected_steps()

        self.pipeline.run(targets=steps)

        self.update_step_display()

        img = self.pipeline.ctx.get("M0_ff_image")
        art = self.pipeline.ctx.get("retinal_artery_mask")
        vein = self.pipeline.ctx.get("retinal_vein_mask")

        if img is not None:
            overlay = self.overlay(img, art, vein)
            self.display_image(overlay)

    # -------------------
    # Image utils
    # -------------------

    def display_image(self, img):
        self.image_tk = np_to_tk(img)  # keep reference!
        self.image_label.config(image=self.image_tk)

    def overlay(self, image, artery_mask, vein_mask):
        img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if artery_mask is not None:
            img[artery_mask > 0] = [255, 0, 0]

        if vein_mask is not None:
            img[vein_mask > 0] = [0, 0, 255]

        return img


# -------------------
# Run app
# -------------------

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = MainWindow(root)
    root.mainloop()