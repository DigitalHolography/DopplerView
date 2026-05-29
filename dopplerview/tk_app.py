import sys
import os
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk
from pathlib import Path
import threading
import queue
import matplotlib

from dopplerview.input_output import log_config, user_config
import numpy as np
import cv2
from PIL import Image, ImageTk

from dopplerview.input_output.output_manager import OutputManager
from dopplerview.pipeline.pipeline import Pipeline

import logging
logger = logging.getLogger(__name__)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None
    logger.warning("Warning: tkinterdnd2 not found, drag-and-drop functionality will be disabled.")

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

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
        self.root.title("DopplerView")

        self._minimal_title_font: tkfont.Font | None = None

        # --- pipeline init ---

        h5_schema_path = user_config.ensure_config_file("h5_schema.json")
        output_config_path = user_config.ensure_config_file("output_config.json")
        self.output_manager = OutputManager(h5_schema_path, output_config_path)
        self.pipeline = Pipeline(output_manager=self.output_manager)

        models_config = user_config.ensure_config_file("models.yaml")
        self.pipeline.load_model_registry(models_config)
        
        config_path = user_config.ensure_config_file("default_DV_params.json")
        # self.pipeline.load_dopplerview_config(config_path)
        self.config_path = tk.StringVar(value=str(config_path))

        self.image_tk = None  # keep reference (IMPORTANT)

        self.queue = queue.Queue()

        self._apply_theme()
        self._set_window_icon()

        # --- UI layout --
        self._build_ui()
        self._install_drop_targets()
        self.update_mode()  # set initial mode

        self.config_mode_var = tk.StringVar(value="default")
        self.update_config_mode() # set initial config mode

        self.step_index = 0
        self.measure_index = 0


    def _apply_theme(self) -> None:
        """
        Apply the Sun Valley ttk theme when available; otherwise fall back to a simple dark palette.
        """
        style = ttk.Style(self.root)
        self._style = style
        if sv_ttk:
            try:
                sv_ttk.set_theme("dark")
            except Exception:
                pass

        # Fallback palette aligned with Sun Valley dark.
        fallback_bg = "#0f1116"
        fallback_surface = "#1b1f27"
        fallback_fg = "#e8eef5"
        fallback_muted = "#9aa6b5"
        fallback_accent = "#4f9dff"

        # Derive colors from the active theme when possible to keep consistency.
        bg = style.lookup("TFrame", "background") or fallback_bg
        fg = style.lookup("TLabel", "foreground") or fallback_fg
        surface = (
            style.lookup("TEntry", "fieldbackground")
            or style.lookup("TEntry", "background")
            or fallback_surface
        )
        muted = (
            style.lookup("TLabel", "foreground", state=("disabled",)) or fallback_muted
        )
        accent = (
            style.lookup("TButton", "bordercolor")
            or style.lookup("TNotebook", "foreground")
            or fallback_accent
        )
        select = (
            style.lookup("TButton", "foreground", state=("selected",))
        )

        self.root.configure(bg=bg)
        # set texts colors when created.
        self._text_bg = surface
        self._text_fg = fg
        self._muted_fg = muted
        self._bg_color = bg
        self._surface_color = surface
        self._accent_color = accent
        self._select_color = select

    # -------------------
    # UI
    # -------------------

    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        self.minimal_frame = ttk.Frame(container, padding=10)
        self.advanced_frame = ttk.Frame(container, padding=10)

        self._build_minimal_ui()
        self._build_advanced_ui()

    def _build_menu(self) -> None:
        self.ui_mode_var = tk.StringVar(value="minimal")

        menu_bar = tk.Menu(self.root, bg=self._bg_color)

        view_menu = tk.Menu(menu_bar, tearoff=False, bg=self._bg_color)
        view_menu.add_radiobutton(
            label="Minimal UI",
            value="minimal",
            variable=self.ui_mode_var,
            command=self.update_mode,
        )
        view_menu.add_radiobutton(
            label="Advanced UI",
            value="advanced",
            variable=self.ui_mode_var,
            command=self.update_mode,
        )

        menu_bar.add_cascade(label="View", menu=view_menu)

        config_menu = tk.Menu(menu_bar, tearoff=False, bg=self._bg_color)
        config_menu.add_command(label="Open Configuration", command=self.show_config)
        config_menu.add_separator()
        config_menu.add_command(label="Modify dopplerview config", command=self.modify_dopplerview_config)
        config_menu.add_command(label="Modify models registry", command=self.modify_models_registry)
        config_menu.add_command(label="Modify h5 schema", command=self.modify_h5_schema)
        config_menu.add_command(label="Modify output config", command=self.modify_output_config)

        menu_bar.add_cascade(label="Config", menu=config_menu)

        menu_bar.add_command(label="Help", command=self.show_help)

        self.root.configure(menu=menu_bar)

    def _get_minimal_title_font(self) -> tkfont.Font:
        if self._minimal_title_font is None:
            title_font = tkfont.nametofont("TkDefaultFont").copy()
            base_size = int(title_font.cget("size")) or 10
            title_font.configure(size=base_size * 2)
            self._minimal_title_font = title_font
        return self._minimal_title_font

    def _build_minimal_ui(self):
        frame = self.minimal_frame

        container = tk.Frame(frame)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.minimal_title_label = tk.Label(
            container, 
            text="DopplerView", 
            font=self._get_minimal_title_font(),
        )
        self.minimal_title_label.grid(row=0, column=0, pady=(0, 10))

        minimal_logo = self._load_scaled_logo_image(max_width=360, max_height=144)
        if minimal_logo is not None:
            self._minimal_logo_image = minimal_logo
            self.minimal_logo_label = ttk.Label(container, image=self._minimal_logo_image)
            self.minimal_logo_label.grid(row=1, column=0, pady=(0, 20))

        self.btn_load = ttk.Button(container, text="Select .holo file(s)", command=self.load_holo)
        self.btn_load.grid(row=2, column=0, pady=(0, 10))

        # -------------------------------------------------
        # Input measures list
        # -------------------------------------------------

        list_container = ttk.Frame(container)
        list_container.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        list_container.grid_columnconfigure(0, weight=1)
        list_container.grid_rowconfigure(0, weight=1)

        self.minimal_input_listbox = tk.Listbox(
            list_container,
            height=3,
            width=50,
            bg=self._surface_color,
            fg=self._text_fg,
            selectbackground=self._accent_color,
            activestyle="none",
        )

        self.minimal_input_listbox.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        minimal_scrollbar = ttk.Scrollbar(
            list_container,
            orient="vertical",
            command=self.minimal_input_listbox.yview,
        )

        minimal_scrollbar.grid(row=0, column=1, sticky="ns")

        self.minimal_input_listbox.config(
            yscrollcommand=minimal_scrollbar.set
        )

        state = "disabled"
        self.btn_run_minimal = ttk.Button(container, text="Run Full Pipeline", command=self.run_full_pipelines, state=state)
        self.btn_run_minimal.grid(row=4, column=0, pady=10)

        self.progress_minimal = ttk.Progressbar(container, maximum=100)
        self.progress_minimal.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _build_advanced_ui(self):
        frame = self.advanced_frame

        # Make frame expandable
        # Main frame stays 1 column
        frame.grid_columnconfigure(0, weight=1)

        row = 0

        # --- Buttons sub-frame (2 columns ONLY here) ---
        self.buttons_frame = tk.Frame(frame, bg=self._bg_color)
        self.buttons_frame.grid(row=row, column=0, sticky="ew", pady=5)

        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)

        self.btn_load = ttk.Button(
            self.buttons_frame,
            text="Select .holo file(s)",
            command=self.load_holo
        )
        self.btn_load.grid(row=0, column=0, padx=5, sticky="ew")

        self.btn_select_config = ttk.Button(
            self.buttons_frame,
            text="Select config",
            command=self.load_config
        )
        self.btn_select_config.grid(row=0, column=1, padx=5, sticky="ew")

        row += 1

        self.config_path_label = tk.Label(
            self.buttons_frame,
            textvariable=self.config_path,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=600,
        )
        self.config_path_label.grid(row=row, column=1, pady=5, sticky="ew")
        row += 1

        # -------------------------------------------------
        # Input measures panel
        # -------------------------------------------------

        self.input_panel = tk.LabelFrame(frame, text="Input Measures")
        self.input_panel.grid(
            row=row,
            column=0,
            padx=5,
            pady=5,
            sticky="nsew"
        )

        self.input_panel.grid_columnconfigure(0, weight=1)
        self.input_panel.grid_rowconfigure(0, weight=1)

        # Listbox + scrollbar container
        list_container = ttk.Frame(self.input_panel)
        list_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        list_container.grid_columnconfigure(0, weight=1)
        list_container.grid_rowconfigure(0, weight=1)

        self.input_listbox = tk.Listbox(
            list_container,
            height=6,
            bg=self._surface_color,
            fg=self._text_fg,
            selectbackground=self._accent_color,
            activestyle="none",
        )

        self.input_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            list_container,
            orient="vertical",
            command=self.input_listbox.yview
        )

        scrollbar.grid(row=0, column=1, sticky="ns")

        self.input_listbox.config(yscrollcommand=scrollbar.set)

        self.progress_batch = ttk.Progressbar(
            self.input_panel,
            maximum=100
        )

        self.progress_batch.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=5,
            pady=(0, 5)
        )

        row += 1

        row += 1

        # --- Steps frame ---
        self.steps_frame = tk.LabelFrame(frame, text="Pipeline Steps")
        self.steps_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        self.steps_frame.grid_columnconfigure(0, weight=1)
        row += 1

        self.step_vars = {}
        self.step_checkboxes = {}
        
        steps = self.pipeline.get_step_names()
        waves = self.pipeline.engine.build_execution_waves(steps)

        for i, wave in enumerate(waves):
            for j, step in enumerate(wave):
                var = tk.BooleanVar(value=True)

                cb = tk.Checkbutton(
                    self.steps_frame,
                    text=step,
                    variable=var,
                    command=lambda s=step: self.on_step_toggle(s),
                    fg=self._text_fg,
                )
                cb.grid(row=i, column=j, sticky="w")

                self.step_vars[step] = var
                self.step_checkboxes[step] = cb

        # --- Run button ---
        state = "disabled"
        self.btn_run = ttk.Button(frame, text="Run Pipeline", command=self.run_pipelines_with_steps, state=state)
        self.btn_run.grid(row=row, column=0, pady=5, sticky="ew")
        row += 1

        # --- Progress bar ---
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.grid(row=row, column=0, sticky="ew", padx=5)
        row += 1

        # --- Image display ---
        self.image_label = tk.Label(frame)
        self.image_label.grid(row=row, column=0, pady=10, sticky="nsew")

    def _install_drop_targets(self) -> None:
        if DND_FILES is None:
            return
        self._register_drop_target_tree(self.root)

    def _register_drop_target_tree(self, widget: tk.Misc) -> None:
        if DND_FILES is None:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self.on_drop)
        except (AttributeError, tk.TclError):
            pass

        for child in widget.winfo_children():
            self._register_drop_target_tree(child)

    def _populate_configuration_frame(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # -----------------------
        # LEFT: Models frame
        # -----------------------
        models_frame = tk.LabelFrame(parent, text="Models")
        models_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        models_frame.grid_columnconfigure(0, weight=1)

        # -----------------------
        # RIGHT: Config panel
        # -----------------------
        config_panel = tk.LabelFrame(parent, text="Configuration")
        config_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        config_panel.grid_columnconfigure(0, weight=1)

        # --- Radio buttons ---
        radio_frame = tk.Frame(config_panel, bg=self._bg_color)
        radio_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        radio_frame.grid_columnconfigure(0, weight=1)
        radio_frame.grid_columnconfigure(1, weight=1)

        rb_default = tk.Radiobutton(
            radio_frame,
            text="Use default config",
            variable=self.config_mode_var,
            value="default",
            anchor="w",
            command=self.update_config_mode,
        )
        rb_default.grid(row=0, column=0, sticky="w")

        rb_local = tk.Radiobutton(
            radio_frame,
            text="Use local config",
            variable=self.config_mode_var,
            value="local",
            anchor="w",
            command=self.update_config_mode,
        )
        rb_local.grid(row=0, column=1, sticky="w")

        # --- Buttons ---
        ttk.Button(
            config_panel,
            text="Modify dopplerview config",
            command=self.modify_dopplerview_config
        ).grid(row=1, column=0, sticky="ew", pady=5)

        ttk.Button(
            config_panel,
            text="Modify models registry",
            command=self.modify_models_registry
        ).grid(row=2, column=0, sticky="ew", pady=5)

        ttk.Button(
            config_panel,
            text="Modify h5 schema",
            command=self.modify_h5_schema
        ).grid(row=3, column=0, sticky="ew", pady=5)

        ttk.Button(
            config_panel,
            text="Modify output config",
            command=self.modify_output_config
        ).grid(row=4, column=0, sticky="ew", pady=5)

        ctx = self.pipeline.ctx
        mm = ctx.model_manager

        def create_model_selector(parent_widget, label_text, task_name, r):
            tk.Label(parent_widget, text=label_text).grid(row=r, column=0, sticky="w")

            values = mm.get_model_name_list_for_task(task_name)
            var = tk.StringVar(value=values[0] if values else "")

            combo = ttk.Combobox(
                parent_widget,
                textvariable=var,
                values=values,
                state="readonly"
            )
            combo.grid(row=r + 1, column=0, sticky="ew", pady=2)

            def on_change(event=None):
                ctx.change_model_for_task(task_name, var.get())

            combo.bind("<<ComboboxSelected>>", on_change)

            if values:
                ctx.change_model_for_task(task_name, var.get())

            return r + 2

        r = 0

        r = create_model_selector(
            models_frame,
            "Binary vessel segmentation model",
            "retinal_vessel_segmentation",
            r,
        )

        r = create_model_selector(
            models_frame,
            "Artery/Vein segmentation model",
            "retinal_artery_vein_segmentation",
            r,
        )

        r = create_model_selector(
            models_frame,
            "Optic disc segmentation model",
            "optic_disc_segmentation",
            r,
        )

        r = create_model_selector(
            models_frame,
            "Eye laterality classification model",
            "eye_laterality_classification",
            r,
        )

    def show_config(self):
        if hasattr(self, "config_window") and self.config_window.winfo_exists():
            self.config_window.lift()
            self.config_window.focus_force()
            return

        self.config_window = tk.Toplevel(self.root)
        self.config_window.title("DopplerView Configuration")
        self.config_window.geometry("600x240")
        self.config_window.configure(bg=self._bg_color)

        container = ttk.Frame(self.config_window, padding=10)
        container.pack(fill="both", expand=True)

        self._populate_configuration_frame(container)

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
        mode = self.ui_mode_var.get()

        self.minimal_frame.pack_forget()
        self.advanced_frame.pack_forget()

        if mode == "minimal":
            self.minimal_frame.pack(fill="both", expand=True)
            self.root.geometry("600x400")

        elif mode == "advanced":
            self.advanced_frame.pack(fill="both", expand=True)
            self.root.geometry("900x650")
            self.resize_window()

    def resize_window(self):
        image_height = self.image_tk.height() if self.image_tk else 0
        window_height = 650 + image_height  # base height + image height
        self.root.geometry(f"{self.root.winfo_width()}x{window_height}")

    def update_step_color(self, step, state):
        cb = self.step_checkboxes[step]
        if state == "done" or state == "cached":
            color =  "#26ac5c"
        elif state == "running":
            color = "#d7a61e"

        cb.config(selectcolor=color)

    def update_step_display(self):
        pipeline = self.pipeline

        selected = self.get_selected_steps()

        # Steps that will actually run
        pipeline.set_targets(selected)

        for step, cb in self.step_checkboxes.items():
            is_checked = self.step_vars[step].get()
            is_cached = pipeline.is_cached(step)

            # -------- label logic --------
            if is_checked:
                if is_cached:
                    color =  "#26ac5c"
                else:
                    color = "#d7a61e"
            else:
                color = "#ffffff"

            cb.config(selectcolor=color)

    def load_input(self, folders):
        # self.input_folder.set(folders)
        self.cleanup_image()
        self.progress["value"] = 0
        self.progress_minimal["value"] = 0
        self.progress_batch["value"] = 0
        self.pipeline.ctx.clear_input_list()

        if isinstance(folders, str):
            folder_list = [Path(f) for f in folders.split() if f]
        else:
            folder_list = [Path(f) for f in folders]

        self.pipeline.load_input_list_from_list(folder_list)
        # self.pipeline
        # if folder_list[0].suffix == ".holo":
        #     self.pipeline.ctx.load_input_folder(folder_list[0])  # load first by default, pipeline will handle the rest in batch mode
        # self.config_path.set(self.pipeline.ctx.dopplerview_config_path)
        self.update_step_display()

        self.btn_run.config(state="enabled")
        self.btn_run_minimal.config(state="enabled")

        self.refresh_input_listbox()

    def refresh_input_listbox(self):
        # Advanced UI list
        if hasattr(self, "input_listbox"):
            self.input_listbox.delete(0, tk.END)

            for path in self.pipeline.ctx.input_list:
                self.input_listbox.insert(tk.END, str(path))

        # Minimal UI list
        if hasattr(self, "minimal_input_listbox"):
            self.minimal_input_listbox.delete(0, tk.END)

            for path in self.pipeline.ctx.input_list:
                self.minimal_input_listbox.insert(tk.END, str(path))

    def load_holo(self):
        file_path = filedialog.askopenfilenames(filetypes=[("Holo files", "*.holo")], defaultextension=".holo")
        if file_path:
            self.load_input(list(file_path))

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], defaultextension=".json")
        if file_path:
            self.pipeline.load_dopplerview_config(file_path)
            self.config_path.set(file_path)
    
    def open_with_default_app(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if sys.platform.startswith("win"):
            logger.info(f"Opening {path} with default application...")
            os.startfile(path)  # Windows
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])  # macOS
        else:
            subprocess.run(["xdg-open", path])  # Linux

    def modify_models_registry(self):
        self.open_with_default_app(self.pipeline.ctx.model_registry_path)
        self.pipeline.load_model_registry(self.pipeline.ctx.model_registry_path)
        self._build_advanced_ui()  # rebuild to update model lists in dropdowns

    def modify_dopplerview_config(self):
        self.open_with_default_app(self.pipeline.ctx.dopplerview_config_path)
        self.pipeline.load_dopplerview_config(self.pipeline.ctx.dopplerview_config_path)

    def modify_h5_schema(self):
        self.open_with_default_app(self.output_manager.schema_path)
        self.output_manager.load_h5_schema(self.output_manager.schema_path)

    def modify_output_config(self):
        self.open_with_default_app(self.output_manager.output_config_path)
        self.output_manager.load_output_config(self.output_manager.output_config_path)

    def update_config_mode(self):
        mode = self.config_mode_var.get()
        self.pipeline.set_config_mode(mode)
        if mode == "local":
            if self.pipeline.ctx.DV_folder is not None:
                config_path = self.pipeline.ctx.DV_folder.dopplerview_config
                self.pipeline.load_dopplerview_config(config_path)
            else:
                config_path = "No config loaded"
        else:
            config_path = user_config.ensure_config_file("default_DV_params.json")
            self.pipeline.load_dopplerview_config(config_path)

        self.config_path.set(config_path)

    def get_selected_steps(self):
        return [step for step, var in self.step_vars.items() if var.get()]

    def on_drop(self, event):
        path = event.data.strip("{}")  # windows fix
        self.load_input(path)

    def run_full_pipelines(self):
        # full pipeline
        self.run_pipelines(None)

    def run_pipelines_with_steps(self):
        steps = self.get_selected_steps()
        self.run_pipelines(steps=steps)

    def run_pipelines(self, steps=None):
        self.btn_run.config(state="disabled")
        self.btn_run_minimal.config(state="disabled")
        thread = threading.Thread(
            target=self._run_pipelines_worker,
            args=(steps,),
            daemon=True
        )
        thread.start()

        self.root.after(100, self.check_queue)

    def _run_pipelines_worker(self, steps):
        def callback(event, *args):
            self.queue.put((event, args))
        try:
            self.pipeline.run_batch(targets=steps, callback=callback)
        except Exception as e:
            self.queue.put(("error", str(e)))

    def step_done_output(self, step_name):
        if step_name == "preprocess":
            img = self.pipeline.ctx.get("M0_ff_image")
            if img is not None:
                self.display_image(img)

        elif step_name == "retinal_vessel_segmentation":
            img = self.pipeline.ctx.get("M0_ff_image")
            vessel = self.pipeline.ctx.get("retinal_vessel_mask")
            if img is not None and vessel is not None:
                overlay = self.overlay(img, vessel, None)
                self.display_image(overlay)

        elif step_name == "retinal_artery_vein_segmentation":
            img = self.pipeline.ctx.get("M0_ff_image")
            art = self.pipeline.ctx.get("retinal_artery_mask")
            vein = self.pipeline.ctx.get("retinal_vein_mask")
            if img is not None and art is not None and vein is not None:
                overlay = self.overlay(img, art, vein)
                self.display_image(overlay)
        
    def check_queue(self):
        try:
            while True:
                event, data = self.queue.get_nowait()
                if event == "pipeline_start":

                    self.config_path.set(self.pipeline.ctx.dopplerview_config_path) # refresh config path

                    i, total = data
                    self.measure_index = i
                    self.input_listbox.selection_clear(0, tk.END)
                    self.minimal_input_listbox.selection_clear(0, tk.END)

                    if i < self.input_listbox.size():
                        self.input_listbox.selection_set(i)
                        self.input_listbox.activate(i)
                        self.input_listbox.see(i)

                    if i < self.minimal_input_listbox.size():
                        self.minimal_input_listbox.selection_set(i)
                        self.minimal_input_listbox.activate(i)
                        self.minimal_input_listbox.see(i)

                    self.progress["value"] = 0
                    progress = (i / total) * 100
                    self.progress_batch["value"] = progress
                    self.progress_minimal["value"] = progress

                elif event == "batch_start":
                    self.progress_batch["value"] = 0

                elif event == "step_start":
                    step_name, i, total = data
                    step_ratio = i / total
                    measure_ratio = self.measure_index / len(self.pipeline.ctx.input_list)
                    self.progress["value"] = step_ratio * 100

                    self.progress_minimal["value"] = measure_ratio * 100 + step_ratio * 100 / len(self.pipeline.ctx.input_list)
                    self.update_step_color(step_name, "running")

                elif event == "step_done":
                    step_name, elapsed = data
                    self.update_step_color(step_name, "done")
                    self.step_done_output(step_name)

                elif event == "step_skipped":
                    step_name = data[0]
                    self.update_step_color(step_name, "cached")

                elif event == "pipeline_done":
                    self.progress["value"] = 100
                    self.btn_run.config(state="enabled")

                    self.btn_run_minimal.config(state="enabled")

                    self.update_step_display()  # refresh colors to reflect final cache status

                elif event == "batch_done":
                    self.progress_batch["value"] = 100
                    self.progress_minimal["value"] = 100

                elif event == "error":
                    logger.error("Error:", data)

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    # -------------------
    # Logo
    # -------------------

    def _resource_roots(self) -> list[Path]:
        roots: list[Path] = []
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            roots.append(Path(frozen_root))
        roots.append(Path(__file__).resolve().parents[1])
        roots.append(Path.cwd())
        return roots

    def _resolve_logo_path(self) -> Path | None:
        for root in self._resource_roots():
            candidate = root / "DopplerView.png"
            if candidate.is_file():
                return candidate
        return None

    def _load_logo_image(self) -> tk.PhotoImage | None:
        logo_path = self._resolve_logo_path()
        if logo_path is None:
            return None
        try:
            return tk.PhotoImage(file=str(logo_path))
        except tk.TclError:
            return None

    def _load_scaled_logo_image(
        self,
        *,
        max_width: int,
        max_height: int,
    ) -> tk.PhotoImage | None:
        image = self._load_logo_image()
        if image is None:
            return None

        scale_x = max(1, (image.width() + max_width - 1) // max_width)
        scale_y = max(1, (image.height() + max_height - 1) // max_height)
        scale = max(scale_x, scale_y)
        if scale > 1:
            image = image.subsample(scale, scale)
        return image

    def _set_window_icon(self) -> None:
        image = self._load_logo_image()
        if image is None:
            return
        self._window_icon_image = image
        try:
            self.root.iconphoto(True, self._window_icon_image)
        except tk.TclError:
            pass

    # -------------------
    # Image utils
    # -------------------

    def display_image(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.image_tk = np_to_tk(img)  # keep reference!
        self.image_label.config(image=self.image_tk)
        if self.ui_mode_var.get() == "advanced":
            self.resize_window()

    def cleanup_image(self):
        self.image_label.config(image="")

    def overlay(self, image, artery_mask, vein_mask):
        img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if artery_mask is not None:
            if vein_mask is not None:
                img[artery_mask > 0] = [255, 0, 0]
            else:
                img[artery_mask > 0] = [255, 250, 250]

        if vein_mask is not None:
            img[vein_mask > 0] = [0, 0, 255]

        return img
    
    # -------------------
    # Help
    # -------------------

    def show_help(self):
        help_text = (
            "DopplerView is a tool for segmentation, classification and analysis of diverse structures and signals on data issued from laser doppler holography.\n"
            "It takes as input measure.holo file(s) with a corresponding measure/measure_HD folder containing the hologram data resulting from holodoppler processing of raw videos, and produces a variety of outputs including artery/vein segmentation masks, velocity estimates, waveform analyses, and more.\n\n"
            "1. Load a .holo file, or drag-and-drop it into the application. You can also load a batch folder containing multiple .holo files, or a .txt file containing a list of paths to .holo files.\n"
            "2. In advanced UI (View -> Advanced UI), select which pipeline steps to run or run the full pipeline.\n"
            "3. View the results, including artery/vein segmentation overlays.\n\n"
            "For more information, visit our GitHub repository: https://github.com/DigitalHolography/DopplerView"
        )
        tk.messagebox.showinfo("Help - DopplerView", help_text)


# -------------------
# Run app
# -------------------

if __name__ == "__main__":
    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    log_config.setup_logging()

    matplotlib.use('Agg')

    app = MainWindow(root)
    root.mainloop()