import sys
import os
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import threading
import queue
import multiprocessing
import traceback
import matplotlib

import numpy as np
import cv2
from PIL import Image, ImageTk

from dopplerview.input_output import log_config, user_config
from dopplerview.input_output.output_manager import OutputManager
from dopplerview.pipeline.pipeline import Pipeline
from dopplerview._version import __version__ as app_version

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


def _overlay_preview(image, artery_mask=None, vein_mask=None):
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


def _resize_preview_for_queue(img, max_side=900):
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


def _build_step_preview(pipeline, step_name):
    """Extract the same previews as the Tk app, but from the child pipeline context."""
    ctx = pipeline.ctx

    if step_name == "preprocess":
        return _resize_preview_for_queue(ctx.get("M0_ff_image"))

    if step_name == "retinal_vessel_segmentation":
        img = ctx.get("M0_ff_image")
        vessel = ctx.get("retinal_vessel_mask")
        return _resize_preview_for_queue(_overlay_preview(img, vessel, None))

    if step_name == "retinal_artery_vein_segmentation":
        img = ctx.get("M0_ff_image")
        art = ctx.get("retinal_artery_mask")
        vein = ctx.get("retinal_vein_mask")
        return _resize_preview_for_queue(_overlay_preview(img, art, vein))

    return None



class _MultiprocessingQueueLogHandler(logging.Handler):
    """Forward log messages from the pipeline child process to the Tk parent process."""

    def __init__(self, queue_out):
        super().__init__()
        self.queue_out = queue_out

    def emit(self, record):
        try:
            self.queue_out.put((
                "log",
                {
                    "name": record.name,
                    "levelno": record.levelno,
                    "levelname": record.levelname,
                    "message": self.format(record),
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                },
            ))
        except Exception:
            # Logging must never crash the worker.
            pass


class _QueueTextStream:
    """Redirect stdout/stderr lines from the child process to the parent logger."""

    def __init__(self, queue_out, *, levelno: int, name: str):
        self.queue_out = queue_out
        self.levelno = levelno
        self.name = name
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.queue_out.put((
                    "log",
                    {
                        "name": self.name,
                        "levelno": self.levelno,
                        "levelname": logging.getLevelName(self.levelno),
                        "message": line,
                        "pathname": "",
                        "lineno": 0,
                    },
                ))

    def flush(self):
        if self._buffer.strip():
            self.queue_out.put((
                "log",
                {
                    "name": self.name,
                    "levelno": self.levelno,
                    "levelname": logging.getLevelName(self.levelno),
                    "message": self._buffer.strip(),
                    "pathname": "",
                    "lineno": 0,
                },
            ))
        self._buffer = ""


def _configure_child_process_logging(queue_out):
    """
    Configure logging in the spawned pipeline process.

    On Windows, multiprocessing uses spawn: the child process starts with a fresh
    interpreter and does not reliably inherit the parent logging handlers.  This
    handler forwards child logs to the parent process through the same event queue
    used for pipeline progress.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    handler = _MultiprocessingQueueLogHandler(queue_out)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    sys.stdout = _QueueTextStream(queue_out, levelno=logging.INFO, name="pipeline.stdout")
    sys.stderr = _QueueTextStream(queue_out, levelno=logging.ERROR, name="pipeline.stderr")

def pipeline_process_worker(run_spec, queue_out):
    """
    Run the heavy DopplerView pipeline in a child process.

    The Tk process must remain UI-only.  This isolates native crashes/hangs from
    OpenCV, ONNXRuntime, PyTorch, h5py, etc. from Tkinter's event loop.
    """
    _configure_child_process_logging(queue_out)

    try:
        h5_schema_path = run_spec["h5_schema_path"]
        output_config_path = run_spec["output_config_path"]
        models_config_path = run_spec["models_config_path"]
        dopplerview_config_path = run_spec.get("dopplerview_config_path")
        input_list = [Path(p) for p in run_spec["input_list"]]
        targets = run_spec.get("steps")
        selected_models = run_spec.get("selected_models", {})
        config_mode = run_spec.get("config_mode", "default")
        output_enabled = bool(run_spec.get("output_enabled", False))

        output_manager = OutputManager(
            h5_schema_path,
            output_config_path,
            output_enabled=output_enabled,
        )
        pipeline = Pipeline(output_manager=output_manager)

        pipeline.load_model_registry(models_config_path)
        if dopplerview_config_path and Path(dopplerview_config_path).exists():
            pipeline.load_dopplerview_config(dopplerview_config_path)

        try:
            pipeline.set_config_mode(config_mode)
        except Exception:
            logger.exception("Failed to set pipeline config mode in worker")

        for task_name, model_name in selected_models.items():
            if model_name:
                try:
                    pipeline.ctx.change_model_for_task(task_name, model_name)
                except Exception:
                    logger.exception("Failed to select model %s for task %s", model_name, task_name)

        pipeline.ctx.clear_input_list()
        pipeline.load_input_list_from_list(input_list)

        def callback(event, *args):
            queue_out.put((event, args))

            if event == "step_done" and args:
                step_name = args[0]
                preview = _build_step_preview(pipeline, step_name)
                if preview is not None:
                    queue_out.put(("preview_image", (preview,)))

        pipeline.run_batch(targets=targets, callback=callback)
        queue_out.put(("worker_done", None))

    except BaseException:
        queue_out.put(("error", traceback.format_exc()))


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title(f"DopplerView {app_version}")

        self._minimal_title_font: tkfont.Font | None = None

        self._config_mtimes = {}
        self._config = {}
        self.root.bind("<FocusIn>", self.on_focus)

        # --- pipeline init ---

        h5_schema_path = user_config.ensure_config_file("h5_schema.json")
        self.register_config_file(h5_schema_path, "h5_schema")
        output_config_path = user_config.ensure_config_file("output_config.json")
        self.register_config_file(output_config_path, "output_config")
        self.output_manager = OutputManager(h5_schema_path, output_config_path, output_enabled=False)
        self.pipeline = Pipeline(output_manager=self.output_manager)

        models_config = user_config.ensure_config_file("models.yaml")
        self.pipeline.load_model_registry(models_config)
        self.register_config_file(models_config, "models_config")

        config_path = user_config.ensure_config_file("default_DV_params.json")
        # self.pipeline.load_dopplerview_config(config_path)
        self.config_path = tk.StringVar(value=str(config_path))
        self.status_var = tk.StringVar(value="Ready")
        self.register_config_file(config_path, "dopplerview_config")

        self.image_tk = None  # keep reference (IMPORTANT)

        self.queue = queue.Queue()
        self.pipeline_worker = None
        self.output_worker = None
        self.mp_context = multiprocessing.get_context("spawn")
        self.selected_models: dict[str, str] = {}

        self.theme_var = tk.StringVar(value="dark")
        self._menus: list[tk.Menu] = []

        self._apply_theme()
        self._set_window_icon()

        # --- UI layout --
        self._build_ui()
        self._darken_tk_widget(self.root)
        self._install_drop_targets()
        self.update_mode()  # set initial mode

        self.config_mode_var = tk.StringVar(value="default")
        self.update_config_mode() # set initial config mode

        self.step_index = 0
        self.measure_index = 0

        self.enable_debug_output = False


    def _apply_theme(self, theme: str | None = None) -> None:
        """
        Apply the selected application theme.

        This controls both ttk widgets and classic tk widgets.  ttk styling is
        applied through ttk.Style; classic tk widgets are handled through the Tk
        option database and by recursively re-configuring existing widgets.
        """
        if theme is None:
            theme = self.theme_var.get() if hasattr(self, "theme_var") else "dark"
        theme = theme.lower()
        if theme not in {"dark", "light"}:
            theme = "dark"

        if hasattr(self, "theme_var"):
            self.theme_var.set(theme)

        style = ttk.Style(self.root)
        self._style = style
        self._theme_name = theme

        palettes = {
            "dark": {
                "bg": "#0f1116",
                "surface": "#1b1f27",
                "surface_alt": "#242a35",
                "text": "#e8eef5",
                "muted": "#9aa6b5",
                "accent": "#4f9dff",
                "select": "#2d5f9a",
                "disabled": "#6f7a88",
                "border": "#303746",
            },
            "light": {
                "bg": "#f5f7fb",
                "surface": "#ffffff",
                "surface_alt": "#e9eef7",
                "text": "#1f2937",
                "muted": "#667085",
                "accent": "#2563eb",
                "select": "#cfe1ff",
                "disabled": "#9ca3af",
                "border": "#cfd7e6",
            },
        }
        palette = palettes[theme]

        self._bg_color = palette["bg"]
        self._surface_color = palette["surface"]
        self._surface_alt_color = palette["surface_alt"]
        self._text_fg = palette["text"]
        self._muted_fg = palette["muted"]
        self._accent_color = palette["accent"]
        self._select_color = palette["select"]
        self._disabled_fg = palette["disabled"]
        self._border_color = palette["border"]
        self._text_bg = self._surface_color

        # Use a theme that accepts color customization.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # sv_ttk can improve widget metrics; colors are still forced below.
        if sv_ttk:
            try:
                sv_ttk.set_theme(theme)
                style.theme_use("clam")
            except Exception:
                pass

        self.root.configure(bg=self._bg_color)

        # Tk option database: affects widgets created later unless explicitly overridden.
        self.root.option_add("*Background", self._bg_color)
        self.root.option_add("*Foreground", self._text_fg)
        self.root.option_add("*activeBackground", self._select_color)
        self.root.option_add("*activeForeground", self._text_fg)
        self.root.option_add("*selectBackground", self._select_color)
        self.root.option_add("*selectForeground", self._text_fg)
        self.root.option_add("*insertBackground", self._text_fg)
        self.root.option_add("*disabledForeground", self._disabled_fg)
        self.root.option_add("*Menu.Background", self._surface_color)
        self.root.option_add("*Menu.Foreground", self._text_fg)
        self.root.option_add("*Menu.activeBackground", self._select_color)
        self.root.option_add("*Menu.activeForeground", self._text_fg)

        # ---------- ttk styles ----------
        style.configure(".", background=self._bg_color, foreground=self._text_fg)

        style.configure("TFrame", background=self._bg_color)
        style.configure("Dark.TFrame", background=self._bg_color)
        style.configure("Surface.TFrame", background=self._surface_color)

        style.configure("TLabel", background=self._bg_color, foreground=self._text_fg)
        style.configure("Muted.TLabel", background=self._bg_color, foreground=self._muted_fg)

        for labelframe_style in ("TLabelframe", "TLabelFrame"):
            style.configure(
                labelframe_style,
                background=self._bg_color,
                foreground=self._text_fg,
                bordercolor=self._border_color,
                lightcolor=self._border_color,
                darkcolor=self._border_color,
            )
        for label_style in ("TLabelframe.Label", "TLabelFrame.Label"):
            style.configure(label_style, background=self._bg_color, foreground=self._text_fg)

        style.configure(
            "TButton",
            background=self._surface_alt_color,
            foreground=self._text_fg,
            bordercolor=self._border_color,
            focuscolor=self._accent_color,
            padding=(10, 5),
        )
        style.map(
            "TButton",
            background=[
                ("disabled", self._surface_color),
                ("active", self._select_color),
                ("pressed", self._select_color),
            ],
            foreground=[
                ("disabled", self._disabled_fg),
                ("active", self._text_fg),
                ("pressed", self._text_fg),
            ],
        )

        style.configure(
            "TEntry",
            fieldbackground=self._surface_color,
            background=self._surface_color,
            foreground=self._text_fg,
            insertcolor=self._text_fg,
            bordercolor=self._border_color,
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", self._surface_color), ("readonly", self._surface_color)],
            foreground=[("disabled", self._disabled_fg)],
        )

        style.configure(
            "TCombobox",
            fieldbackground=self._surface_color,
            background=self._surface_alt_color,
            foreground=self._text_fg,
            arrowcolor=self._text_fg,
            bordercolor=self._border_color,
            selectbackground=self._select_color,
            selectforeground=self._text_fg,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self._surface_color)],
            foreground=[("readonly", self._text_fg)],
            background=[("active", self._surface_alt_color)],
            arrowcolor=[("active", self._text_fg)],
        )

        style.configure("TCheckbutton", background=self._bg_color, foreground=self._text_fg, focuscolor=self._accent_color)
        style.map(
            "TCheckbutton",
            background=[("active", self._bg_color)],
            foreground=[("disabled", self._disabled_fg), ("active", self._text_fg)],
        )

        style.configure("TRadiobutton", background=self._bg_color, foreground=self._text_fg, focuscolor=self._accent_color)
        style.map(
            "TRadiobutton",
            background=[("active", self._bg_color)],
            foreground=[("disabled", self._disabled_fg), ("active", self._text_fg)],
        )

        style.configure(
            "Vertical.TScrollbar",
            background=self._surface_alt_color,
            troughcolor=self._bg_color,
            bordercolor=self._border_color,
            arrowcolor=self._text_fg,
            gripcount=0,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", self._select_color)],
            arrowcolor=[("active", self._text_fg)],
        )

        style.configure(
            "TProgressbar",
            background=self._accent_color,
            troughcolor=self._surface_color,
            bordercolor=self._border_color,
            lightcolor=self._accent_color,
            darkcolor=self._accent_color,
        )

    def set_theme(self, theme: str) -> None:
        """Switch the whole application between the available themes."""
        self._apply_theme(theme)
        self._style_existing_widgets()
        self.update_step_display()

    def _style_existing_widgets(self) -> None:
        """Re-apply current colors to all already-created widgets and menus."""
        self._darken_tk_widget(self.root)

        if hasattr(self, "config_window") and self.config_window.winfo_exists():
            self._darken_tk_widget(self.config_window)

        for menu in getattr(self, "_menus", []):
            self._darken_menu(menu)

    def _darken_menu(self, menu: tk.Menu) -> None:
        """Best-effort dark styling for classic Tk menus."""
        try:
            menu.configure(
                bg=self._surface_color,
                fg=self._text_fg,
                activebackground=self._select_color,
                activeforeground=self._text_fg,
                disabledforeground=self._disabled_fg,
                selectcolor=self._accent_color,
            )
        except tk.TclError:
            pass

    def _darken_tk_widget(self, widget: tk.Misc) -> None:
        """
        Recursively apply dark colors to classic Tk widgets.

        This is necessary because ttk themes do not affect tk.Frame, tk.Label,
        tk.LabelFrame, tk.Listbox, tk.Checkbutton, tk.Radiobutton, tk.Canvas,
        tk.Text, or tk.Menu.
        """
        cls = widget.winfo_class()

        try:
            if isinstance(widget, tk.Menu):
                self._darken_menu(widget)

            elif cls in {"Frame", "TFrame"}:
                # ttk.Frame does not accept bg; tk.Frame does.
                if not isinstance(widget, ttk.Frame):
                    widget.configure(bg=self._bg_color)

            elif cls in {"Labelframe", "LabelFrame"}:
                widget.configure(
                    bg=self._bg_color,
                    fg=self._text_fg,
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                )

            elif cls == "Label":
                widget.configure(
                    bg=self._bg_color,
                    fg=self._text_fg,
                    activebackground=self._bg_color,
                    activeforeground=self._text_fg,
                )

            elif cls == "Button":
                widget.configure(
                    bg=self._surface_alt_color,
                    fg=self._text_fg,
                    activebackground=self._select_color,
                    activeforeground=self._text_fg,
                    disabledforeground=self._disabled_fg,
                    highlightbackground=self._bg_color,
                    highlightcolor=self._accent_color,
                )

            elif cls in {"Checkbutton", "Radiobutton"}:
                widget.configure(
                    bg=self._bg_color,
                    fg=self._text_fg,
                    activebackground=self._bg_color,
                    activeforeground=self._text_fg,
                    disabledforeground=self._disabled_fg,
                    selectcolor=self._surface_color,
                    highlightbackground=self._bg_color,
                    highlightcolor=self._accent_color,
                )

            elif cls == "Listbox":
                widget.configure(
                    bg=self._surface_color,
                    fg=self._text_fg,
                    selectbackground=self._select_color,
                    selectforeground=self._text_fg,
                    activestyle="none",
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                    relief="flat",
                )

            elif cls == "Text":
                widget.configure(
                    bg=self._surface_color,
                    fg=self._text_fg,
                    insertbackground=self._text_fg,
                    selectbackground=self._select_color,
                    selectforeground=self._text_fg,
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                    relief="flat",
                )

            elif cls == "Entry":
                widget.configure(
                    bg=self._surface_color,
                    fg=self._text_fg,
                    insertbackground=self._text_fg,
                    selectbackground=self._select_color,
                    selectforeground=self._text_fg,
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                    relief="flat",
                )

            elif cls == "Canvas":
                widget.configure(
                    bg=self._bg_color,
                    highlightbackground=self._bg_color,
                    highlightcolor=self._accent_color,
                )

            elif cls == "Toplevel":
                widget.configure(bg=self._bg_color)

        except tk.TclError:
            # Some widgets/classes do not support all options above.
            pass

        for child in widget.winfo_children():
            self._darken_tk_widget(child)

    # -------------------
    # UI
    # -------------------

    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self.root, padding=10, style="Dark.TFrame")
        container.pack(fill="both", expand=True)

        self.minimal_frame = ttk.Frame(container, padding=10, style="Dark.TFrame")
        self.advanced_frame = ttk.Frame(container, padding=10, style="Dark.TFrame")

        self._build_minimal_ui()
        self._build_advanced_ui()

    def _build_menu(self) -> None:
        self.ui_mode_var = tk.StringVar(value="minimal")
        self._menus = []

        menu_bar = tk.Menu(
            self.root,
            bg=self._surface_color,
            fg=self._text_fg,
            activebackground=self._select_color,
            activeforeground=self._text_fg,
            disabledforeground=self._disabled_fg,
        )
        self._menus.append(menu_bar)

        view_menu = tk.Menu(
            menu_bar,
            tearoff=False,
            bg=self._surface_color,
            fg=self._text_fg,
            activebackground=self._select_color,
            activeforeground=self._text_fg,
            disabledforeground=self._disabled_fg,
            selectcolor=self._accent_color,
        )
        self._menus.append(view_menu)
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

        config_menu = tk.Menu(
            menu_bar,
            tearoff=False,
            bg=self._surface_color,
            fg=self._text_fg,
            activebackground=self._select_color,
            activeforeground=self._text_fg,
            disabledforeground=self._disabled_fg,
        )
        self._menus.append(config_menu)
        config_menu.add_command(label="Open Configuration", command=self.show_config)
        config_menu.add_separator()
        config_menu.add_command(label="Modify dopplerview config", command=self.modify_dopplerview_config)
        config_menu.add_command(label="Modify models registry", command=self.modify_models_registry)
        config_menu.add_command(label="Modify h5 schema", command=self.modify_h5_schema)
        config_menu.add_command(label="Modify output config", command=self.modify_output_config)
        menu_bar.add_cascade(label="Config", menu=config_menu)

        theme_menu = tk.Menu(
            menu_bar,
            tearoff=False,
            bg=self._surface_color,
            fg=self._text_fg,
            activebackground=self._select_color,
            activeforeground=self._text_fg,
            disabledforeground=self._disabled_fg,
            selectcolor=self._accent_color,
        )
        self._menus.append(theme_menu)
        theme_menu.add_radiobutton(
            label="Light",
            value="light",
            variable=self.theme_var,
            command=lambda: self.set_theme("light"),
        )
        theme_menu.add_radiobutton(
            label="Dark",
            value="dark",
            variable=self.theme_var,
            command=lambda: self.set_theme("dark"),
        )
        menu_bar.add_cascade(label="Theme", menu=theme_menu)

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

        container = tk.Frame(frame, bg=self._bg_color)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.minimal_title_label = tk.Label(
            container,
            text="DopplerView",
            font=self._get_minimal_title_font(),
            bg=self._bg_color,
            fg=self._text_fg,
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

        list_container = ttk.Frame(container, style="Dark.TFrame")
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
            selectbackground=self._select_color,
            selectforeground=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
            relief="flat",
            activestyle="none",
            exportselection=False,
        )

        self.minimal_input_listbox.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        self.minimal_input_listbox.bind("<Button-1>", lambda e: "break")
        self.minimal_input_listbox.bind("<B1-Motion>", lambda e: "break")
        self.minimal_input_listbox.bind("<Key>", lambda e: "break")

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
        self.btn_run_minimal = ttk.Button(container, text="Run Full Pipeline", command=self.run_pipelines_with_steps, state=state)
        self.btn_run_minimal.grid(row=4, column=0, pady=10)

        self.progress_minimal = ttk.Progressbar(container, maximum=100)
        self.progress_minimal.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 4))

        self.status_label_minimal = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="center",
            style="Muted.TLabel",
        )
        self.status_label_minimal.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))

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
            command=self.load_dopplerview_config
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

        self.input_panel = tk.LabelFrame(
            frame,
            text="Input Measures",
            bg=self._bg_color,
            fg=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
        )
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
        list_container = ttk.Frame(self.input_panel, style="Dark.TFrame")
        list_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        list_container.grid_columnconfigure(0, weight=1)
        list_container.grid_rowconfigure(0, weight=1)

        self.input_listbox = tk.Listbox(
            list_container,
            height=6,
            bg=self._surface_color,
            fg=self._text_fg,
            selectbackground=self._select_color,
            selectforeground=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
            relief="flat",
            activestyle="none",
            exportselection=False,
        )
        self.input_listbox.grid(row=0, column=0, sticky="nsew")
        self.input_listbox.bind("<Button-1>", lambda e: "break")
        self.input_listbox.bind("<B1-Motion>", lambda e: "break")
        self.input_listbox.bind("<Key>", lambda e: "break")

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
        self.steps_frame = tk.LabelFrame(
            frame,
            text="Pipeline Steps",
            bg=self._bg_color,
            fg=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
        )
        self.steps_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        self.steps_frame.grid_columnconfigure(0, weight=1)
        row += 1

        self.step_vars = {}
        self.step_checkboxes = {}
        
        steps = self.pipeline.get_step_names()
        waves = self.pipeline.engine.build_execution_waves(steps)

        optional_steps = ["retinal_vessel_velocity_estimator", "arterial_waveform_analysis"]

        for i, wave in enumerate(waves):
            for j, step in enumerate(wave):
                var = tk.BooleanVar(value=step not in optional_steps)

                cb = tk.Checkbutton(
                    self.steps_frame,
                    text=step,
                    variable=var,
                    command=lambda s=step: self.on_step_toggle(s),
                    bg=self._bg_color,
                    fg=self._text_fg,
                    activebackground=self._bg_color,
                    activeforeground=self._text_fg,
                    selectcolor=self._surface_color,
                    highlightbackground=self._bg_color,
                    highlightcolor=self._accent_color,
                )
                cb.grid(row=i, column=j, sticky="w")

                self.step_vars[step] = var
                self.step_checkboxes[step] = cb
        self.update_step_display()

        # --- Run button ---
        state = "disabled"
        self.btn_run = ttk.Button(frame, text="Run Pipeline", command=self.run_pipelines_with_steps, state=state)
        self.btn_run.grid(row=row, column=0, pady=5, padx=3, sticky="ew")
        row += 1

        # --- Progress bar ---
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.grid(row=row, column=0, sticky="ew", padx=5, pady=(0, 4))
        row += 1

        self.status_label = ttk.Label(
            frame,
            textvariable=self.status_var,
            anchor="center",
            style="Muted.TLabel",
        )
        self.status_label.grid(row=row, column=0, sticky="ew", padx=5, pady=(0, 6))
        row += 1

        # --- Image display ---
        self.image_label = tk.Label(frame, bg=self._bg_color, fg=self._text_fg)
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
        models_frame = tk.LabelFrame(
            parent,
            text="Models",
            bg=self._bg_color,
            fg=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
        )
        models_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        models_frame.grid_columnconfigure(0, weight=1)

        steps_frame = tk.LabelFrame(
            parent,
            text="Steps",
            bg=self._bg_color,
            fg=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
        )
        steps_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        steps_frame.grid_columnconfigure(0, weight=1)

        debug_output_button = ttk.Checkbutton(
            steps_frame,
            text="Enable Debug Output",
            command=self.toggle_debug_output
        )
        debug_output_button.grid(row=0, column=0, sticky="ew", pady=5)

        # -----------------------
        # RIGHT: Config panel
        # -----------------------
        config_panel = tk.LabelFrame(
            parent,
            text="Configuration",
            bg=self._bg_color,
            fg=self._text_fg,
            highlightbackground=self._border_color,
            highlightcolor=self._accent_color,
        )
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
            bg=self._bg_color,
            fg=self._text_fg,
            activebackground=self._bg_color,
            activeforeground=self._text_fg,
            selectcolor=self._surface_color,
        )
        rb_default.grid(row=0, column=0, sticky="w")

        rb_local = tk.Radiobutton(
            radio_frame,
            text="Use local config",
            variable=self.config_mode_var,
            value="local",
            anchor="w",
            command=self.update_config_mode,
            bg=self._bg_color,
            fg=self._text_fg,
            activebackground=self._bg_color,
            activeforeground=self._text_fg,
            selectcolor=self._surface_color,
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
            tk.Label(
                parent_widget,
                text=label_text,
                bg=self._bg_color,
                fg=self._text_fg,
            ).grid(row=r, column=0, sticky="w")

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
                model_name = var.get()
                self.selected_models[task_name] = model_name
                ctx.change_model_for_task(task_name, model_name)

            combo.bind("<<ComboboxSelected>>", on_change)

            if values:
                self.selected_models[task_name] = var.get()
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
        self.config_window.geometry("600x300")
        self.config_window.configure(bg=self._bg_color)

        container = ttk.Frame(self.config_window, padding=10, style="Dark.TFrame")
        container.pack(fill="both", expand=True)

        self._populate_configuration_frame(container)
        self._darken_tk_widget(self.config_window)

    # -------------------
    # Actions
    # -------------------

    def toggle_debug_output(self):
        self.enable_debug_output = not self.enable_debug_output
        if self.enable_debug_output:
            self.output_manager.enable_output()
        else:
            self.output_manager.disable_output()

    def on_focus(self, event=None):
        self.check_config_updates()

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
            self.root.geometry("600x420")

        elif mode == "advanced":
            self.advanced_frame.pack(fill="both", expand=True)
            self.root.geometry("850x650")
            self.resize_window()

    def resize_window(self):
        image_height = self.image_tk.height() if self.image_tk else 0
        window_height = 580 + image_height  # base height + image height
        self.root.geometry(f"{self.root.winfo_width()}x{window_height}")

    def update_step_color(self, step, state):
        cb = self.step_checkboxes[step]
        if state == "done" or state == "cached":
            color = "#26ac5c"
        elif state == "running":
            color = "#d7a61e"
        else:
            color = self._surface_color

        cb.config(selectcolor=color, bg=self._bg_color, fg=self._text_fg)

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
                color = self._surface_color

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

        self.update_step_display()

        self.btn_run.config(state="enabled")
        self.btn_run_minimal.config(state="enabled")

        self.refresh_input_listbox()

        n_inputs = len(self.pipeline.ctx.input_list)
        if n_inputs == 0:
            self.status_var.set("Ready")
        else:
            self.status_var.set(f"Loaded {n_inputs} input file(s)")

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

    # -------------------
    # Configuration
    # -------------------

    def load_dopplerview_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], defaultextension=".json")
        if file_path:
            self.pipeline.load_dopplerview_config(file_path)
            self.config_path.set(file_path)

    def reload_config(self, path):
        config_type = self._config.get(path)

        if config_type == "dopplerview_config":
            self.pipeline.load_dopplerview_config(path)

        elif config_type == "models_config":
            self.pipeline.load_model_registry(path)
            self._build_advanced_ui()  # rebuild to update model lists in dropdowns
            self._darken_tk_widget(self.advanced_frame)

        elif config_type == "h5_schema":
            self.output_manager.load_h5_schema(path)

        elif config_type == "output_config":
            self.output_manager.load_output_config(path)

    def check_config_updates(self):
        for path, old_mtime in self._config_mtimes.items():
            try:
                new_mtime = path.stat().st_mtime

                if new_mtime > old_mtime:
                    logger.info(f"Configuration changed: {path}")

                    self.reload_config(path)

                    self._config_mtimes[path] = new_mtime

            except Exception:
                logger.exception(f"Failed checking {path}")

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

    def modify_dopplerview_config(self):
        self.open_with_default_app(self.pipeline.ctx.dopplerview_config_path)

    def modify_h5_schema(self):
        self.open_with_default_app(self.output_manager.schema_path)

    def modify_output_config(self):
        self.open_with_default_app(self.output_manager.output_config_path)

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
    
    def register_config_file(self, path, config_type):
        path = Path(path)
        self._config_mtimes[path] = path.stat().st_mtime
        self._config[path] = config_type

    # -------------------
    # Pipeline execution
    # -------------------

    def get_selected_steps(self):
        return [step for step, var in self.step_vars.items() if var.get()]

    def on_drop(self, event):
        path = event.data.strip("{}")  # windows fix
        self.load_input(path)

    def run_pipelines_with_steps(self):
        steps = self.get_selected_steps()
        self.run_pipelines(steps=steps)

    def _make_pipeline_run_spec(self, steps=None):
        """Create a picklable description of the run for the child process."""
        return {
            "steps": steps,
            "input_list": [str(p) for p in self.pipeline.ctx.input_list],
            "h5_schema_path": str(self.output_manager.schema_path),
            "output_config_path": str(self.output_manager.output_config_path),
            "models_config_path": str(self.pipeline.ctx.model_registry_path),
            "dopplerview_config_path": str(self.config_path.get()),
            "config_mode": self.config_mode_var.get(),
            "selected_models": dict(self.selected_models),
            "output_enabled": self.enable_debug_output,
        }

    def run_pipelines(self, steps=None):
        if self.pipeline_worker is not None and self.pipeline_worker.is_alive():
            return

        self.btn_run.config(state="disabled")
        self.btn_run_minimal.config(state="disabled")
        self.progress["value"] = 0
        self.progress_batch["value"] = 0
        self.progress_minimal["value"] = 0

        self.queue = self.mp_context.Queue()
        run_spec = self._make_pipeline_run_spec(steps)

        self.pipeline_worker = self.mp_context.Process(
            target=pipeline_process_worker,
            args=(run_spec, self.queue),
            daemon=True,
        )
        self.pipeline_worker.start()

        self.root.after(100, self.check_queue)

    def _finish_pipeline_ui(self):
        self.btn_run.config(state="enabled")
        self.btn_run_minimal.config(state="enabled")
        self.update_step_display()

    def _terminate_pipeline_worker(self):
        worker = self.pipeline_worker
        if worker is None:
            return

        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
            if worker.is_alive() and hasattr(worker, "kill"):
                worker.kill()
                worker.join(timeout=2)
        else:
            worker.join(timeout=0)

        self.pipeline_worker = None

    def _drain_pipeline_queue(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

    def handle_pipeline_error(self, error_text: str):
        logger.error("Pipeline failed:\n%s", error_text)

        self._terminate_pipeline_worker()
        self._drain_pipeline_queue()

        self.progress.stop()
        self.progress_batch.stop()
        self.progress_minimal.stop()
        self.progress["value"] = 0
        self.progress_batch["value"] = 0
        self.progress_minimal["value"] = 0

        self._finish_pipeline_ui()

        self.root.after_idle(
            lambda: messagebox.showerror(
                "Pipeline error",
                "The pipeline failed. The application was restored to an idle state.\n\n"
                f"{error_text[-3000:]}",
            )
        )

    def _handle_worker_exit_without_message(self):
        worker = self.pipeline_worker
        if worker is None:
            return False

        if worker.is_alive():
            return False

        exit_code = worker.exitcode
        worker.join(timeout=0)
        self.pipeline_worker = None

        if exit_code not in (0, None):
            self._finish_pipeline_ui()
            message = f"The pipeline process stopped unexpectedly with exit code {exit_code}."
            logger.error(message)
            self.root.after_idle(lambda: messagebox.showerror("Pipeline error", message))
            return True

        return False

    def _handle_pipeline_log(self, payload):
        """Replay a log record produced by the pipeline child process in the parent logger."""
        if isinstance(payload, tuple) and payload:
            payload = payload[0]

        if not isinstance(payload, dict):
            logger.info("%s", payload)
            return

        name = payload.get("name") or "pipeline"
        levelno = int(payload.get("levelno", logging.INFO))
        message = payload.get("message", "")

        logging.getLogger(name).log(levelno, "%s", message)

    def check_queue(self):
        max_events_per_tick = 50

        try:
            for _ in range(max_events_per_tick):
                try:
                    event, data = self.queue.get_nowait()
                except queue.Empty:
                    break

                if event == "log":
                    self._handle_pipeline_log(data)

                elif event == "pipeline_start":
                    self.config_path.set(self.config_path.get())

                    i, total = data
                    self.measure_index = i

                    try:
                        measure_name = Path(self.pipeline.ctx.input_list[i]).name
                    except Exception:
                        measure_name = "unknown input"
                    self.status_var.set(f"Processing {i + 1} / {total}: {measure_name}")

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
                    progress = (i / total) * 100 if total else 0
                    self.progress_batch["value"] = progress
                    self.progress_minimal["value"] = progress

                elif event == "batch_start":
                    self.progress_batch["value"] = 0

                elif event == "step_start":
                    step_name, i, total = data
                    step_ratio = i / total if total else 0
                    input_count = max(1, len(self.pipeline.ctx.input_list))
                    measure_ratio = self.measure_index / input_count
                    self.progress["value"] = step_ratio * 100
                    self.progress_minimal["value"] = measure_ratio * 100 + step_ratio * 100 / input_count
                    self.update_step_color(step_name, "running")

                elif event == "step_done":
                    step_name, elapsed = data
                    self.update_step_color(step_name, "done")

                elif event == "preview_image":
                    img = data[0]
                    self.display_image(img)

                elif event == "step_skipped":
                    step_name = data[0]
                    self.update_step_color(step_name, "cached")

                elif event == "pipeline_done":
                    self.progress["value"] = 100
                    self._finish_pipeline_ui()

                elif event == "batch_done":
                    results = data[0] if data else []

                    self.progress_batch["value"] = 100
                    self.progress_minimal["value"] = 100
                    self.status_var.set("Finished")

                    self.btn_run.config(state="enabled")
                    self.btn_run_minimal.config(state="enabled")

                    failed = [r for r in results if r["status"] == "failed"]

                    if failed:
                        tk.messagebox.showwarning(
                            "Batch finished with errors",
                            f"{len(failed)} file(s) failed, "
                            f"{len(results) - len(failed)} succeeded.\n\n"
                            "See logs for details."
                        )

                elif event == "worker_done":
                    if self.pipeline_worker is not None:
                        self.pipeline_worker.join(timeout=0)
                        self.pipeline_worker = None
                    self._finish_pipeline_ui()

                elif event == "pipeline_failed":
                    i, total, input_path, error_text = data

                    logger.error("Failed input %s:\n%s", input_path, error_text)

                    self.progress["value"] = 0
                    self.progress_batch["value"] = ((i + 1) / total) * 100
                    self.progress_minimal["value"] = ((i + 1) / total) * 100

                    self.update_step_display()

                elif event == "error":
                    error_text = data[0] if isinstance(data, tuple) else str(data)
                    self.handle_pipeline_error(error_text)
                    return

        except Exception:
            logger.exception("Error while processing pipeline UI queue")

        if self._handle_worker_exit_without_message():
            return

        if self.pipeline_worker is not None:
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
    multiprocessing.freeze_support()

    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    log_config.setup_logging()

    matplotlib.use('Agg')

    app = MainWindow(root)
    root.mainloop()