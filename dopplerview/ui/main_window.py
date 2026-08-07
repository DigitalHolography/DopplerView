import logging
import os
import queue
import subprocess
import sys
import multiprocessing
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from dopplerview._version import __version__ as app_version
from dopplerview.input_output import log_config, user_config
from dopplerview.input_output.output_manager import OutputManager
from dopplerview.input_output import read_folder
from dopplerview.models.registry import ModelRegistryConfig
from dopplerview.pipeline.definition import PipelineDefinition
from dopplerview.pipeline.execution_profile import ExecutionProfile

from dopplerview.ui.image_utils import np_to_tk, resize_preview_to_fit
from dopplerview.ui.theme import ThemeMixin
from dopplerview.ui.worker import pipeline_process_worker

logger = logging.getLogger(__name__)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None
    logger.warning("Warning: tkinterdnd2 not found, drag-and-drop functionality will be disabled.")


class _UILogQueueHandler(logging.Handler):
    """Move log records to a queue so Tk only ever updates on its main thread."""

    def __init__(self, queue_out: queue.Queue):
        super().__init__()
        self.queue_out = queue_out

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue_out.put_nowait((self.format(record), record.levelno))
        except Exception:
            pass


class MainWindow(ThemeMixin):
    def __init__(self, root):
        self.root = root
        self.root.title(f"DopplerView {app_version}")

        self._config_mtimes = {}
        self._config = {}
        self.root.bind("<FocusIn>", self.on_focus)

        # --- pipeline init ---

        h5_schema_path = user_config.ensure_config_file("h5_schema.json")
        self.register_config_file(h5_schema_path, "h5_schema")
        output_config_path = user_config.ensure_config_file("output_config.json")
        self.register_config_file(output_config_path, "output_config")
        self.output_manager = OutputManager(h5_schema_path, output_config_path, output_enabled=False)
        self.pipeline_definition = PipelineDefinition.default()
        self.input_list = []
        self.execution_profile = ExecutionProfile.DEFAULT

        models_config = user_config.ensure_config_file("models.yaml")
        self.models_config_path = Path(models_config)
        self.model_registry = ModelRegistryConfig(self.models_config_path)
        self.register_config_file(models_config, "models_config")

        config_path = user_config.ensure_config_file("default_DV_params.json")
        self.config_path = tk.StringVar(value=str(config_path))
        self.status_var = tk.StringVar(value="Ready")
        self.register_config_file(config_path, "dopplerview_config")

        self.image_tk = None  # keep reference (IMPORTANT)
        self.preview_image = None
        self._preview_resize_job = None

        self.queue = queue.Queue()
        self._ui_log_queue = queue.Queue()
        self._ui_log_handler: logging.Handler | None = None
        self.pipeline_worker = None
        self.pipeline_commands = None
        self._validated_steps: set[str] = set()
        self.output_worker = None
        self.mp_context = multiprocessing.get_context("spawn")
        self.selected_models: dict[str, str] = {}

        self.theme_var = tk.StringVar(value="dark")
        self.ui_mode_var = tk.StringVar(value="minimal")
        self.show_logs_var = tk.BooleanVar(value=False)
        self._logs_visible = False
        self._height_before_logs: int | None = None
        self.config_mode_var = tk.StringVar(value="default")
        self.enable_debug_output = False

        self._apply_theme()
        self._set_window_icon()

        # --- UI layout --
        self._build_ui()
        self._install_ui_log_handler()
        self._darken_tk_widget(self.root)
        self._install_drop_targets()
        self.update_mode()  # set initial mode
        self.update_config_mode() # set initial config mode

        self.step_index = 0
        self.measure_index = 0

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._report_callback_exception
        self.root.after(100, self._poll_ui_logs)
        logger.info("DopplerView %s ready", app_version)


    # -------------------
    # UI
    # -------------------

    def _build_ui(self) -> None:
        self.root.geometry("900x680")
        self.root.minsize(720, 540)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.shell = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        self.shell.grid(row=0, column=0, sticky="nsew")
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(0, weight=1)

        self.mode_notebook = ttk.Notebook(self.shell)
        self.mode_notebook.grid(row=0, column=0, sticky="nsew")

        self.minimal_frame = ttk.Frame(self.mode_notebook, padding=22)
        self.advanced_frame = ttk.Frame(self.mode_notebook, padding=16)
        self.mode_notebook.add(self.minimal_frame, text="Minimal")
        self.mode_notebook.add(self.advanced_frame, text="Advanced")
        self.mode_notebook.bind("<<NotebookTabChanged>>", self._on_mode_tab_changed)

        self._build_minimal_ui()
        self._build_advanced_ui()
        self._build_log_panel()

        footer = ttk.Frame(self.shell)
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)

        self.footer_theme_button = ttk.Button(
            footer,
            text="Light mode",
            command=self._toggle_theme,
            style="Toolbar.TButton",
        )
        self.footer_theme_button.grid(row=0, column=1, padx=(8, 0))

        self.footer_help_button = ttk.Button(
            footer,
            text="Help",
            command=self.show_help,
            style="Toolbar.TButton",
        )
        self.footer_help_button.grid(row=0, column=2, padx=(8, 0))

        self.footer_logs_button = ttk.Button(
            footer,
            text="Logs",
            command=self._toggle_logs_from_button,
            style="Toolbar.TButton",
        )
        self.footer_logs_button.grid(row=0, column=3, padx=(8, 0))

    def _toggle_theme(self) -> None:
        new_theme = "light" if self.theme_var.get() == "dark" else "dark"
        self.set_theme(new_theme)
        self.footer_theme_button.configure(
            text="Dark mode" if new_theme == "light" else "Light mode"
        )

    def _build_minimal_ui(self):
        frame = self.minimal_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        container = ttk.Frame(frame)
        container.grid(row=0, column=0, sticky="nsew", padx=34, pady=8)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)

        self.minimal_title_label = ttk.Label(
            header,
            text="DopplerView",
            style="Hero.TLabel",
            font=("Segoe UI", 28, "bold"),
            anchor="center",
        )
        self.minimal_title_label.grid(row=0, column=0, sticky="ew")
        minimal_logo = self._load_scaled_logo_image(max_width=320, max_height=112)
        if minimal_logo is not None:
            self._minimal_logo_image = minimal_logo
            self.minimal_logo_label = ttk.Label(header, image=self._minimal_logo_image, anchor="center")
            self.minimal_logo_label.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.btn_load_minimal = ttk.Button(container, text="Load input", command=self.load_holo)
        self.btn_load_minimal.grid(row=1, column=0, pady=(0, 12))

        self.minimal_file_count_var = tk.StringVar(value="No input selected")
        self.minimal_file_detail_var = tk.StringVar(value="Drop one or more .holo files here.")
        drop_frame = ttk.Frame(container, style="Drop.TFrame", padding=14)
        drop_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 14))
        drop_frame.columnconfigure(0, weight=1)
        drop_frame.rowconfigure(2, weight=1)
        ttk.Label(
            drop_frame,
            textvariable=self.minimal_file_count_var,
            style="Section.TLabel",
            font=("Segoe UI", 11, "bold"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            drop_frame,
            textvariable=self.minimal_file_detail_var,
            style="Muted.TLabel",
            anchor="center",
            wraplength=620,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 10))

        list_container = ttk.Frame(drop_frame)
        list_container.grid(row=2, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)

        self.minimal_input_listbox = tk.Listbox(
            list_container,
            height=4,
            activestyle="none",
            exportselection=False,
        )
        self.minimal_input_listbox.grid(row=0, column=0, sticky="nsew")
        self.minimal_input_listbox.bind("<Button-1>", lambda e: "break")
        self.minimal_input_listbox.bind("<B1-Motion>", lambda e: "break")
        self.minimal_input_listbox.bind("<Key>", lambda e: "break")

        minimal_scrollbar = ttk.Scrollbar(
            list_container, orient="vertical", command=self.minimal_input_listbox.yview
        )
        minimal_scrollbar.grid(row=0, column=1, sticky="ns")
        self.minimal_input_listbox.config(yscrollcommand=minimal_scrollbar.set)

        self.btn_run_minimal = ttk.Button(
            container,
            text="Run full pipeline",
            command=self.run_pipelines_with_steps,
            state="disabled",
            style="Accent.TButton",
        )
        self.btn_run_minimal.grid(row=3, column=0, pady=(0, 14))

        progress_frame = ttk.Frame(container)
        progress_frame.grid(row=4, column=0, sticky="ew")
        progress_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_frame, text="Overall progress", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.progress_minimal = ttk.Progressbar(progress_frame, maximum=100, mode="determinate")
        self.progress_minimal.grid(row=1, column=0, sticky="ew")

        self.status_label_minimal = ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            anchor="center",
            style="Muted.TLabel",
        )
        self.status_label_minimal.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _build_advanced_ui(self):
        frame = self.advanced_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        self.buttons_frame = ttk.Frame(frame)
        self.buttons_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.buttons_frame.columnconfigure(5, weight=1)

        self.btn_load = ttk.Button(
            self.buttons_frame,
            text="Load input",
            command=self.load_holo,
            style="Toolbar.TButton",
        )
        self.btn_load.grid(row=0, column=0, padx=(0, 6))

        self.btn_select_config = ttk.Button(
            self.buttons_frame,
            text="Load config",
            command=self.load_dopplerview_config,
            style="Toolbar.TButton",
        )
        self.btn_select_config.grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            self.buttons_frame,
            text="Settings",
            command=self.show_config,
            style="Toolbar.TButton",
        ).grid(row=0, column=2, padx=(0, 12))
        self.btn_run = ttk.Button(
            self.buttons_frame,
            text="Run pipeline",
            command=self.run_pipelines_with_steps,
            state="disabled",
            style="Accent.TButton",
        )
        self.btn_run.grid(row=0, column=3, padx=(0, 8))
        ttk.Button(
            self.buttons_frame,
            text="Logs",
            command=self.show_logs,
            style="Toolbar.TButton",
        ).grid(row=0, column=4)
        self.status_label = ttk.Label(
            self.buttons_frame,
            textvariable=self.status_var,
            style="Muted.TLabel",
            anchor="e",
        )
        self.status_label.grid(row=0, column=5, sticky="e", padx=(12, 0))
        self.config_path_label = ttk.Label(
            self.buttons_frame,
            textvariable=self.config_path,
            style="Muted.TLabel",
            anchor="w",
        )
        self.config_path_label.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0))

        content = ttk.Frame(frame)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1, minsize=330)
        content.columnconfigure(1, weight=2, minsize=380)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        self.input_panel = ttk.LabelFrame(left, text="Inputs", padding=8)
        self.input_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.input_panel.columnconfigure(0, weight=1)
        self.input_panel.rowconfigure(0, weight=1)
        list_container = ttk.Frame(self.input_panel)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)

        self.input_listbox = tk.Listbox(
            list_container,
            height=5,
            activestyle="none",
            exportselection=False,
        )
        self.input_listbox.grid(row=0, column=0, sticky="nsew")
        self.input_listbox.bind("<Button-1>", lambda e: "break")
        self.input_listbox.bind("<B1-Motion>", lambda e: "break")
        self.input_listbox.bind("<Key>", lambda e: "break")

        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.input_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.input_listbox.config(yscrollcommand=scrollbar.set)

        self.steps_frame = ttk.LabelFrame(left, text="Pipeline steps", padding=8)
        self.steps_frame.grid(row=1, column=0, sticky="nsew")
        self.steps_frame.columnconfigure(0, weight=1)
        self.steps_frame.columnconfigure(1, weight=1)

        self.step_vars = {}
        self.step_checkboxes = {}
        steps = self.pipeline_definition.execution_order
        optional_steps = {
            "retinal_vessel_velocity_estimator",
            "arterial_waveform_analysis",
            "choroidal_artery_vein_segmentation",
        }

        for index, step in enumerate(steps):
            var = tk.BooleanVar(value=step not in optional_steps)
            label = step.replace("_", " ").title()
            cb = ttk.Checkbutton(
                self.steps_frame,
                text=label,
                variable=var,
                command=lambda s=step: self.on_step_toggle(s),
                style="Step.TCheckbutton",
            )
            cb.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 8))
            self.step_vars[step] = var
            self.step_checkboxes[step] = cb
        self.update_step_display()

        preview_panel = ttk.LabelFrame(content, text="Preview", padding=8)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)
        self.preview_frame = ttk.Frame(
            preview_panel,
            style="Preview.TFrame",
            padding=12,
        )
        self.preview_frame.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_frame.bind("<Configure>", self._schedule_preview_resize)
        self.image_label = ttk.Label(
            self.preview_frame,
            text="A preview will appear here while the pipeline is running.",
            anchor="center",
            style="Muted.TLabel",
            wraplength=480,
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        progress_panel = ttk.Frame(preview_panel)
        progress_panel.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        progress_panel.columnconfigure(0, weight=1)
        ttk.Label(progress_panel, text="Current pipeline", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.progress = ttk.Progressbar(progress_panel, maximum=100, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(progress_panel, text="Batch", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        self.progress_batch = ttk.Progressbar(progress_panel, maximum=100, mode="determinate")
        self.progress_batch.grid(row=3, column=0, sticky="ew")

    def _build_log_panel(self) -> None:
        self.log_panel = ttk.LabelFrame(self.shell, text="Application logs", padding=8)
        self.log_panel.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.log_panel.columnconfigure(0, weight=1)
        self.log_panel.rowconfigure(1, weight=1)

        actions = ttk.Frame(self.log_panel)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        actions.columnconfigure(0, weight=1)
        ttk.Label(
            actions,
            text=str(log_config.get_log_file()),
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Open file", command=self.open_log_file).grid(
            row=0, column=1, padx=(8, 6)
        )
        ttk.Button(actions, text="Clear view", command=self.clear_log_view).grid(row=0, column=2)

        text_frame = ttk.Frame(self.log_panel, style="Log.TFrame")
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            text_frame,
            height=7,
            wrap="none",
            state="disabled",
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        log_horizontal = ttk.Scrollbar(text_frame, orient="horizontal", command=self.log_text.xview)
        log_horizontal.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(
            yscrollcommand=log_scrollbar.set,
            xscrollcommand=log_horizontal.set,
        )
        self._configure_log_tags()
        self.log_panel.grid_remove()

    def _on_mode_tab_changed(self, _event=None) -> None:
        try:
            index = self.mode_notebook.index("current")
        except tk.TclError:
            return
        mode = "advanced" if index == 1 else "minimal"
        self.ui_mode_var.set(mode)
        if mode == "advanced" and self.root.winfo_width() < 1000:
            self.root.geometry("1060x720")

    def _install_ui_log_handler(self) -> None:
        handler = _UILogQueueHandler(self._ui_log_queue)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        self._ui_log_handler = handler

    def _poll_ui_logs(self) -> None:
        try:
            for _ in range(200):
                try:
                    message, levelno = self._ui_log_queue.get_nowait()
                except queue.Empty:
                    break
                self._append_log_line(message, levelno)
            self.root.after(100, self._poll_ui_logs)
        except tk.TclError:
            return

    def _append_log_line(self, message: str, levelno: int) -> None:
        if not hasattr(self, "log_text"):
            return
        if levelno >= logging.ERROR:
            tag = "error"
        elif levelno >= logging.WARNING:
            tag = "warning"
        elif levelno <= logging.DEBUG:
            tag = "debug"
        else:
            tag = "info"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n", tag)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 4000:
            self.log_text.delete("1.0", f"{line_count - 3500}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _configure_log_tags(self) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.tag_configure("debug", foreground=self._muted_fg)
        self.log_text.tag_configure("info", foreground=self._text_fg)
        self.log_text.tag_configure("warning", foreground=self._warning_color)
        self.log_text.tag_configure("error", foreground=self._error_color)

    def toggle_logs(self) -> None:
        if self.show_logs_var.get():
            if not self._logs_visible:
                self._height_before_logs = self.root.winfo_height()
            self.log_panel.grid()
            self.footer_logs_button.configure(text="Hide logs")
            current_height = self.root.winfo_height()
            available_height = max(540, self.root.winfo_screenheight() - 80)
            target_height = min(840, available_height)
            if current_height < target_height:
                self.root.geometry(f"{self.root.winfo_width()}x{target_height}")
            self._logs_visible = True
        else:
            self.log_panel.grid_remove()
            self.footer_logs_button.configure(text="Logs")
            if self._logs_visible and self._height_before_logs is not None:
                self.root.geometry(
                    f"{self.root.winfo_width()}x{self._height_before_logs}"
                )
            self._logs_visible = False
            self._height_before_logs = None

    def _toggle_logs_from_button(self) -> None:
        self.show_logs_var.set(not self.show_logs_var.get())
        self.toggle_logs()

    def show_logs(self) -> None:
        self.show_logs_var.set(True)
        self.toggle_logs()
        self.log_text.focus_set()

    def clear_log_view(self) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def open_log_file(self) -> None:
        path = log_config.get_log_file()
        if not path.exists():
            messagebox.showinfo("Logs", "No log file has been created yet.", parent=self.root)
            return
        self.open_with_default_app(path)

    def _report_callback_exception(self, exc_type, value, tb) -> None:
        details = "".join(traceback.format_exception(exc_type, value, tb))
        logger.error("Unhandled UI error:\n%s", details)
        self.show_logs()
        messagebox.showerror(
            "DopplerView error",
            "An unexpected error occurred. The details are available in the Logs panel.",
            parent=self.root,
        )

    def _on_close(self) -> None:
        if self.pipeline_worker is not None:
            self._terminate_pipeline_worker()
        if self._ui_log_handler is not None:
            logging.getLogger().removeHandler(self._ui_log_handler)
            self._ui_log_handler.close()
            self._ui_log_handler = None
        self.root.destroy()

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
        models_frame = ttk.LabelFrame(parent, text="Models", padding=10)
        models_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        models_frame.grid_columnconfigure(0, weight=1)

        steps_frame = ttk.LabelFrame(parent, text="Outputs", padding=10)
        steps_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        steps_frame.grid_columnconfigure(0, weight=1)

        debug_output_button = ttk.Checkbutton(
            steps_frame,
            text="Enable debug output",
            command=self.toggle_debug_output
        )
        debug_output_button.grid(row=0, column=0, sticky="ew", pady=5)

        # -----------------------
        # RIGHT: Config panel
        # -----------------------
        config_panel = ttk.LabelFrame(parent, text="Configuration", padding=10)
        config_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        config_panel.grid_columnconfigure(0, weight=1)

        # --- Radio buttons ---
        radio_frame = ttk.Frame(config_panel)
        radio_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        radio_frame.grid_columnconfigure(0, weight=1)
        radio_frame.grid_columnconfigure(1, weight=1)

        rb_default = ttk.Radiobutton(
            radio_frame,
            text="Use default config",
            variable=self.config_mode_var,
            value="default",
            command=self.update_config_mode,
        )
        rb_default.grid(row=0, column=0, sticky="w")

        rb_local = ttk.Radiobutton(
            radio_frame,
            text="Use local config",
            variable=self.config_mode_var,
            value="local",
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

        def create_model_selector(parent_widget, label_text, task_name, r):
            ttk.Label(
                parent_widget,
                text=label_text,
            ).grid(row=r, column=0, sticky="w")

            values = self.model_registry.list_models_for_task(task_name)
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
                self._invalidate_step_colors(task_name)

            combo.bind("<<ComboboxSelected>>", on_change)

            if values:
                self.selected_models[task_name] = var.get()

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
        self.config_window.geometry("720x440")
        self.config_window.minsize(640, 380)
        self.config_window.transient(self.root)
        self.config_window.configure(bg=self._bg_color)

        container = ttk.Frame(self.config_window, padding=16)
        container.pack(fill="both", expand=True)
        self.config_container = container

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
        selected = self.get_selected_steps()

        if self.step_vars[step].get():
            # ADD step → recompute full dependency closure
            resolved = self.pipeline_definition.resolve_execution_graph(selected)

            for s in self.pipeline_definition.execution_order:
                self.step_vars[s].set(s in resolved)
        else:
            # REMOVE step + downstream
            downstream = self.pipeline_definition.get_downstream_steps(step)

            for s in downstream:
                self.step_vars[s].set(False)

            self.step_vars[step].set(False)

        self.update_step_display()

    def update_mode(self):
        mode = self.ui_mode_var.get()
        self.mode_notebook.select(self.advanced_frame if mode == "advanced" else self.minimal_frame)
        if mode == "advanced" and self.root.winfo_width() < 1000:
            self.root.geometry("1060x720")

    def resize_window(self):
        # The preview now lives in an expanding panel; keep the window stable.
        self.root.update_idletasks()

    def update_step_color(self, step, state):
        cb = self.step_checkboxes[step]
        if state == "done" or state == "cached":
            style = "Done.Step.TCheckbutton"
        elif state == "running":
            style = "Running.Step.TCheckbutton"
        else:
            style = "Step.TCheckbutton"
        cb.configure(style=style)

    def _invalidate_step_colors(self, step):
        affected = {step, *self.pipeline_definition.get_downstream_steps(step)}
        self._validated_steps.difference_update(affected)
        self.update_step_display()

    def update_step_display(self):
        for step, cb in self.step_checkboxes.items():
            is_checked = self.step_vars[step].get()
            is_cached = step in self._validated_steps

            # -------- label logic --------
            if is_checked:
                if is_cached:
                    style = "Done.Step.TCheckbutton"
                else:
                    style = "Step.TCheckbutton"
            else:
                style = "Inactive.Step.TCheckbutton"

            cb.configure(style=style)

    def load_input(self, folders):
        # self.input_folder.set(folders)
        self.cleanup_image()
        self.progress["value"] = 0
        self.progress_minimal["value"] = 0
        self.progress_batch["value"] = 0
        self.input_list.clear()
        self._validated_steps.clear()

        if isinstance(folders, str):
            folder_list = [Path(f) for f in folders.split() if f]
        else:
            folder_list = [Path(f) for f in folders]

        for path in folder_list:
            self._add_input_path(path)
        self.update_step_display()

        self.btn_run.config(state="enabled")
        self.btn_run_minimal.config(state="enabled")

        self.refresh_input_listbox()

        n_inputs = len(self.input_list)
        if n_inputs == 0:
            self.status_var.set("Ready")
        else:
            self.status_var.set(f"Loaded {n_inputs} input file(s)")

    def _add_input_path(self, path):
        """Resolve a UI selection without constructing an execution context."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_dir():
            holo_files = read_folder.search_holo_files(path)
            if not holo_files:
                raise FileNotFoundError(f"No .holo file found in {path}")
            self.input_list.extend(holo_files)
            return
        if path.suffix.lower() == ".txt":
            with path.open("r") as handle:
                for line in handle:
                    candidate = Path(line.strip())
                    if candidate.is_dir() or (
                        candidate.suffix.lower() == ".holo" and candidate.is_file()
                    ):
                        self.input_list.append(candidate)
            return
        if path.suffix.lower() == ".holo" and path.is_file():
            self.input_list.append(path)

    def refresh_input_listbox(self):
        # Advanced UI list
        if hasattr(self, "input_listbox"):
            self.input_listbox.delete(0, tk.END)

            for path in self.input_list:
                self.input_listbox.insert(tk.END, str(path))

        # Minimal UI list
        if hasattr(self, "minimal_input_listbox"):
            self.minimal_input_listbox.delete(0, tk.END)

            for path in self.input_list:
                self.minimal_input_listbox.insert(tk.END, str(path))

        input_list = list(self.input_list)
        if hasattr(self, "minimal_file_count_var"):
            count = len(input_list)
            if count:
                suffix = "s" if count != 1 else ""
                self.minimal_file_count_var.set(f"{count} input file{suffix} selected")
                first_path = input_list[0]
                detail = str(first_path) if count == 1 else f"First file: {first_path}"
                self.minimal_file_detail_var.set(detail)
            else:
                self.minimal_file_count_var.set("No input selected")
                self.minimal_file_detail_var.set("Drop one or more .holo files here.")

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
            self.config_path.set(file_path)

    def reload_config(self, path):
        config_type = self._config.get(path)

        if config_type == "dopplerview_config":
            self.config_path.set(str(path))

        elif config_type == "models_config":
            self.models_config_path = Path(path)
            self.model_registry = ModelRegistryConfig(self.models_config_path)
            if hasattr(self, "config_container") and self.config_container.winfo_exists():
                for child in self.config_container.winfo_children():
                    child.destroy()
                self._populate_configuration_frame(self.config_container)

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
        self.open_with_default_app(self.models_config_path)

    def modify_dopplerview_config(self):
        path = self.config_path.get()
        if path == "No config loaded":
            raise FileNotFoundError("No DopplerView configuration is currently selected")
        self.open_with_default_app(path)

    def modify_h5_schema(self):
        self.open_with_default_app(self.output_manager.schema_path)

    def modify_output_config(self):
        self.open_with_default_app(self.output_manager.output_config_path)

    def update_config_mode(self):
        mode = self.config_mode_var.get()
        if mode == "local":
            config_path = "No config loaded"
        else:
            config_path = user_config.ensure_config_file("default_DV_params.json")

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
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except tk.TclError:
            paths = [event.data.strip("{}")]
        if paths:
            self.load_input(paths)

    def run_pipelines_with_steps(self):
        steps = self.get_selected_steps()
        self.run_pipelines(steps=steps)

    def _make_pipeline_run_spec(self, steps=None):
        """Create a picklable description of the run for the child process."""
        return {
            "steps": steps,
            "input_list": [str(p) for p in self.input_list],
            "h5_schema_path": str(self.output_manager.schema_path),
            "output_config_path": str(self.output_manager.output_config_path),
            "models_config_path": str(self.models_config_path),
            "dopplerview_config_path": str(self.config_path.get()),
            "config_mode": self.config_mode_var.get(),
            "selected_models": dict(self.selected_models),
            "output_enabled": self.enable_debug_output,
            "execution_profile": self.execution_profile.value,
        }

    def run_pipelines(self, steps=None):
        self.btn_run.config(state="disabled")
        self.btn_run_minimal.config(state="disabled")
        self.progress["value"] = 0
        self.progress_batch["value"] = 0
        self.progress_minimal["value"] = 0
        self.status_var.set("Starting pipeline…")
        logger.info("Starting pipeline for %d input(s)", len(self.input_list))

        run_spec = self._make_pipeline_run_spec(steps)

        if self.pipeline_worker is None or not self.pipeline_worker.is_alive():
            self.queue = self.mp_context.Queue()
            self.pipeline_commands = self.mp_context.Queue()
            self.pipeline_worker = self.mp_context.Process(
                target=pipeline_process_worker,
                args=(self.pipeline_commands, self.queue),
                daemon=True,
            )
            self.pipeline_worker.start()

        self.pipeline_commands.put(run_spec)

        self.root.after(100, self.check_queue)

    def _finish_pipeline_ui(self):
        self.btn_run.config(state="enabled")
        self.btn_run_minimal.config(state="enabled")
        # The authoritative runtime cache lives in the persistent child
        # process.  Recomputing the display here from the UI process's empty
        # Pipeline context would overwrite the green step_done/step_skipped
        # states received from that child.

    def _terminate_pipeline_worker(self):
        worker = self.pipeline_worker
        if worker is None:
            return

        if worker.is_alive():
            if self.pipeline_commands is not None:
                try:
                    self.pipeline_commands.put(None)
                except Exception:
                    pass
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            if worker.is_alive() and hasattr(worker, "kill"):
                worker.kill()
                worker.join(timeout=2)
        else:
            worker.join(timeout=0)

        self.pipeline_worker = None
        self.pipeline_commands = None

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
        self.show_logs()

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
            self.show_logs()
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
                        measure_name = Path(self.input_list[i]).stem
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
                    self._validated_steps.discard(step_name)
                    step_ratio = i / total if total else 0
                    input_count = max(1, len(self.input_list))
                    measure_ratio = self.measure_index / input_count
                    self.progress["value"] = step_ratio * 100
                    self.progress_minimal["value"] = measure_ratio * 100 + step_ratio * 100 / input_count
                    self.update_step_color(step_name, "running")

                elif event == "step_done":
                    step_name, elapsed = data
                    self._validated_steps.add(step_name)
                    self.update_step_color(step_name, "done")

                elif event == "preview_image":
                    img = data[0]
                    self.display_image(img)

                elif event == "step_skipped":
                    step_name = data[0]
                    self._validated_steps.add(step_name)
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
        self.preview_image = img.copy()
        self._render_preview()
        if self.ui_mode_var.get() == "advanced":
            self.resize_window()

    def _schedule_preview_resize(self, _event=None):
        if self.preview_image is None:
            return
        if self._preview_resize_job is not None:
            self.root.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.root.after(50, self._render_preview)

    def _render_preview(self):
        self._preview_resize_job = None
        if self.preview_image is None:
            return

        # The frame has 12 px of padding on each side.
        max_width = max(1, self.preview_frame.winfo_width() - 24)
        max_height = max(1, self.preview_frame.winfo_height() - 24)
        fitted = resize_preview_to_fit(
            self.preview_image,
            max_width,
            max_height,
        )
        self.image_tk = np_to_tk(fitted)  # keep reference!
        self.image_label.config(image=self.image_tk, text="")

    def cleanup_image(self):
        if self._preview_resize_job is not None:
            self.root.after_cancel(self._preview_resize_job)
            self._preview_resize_job = None
        self.preview_image = None
        self.image_tk = None
        self.image_label.config(
            image="",
            text="A preview will appear here while the pipeline is running.",
        )

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
            "2. In the Advanced tab, select which pipeline steps to run or run the full pipeline.\n"
            "3. View the results, including artery/vein segmentation overlays.\n"
            "4. Open the Logs panel from the footer to follow processing details.\n\n"
            "For more information, visit our GitHub repository: https://github.com/DigitalHolography/DopplerView"
        )
        tk.messagebox.showinfo("Help - DopplerView", help_text)
