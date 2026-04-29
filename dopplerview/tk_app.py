import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk
from pathlib import Path
import threading
import queue

from dopplerview.input_output import user_config
import numpy as np
import cv2
from PIL import Image, ImageTk

from dopplerview.pipeline.pipeline import Pipeline

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None
    print("Warning: tkinterdnd2 not found, drag-and-drop functionality will be disabled.")

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
        self.input_folder = tk.StringVar(value="No input selected")
        self.folder_loaded=False

        # --- pipeline init ---

        self.pipeline = Pipeline()

        models_config = user_config.ensure_config_file("models.yaml")
        self.pipeline.load_model_registry(models_config)
        
        h5_schema_config = user_config.ensure_config_file("h5_schema.json")
        output_config = user_config.ensure_config_file("output_config.json")
        self.pipeline.load_h5_schema(h5_schema_config)
        self.pipeline.load_output_config(output_config)

        self.image_tk = None  # keep reference (IMPORTANT)

        self.queue = queue.Queue()

        self._apply_theme()
        self._set_window_icon()

        # --- UI layout --
        self._build_ui()
        self._install_drop_targets()
        self.update_mode()  # set initial mode


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
        self.main_container = container

        self.minimal_view = ttk.Frame(container, padding=10)
        self.advanced_view = ttk.Frame(container, padding=10)

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
            command=lambda: self.update_mode(),
        )
        view_menu.add_radiobutton(
            label="Advanced UI",
            value="advanced",
            variable=self.ui_mode_var,
            command=lambda: self.update_mode(),
        )
        menu_bar.add_cascade(label="View", menu=view_menu)
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
        frame = self.minimal_view

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

        self.btn_load = ttk.Button(container, text="Select .holo File", command=self.load_holo)
        self.btn_load.grid(row=2, column=0, pady=(0, 10))

        self.minimal_input_path_label = tk.Label(
            container,
            textvariable=self.input_folder,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        ).grid(row=3, column=0, pady=(0, 10))

        self.btn_run_minimal = ttk.Button(container, text="Run Full Pipeline", command=self.run_full_pipeline)
        self.btn_run_minimal.grid(row=4, column=0, pady=10)

        self.progress_minimal = ttk.Progressbar(container, maximum=100)
        self.progress_minimal.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _build_advanced_ui(self):
        frame = self.advanced_view

        # Buttons
        self.btn_load = ttk.Button(frame, text="Select .holo File", command=self.load_holo)
        self.btn_load.pack(pady=5)

        self.minimal_input_path_label = tk.Label(
            frame,
            textvariable=self.input_folder,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        ).pack(pady=5)

        # Model selection
         # --- Model selection frame ---
        self.models_frame = tk.LabelFrame(frame, text="Models")
        self.models_frame.pack(fill="x", padx=5, pady=5)

        ctx = self.pipeline.ctx
        mm = ctx.model_manager

        # Helper to create one dropdown
        def create_model_selector(parent, label_text, task_name):
            tk.Label(parent, text=label_text).pack(anchor="w")

            values = mm.get_model_name_list_for_task(task_name)
            var = tk.StringVar(value=values[0] if values else "")

            combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
            combo.pack(fill="x", pady=2)

            def on_change(event=None):
                ctx.change_model_for_task(task_name, var.get())

            combo.bind("<<ComboboxSelected>>", on_change)

            # Initialize state (like Streamlit does implicitly)
            if values:
                ctx.change_model_for_task(task_name, var.get())

            return var, combo

        # Create the three selectors
        self.binary_model_var, self.binary_model_combo = create_model_selector(
            self.models_frame,
            "Binary vessel segmentation model",
            "retinal_vessel_segmentation"
        )

        self.av_model_var, self.av_model_combo = create_model_selector(
            self.models_frame,
            "Artery/Vein segmentation model",
            "retinal_artery_vein_segmentation"
        )

        self.optic_disc_model_var, self.optic_disc_model_combo = create_model_selector(
            self.models_frame,
            "Optic disc detection model",
            "optic_disc_detection"
        )

        # Step list (checkboxes)
        self.step_vars = {}
        self.step_checkboxes = {}

        self.steps_frame = tk.LabelFrame(self.advanced_view, text="Pipeline Steps")
        self.steps_frame.pack(fill="x", padx=5, pady=5)

        for step in self.pipeline.get_step_names():
            var = tk.BooleanVar(value=True)

            cb = tk.Checkbutton(
                self.steps_frame,
                text=step,
                variable=var,
                command=lambda s=step: self.on_step_toggle(s),
                fg=self._text_fg,
            )
            cb.pack(anchor="w")

            self.step_vars[step] = var
            self.step_checkboxes[step] = cb

        self.btn_run = ttk.Button(self.advanced_view, text="Run Pipeline", command=self.run_pipeline)
        self.btn_run.pack(pady=5)

        self.progress = ttk.Progressbar(self.advanced_view, maximum=100)
        self.progress.pack()
    
        # Image display
        self.image_label = tk.Label(self.advanced_view)
        self.image_label.pack(pady=10)

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

        self.minimal_view.pack_forget()
        self.advanced_view.pack_forget()

        if mode == "minimal":
            self.minimal_view.pack(fill="both", expand=True)
            self.root.geometry("600x400")
        else:
            self.advanced_view.pack(fill="both", expand=True)
            self.root.geometry("900x700")

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

    def load_input(self, folder):
        self.input_folder.set(folder)
        self.folder_loaded = False
        self.cleanup_image()
        self.pipeline.ctx.clear()
        self.update_step_display()

    def load_holo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Holo files", "*.holo")], defaultextension=".holo")
        if file_path:
            self.load_input(file_path)

    def get_selected_steps(self):
        return [step for step, var in self.step_vars.items() if var.get()]

    def on_drop(self, event):
        path = event.data.strip("{}")  # windows fix
        self.load_input(path)

    def run_full_pipeline(self):
        # full pipeline
        self.run_pipeline(steps=None)

    def run_pipeline_with_steps(self):
        steps = self.get_selected_steps()
        self.run_pipeline(steps=steps)

    def run_pipeline(self, steps=None):
        self.btn_run.config(state="disabled")
        self.btn_run_minimal.config(state="disabled")
        thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(steps,),
            daemon=True
        )
        thread.start()

        self.root.after(100, self.check_queue)

    def _run_pipeline_worker(self, steps):
        def callback(event, *args):
            self.queue.put((event, args))
        try:
            if not self.folder_loaded:
                self.pipeline.load_input(Path(self.input_folder.get()))
                self.folder_loaded=True

            self.pipeline.run(targets=steps, callback=callback)

            img = self.pipeline.ctx.get("M0_ff_image")
            art = self.pipeline.ctx.get("retinal_artery_mask")
            vein = self.pipeline.ctx.get("retinal_vein_mask")

            self.queue.put(("finished", (img, art, vein)))

        except Exception as e:
            self.queue.put(("error", str(e)))

    # def check_queue(self):
    #     try:
    #         msg, data = self.queue.get_nowait()

    #         if msg == "finished":
    #             self.update_step_display()

    #             img, art, vein = data

    #             if img is not None:
    #                 overlay = self.overlay(img, art, vein)
    #                 self.display_image(overlay)

    #         elif msg == "error":
    #             print("Pipeline error:", data)

    #         self.btn_run.config(state="enabled")
            
    #     except queue.Empty:
    #         self.root.after(100, self.check_queue)

    def check_queue(self):
        try:
            while True:
                event, data = self.queue.get_nowait()

                if event == "step_start":
                    step_name, i, total = data
                    progress = (i / total) * 100
                    self.progress["value"] = progress
                    self.progress_minimal["value"] = progress

                    self.update_step_color(step_name, "running")

                elif event == "step_done":
                    step_name, elapsed = data
                    self.update_step_color(step_name, "done")

                elif event == "step_skipped":
                    step_name = data[0]
                    self.update_step_color(step_name, "cached")

                elif event == "finished":
                    self.progress["value"] = 100
                    self.btn_run.config(state="enabled")

                    self.progress_minimal["value"] = 100
                    self.btn_run_minimal.config(state="enabled")

                    img, art, vein = data

                    if img is not None:
                        overlay = self.overlay(img, art, vein)
                        self.display_image(overlay)

                elif event == "error":
                    print("Error:", data)

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
        self.image_tk = np_to_tk(img)  # keep reference!
        self.image_label.config(image=self.image_tk)

    def cleanup_image(self):
        self.image_label.config(image="")

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
    # Help
    # -------------------

    def show_help(self):
        help_text = (
            "DopplerView is a tool for segmentation, classification and analysis of diverse structures and signals on data issued from laser doppler holography.\n"
            "It takes as input .h5 file(s) resulting from holodoppler processing of raw videos, and produces a variety of outputs including artery/vein segmentation masks, velocity estimates, waveform analyses, and more.\n\n"
            "1. Load a folder containing your .h5 file\n"
            "2. In advanced UI (View -> Advanced UI), select which pipeline steps to run or run the full pipeline.\n"
            "3. View the results, including artery/vein segmentation overlays.\n\n"
            "For more information, visit our GitHub repository."
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
    app = MainWindow(root)
    root.mainloop()