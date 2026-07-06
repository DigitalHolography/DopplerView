import tkinter as tk
from tkinter import ttk

try:
    import sv_ttk
except ImportError:  # optional dependency
    sv_ttk = None


class ThemeMixin:
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
