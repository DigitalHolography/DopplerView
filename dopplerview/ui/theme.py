from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    import sv_ttk
except ImportError:  # Keep source checkouts usable before optional UI deps are installed.
    sv_ttk = None


PALETTES = {
    "dark": {
        "bg": "#1f1f1f",
        "surface": "#202020",
        "surface_alt": "#2d2d2d",
        "text": "#f2f2f2",
        "muted": "#a5a5a5",
        "accent": "#60cdff",
        "select": "#3a6ea5",
        "disabled": "#777777",
        "border": "#3a3a3a",
        "success": "#6ccb5f",
        "warning": "#f0c75e",
        "error": "#ff7b72",
    },
    "light": {
        "bg": "#f5f5f5",
        "surface": "#ffffff",
        "surface_alt": "#eeeeee",
        "text": "#202020",
        "muted": "#6b7280",
        "accent": "#0067c0",
        "select": "#cde8ff",
        "disabled": "#9a9a9a",
        "border": "#d6d6d6",
        "success": "#107c10",
        "warning": "#9a6700",
        "error": "#c42b1c",
    },
}


class ThemeMixin:
    """Apply Sun Valley and keep the few classic Tk widgets in sync with it."""

    def _apply_theme(self, theme: str | None = None) -> None:
        if theme is None:
            theme = self.theme_var.get() if hasattr(self, "theme_var") else "dark"
        theme = theme.lower()
        if theme not in PALETTES:
            theme = "dark"

        if hasattr(self, "theme_var"):
            self.theme_var.set(theme)

        palette = PALETTES[theme]
        self._theme_name = theme
        self._bg_color = palette["bg"]
        self._surface_color = palette["surface"]
        self._surface_alt_color = palette["surface_alt"]
        self._text_fg = palette["text"]
        self._muted_fg = palette["muted"]
        self._accent_color = palette["accent"]
        self._select_color = palette["select"]
        self._disabled_fg = palette["disabled"]
        self._border_color = palette["border"]
        self._success_color = palette["success"]
        self._warning_color = palette["warning"]
        self._error_color = palette["error"]
        self._text_bg = self._surface_color

        style = ttk.Style(self.root)
        self._style = style
        self._sun_valley_enabled = False

        if sv_ttk is not None:
            try:
                sv_ttk.set_theme(theme)
                self._sun_valley_enabled = True
            except (tk.TclError, RuntimeError):
                self._configure_fallback_theme(style, theme)
        else:
            self._configure_fallback_theme(style, theme)

        self.root.configure(bg=self._bg_color)
        self.root.option_add("*Font", ("Segoe UI", 10))
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
        self.root.option_add("*selectBackground", self._select_color)
        self.root.option_add("*selectForeground", self._text_fg)
        self.root.option_add("*insertBackground", self._text_fg)
        self.root.option_add("*disabledForeground", self._disabled_fg)

        # Sun Valley owns the base widget rendering. These styles only add the
        # hierarchy, spacing and states used by DopplerView.
        style.configure("Hero.TLabel", font=("Segoe UI", 28, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11), foreground=self._muted_fg)
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground=self._muted_fg)
        style.configure("Status.TLabel", foreground=self._muted_fg, padding=(2, 3))
        style.configure("Accent.TButton", padding=(14, 7))
        style.configure("Toolbar.TButton", padding=(10, 6))
        style.configure("Drop.TFrame", borderwidth=1, relief="solid")
        style.configure("Preview.TFrame", borderwidth=1, relief="solid")
        style.configure("Log.TFrame", borderwidth=1, relief="solid")
        style.configure("Step.TCheckbutton", padding=(4, 2))
        style.configure("Running.Step.TCheckbutton", foreground=self._warning_color, padding=(4, 2))
        style.configure("Done.Step.TCheckbutton", foreground=self._success_color, padding=(4, 2))
        style.configure("Inactive.Step.TCheckbutton", foreground=self._muted_fg, padding=(4, 2))

    def _configure_fallback_theme(self, style: ttk.Style, theme: str) -> None:
        """Small source-checkout fallback; packaged builds always include sv-ttk."""
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self._bg_color, foreground=self._text_fg)
        style.configure("TFrame", background=self._bg_color)
        style.configure("TLabel", background=self._bg_color, foreground=self._text_fg)
        style.configure("TLabelframe", background=self._bg_color, foreground=self._text_fg)
        style.configure("TLabelframe.Label", background=self._bg_color, foreground=self._text_fg)
        style.configure("TEntry", fieldbackground=self._surface_color, foreground=self._text_fg)
        style.configure("TCombobox", fieldbackground=self._surface_color, foreground=self._text_fg)
        style.configure("TButton", background=self._surface_alt_color, foreground=self._text_fg, padding=(10, 6))
        style.configure("Accent.TButton", background=self._accent_color, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", self._accent_color)])
        style.configure("TProgressbar", background=self._accent_color, troughcolor=self._surface_alt_color)

    def set_theme(self, theme: str) -> None:
        self._apply_theme(theme)
        self._style_existing_widgets()
        if hasattr(self, "update_step_display"):
            self.update_step_display()

    def _style_existing_widgets(self) -> None:
        self._style_classic_widget_tree(self.root)
        if hasattr(self, "config_window") and self.config_window.winfo_exists():
            self._style_classic_widget_tree(self.config_window)
        for menu in getattr(self, "_menus", []):
            self._style_menu(menu)

    def _style_menu(self, menu: tk.Menu) -> None:
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

    # Compatibility aliases used by the existing window code.
    def _darken_menu(self, menu: tk.Menu) -> None:
        self._style_menu(menu)

    def _darken_tk_widget(self, widget: tk.Misc) -> None:
        self._style_classic_widget_tree(widget)

    def _style_classic_widget_tree(self, widget: tk.Misc) -> None:
        try:
            if isinstance(widget, tk.Menu):
                self._style_menu(widget)
            elif isinstance(widget, tk.Listbox):
                widget.configure(
                    bg=self._surface_color,
                    fg=self._text_fg,
                    selectbackground=self._select_color,
                    selectforeground=self._text_fg,
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                    relief="flat",
                )
            elif isinstance(widget, tk.Text):
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
                if hasattr(self, "_configure_log_tags") and widget is getattr(self, "log_text", None):
                    self._configure_log_tags()
            elif isinstance(widget, tk.Entry):
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
            elif isinstance(widget, tk.Canvas):
                widget.configure(
                    bg=self._bg_color,
                    highlightbackground=self._border_color,
                    highlightcolor=self._accent_color,
                )
            elif isinstance(widget, tk.Toplevel):
                widget.configure(bg=self._bg_color)
            elif isinstance(widget, tk.Frame) and not isinstance(widget, ttk.Frame):
                widget.configure(bg=self._bg_color)
            elif isinstance(widget, tk.Label) and not isinstance(widget, ttk.Label):
                widget.configure(bg=self._bg_color, fg=self._text_fg)
        except tk.TclError:
            pass

        for child in widget.winfo_children():
            self._style_classic_widget_tree(child)
