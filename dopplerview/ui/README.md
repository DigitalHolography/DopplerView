# DopplerView GUI

This folder splits the former single Tkinter GUI file into focused modules:

- `app.py`: application entry point.
- `main_window.py`: `MainWindow` orchestration, UI construction, config handling, and queue handling.
- `theme.py`: Sun Valley dark/light theme management and styling for classic Tk widgets.
- `worker.py`: multiprocessing pipeline worker and child-process log forwarding.
- `image_utils.py`: Tk image conversion and preview/overlay helpers.

The main window follows the same layout language as HoloDoppler: Minimal and
Advanced tabs, compact toolbars, bordered drop/preview areas, Segoe UI type and
Sun Valley accent controls.

The collapsible **Logs** panel mirrors parent and pipeline-process log messages.
The same readable messages are persisted to
`%APPDATA%\\DopplerView\\logs\\last_run.log`. A complete trace, including DEBUG
records and runtime metrics, is written alongside it as `last_run_debug.log`.
Both keep diagnostics available in the windowed (console-free) Windows build.
