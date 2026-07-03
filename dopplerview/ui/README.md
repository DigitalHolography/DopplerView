# DopplerView GUI refactor

This folder splits the former single Tkinter GUI file into focused modules:

- `app.py`: application entry point.
- `main_window.py`: `MainWindow` orchestration, UI construction, config handling, and queue handling.
- `theme.py`: dark/light theme management and recursive styling for classic Tk widgets.
- `worker.py`: multiprocessing pipeline worker and child-process log forwarding.
- `image_utils.py`: Tk image conversion and preview/overlay helpers.

The behavior is intended to match the uploaded `DopplerView_status.py`, including the status label below the progress bars.
