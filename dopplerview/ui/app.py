import logging
import multiprocessing
import tkinter as tk
from tkinter import messagebox

from dopplerview.input_output import log_config

from dopplerview.ui.main_window import MainWindow, TkinterDnD


logger = logging.getLogger(__name__)


def main() -> None:
    multiprocessing.freeze_support()

    log_config.setup_logging()

    root = None
    try:
        if TkinterDnD:
            root = TkinterDnD.Tk()
        else:
            root = tk.Tk()

        MainWindow(root)
        root.mainloop()
    except Exception:
        logger.exception("DopplerView failed to start or stopped unexpectedly")
        if root is not None:
            try:
                messagebox.showerror(
                    "DopplerView error",
                    "DopplerView stopped unexpectedly. Details were written to:\n"
                    f"{log_config.get_log_file()}",
                    parent=root,
                )
            except tk.TclError:
                pass
        raise


if __name__ == "__main__":
    main()
