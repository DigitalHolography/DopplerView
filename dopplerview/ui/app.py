import multiprocessing
import tkinter as tk

import matplotlib

from dopplerview.input_output import log_config

from dopplerview.ui.main_window import MainWindow, TkinterDnD


def main() -> None:
    multiprocessing.freeze_support()

    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    log_config.setup_logging()
    matplotlib.use("Agg")

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
