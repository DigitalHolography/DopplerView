"""Headless, thread-safe Matplotlib helpers for pipeline diagnostics."""

import threading
from functools import wraps

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Matplotlib also has shared caches (for example, font discovery). Serialize
# diagnostic figure creation performed by background output threads.
render_lock = threading.RLock()


def new_agg_figure(*args, **kwargs):
    """Create a figure attached directly to a non-GUI Agg canvas.

    This avoids importing ``matplotlib.pyplot`` and therefore never changes
    the process-global backend selected by Tk or an IPython notebook.
    """
    figure = Figure(*args, **kwargs)
    FigureCanvasAgg(figure)
    return figure


def serialized_render(function):
    """Serialize a function that creates or manipulates Matplotlib figures."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with render_lock:
            return function(*args, **kwargs)

    return wrapped
