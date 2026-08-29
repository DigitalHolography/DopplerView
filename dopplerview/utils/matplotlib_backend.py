"""Headless, thread-safe Matplotlib setup for pipeline diagnostics."""

import threading
from functools import wraps

import matplotlib


# # Rendering happens in a spawned compute process and background writer thread;
# # it must never create a Tk window or depend on the GUI event loop.
# matplotlib.use("Agg", force=True)

# pyplot owns process-global state. Serialize diagnostic figure creation even
# with the non-interactive backend.
render_lock = threading.RLock()


def serialized_render(function):
    """Serialize a function that creates or manipulates Matplotlib figures."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with render_lock:
            return function(*args, **kwargs)

    return wrapped
