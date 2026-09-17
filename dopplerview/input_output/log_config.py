from pathlib import Path
import os
import logging
import sys

def get_log_dir():
    if os.name == "nt":
        base = Path(os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path.home() / ".config"

    log_dir = base / "DopplerView" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_log_file():
    return get_log_dir() / "last_run.log"


def get_debug_log_file():
    return get_log_dir() / "last_run_debug.log"


def setup_logging():
    logger = logging.getLogger()
    # Accept every record at the root. Individual handlers decide what belongs
    # in the readable log, the console/UI, and the complete debug log.
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers (important if re-running in same session)
    logger.handlers.clear()

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_formatter = logging.Formatter("%(message)s")

    # Human-readable log used by the GUI and for routine diagnostics.
    file_handler = logging.FileHandler(get_log_file(), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Complete per-run trace, including metrics and third-party DEBUG records.
    debug_file_handler = logging.FileHandler(
        get_debug_log_file(), mode="w", encoding="utf-8"
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(file_formatter)

    # Windowed PyInstaller builds intentionally have no stderr stream. Keep a
    # console handler for source/CLI runs only when the stream is writable.
    stream = sys.stderr
    if stream is not None and hasattr(stream, "write"):
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(debug_file_handler)
