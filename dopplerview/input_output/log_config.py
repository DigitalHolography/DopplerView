from pathlib import Path
import os
import logging

def get_log_dir():
    if os.name == "nt":
        base = Path(os.getenv("APPDATA"))
    else:
        base = Path.home() / ".config"

    log_dir = base / "DopplerView" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_log_file():
    return get_log_dir() / "last_run.log"


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (important if re-running in same session)
    logger.handlers.clear()

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_formatter = logging.Formatter("%(message)s")

    # --- Terminal handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # --- File handler ---
    file_handler = logging.FileHandler(get_log_file(), mode="w")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)