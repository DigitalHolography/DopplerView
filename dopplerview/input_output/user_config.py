import os
from pathlib import Path
import shutil
import sys
from importlib.resources import files



def get_user_config_dir():
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA"))
    else:
        base = Path.home() / ".config"

    path = base / "DopplerView"
    path.mkdir(parents=True, exist_ok=True)
    return path

# def get_resource_path(filename):
#     if hasattr(sys, "_MEIPASS"):
#         base = Path(sys._MEIPASS)
#     else:
#         base = Path(__file__).parent

#     return base / "config" / filename

def get_resource_path(filename):
    return files("dopplerview.resources") / filename

def ensure_config_file(filename):
    user_dir = get_user_config_dir()
    user_file = user_dir / filename

    if not user_file.exists():
        default_file = get_resource_path(filename)
        shutil.copy(default_file, user_file)

    return user_file