import json
import os
from pathlib import Path
import shutil
import sys
from importlib.resources import files
from importlib.metadata import version


def get_user_config_dir():
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA"))
    else:
        base = Path.home() / ".config"

    path = base / "DopplerView" / version("dopplerview")
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

def get_version(config):
    with json.load(open(config, "r")) as data:
        return data.get("Version", 0)
    
def get_latest_config(user_config, default_config):
    if get_version(user_config) < get_version(default_config):
        return default_config
    return user_config

def ensure_latest_DV_config(config_path):
    default_config = get_resource_path("default_dv_params.json")
    user_config = ensure_config_file("DV_params.json", versioning=True)
    latest_config = get_latest_config(user_config, default_config)

    if latest_config != user_config:
        shutil.copy(latest_config, user_config)

    return latest_config

def ensure_config_file(filename, versioning=False):
    user_dir = get_user_config_dir()
    user_file = user_dir / filename

    default_file = get_resource_path(filename)
    if not user_file.exists() or (versioning and get_version(user_file) < get_version(default_file)):
        shutil.copy(default_file, user_file)

    return user_file