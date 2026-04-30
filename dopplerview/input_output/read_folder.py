"""
Read a holodoppler folder
"""

import json
import os
import glob
import shutil
from pathlib import Path
import sys
import dopplerview.input_output.user_config as user_config

class HolodopplerFolder:
    def __init__(self, parent_directory):
        self.directory = None
        self.holodoppler_config = None
        self.raw_folder = None
        self.input_file = None
        self.measure_name = None

        self.read(parent_directory)

    def get_HD_folder(self, parent_directory):
        self.measure_name = parent_directory.name
        self.directory = Path(parent_directory) / (self.measure_name + "_HD")
        if not self.directory.exists():
            raise FileNotFoundError(f"Holodoppler folder not found: {self.directory}")

    def get_HD_config(self):
        config_directory = self.directory / "json"
        config_name = "parameters.json"
        json_files = [f for f in os.listdir(config_directory) if f.endswith(".json")]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON configuration file found in {self.directory}")
        json_path = config_directory / config_name if config_name in json_files else config_directory / json_files[0]
        return json_path
    
    def get_input_folder(self):
        raw_folder = self.directory / "h5"
        if not os.path.exists(raw_folder):
            raise FileNotFoundError(f"Raw folder not found in {self.directory}")
        return raw_folder
    
    def find_input_file(self):
        output_files = glob.glob(os.path.join(self.raw_folder, "*output.h5"))
        raw_files = glob.glob(os.path.join(self.raw_folder, "*raw.h5"))
        if output_files:
            return Path(output_files[0])
        if raw_files:
            return Path(raw_files[0])
        # If expected .h5 file is not found, take the first .h5 file found
        input_files = [f for f in os.listdir(self.raw_folder) if f.endswith(".h5")]
        if len(input_files) == 0:
            raise FileNotFoundError(f"No HDF5 file found in {self.raw_folder}.")
        return self.raw_folder / input_files[0]

    def read(self, parent_directory):
        self.get_HD_folder(parent_directory)
        self.raw_folder = self.get_input_folder()
        self.holodoppler_config = self.get_HD_config()
        self.input_file = self.find_input_file()

class DopplerViewFolder():
    def __init__(self, parent_directory):
        self.measure_name = parent_directory.name
        self.directory = Path(parent_directory) / (self.measure_name + "_DV")
        self.directory.mkdir(exist_ok=True)

        self.h5_folder = self.directory / "h5"
        self.h5_folder.mkdir(exist_ok=True)
        self.h5_path = self.h5_folder / f"{self.measure_name}_DV.h5"

        self.output_parent_folder = self.directory / "output"
        self.output_parent_folder.mkdir(exist_ok=True)
        self.output_folder = None  # Will be set when creating a new output folder

        self.cache_folder = self.directory / "cache"
        self.cache_folder.mkdir(exist_ok=True)

        self.config_folder = self.directory / "config"
        self.config_folder.mkdir(exist_ok=True)
        self.config_name = "DV_params.json"

        self.dopplerview_config = self.get_dopplerview_config()

    def get_cache_folder(self):
        return self.cache_folder

    def get_dopplerview_config(self):
        config_path = self.config_folder / self.config_name
        if not config_path.exists():
            default_config_path = user_config.ensure_config_file("default_DV_params.json")
            shutil.copy(default_config_path, config_path)
        return config_path

    def _select_highest_subdirectory(self, base_path: str):
        # Filter entries to include only subdirectories
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        # Use a regex to extract the digit at the end of each subdirectory name
        digit_subdirs = []
        for subdir in subdirs:
            parts = subdir.split('_')
            if parts[-1].isdigit():
                digit_subdirs.append((subdir, int(parts[-1])))
        
        if not digit_subdirs:
            return None, -1  # No subdirectories with digits found
        return max(digit_subdirs, key=lambda x: x[1])

    def get_output_folder(self):
        """
        Creates a new output folder with an incremented index if it already exists.
        """
        if self.output_folder is not None:
            return self.output_folder, int(self.output_folder.name.split('_')[-1])
        former_output_folder, index = self._select_highest_subdirectory(self.output_parent_folder)
        if former_output_folder is None:
            return None, -1
        return self.output_parent_folder / f"output_{index}", index

    def create_output_folder(self):
        _, index = self.get_output_folder()
        new_output_folder = self.output_parent_folder / f"output_{index + 1}"
        # input_folder = new_output_folder / "h5"
        os.makedirs(new_output_folder, exist_ok=True)
        # os.makedirs(input_folder, exist_ok=True)
        
        self.output_folder = new_output_folder
        return new_output_folder
    
    def get_h5_path(self):
        return self.h5_folder / f"{self.measure_name}_DV.h5"