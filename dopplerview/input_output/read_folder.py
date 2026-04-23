"""
Read a holodoppler folder
"""

import os
import glob
import shutil
from pathlib import Path

class HolodopplerFolder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.eyeflow_config = None
        self.holodoppler_config = None
        self.output_folder = None
        self.raw_folder = None
        self.cache_folder = None

        self.read()

    def get_EF_config(self):
        json_path = Path(self.directory) / "eyeflow" / "json"
        if not os.path.exists(json_path) or not os.path.isdir(json_path):
            os.makedirs(json_path)
        json_files = [f for f in os.listdir(json_path) if f.endswith(".json")]
        if len(json_files) == 0:
            config_path = Path("config") / "DefaultEyeflowParams.json"
            config_file = shutil.copy(config_path, json_path / "input_EF_params.json")
        else:
            config_file = json_path / json_files[0]
        return config_file

    def get_HD_config(self):
        config_name = self.directory.name + "_input_HD_params.json"
        json_files = [f for f in os.listdir(self.directory) if f.endswith(".json")]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON configuration file found in {self.directory}")
        json_path = self.directory / config_name if config_name in json_files else self.directory / json_files[0]
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Holodoppler config file not found: {json_path}")
        return json_path

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
        dopplerview_folder = self.directory / "dopplerview"
        if not os.path.exists(dopplerview_folder):
            os.makedirs(dopplerview_folder)
        
        former_output_folder, index = self._select_highest_subdirectory(dopplerview_folder)
        if former_output_folder is None:
            return None, -1
        return dopplerview_folder / f"output_{index}", index

    def create_output_folder(self):
        _, index = self.get_output_folder()
        new_output_folder = self.directory / "dopplerview" / f"output_{index + 1}"
        debug_folder = new_output_folder
        # input_folder = new_output_folder / "h5"
        os.makedirs(debug_folder, exist_ok=True)
        # os.makedirs(input_folder, exist_ok=True)
        
        self.output_folder = new_output_folder
        return new_output_folder
    
    def get_cache_folder(self):
        self.cache_folder = self.directory / "dopplerview" / "cache"
        os.makedirs(self.cache_folder, exist_ok=True)
        return self.cache_folder
    
    def get_input_folder(self):
        raw_folder = self.directory / "raw"
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

    def read(self):
        self.raw_folder = self.get_input_folder()
        self.eyeflow_config = self.get_EF_config()
        self.holodoppler_config = self.get_HD_config()
        self.output_folder = self.get_output_folder()
        self.input_file = self.find_input_file()