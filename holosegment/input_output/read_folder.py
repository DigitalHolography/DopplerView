"""
Read a holodoppler folder
"""

import os
import shutil
from pathlib import Path

class HolodopplerFolder:
    def __init__(self, directory):
        self.directory = directory
        self.eyeflow_config = None
        self.holodoppler_config = None
        self.output_folder = None
        self.raw_folder = None

        self.read()

    def get_EF_config(self):
        json_path = os.path.join(self.directory, "eyeflow", "json")
        if not os.path.exists(json_path) or not os.path.isdir(json_path):
            os.makedirs(json_path)
        json_files = [f for f in os.listdir(json_path) if f.endswith(".json")]
        if len(json_files) == 0:
            config_file = shutil.copy("DefaultEyeflowParams.json", os.path.join(json_path, "input_EF_params.json"))
        else:
            config_file = os.path.join(json_path, json_files[0])
        return config_file

    def get_HD_config(self):
        config_name = self.directory.name + "_input_HD_params.json"
        json_files = [f for f in os.listdir(self.directory) if f.endswith(".json")]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON configuration file found in {self.directory}")
        json_path = os.path.join(self.directory, config_name) if config_name in json_files else os.path.join(self.directory, json_files[0])
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
        holosegment_folder = os.path.join(self.directory, "holosegment")
        if not os.path.exists(holosegment_folder):
            os.makedirs(holosegment_folder)
        
        former_output_folder, index = self._select_highest_subdirectory(holosegment_folder)
        if former_output_folder is None:
            return None, -1
        return os.path.join(holosegment_folder, f"output_{index}"), index
    
    def create_output_folder(self):
        _, index = self.get_output_folder()
        new_output_folder = os.path.join(self.directory, "holosegment", f"output_{index + 1}")
        debug_folder = os.path.join(new_output_folder, "debug")
        # h5_folder = os.path.join(new_output_folder, "h5")
        os.makedirs(debug_folder, exist_ok=True)
        # os.makedirs(h5_folder, exist_ok=True)
        
        print(f"Created output folder: {new_output_folder}")
        self.output_folder = new_output_folder
        return new_output_folder

    def get_input_folder(self):
        self.raw_folder = os.path.join(self.directory, "raw")
        if not os.path.exists(self.raw_folder):
            raise FileNotFoundError(f"Raw folder not found in {self.directory}")
        return self.raw_folder
    
    def find_h5_file(self):
        raw_h5_filename = self.directory.name + "_raw.h5"
        if os.path.exists(os.path.join(self.raw_folder, raw_h5_filename)):
            return os.path.join(self.raw_folder, raw_h5_filename)
        
        # If expected .h5 file is not found, take the first .h5 file found
        h5_files = [f for f in os.listdir(self.raw_folder) if f.endswith(".h5")]
        if len(h5_files) == 0:
            raise FileNotFoundError(f"No HDF5 file found in {self.raw_folder}.")
        return os.path.join(self.raw_folder, h5_files[0])

    def read(self):
        self.raw_folder = self.get_input_folder()
        self.eyeflow_config = self.get_EF_config()
        self.holodoppler_config = self.get_HD_config()
        self.output_folder = self.get_output_folder()
        self.h5_file = self.find_h5_file()