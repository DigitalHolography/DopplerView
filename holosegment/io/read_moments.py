import os
import h5py
import numpy as np

class Moments:
    def __init__(self, directory):
        self.directory = directory
        self.M0 = None
        self.M1 = None
        self.M2 = None
        self.SH = None

    def read_holo(self, file_path):
        pass

    def read_hdf5(self, file_path):
        print(f"    - Reading the HDF5 file: {file_path}")

        try:
            with h5py.File(file_path, "r") as f:

                dataset_names = list(f.keys())

                if "moment0" in dataset_names:
                    print("    - Reading the M0 data")
                    self.M0 = np.squeeze(f["moment0"][()])
                else:
                    print("Warning: moment0 dataset not found")

                if "moment1" in dataset_names:
                    print("    - Reading the M1 data")
                    self.M1 = np.squeeze(f["moment1"][()])
                else:
                    print("Warning: moment1 dataset not found")

                if "moment2" in dataset_names:
                    print("    - Reading the M2 data")
                    self.M2 = np.squeeze(f["moment2"][()])
                else:
                    print("Warning: moment2 dataset not found")

                if "SH" in dataset_names:
                    print("    - Reading the SH data")
                    self.SH = np.squeeze(f["SH"][()])
                else:
                    print("Warning: SH dataset not found")

        except Exception as e:
            print(f"ID: {type(e).__name__}")
            raise

    def read_moments(self):
        dir_path_raw = os.path.join(self.directory, "raw")

        # Search for all .h5 files in the folder
        h5_files = [f for f in os.listdir(dir_path_raw) if f.endswith(".h5")]

        if len(h5_files) == 0:
            holo_files = [f for f in os.listdir(dir_path_raw) if f.endswith(".holo")]

            if len(holo_files) == 0:
                raise FileNotFoundError(f"No HDF5 or Holo file was found in the folder: {dir_path_raw}")
            
            # Takes the first .holo file found
            ref_raw_file_path = os.path.join(dir_path_raw, holo_files[0])
            raise NotImplementedError("Holo file format is not supported yet. Please provide an HDF5 (.h5) file.")

        # Takes the first .h5 file found
        ref_raw_file_path = os.path.join(dir_path_raw, h5_files[0])
        self.read_hdf5(ref_raw_file_path)
