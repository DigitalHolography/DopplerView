import os
import h5py
import numpy as np

def read_hdf5(self):
    dir_path_raw = os.path.join(self.directory, "raw")

    # Search for all .h5 files in the folder
    h5_files = [f for f in os.listdir(dir_path_raw) if f.endswith(".h5")]

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No HDF5 file was found in the folder: {dir_path_raw}")

    # Takes the first .h5 file found
    ref_raw_file_path = os.path.join(dir_path_raw, h5_files[0])

    print(f"    - Reading the HDF5 file: {h5_files[0]}")

    try:
        with h5py.File(ref_raw_file_path, "r") as f:

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
