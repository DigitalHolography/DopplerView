import os
import h5py
import numpy as np

from dopplerview.pipeline.step import BaseStep

class Moments:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.M0 = None
        self.M1 = None
        self.M2 = None

    def read_holo(self, file_path):
        pass

    def read_hdf5(self, file_path):
        print(f"    - Reading the HDF5 file: {file_path}")

        try:
            with h5py.File(file_path, "r") as f:

                dataset_names = list(f.keys())

                if "moment0" in dataset_names:
                    print("    - Reading the M0 data")
                    self.M0 = np.transpose(np.squeeze(f["moment0"][()]), (0, 2, 1))
                else:
                    print("Warning: moment0 dataset not found")

                if "moment1" in dataset_names:
                    print("    - Reading the M1 data")
                    self.M1 = np.transpose(np.squeeze(f["moment1"][()]), (0, 2, 1))
                else:
                    print("Warning: moment1 dataset not found")

                if "moment2" in dataset_names:
                    print("    - Reading the M2 data")
                    self.M2 = np.transpose(np.squeeze(f["moment2"][()]), (0, 2, 1))
                else:
                    print("Warning: moment2 dataset not found")

        except Exception as e:
            print(f"ID: {type(e).__name__}")
            raise

    def read_moments(self):
        self.read_hdf5(self.input_file_path)

class ReadMomentsStep(BaseStep):
    requires = {"input_file"}
    produces = {"moment0", "moment1", "moment2"}
    name = "read_moments"

    def run(self, ctx):
        reader = Moments(ctx.require("input_file"))
        reader.read_moments()
        ctx.cache["moment0"] = reader.M0
        ctx.cache["moment1"] = reader.M1
        ctx.cache["moment2"] = reader.M2