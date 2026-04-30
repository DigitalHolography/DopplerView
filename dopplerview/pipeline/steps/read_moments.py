import h5py
import numpy as np

from dopplerview.pipeline.step import BaseStep

class ReadMomentsStep(BaseStep):
    requires = {"input_file"}
    produces = {"moment0", "moment1", "moment2"}
    name = "read_moments"

    def _relevant_config(self, ctx):
        return {}

    def read_holo(self, file_path):
        pass

    def read_hdf5(self, file_path):
        self.logger.info(f"    - Reading the HDF5 file: {file_path}")
        M0, M1, M2 = None, None, None

        try:
            with h5py.File(file_path, "r") as f:

                dataset_names = list(f.keys())

                if "moment0" in dataset_names:
                    self.logger.info("    - Reading the M0 data")
                    M0 = np.transpose(np.squeeze(f["moment0"][()]), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment0 dataset not found")

                if "moment1" in dataset_names:
                    self.logger.info("    - Reading the M1 data")
                    M1 = np.transpose(np.squeeze(f["moment1"][()]), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment1 dataset not found")

                if "moment2" in dataset_names:
                    self.logger.info("    - Reading the M2 data")
                    M2 = np.transpose(np.squeeze(f["moment2"][()]), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment2 dataset not found")

        except Exception as e:
            self.logger.info(f"ID: {type(e).__name__}")
            raise

        return M0, M1, M2

    def run(self, ctx):
        input_file = ctx.require("input_file")
        M0, M1, M2 = self.read_hdf5(input_file)
        ctx.cache["moment0"] = M0
        ctx.cache["moment1"] = M1
        ctx.cache["moment2"] = M2