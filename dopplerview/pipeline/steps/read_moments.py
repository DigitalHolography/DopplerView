import h5py
import numpy as np

from dopplerview.pipeline.step import BaseStep
from dopplerview.input_output.h5_file import read_bands

class ReadMomentsStep(BaseStep):
    requires = {"input_file"}
    produces = {"moment0", "moment1", "moment2", "LF_M0", "HF_M0"}
    name = "read_moments"

    def _relevant_config(self, ctx):
        return {}

    def read_holo(self, file_path):
        pass

    def read_hdf5(self, file_path):
        self.logger.info(f"    - Reading the HDF5 file: {file_path}")
        M0, M1, M2, LF_M0, HF_M0 = None, None, None, None, None

        try:
            with h5py.File(file_path, "r") as f:
                self.logger.info(f"    - Available datasets in the HDF5 file: {list(f.keys())}")

                try:
                    self.logger.info("    - Reading the M0 data")
                    M0 = np.squeeze(np.array(f["moment0" if "moment0" in f else "M0"][()]))
                except:
                    self.logger.info("Warning: moment0 or M0 dataset not found")

                try:
                    self.logger.info("    - Reading the M1 data")
                    M1 = np.squeeze(np.array(f["moment1" if "moment1" in f else "M1"][()]))
                except:
                    self.logger.info("Warning: moment1 or M1 dataset not found")

                try:
                    self.logger.info("    - Reading the M2 data")
                    M2 = np.squeeze(np.array(f["moment2" if "moment2" in f else "M2"][()]))
                except:
                    self.logger.info("Warning: moment2 or M2 dataset not found")

                self.logger.info("    - Reading the LF_M0 and HF_M0 data")
                bands = read_bands(f, generic_band_names=["LF_M0", "HF_M0"])
                if len(bands) < 2:
                    self.logger.info(f"Warning: {len(bands)} found in the HDF5 file. Expected at least 2 bands.")
                else:
                    self.logger.info(f"    - Using {bands[0][0]} as LF_M0 and {bands[1][0]} as HF_M0")
                    LF_M0 = np.squeeze(np.array(bands[0][1]))
                    HF_M0 = np.squeeze(np.array(bands[1][1]))

        except Exception as e:
            self.logger.info(f"ID: {type(e).__name__}")
            raise

        return M0, M1, M2, LF_M0, HF_M0

    def run(self, ctx):
        input_file = ctx.require("input_file")
        M0, M1, M2, LF_M0, HF_M0 = self.read_hdf5(input_file)
        ctx.set("moment0", M0)
        ctx.set("moment1", M1)
        ctx.set("moment2", M2)
        
        if LF_M0 is not None:
            ctx.set("LF_M0", LF_M0)
        if HF_M0 is not None:
            ctx.set("HF_M0", HF_M0)

        ctx.output_manager.output(self.name, "moment0", M0, "video")
