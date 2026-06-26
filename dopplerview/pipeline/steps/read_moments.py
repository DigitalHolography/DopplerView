import h5py
import numpy as np

from dopplerview.pipeline.step import BaseStep

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
        M0, M1, M2, band_ratio, LF_M0, HF_M0 = None, None, None, None, None, None

        try:
            with h5py.File(file_path, "r") as f:

                dataset_names = list(f.keys())

                if "moment0" in dataset_names:
                    self.logger.info("    - Reading the M0 data")
                    M0 = np.transpose(np.squeeze(np.array(f["moment0"][()])), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment0 dataset not found")

                if "moment1" in dataset_names:
                    self.logger.info("    - Reading the M1 data")
                    M1 = np.transpose(np.squeeze(np.array(f["moment1"][()])), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment1 dataset not found")

                if "moment2" in dataset_names:
                    self.logger.info("    - Reading the M2 data")
                    M2 = np.transpose(np.squeeze(np.array(f["moment2"][()])), (0, 2, 1))
                else:
                    self.logger.info("Warning: moment2 dataset not found")

                if "LF_M0" in dataset_names:
                    self.logger.info("    - Reading the LF_M0 data")
                    try:
                        lf_m0_data = f["LF_M0"][()]
                    except:
                        try:
                            lf_m0_data = f["band_3000_9000"][:]
                        except:
                            self.logger.info("Warning: LF_M0 dataset not found")
                            lf_m0_data = None
                    if lf_m0_data is not None:
                        LF_M0 = np.transpose(np.squeeze(np.array(f["LF_M0"][()])), (0, 2, 1))
                else:
                    self.logger.info("Warning: LF_M0 dataset not found")
                
                if "HF_M0" in dataset_names:
                    self.logger.info("    - Reading the HF_M0 data")
                    try:
                        hf_m0_data = f["HF_M0"][()]
                    except:
                        try:
                            hf_m0_data = f["band_9000_18000"][:]
                        except:
                            self.logger.info("Warning: HF_M0 dataset not found")
                            hf_m0_data = None
                    if hf_m0_data is not None:
                        HF_M0 = np.transpose(np.squeeze(np.array(f["HF_M0"][()])), (0, 2, 1))

        except Exception as e:
            self.logger.info(f"ID: {type(e).__name__}")
            raise

        return M0, M1, M2, band_ratio, LF_M0, HF_M0

    def run(self, ctx):
        input_file = ctx.require("input_file")
        M0, M1, M2, band_ratio, LF_M0, HF_M0 = self.read_hdf5(input_file)
        ctx.set("moment0", M0)
        ctx.set("moment1", M1)
        ctx.set("moment2", M2)
        ctx.set("LF_M0", LF_M0)
        ctx.set("HF_M0", HF_M0)