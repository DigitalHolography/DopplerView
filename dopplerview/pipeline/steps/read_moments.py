import h5py
import numpy as np

from dopplerview.pipeline.step import BaseStep
from dopplerview.input_output.h5_file import read_bands
from dopplerview.segmentation import pulse_analysis

class ReadMomentsStep(BaseStep):
    requires = {"input_file"}
    produces = {"moment0", "moment1", "moment2", "LF_M0", "HF_M0", "sampling_freq"}
    name = "read_moments"

    def _relevant_config(self, ctx):
        return {
            "sampling_freq": ctx.holodoppler_config.get("sampling_freq"),
            "batch_stride": ctx.holodoppler_config.get("batch_stride"),
            }

    def read_holo(self, file_path):
        pass

    def _read_required_moment(self, h5_file, names):
        last_error = KeyError(names)
        for name in names:
            if name not in h5_file:
                continue
            try:
                return np.squeeze(np.asarray(h5_file[name][()]))
            except Exception as exc:
                last_error = exc
        message = f"Cannot read {names} from '{h5_file.filename}'"
        self.logger.error(message)
        raise RuntimeError(message) from last_error

    def read_hdf5(self, file_path):
        self.logger.info(f"    - Reading the HDF5 file: {file_path}")
        M0, M1, M2, LF_M0, HF_M0 = None, None, None, None, None

        try:
            with h5py.File(file_path, "r") as f:
                self.logger.info(f"    - Available datasets in the HDF5 file: {list(f.keys())}")

                self.logger.info("    - Reading the M0 data")
                M0 = self._read_required_moment(f, ["moment0", "M0"])

                self.logger.info("    - Reading the M1 data")
                M1 = self._read_required_moment(f, ["moment1", "M1"])

                self.logger.info("    - Reading the M2 data")
                M2 = self._read_required_moment(f, ["moment2", "M2"])

                self.logger.info("    - Reading the LF_M0 and HF_M0 data")
                bands = read_bands(f, generic_band_names=["LF_M0", "HF_M0"])
                if len(bands) < 2:
                    self.logger.info(f"Warning: {len(bands)} found in the HDF5 file. Expected at least 2 bands.")
                else:
                    self.logger.info(f"    - Using {bands[0][0]} as LF_M0 and {bands[1][0]} as HF_M0")
                    LF_M0 = np.squeeze(np.asarray(bands[0][1]))
                    HF_M0 = np.squeeze(np.asarray(bands[1][1]))

                    # band name has the form band_low_high, e.g., band_0_1000, band_1000_2000, etc.
                    # extract the low and high frequencies from the band names
                    lf_low, lf_high = map(int, bands[0][0].split("_")[-2:])
                    hf_low, hf_high = map(int, bands[1][0].split("_")[-2:])
                    if lf_high > 9000:
                        self.logger.info(f"Warning: LF_M0 band has a high frequency of {lf_high} Hz, which is above the expected threshold of 9 kHz. Choroid segmentation may not work properly.")
                    if hf_low < 16000:
                        self.logger.info(f"Warning: HF_M0 band has a low frequency of {hf_low} Hz, which is below the expected threshold of 16 kHz. Choroid segmentation may not work properly.")

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

        fs = ctx.holodoppler_config.get("sampling_freq")
        stride = ctx.holodoppler_config.get("batch_stride")

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)
        self.logger.info(f"    - Camera sampling frequency: {fs} Hz, batch stride: {stride}. Effective sampling frequency after accounting for batch stride: {sampling_frequency:.2f} Hz")

        ctx.set("sampling_freq", sampling_frequency)