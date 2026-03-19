import numpy as np

from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.signal import filtfilt, find_peaks, butter
from skimage.filters import frangi
from skimage.morphology import disk, dilation
from skimage.restoration import inpaint
from holosegment.pipeline.step import BaseStep

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class ArterialWaveformAnalysisStep(BaseStep):
    name = "arterial_waveform_analysis"
    requires = {"retinal_artery_velocity_signal"}
    produces = {"retinal_artery_velocity_signal_filtered","retinal_artery_velocity_signal_filtered_perbeat","beat_indices","time_per_beat"}

    def find_systole_index(self, pulse_artery):
        Wn = 0.5  # cutoff frequency in 0 1 Niquist freq TODO parametrize
        N = 1     # filter order TODO parametrize
        b, a = butter(N, Wn, btype="low")
        pulse_filtered = filtfilt(b, a, pulse_artery)
        validation_distance = 10 # distance minimal to be accepted TODO parametrize
        min_peak_distance = 40 # distance between two peaks minimal TODO parametrize

        diff_signal = np.gradient(pulse_filtered)

        # Step 3: Detect peaks
        min_peak_height = np.percentile(diff_signal, 95) # TODO parametrize

        peaks, _ = find_peaks(
            diff_signal,
            height = min_peak_height,
            distance = min_peak_distance,
        )

        # Step 4: Validate peaks
        def validate_peaks(peaks, min_distance):
            if len(peaks) == 0:
                return peaks
            validated = [peaks[0]]
            for idx in peaks[1:]:
                if idx - validated[-1] >= min_distance:
                    validated.append(idx)
            return np.array(validated)
        
        peaks = validate_peaks(peaks, validation_distance) # 

        return peaks, pulse_filtered
    
    def slice_interp_beats(self, peaks, sig):
        nbeat = len(peaks)

        ninterp = 128 # TODO parametrize

        sig_perbeat = np.zeros(shape=(nbeat,ninterp))

        for i in range(nbeat-1):
            beat_sig = sig[peaks[i]:peaks[i+1]]
            beat_sig_interp = np.interp(np.linspace(0,1,ninterp),np.linspace(0,1,len(beat_sig)),beat_sig)
            sig_perbeat[i,:] = beat_sig_interp
        
        return sig_perbeat

    def run(self, ctx):
        # ---- Requires ----
        sig = ctx.require("retinal_artery_velocity_signal")

        peaks, sig_filtered = self.find_systole_index(sig)

        sig_perbeat = self.slice_interp_beats(peaks, sig_filtered)

        ctx.set("retinal_artery_velocity_signal_filtered_perbeat",sig_perbeat)
        ctx.set("retinal_artery_velocity_signal_filtered",sig_filtered)
        ctx.set("beat_indices", peaks)
        ctx.set("time_per_beat", np.diff(peaks) * 256/37000) # TODO parametrize look for params