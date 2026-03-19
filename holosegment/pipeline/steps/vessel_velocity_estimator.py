import numpy as np

from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.signal import filtfilt, find_peaks, butter
from skimage.filters import frangi
from skimage.morphology import disk, dilation
from skimage.restoration import inpaint
from holosegment.pipeline.step import BaseStep
import joblib
from holosegment.segmentation.process_masks import elliptical_mask
from holosegment.utils.parallelization_utils import run_in_parallel
from functools import partial

import matplotlib.pyplot as plt

class VesselVelocityEstimatorStep(BaseStep):
    name = "retinal_vessel_velocity_estimator"
    requires = {"moments", "retinal_artery_mask", "retinal_vein_mask", "optic_disc_center"}
    produces = {"retinal_vessel_velocity","velocity_map_avg","fRMS_avg","fRMS_bkg_avg","retinal_artery_velocity_signal","retinal_vein_velocity_signal"}

    def _relevant_config(self, ctx):
        return {"LocalBackgroundDist": ctx.eyeflow_config["PulseAnalysis"]["LocalBackgroundDist"]}

    def run(self, ctx):

        # ---- Requires ----
        moments = ctx.require("moments")
        moment2 = moments.M2
        moment0 = moments.M0

        artery_mask = ctx.require("retinal_artery_mask")
        vein_mask = ctx.require("retinal_vein_mask")
        vessel_mask = artery_mask | vein_mask

        # Compute fRMS
        mean_m0 = np.mean(moment0, axis=(-1, -2), keepdims=True)
        fRMS = np.sqrt(moment2 / mean_m0)

        # Inpaint fRMS to estimate background
        local_background_dist = ctx.eyeflow_config["PulseAnalysis"]["LocalBackgroundDist"]
        print(local_background_dist)
        mask = dilation(vessel_mask, disk(3)) #TODO add parameter

        n_jobs = joblib.cpu_count() #TODO add parameter for number of parallel jobs

        print(f"    - Inpainting fRMS with {n_jobs} parallel jobs")

        def _inpaint_frame(frame, mask):
            return inpaint.inpaint_biharmonic(frame, mask)
        
        fRMSbkg = run_in_parallel(partial(_inpaint_frame, mask=mask), fRMS, n_jobs=n_jobs)

        # fRMSbkg = np.stack(np.array([inpaint.inpaint_biharmonic(frame, mask) for frame in fRMS]), axis=0)

        # Velocity estimation
        A = fRMS**2 - fRMSbkg**2
        deltafRMS = np.sign(A) * np.sqrt(np.abs(A))

        velocity_map = 2 * 852e-9 / np.sin(0.25) * deltafRMS * 1e6  # mm/s

        ctx.set("velocity_map", velocity_map)

        # num_bins = 256  # for 8-bit grayscale
        # hist_matrix = np.zeros((velocity_map.shape[2], num_bins))
        # v_range = (velocity_map.min(),velocity_map.max())

        # for i in range(velocity_map.shape[2]):
        #     masked_pixels = velocity_map[:,:,i][mask]  # select only pixels under mask
        #     hist, _ = np.histogram(masked_pixels, bins=num_bins, range=v_range)
        #     hist_matrix[i,:] = hist

        # ctx.set("hist_matrix", hist_matrix)
        print("setting debug variables")
        ctx.set("velocity_map_avg", np.mean(velocity_map,axis=0))
        ctx.set("fRMS_avg", np.mean(fRMS,axis=0))
        ctx.set("fRMS_bkg_avg", np.mean(fRMSbkg,axis=0))

        sz = velocity_map.shape

        section_mask = elliptical_mask(sz[-2], sz[-1], 0.5) & (~(elliptical_mask(sz[-2], sz[-1], 0.2)))

        artery_sig = np.sum(velocity_map * section_mask * artery_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * artery_mask)

        vein_sig = np.sum(velocity_map * section_mask * vein_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * vein_mask)

        ctx.set("retinal_vessel_velocity", velocity_map)
        ctx.set("retinal_artery_velocity_signal", artery_sig)
        ctx.set("retinal_vein_velocity_signal", vein_sig)