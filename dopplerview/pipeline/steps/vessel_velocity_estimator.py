import numpy as np

from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.signal import filtfilt, find_peaks, butter
from skimage.filters import frangi
from skimage.morphology import disk, dilation
from skimage.restoration import inpaint
from dopplerview.pipeline.step import BaseStep
import joblib
from dopplerview.segmentation.process_masks import elliptical_mask
from dopplerview.utils.parallelization_utils import run_in_parallel
from functools import partial

import matplotlib.pyplot as plt

class VesselVelocityEstimatorStep(BaseStep):
    name = "retinal_vessel_velocity_estimator"
    requires = {"M0_ff_video", "M2_ff_video", "retinal_artery_mask", "retinal_vein_mask", "optic_disc_center"}
    produces = {"velocity_map_avg","fRMS_avg","fRMS_bkg_avg","retinal_artery_velocity_signal","retinal_vein_velocity_signal"}

    def _relevant_config(self, ctx):
        return {
            "LocalBackgroundDist": ctx.dopplerview_config["VelocityEstimation"]["LocalBackgroundDist"],
            "NumberOfWorkers": ctx.dopplerview_config["NumberOfWorkers"]
        }

    def run(self, ctx):

        # ---- Requires ----
        moment0 = ctx.require("M0_ff_video")
        moment2 = ctx.require("M2_ff_video")

        artery_mask = ctx.require("retinal_artery_mask")
        vein_mask = ctx.require("retinal_vein_mask")
        vessel_mask = artery_mask | vein_mask

        # Compute fRMS
        mean_m0 = np.mean(moment0, axis=(-1, -2), keepdims=True)
        fRMS = np.sqrt(moment2 / mean_m0)

        # Inpaint fRMS to estimate background
        local_background_dist = ctx.dopplerview_config["VelocityEstimation"]["LocalBackgroundDist"]
        mask = dilation(vessel_mask, disk(local_background_dist)) #TODO add parameter

        n_jobs = ctx.dopplerview_config["NumberOfWorkers"]

        def _inpaint_frame(frame, mask):
            return inpaint.inpaint_biharmonic(frame, mask)
        
        fRMSbkg = run_in_parallel(partial(_inpaint_frame, mask=mask), fRMS, n_jobs=n_jobs, chunking=False, task_name="fRMS inpainting")

        # fRMSbkg = np.stack(np.array([inpaint.inpaint_biharmonic(frame, mask) for frame in fRMS]), axis=0)

        # Velocity estimation
        A = fRMS**2 - fRMSbkg**2
        deltafRMS = np.sign(A) * np.sqrt(np.abs(A))

        velocity_map = 2 * 852e-9 / np.sin(0.25) * deltafRMS * 1e6  # mm/s

        ctx.set("velocity_map_avg", np.mean(velocity_map,axis=0))
        ctx.set("fRMS_avg", np.mean(fRMS,axis=0))
        ctx.set("fRMS_bkg_avg", np.mean(fRMSbkg,axis=0))

        sz = velocity_map.shape

        section_mask = elliptical_mask(sz[-2], sz[-1], 0.5) & (~(elliptical_mask(sz[-2], sz[-1], 0.2)))
        
        ctx.set("section_mask", section_mask)
        
        from joblib import Parallel, delayed

        def calculate_velocity_histogram(velocity_map, mask, n_jobs=-1):
            num_bins = 512
            masked_data = velocity_map[:, mask]
            v_min, v_max = velocity_map.min(), velocity_map.max()
            
            def hist_parallel(row):
                return np.histogram(row, bins=num_bins, range=(v_min, v_max))[0]
            
            hist_matrix = Parallel(n_jobs=n_jobs)(
                delayed(hist_parallel)(masked_data[i]) 
                for i in range(velocity_map.shape[0])
            )
            
            return np.array(hist_matrix)
        
        hist_matrix_artery = calculate_velocity_histogram(velocity_map, artery_mask * section_mask)
        hist_matrix_vein = calculate_velocity_histogram(velocity_map, vein_mask * section_mask)
        
        ctx.set("velocity_histogram_artery", hist_matrix_artery)
        ctx.set("velocity_histogram_vein", hist_matrix_vein)

        artery_sig = np.sum(velocity_map * section_mask * artery_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * artery_mask)

        vein_sig = np.sum(velocity_map * section_mask * vein_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * vein_mask)
        
        
        ctx.set("section_mask", section_mask)

        # too big to save
        # ctx.set("retinal_vessel_velocity", velocity_map) 
        ctx.set("retinal_artery_velocity_signal", artery_sig)
        ctx.set("retinal_vein_velocity_signal", vein_sig)
        