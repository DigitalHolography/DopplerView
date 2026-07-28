import numpy as np

from skimage.morphology import disk, dilation
from skimage.restoration import inpaint
from dopplerview.pipeline.step import BaseStep
import joblib
from dopplerview.segmentation.process_masks import elliptical_mask
from dopplerview.utils.parallelization_utils import run_in_parallel
from functools import partial

class VesselVelocityEstimatorStep(BaseStep):
    name = "retinal_vessel_velocity_estimator"
    requires = {"moment0", "moment2", "M0_ff_video", "retinal_artery_mask", "retinal_vein_mask", "optic_disc_center"} # ,"optic_disc_mask"
    produces = {"retinal_vessel_velocity","velocity_map_avg","fRMS_avg","fRMS_bkg_avg","retinal_artery_velocity_signal","retinal_vein_velocity_signal",
    "artery_section_mask", "vein_section_mask", "retinal_artery_M0ff_signal", "retinal_vein_M0ff_signal"}

    def _relevant_config(self, ctx):
        return {
            "LocalBackgroundDist": ctx.dopplerview_config.get("VelocityEstimation", {}).get("LocalBackgroundDist", 2),
            "NumberOfWorkers": ctx.dopplerview_config.get("NumberOfWorkers", 0.5)
        }

    def run(self, ctx):

        # ---- Requires ----
        moment0 = ctx.require("moment0")
        moment0ff = ctx.require("M0_ff_video")
        moment2 = ctx.require("moment2")

        artery_mask = ctx.require("retinal_artery_mask")
        vein_mask = ctx.require("retinal_vein_mask")
        vessel_mask = artery_mask | vein_mask

        optic_disc_center_x, optic_disc_center_y = ctx.require("optic_disc_center")
        # optic_disk_mask = ctx.require("optic_disk_mask")

        # Compute fRMS
        mean_m0 = np.mean(moment0, axis=(-1, -2), keepdims=True)
        fRMS = np.sqrt(moment2 / mean_m0)

        # Inpaint fRMS to estimate background
        local_background_dist = ctx.dopplerview_config.get("VelocityEstimation", {}).get("LocalBackgroundDist", 2)
        mask = dilation(vessel_mask, disk(local_background_dist)) 

        n_jobs = ctx.dopplerview_config.get("NumberOfWorkers", 0.5)

        def _inpaint_frame(frame, mask):
            return inpaint.inpaint_biharmonic(frame, mask)
        
        fRMSbkg = run_in_parallel(partial(_inpaint_frame, mask=mask), fRMS, n_jobs=n_jobs, chunking=False, task_name="fRMS inpainting")

        # fRMSbkg = np.stack(np.array([inpaint.inpaint_biharmonic(frame, mask) for frame in fRMS]), axis=0)

        # Velocity estimation
        A = fRMS**2 - fRMSbkg**2
        deltafRMS = np.sign(A) * np.sqrt(np.abs(A))

        velocity_map = 2 * 852e-9 / np.sin(0.20) * deltafRMS * 1e3  # mm/s

        ctx.set("velocity_map", velocity_map)

        # num_bins = 256  # for 8-bit grayscale
        # hist_matrix = np.zeros((velocity_map.shape[2], num_bins))
        # v_range = (velocity_map.min(),velocity_map.max())

        # for i in range(velocity_map.shape[2]):
        #     masked_pixels = velocity_map[:,:,i][mask]  # select only pixels under mask
        #     hist, _ = np.histogram(masked_pixels, bins=num_bins, range=v_range)
        #     hist_matrix[i,:] = hist

        # ctx.set("hist_matrix", hist_matrix)
        ctx.set("velocity_map_avg", np.mean(velocity_map,axis=0))
        ctx.set("fRMS_avg", np.mean(fRMS,axis=0))
        ctx.set("fRMS_bkg_avg", np.mean(fRMSbkg,axis=0))

        sz = velocity_map.shape

        radius_out = ctx.dopplerview_config.get("VelocityEstimation", {}).get("SectionRadiusOut", 0.75)
        radius_in = ctx.dopplerview_config.get("VelocityEstimation", {}).get("SectionRadiusIn", 0.15)

        section_mask = elliptical_mask(sz[-2], sz[-1], radius_out, center=(optic_disc_center_y, optic_disc_center_x)) & (~(elliptical_mask(sz[-2], sz[-1], radius_in, center=(optic_disc_center_y, optic_disc_center_x))))
        # section_mask *= ~optic_disk_mask
        artery_sig = np.sum(velocity_map * section_mask * artery_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * artery_mask)

        artery_sigm0ff = np.sum(moment0ff * section_mask * artery_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * artery_mask)
        ctx.set("retinal_artery_M0ff_signal", artery_sigm0ff)
        
        vein_sigm0ff = np.sum(moment0ff * section_mask * vein_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * vein_mask)
        ctx.set("retinal_vein_M0ff_signal", vein_sigm0ff)


        vein_sig = np.sum(velocity_map * section_mask * vein_mask, axis=(-2,-1)) / np.count_nonzero(section_mask * vein_mask)

        ctx.set("artery_section_mask", section_mask * artery_mask)
        ctx.set("vein_section_mask", section_mask * vein_mask)

        ctx.set("retinal_vessel_velocity", velocity_map)
        ctx.set("retinal_artery_velocity_signal", artery_sig)
        ctx.set("retinal_vein_velocity_signal", vein_sig)