import numpy as np

from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.signal import filtfilt, find_peaks, butter
from skimage.filters import frangi
from skimage.morphology import disk, dilation
from skimage.restoration import inpaint

import matplotlib.pyplot as plt

class VesselVelocityEstimatorStep(BaseStep):
    name = "retinal_vessel_velocity_estimator"
    requires = {"moments", "retinal_artery_mask", "retinal_vein_mask", "optic_disc_center"}
    produces = {"retinal_vessel_velocity"}

    def run(self, ctx):
        # ---- Requires ----
        moments = ctx.require("moments")
        moment2 = moments.M2
        moment0 = moments.M0 
        artery_mask = ctx.require("retinal_artery_mask")
        vein_mask = ctx.require("retinal_vein_mask")
        vessel_mask = artery_mask | vein_mask

        fRMS = np.sqrt(moment2 / np.mean(moment0, axis=(-1,-2)))
        fRMSbkg = np.zeros(shape=fRMS.shape)

        mask = dilation(vessel_mask, disk(3)) #TODO add parameter

        for i in range(fRMS.shape[0]):
            fRMSbkg[i,:,:] = inpaint.inpaint_biharmonic(fRMS[i,:,:], mask)

        A = fRMS**2 - fRMSbkg**2
        deltafRMS = np.sign(A) * np.sqrt(np.abs(A))

        velocity_map = 2 * 852e-9 / np.sin(0.25) * deltafRMS * 1e3 #mm/s

        # num_bins = 256  # for 8-bit grayscale
        # hist_matrix = np.zeros((velocity_map.shape[2], num_bins))
        # v_range = (velocity_map.min(),velocity_map.max())

        # for i in range(velocity_map.shape[2]):
        #     masked_pixels = velocity_map[:,:,i][mask]  # select only pixels under mask
        #     hist, _ = np.histogram(masked_pixels, bins=num_bins, range=v_range)
        #     hist_matrix[i,:] = hist

        # ctx.set("hist_matrix", hist_matrix)
        # ctx.set("velocity_map_avg", np.mean(velocity_map,axis=2))
        # ctx.set("fRMS_avg", np.mean(fRMS,axis=2))
        # ctx.set("fRMS_bkg_avg", np.mean(fRMS,axis=2))

        def _elliptical_mask(ny, nx, radius_frac, xp):
            radius_frac = max(0.0, min(1.0, float(radius_frac)))
            a = (nx / 2) * radius_frac
            b = (ny / 2) * radius_frac

            Y, X = xp.ogrid[:ny, :nx]
            cy, cx = ny / 2, nx / 2

            mask = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
            return mask

        sz = velocity_map.shape

        section_mask = _elliptical_mask(sz[-2], sz[-1], 0.5, xp) & (~(_elliptical_mask(sz[-2], sz[-1], 0.2, xp)))

        artery_sig = np.sum(velocity_map * section_mask * mask_artery, axis=(-2,-1)) / np.nonzero(section_mask * mask_artery)

        vein_sig = np.sum(velocity_map * section_mask * mask_vein, axis=(-2,-1)) / np.nonzero(section_mask * mask_vein)

        ctx.set("retinal_vessel_velocity", velocity_map)
        ctx.set("retinal_artery_velocity_signal", artery_sig)
        ctx.set("retinal_vein_velocity_signal", velocity_map)
