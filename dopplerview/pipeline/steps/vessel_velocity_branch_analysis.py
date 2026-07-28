import numpy as np

from skimage.morphology import disk, dilation
from skimage.restoration import inpaint
from dopplerview.pipeline.step import BaseStep
import joblib
from dopplerview.segmentation.process_masks import elliptical_mask, get_labeled_vessels
from dopplerview.utils.parallelization_utils import run_in_parallel
from functools import partial

class VesselVelocityBranchesStep(BaseStep):
    name = "retinal_vessel_branch_analysis"
    requires = {"retinal_vessel_velocity", "M0_ff_video", "retinal_artery_mask", "retinal_vein_mask" "optic_disc_center"} # ,"optic_disc_mask"
    produces = {"retinal_artery_velocity_signal_by_branch", "retinal_vein_velocity_signal_by_branch", "artery_labeled_branches", "vein_labeled_branches"}

    def run(self, ctx):

        # ---- Requires ----
        velocity_map = ctx.require("retinal_vessel_velocity")
        m0ff = ctx.require("M0_ff_video")

        artery_mask = ctx.require("retinal_artery_mask")
        vein_mask = ctx.require("retinal_vein_mask")
        vessel_mask = artery_mask | vein_mask

        optic_disc_center_x, optic_disc_center_y = ctx.require("optic_disc_center")
        # optic_disk_mask = ctx.require("optic_disk_mask")

        # create the labeled branches masks for art and veins

        artery_labeled_branches, _ = get_labeled_vessels(artery_mask, mask_optic_disc=True, x_center=optic_disc_center_x, y_center=optic_disc_center_y, r1=0.1)
        vein_labeled_branches, _ = get_labeled_vessels(vein_mask, mask_optic_disc=True, x_center=optic_disc_center_x, y_center=optic_disc_center_y, r1=0.1)

        ctx.set("artery_labeled_branches", artery_labeled_branches)
        ctx.set("vein_labeled_branches", vein_labeled_branches)

        # extract each branch data

        n_branch_art = np.max(artery_labeled_branches)
        n_branch_vein = np.max(vein_labeled_branches)

        def _extract_sig(data, lab_mask, branch_id):
            mask = (lab_mask == branch_id)
            return np.sum(data * mask , axis=(-2,-1)) / np.count_nonzero(mask)


        art_sig_branches = np.stack([_extract_sig(velocity_map, artery_labeled_branches, branch_id) for branch_id in range(1,n_branch_art)])
        vein_sig_branches = np.stack([_extract_sig(velocity_map, vein_labeled_branches, branch_id) for branch_id in range(1,n_branch_vein)])

        ctx.set("retinal_artery_velocity_signal_by_branch", art_sig_branches)
        ctx.set("retinal_vein_velocity_signal_by_branch", vein_sig_branches)

        art_sig_branches = np.stack([_extract_sig(m0ff, artery_labeled_branches, branch_id) for branch_id in range(1,n_branch_art)])
        vein_sig_branches = np.stack([_extract_sig(m0ff, vein_labeled_branches, branch_id) for branch_id in range(1,n_branch_vein)])

        ctx.set("retinal_artery_M0ff_signal_by_branch", art_sig_branches)
        ctx.set("retinal_vein_M0ff_signal_by_branch", vein_sig_branches)



        