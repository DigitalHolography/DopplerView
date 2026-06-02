from dopplerview.pipeline.step import BaseStep
from dopplerview.segmentation.process_masks import clean_vessel_mask
import numpy as np

class AVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image_cleaned", "correlation", "diasys_image", "optic_disc_center"}
    produces = {"retinal_artery_mask", "retinal_vein_mask", "retinal_artery_mask_clean", "retinal_vein_mask_clean"}
    name = "retinal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
        return { "AVSegmentationMethod": params.get("AVSegmentationMethod", "AI"),
                    "av_segmentation_model": ctx.get_current_model_for_task(self.name),
                    "DiaphragmRadius": params.get("DiaphragmRadius", 0.45),
                    "CenterRadius": params.get("CenterRadius", 0.1)
        }

    def deep_segmentation(self, ctx):
        # model_name = ctx.dopplerview_config["models"]["av"]
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)

        mask = model.predict(input)
        mask = np.squeeze(mask)  # Remove channel dimension if present

        if model.spec.output_activation == "argmax":
            return np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)
        
        return mask[0], mask[1]

    def handmade_segmentation(self, ctx):
        raise NotImplementedError("Handmade artery vein segmentation not implemented yet.")
    
    def clean_mask(self, raw_mask, ctx):
        optic_disc_center = ctx.require("optic_disc_center")
        width, height = raw_mask.shape
        optic_disc_center = (optic_disc_center[0] / width, optic_disc_center[1] / height)

        params = ctx.dopplerview_config["Mask"]

        clean_mask = clean_vessel_mask(
            raw_mask,
            image_shape=raw_mask.shape,
            optic_disc_center=optic_disc_center,
            diaphragm_radius=params.get("DiaphragmRadius", 0.45),
            connect_radius=params.get("CenterRadius", 0.1),
            tolerance=5
        )

        return clean_mask

    def run(self, ctx):
        if ctx.dopplerview_config.get("AVSegmentationMethod", "AI") == "AI":
            self.logger.info("    - Use deep segmentation model for artery vein segmentation.")
            artery_mask, vein_mask = self.deep_segmentation(ctx)
        else:
            self.logger.info("    - Use hand-made heuristics for artery vein segmentation.")
            artery_mask, vein_mask = self.handmade_segmentation(ctx)

        ctx.set("retinal_artery_mask", artery_mask)
        ctx.set("retinal_vein_mask", vein_mask)
        ctx.output_manager.save_overlay(self.name, "av_overlay", ctx.require("M0_ff_image_cleaned"), artery_mask, vein_mask)

        artery_mask_clean = self.clean_mask(artery_mask, ctx)
        self.logger.info(f"    - Artery mask: {artery_mask.sum()} pixels, Cleaned artery mask: {artery_mask_clean.sum()} pixels")
        vein_mask_clean = self.clean_mask(vein_mask, ctx)
        self.logger.info(f"    - Vein mask: {vein_mask.sum()} pixels, Cleaned vein mask: {vein_mask_clean.sum()} pixels")

        ctx.set("retinal_artery_mask_clean", artery_mask_clean)
        ctx.set("retinal_vein_mask_clean", vein_mask_clean)
        ctx.output_manager.save_overlay(self.name, "av_overlay_cleaned", ctx.require("M0_ff_image_cleaned"), artery_mask_clean, vein_mask_clean)
        