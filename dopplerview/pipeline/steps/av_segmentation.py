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
                    "DiaphragmRadius": params["DiaphragmRadius"],
                    "CropChoroidRadius": params["CropChoroidRadius"]
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

        params = ctx.dopplerview_config["Mask"]

        clean_mask = clean_vessel_mask(
            raw_mask,
            image_shape=raw_mask.shape,
            optic_disc_center=optic_disc_center,
            diaphragm_radius=params["DiaphragmRadius"],
            crop_radius=params["CropChoroidRadius"],
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

        artery_mask_clean = self.clean_mask(artery_mask, ctx)
        vein_mask_clean = self.clean_mask(vein_mask, ctx)

        ctx.set("retinal_artery_mask_clean", artery_mask_clean)
        ctx.set("retinal_vein_mask_clean", vein_mask_clean)
        