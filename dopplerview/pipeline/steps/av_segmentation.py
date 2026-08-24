from dopplerview.pipeline.step import BaseStep
import dopplerview.segmentation.process_masks as process_masks
from skimage.morphology import dilation, opening, disk
from skimage.measure import label
import numpy as np

class RetinalAVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image_cleaned", "correlation_M0", "diasys_image", "optic_disc_center"}
    produces = {"retinal_artery_mask", "retinal_vein_mask", "retinal_artery_mask_clean", "retinal_vein_mask_clean"}
    name = "retinal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
        return { "AVSegmentationMethod": params.get("AVSegmentationMethod", "AI"),
                    "av_segmentation_model_name": ctx.get_current_model_name_for_task(self.name),
                    "DiaphragmRadius": params.get("DiaphragmRadius", 0.45),
                    "CenterRadius": params.get("CenterRadius", 0.1)
        }

    def deep_segmentation(self, ctx):
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
        if optic_disc_center is None:
            self.logger.warning("    - Optic disc center not available. Skipping mask cleaning.")
            return raw_mask
        
        width, height = raw_mask.shape
        optic_disc_center = (optic_disc_center[0] / width, optic_disc_center[1] / height)

        params = ctx.dopplerview_config["Mask"]

        clean_mask = process_masks.clean_vessel_mask(
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
        ctx.output_manager.save_overlay(self.name, "av_overlay", ctx.require("M0_ff_image_cleaned"), [artery_mask, vein_mask])

        artery_mask_clean = self.clean_mask(artery_mask, ctx)
        self.logger.info(f"    - Artery mask: {artery_mask.sum()} pixels, Cleaned artery mask: {artery_mask_clean.sum()} pixels")
        vein_mask_clean = self.clean_mask(vein_mask, ctx)
        self.logger.info(f"    - Vein mask: {vein_mask.sum()} pixels, Cleaned vein mask: {vein_mask_clean.sum()} pixels")

        ctx.set("retinal_artery_mask_clean", artery_mask_clean)
        ctx.set("retinal_vein_mask_clean", vein_mask_clean)
        ctx.output_manager.save_overlay(self.name, "av_overlay_cleaned", ctx.require("M0_ff_image_cleaned"), [artery_mask_clean, vein_mask_clean])

class ChoroidalAVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image_cleaned", "HF_M0_FF", "retinal_artery_mask", "retinal_vein_mask", "correlation_M0", "correlation_HF_M0_ff", "optic_disc_center"}
    produces = {"choroidal_artery_mask", "choroidal_vein_mask", "choroidal_aliased_artery_mask", "choroidal_artery_mask_clean", "choroidal_vein_mask_clean", "choroidal_aliased_artery_mask_clean"}
    name = "choroidal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        return {}
    
    def clean_vessel_mask(self, raw_mask, ctx):
        params = ctx.dopplerview_config["Mask"]

        diaphragm_radius = params.get("DiaphragmRadius", 0.45)
        h, w = raw_mask.shape
        mask_diaphragm = process_masks.disk_mask(h, w, diaphragm_radius)

        clean_mask = raw_mask & mask_diaphragm

        # If optic disc center is available, remove vessels in the center region
        optic_disc_center = ctx.require("optic_disc_center")
        if optic_disc_center is not None:
            mask_center = ctx.require("optic_disc_mask")
            clean_mask = clean_mask & ~mask_center

            # If optic disc center is available, retinal masks should be valid, and can be removed
            retinal_artery_mask = ctx.require("retinal_artery_mask")
            retinal_vein_mask = ctx.require("retinal_vein_mask")
            retinal_vessel_mask = retinal_artery_mask | retinal_vein_mask
            dilated_retinal_vessel_mask = dilation(retinal_vessel_mask)
            clean_mask = clean_mask & ~dilated_retinal_vessel_mask

        return clean_mask
    
    def get_retinal_artery_mask(self, ctx):
        # If the retinal artery mask is valid, return it
        if ctx.require("retinal_artery_mask") is not None:
            if ctx.require("retinal_artery_mask").sum() > 0 and ctx.require("optic_disc_center") is not None:
                return ctx.require("retinal_artery_mask")

        corr_HF = ctx.require("correlation_HF_M0_ff")
        corr_M0 = ctx.require("correlation_M0")

        retinal_artery_mask = (corr_HF > 0.3) | (corr_M0 > 0.35)

        return retinal_artery_mask
    
    def get_retinal_vein_mask(self, ctx):
        # If the retinal vein mask is valid, return it
        if ctx.require("retinal_vein_mask") is not None:
            if ctx.require("retinal_vein_mask").sum() > 0 and ctx.require("optic_disc_center") is not None:
                return ctx.require("retinal_vein_mask")

        return np.zeros_like(ctx.require("M0_ff_image_cleaned"), dtype=np.uint8)

    def get_masks(self, ctx):
        # Get artery and vein masks based on the correlation of the M0 and its high frequencies with the retinal aterial signal.
        # The choroidal vessels can be identified as three types:
        # - arteries with positive correlation with the retinal arteries
        # - veins with negative correlation with the retinal arteries, less present in the high frequency signal
        # - aliased arteries with almost perfect negative correlation with the retinal arteries, dominant in the high frequency signal

        corr_M0 = ctx.require("correlation_M0")
        corr_HF = ctx.require("correlation_HF_M0_ff")

        retinal_artery_mask = self.get_retinal_artery_mask(ctx)
        retinal_vein_mask = self.get_retinal_vein_mask(ctx)
        retinal_vessel_mask = dilation(retinal_artery_mask | retinal_vein_mask)

        self.logger.info("    - Identifying choroidal aliased arteries with -0.2 correlation threshold in high frequency signal and -0.25 for M0 correlation.")
        pre_aliased_arteries = corr_HF < -0.2   # Identify aliased arteries based on high frequency correlation
        connected_arteries = process_masks.connect_components(
            pre_aliased_arteries,
            max_distance=5,
            executor=ctx.parallel,
        )  # Connect disconnected aliased arteries that are close to each other
        large_arteries = opening(connected_arteries, disk(2))
        anti_correlated_vessels = corr_M0 < -0.25
        aliased_arteries = process_masks.keep_connected_components(anti_correlated_vessels, large_arteries) # Keep only the vessels that are connected to the aliased arteries
        choroidal_aliased_artery_mask = aliased_arteries & ~retinal_vessel_mask  # Remove retinal vessels

        self.logger.info("    - Identifying choroidal veins with -0.15 correlation threshold in M0 signal.")
        pre_veins = corr_M0 < -0.15
        veins = process_masks.keep_connected_components(pre_veins, aliased_arteries, negative=True)   # Remove aliased arteries from the vein mask
        veins = process_masks.remove_small_vessels(label(veins), min_size=10) > 0
        choroidal_vein_mask = veins & ~retinal_vessel_mask  # Remove retinal vessels

        self.logger.info("    - Identifying choroidal arteries with 0.12 correlation threshold in M0 signal.")
        arteries = corr_M0 > 0.12
        arteries = process_masks.remove_small_vessels(label(arteries), min_size=10) > 0
        choroidal_artery_mask = arteries & ~retinal_vessel_mask  # Remove retinal vessels

        ctx.output_manager.save_overlay(self.name, "overlay", ctx.require("M0_ff_image_cleaned"), [choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask], colors=[(0, 0, 255), (255, 0, 0), (0, 255, 0)])

        return choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask
    
    def run(self, ctx):
        # Implement choroidal artery vein segmentation logic based on ctx and self.dopplerview_config
        # For now, we will just set empty masks as placeholders
        choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask = self.get_masks(ctx)

        ctx.set("choroidal_artery_mask", choroidal_artery_mask)
        ctx.set("choroidal_vein_mask", choroidal_vein_mask)
        ctx.set("choroidal_aliased_artery_mask", choroidal_aliased_artery_mask)

        choroidal_artery_mask_clean = self.clean_vessel_mask(choroidal_artery_mask, ctx)
        choroidal_vein_mask_clean = self.clean_vessel_mask(choroidal_vein_mask, ctx)
        choroidal_aliased_artery_mask_clean = self.clean_vessel_mask(choroidal_aliased_artery_mask, ctx)

        ctx.set("choroidal_artery_mask_clean", choroidal_artery_mask_clean)
        ctx.set("choroidal_vein_mask_clean", choroidal_vein_mask_clean)
        ctx.set("choroidal_aliased_artery_mask_clean", choroidal_aliased_artery_mask_clean)
        
        ctx.output_manager.save_overlay(self.name, "overlay_clean", ctx.require("M0_ff_image_cleaned"), [choroidal_artery_mask_clean, choroidal_vein_mask_clean, choroidal_aliased_artery_mask_clean], colors=[(0, 0, 255), (255, 0, 0), (0, 255, 0)])
