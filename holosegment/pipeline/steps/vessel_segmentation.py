from holosegment.pipeline.step import BaseStep
from abc import abstractmethod
from holosegment.segmentation import process_masks
from holosegment.segmentation.process_masks import clean_vessel_mask
import numpy as np
from skimage.filters import frangi

class VesselSegmentationStep(BaseStep):
    @abstractmethod
    def get_vessel_mask(self, ctx):
        pass

    @abstractmethod
    def clean_vessel_mask(self, raw_mask, ctx):
        pass

class RetinalVesselSegmentationStep(VesselSegmentationStep):
    name = "retinal_vessel_segmentation"
    requires = {"M0_ff_image", "optic_disc_center"}
    produces = {"retinal_vessel_mask", "vessel_segmentation_logits"}

    def _relevant_config(self, ctx):
        params = ctx.eyeflow_config["Mask"]
        d = { "VesselSegmentationMethod": params["VesselSegmentationMethod"],
                 "DiaphragmRadius": params["DiaphragmRadius"],
                 "CropChoroidRadius": params["CropChoroidRadius"],
                 "retinal_vessel_segmentation_model": ctx.get_current_model_for_task(self.name)
        }
        return d

    def frangi_segmentation(self, ctx):
        # Placeholder for traditional Frangi filter-based segmentation

        image = ctx.require("M0_ff_image")
        vesselness = frangi(image)
        mask = ~vesselness.astype(bool)

        return mask

    def deep_segmentation(self, ctx):
        model = ctx.get_current_model_for_task(self.name)
        input = model.prepare_input(ctx)

        logits = np.squeeze(model.predict(input))
        mask = logits > 0.5

        ctx.cache["vessel_segmentation_logits"] = logits

        return mask
    
    def clean_vessel_mask(self, raw_mask, ctx):
        image = ctx.require("M0_ff_image")
        optic_disc_center = ctx.require("optic_disc_center")

        params = ctx.eyeflow_config["Mask"]

        clean_mask = clean_vessel_mask(
            raw_mask,
            image_shape=image.shape,
            optic_disc_center=optic_disc_center,
            diaphragm_radius=params["DiaphragmRadius"],
            crop_radius=params["CropChoroidRadius"],
        )

        return clean_mask

    def get_vessel_mask(self, ctx):
        method = ctx.eyeflow_config.get("Mask", "").get("VesselSegmentationMethod", "AI")

        if method == "AI":
            print("Using deep learning model for vessel segmentation.")
            return self.deep_segmentation(ctx)

        if method == "frangi":
            print("Using Frangi filter for vessel segmentation.")
            return self.frangi_segmentation(ctx)
        
    def run(self, ctx):
        # ---- Segmentation ----
        raw_mask = self.get_vessel_mask(ctx)

        # ---- Postprocessing ----
        clean_mask = self.clean_vessel_mask(raw_mask, ctx)

        ctx.set("retinal_vessel_mask", clean_mask)


class ChoroidalVesselSegmentationStep(VesselSegmentationStep):
    name = "choroidal_vessel_segmentation"
    requires = {"M0_ff_image", "retinal_vessel_mask", "optic_disc_center"}
    produces = {"choroidal_vessel_mask"}

    def _relevant_config(self, ctx):
        params = ctx.eyeflow_config["Mask"]
        d = {"DiaphragmRadius": params["DiaphragmRadius"]}
        return d

    def frangi_segmentation(self, ctx):
        # Placeholder for traditional Frangi filter-based segmentation

        image = ctx.require("M0_ff_image")
        vesselness = frangi(image)
        mask = ~vesselness.astype(bool)

        return mask
    
    def clean_vessel_mask(self, raw_mask, ctx):
        params = ctx.eyeflow_config["Mask"]
        center = ctx.require("optic_disc_center")

        diaphragm_radius = params["DiaphragmRadius"]
        h, w = raw_mask.shape
        mask_diaphragm = process_masks.disk_mask(h, w, diaphragm_radius, center=(center[0] / w, center[1] / h))

        clean_mask = raw_mask & mask_diaphragm

        return clean_mask

    def get_vessel_mask(self, ctx):
        vessel_mask = self.frangi_segmentation(ctx)
        retinal_vessel_mask = ctx.require("retinal_vessel_mask")

        choroidal_vessel_mask = vessel_mask & ~retinal_vessel_mask
        return choroidal_vessel_mask
    
    def run(self, ctx):
        # ---- Segmentation ----
        raw_mask = self.get_vessel_mask(ctx)

        # ---- Postprocessing ----
        clean_mask = self.clean_vessel_mask(raw_mask, ctx)

        ctx.set("choroidal_vessel_mask", clean_mask)