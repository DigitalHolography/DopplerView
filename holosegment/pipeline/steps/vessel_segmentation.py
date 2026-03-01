from holosegment.pipeline.step import BaseStep
from holosegment.segmentation.process_masks import clean_vessel_mask
import numpy as np
from skimage.filters import frangi

class VesselSegmentation(BaseStep):
    name = "vessel_segmentation"
    requires = ["M0_ff_image", "optic_disc_center"]
    produces = ["vessel_mask"]

    def _relevant_config(self, ctx):
        params = ctx.eyeflow_config["Mask"]
        d = { "VesselSegmentationMethod": params.get("VesselSegmentationMethod", "AI"),
                 "DiaphragmRadius": params.get("DiaphragmRadius", 0.1),
                 "CropChoroidRadius": params.get("CropChoroidRadius", 0.45),
                 "vessel_segmentation_model": ctx.get_current_model_for_task(self.name)
        }
        return d

    def frangi_segmentation(self, ctx):
        # Placeholder for traditional Frangi filter-based segmentation

        image = ctx.require("M0_ff_image")
        vesselness = frangi(image)
        mask = vesselness > 0.5  # Example threshold, can be tuned

        ctx.output_manager.save("vessel_segmentation", "vessel_vesselness", vesselness, "png")
        ctx.output_manager.save("vessel_segmentation", "vessel_mask", mask, "png")

        return mask

    def deep_segmentation(self, ctx):
        model = ctx.get_current_model_for_task(self.name)
        input = model.prepare_input(ctx)

        logits = np.squeeze(model.predict(input))
        mask = logits > 0.5

        ctx.output_manager.save("vessel_segmentation", "vessel_logits", logits, "png")
        ctx.output_manager.save("vessel_segmentation", "vessel_mask", mask, "png")

        return mask

    def get_vessel_mask(self, ctx):
        method = ctx.eyeflow_config.get("Mask", "").get("VesselSegmentationMethod", "AI")

        if method == "AI":
            print("Using deep learning model for vessel segmentation.")
            return self.deep_segmentation(ctx)

        if method == "frangi":
            print("Using Frangi filter for vessel segmentation.")
            return self.frangi_segmentation(ctx)

    def run(self, ctx):
        image = ctx.require("M0_ff_image")
        optic_disc_center = ctx.require("optic_disc_center")

        # ---- Segmentation ----
        raw_mask = self.get_vessel_mask(ctx)

        # ---- Postprocessing ----
        params = ctx.eyeflow_config["Mask"]

        clean_mask = clean_vessel_mask(
            raw_mask,
            image_shape=image.shape,
            optic_disc_center=optic_disc_center,
            diaphragm_radius=params["DiaphragmRadius"],
            crop_radius=params["CropChoroidRadius"],
        )

        ctx.set("vessel_mask", clean_mask)

        ctx.output_manager.save("vessel_segmentation", "clean_mask", clean_mask, "png")