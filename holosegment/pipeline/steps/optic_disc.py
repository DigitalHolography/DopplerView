from holosegment.pipeline.step import BaseStep
from holosegment.utils.image_utils import save_bounding_box

import numpy as np

class OpticDiscDetectionStep(BaseStep):
    requires = ["M0_ff_image", "M1_ff_image"]
    produces = ["optic_disc_center"]
    name = "optic_disc_detection"

    def _relevant_config(self, ctx):
        params = ctx.eyeflow_config["Mask"]
        return { 
            "OpticDiskDetectorNet": params.get("OpticDiskDetectorNet"),
            "optic_disc_model": ctx.get_current_model_for_task(self.name)}
    
    def deep_detection(self, ctx):
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)
        boxes = model.predict(input)

        idx = np.argmax(boxes[:, 4, :])  # Assuming the confidence score is in the 5th column
        bestbox = boxes[:, :, idx].flatten()
        x_center = bestbox[0]
        y_center = bestbox[1]
        diameter_x = bestbox[2]
        diameter_y = bestbox[3]

        center = (int(x_center), int(y_center))

        print(f"Optic disc center detected at: {center}")

        save_bounding_box(input, x_center, y_center, diameter_x, diameter_y, ctx.output_manager.output_dir / f"{self.name}")

        return (x_center, y_center), diameter_x, diameter_y
    
        
    def moment1_detection(self, ctx):
        image = ctx.require("M1_ff_image")
        moments = ctx.cache["moments"]
        M1 = moments.M1
        # Implement optic disc detection using M1 moments
        # For example, you could use the location of the maximum value in M1 as the optic disc center
        y_center, x_center = np.unravel_index(np.argmax(M1), M1.shape)
        diameter_x = diameter_y = 100  # Example diameter, adjust as needed

        return (x_center, y_center), diameter_x, diameter_y
        
    def return_image_center(self, ctx):
        image = ctx.require("M0_ff_image")
        height, width = image.shape
        x_center = width // 2
        y_center = height // 2
        diameter_x = diameter_y = 100  # Example diameter, adjust as needed

        return (x_center, y_center), diameter_x, diameter_y

    def run(self, ctx):
        use_optic_disc_detector = ctx.eyeflow_config.get("OpticDiskDetectorNet", True)
        M0 = ctx.cache["M0_ff_image"]

        if use_optic_disc_detector:
            center, diameter_x, diameter_y = self.deep_detection(ctx)
        else:
            center = (M0.shape[1] // 2, M0.shape[0] // 2)  # Fallback to image center if no model is used

        ctx.cache["optic_disc_center"] = center