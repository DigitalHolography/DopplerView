from dopplerview.pipeline.step import BaseStep
from dopplerview.utils.image_utils import save_bounding_box

import numpy as np
import cv2

class OpticDiscDetectionStep(BaseStep):
    requires = {"M0_ff_image", "M1_ff_image"}
    produces = {"optic_disc_center", "optic_disc_axes"}
    name = "optic_disc_detection"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
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

        self.logger.info(f"    - Optic disc center detected at: {center}")

        return (x_center, y_center), diameter_x, diameter_y
    
        
    def moment1_detection(self, ctx):
        M1 = ctx.require("M1_ff_image")
        # Implement optic disc detection using M1 moments
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
        optic_disc_detection_method = ctx.dopplerview_config.get("OpticDiskDetectorMethod", True)

        if optic_disc_detection_method == "deep":
            center, diameter_x, diameter_y = self.deep_detection(ctx)
        elif optic_disc_detection_method == "moment1":
            center, diameter_x, diameter_y = self.moment1_detection(ctx)
        else:
            center, diameter_x, diameter_y = self.return_image_center(ctx)  # Fallback to image center if no model is used

        ctx.set("optic_disc_center", center)
        ctx.set("optic_disc_axes", (diameter_x, diameter_y))


class OpticDiscSegmentationStep(BaseStep):
    requires = {"M0_ff_image_cleaned"}
    produces = {"optic_disc_mask"}
    name = "optic_disc_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
        return { 
            "OpticDiskDetectorNet": params.get("OpticDiskDetectorNet"),
            "optic_disc_segmentation_model": ctx.get_current_model_for_task(self.name)}
    
    def deep_segmentation(self, ctx):
        M0 = ctx.get("M0_ff_image_cleaned")
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)
        pred = model.predict(input)

        predictions = np.squeeze(pred[0]).T
        w, h = M0.shape[1], M0.shape[0]

        # --- 4. Filter and NMS ---

        conf_threshold = 0.25

        scores = predictions[:, 4]  # MATLAB col 5 -> Python index 4

        valid_idx = scores > conf_threshold
        valid_preds = predictions[valid_idx]


        boxes_cxcywh = valid_preds[:, 0:4]

        # Convert to top-left width-height for NMS
        boxes_tlwh = boxes_cxcywh.copy()

        boxes_tlwh[:, 0] = (
            boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        )

        boxes_tlwh[:, 1] = (
            boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        )

        # ---- NMS ----
        # OpenCV NMS expects [x, y, w, h]
        boxes_list = boxes_tlwh.tolist()
        scores_list = valid_preds[:, 4].tolist()

        idx = cv2.dnn.NMSBoxes(
            boxes_list,
            scores_list,
            score_threshold=conf_threshold,
            nms_threshold=0.6
        )

        # OpenCV may return nested arrays
        best_idx = int(np.array(idx).flatten()[0])

        best_box = boxes_cxcywh[best_idx]  # [cx, cy, w, h]

        # --- 5. Scale Box Coordinates ---

        scaleX = w / 1024
        scaleY = h / 1024

        cx = best_box[0] * scaleX
        cy = best_box[1] * scaleY
        width = best_box[2] * scaleX
        height = best_box[3] * scaleY

        # --- 6. Generate Output Mask (Geometric Ellipse) ---

        # Coordinate grid
        X, Y = np.meshgrid(
            np.arange(1, w + 1),
            np.arange(1, h + 1)
        )

        # Ellipse radii
        rx = width / 2
        ry = height / 2

        if rx > 0 and ry > 0:

            norm_dist = (
                ((X - cx) ** 2) / (rx ** 2)
                + ((Y - cy) ** 2) / (ry ** 2)
            )

            optic_disk_mask = norm_dist <= 1

        else:
            optic_disk_mask = np.zeros((h, w), dtype=bool)
    
        return optic_disk_mask
    

    def run(self, ctx):
        optic_disc_mask = self.deep_segmentation(ctx)
        ctx.set("optic_disc_mask", optic_disc_mask)