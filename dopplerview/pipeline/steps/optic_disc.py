from dopplerview.pipeline.step import BaseStep
from dopplerview.segmentation import process_masks

import numpy as np

class OpticDiscSegmentationStep(BaseStep):
    requires = {"M0_ff_image", "M1_ff_image"}
    produces = {"optic_disc_mask", "optic_disc_center", "optic_disc_width", "optic_disc_height"}
    name = "optic_disc_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
        return { 
            "OpticDiskDetectorMethod": params.get("OpticDiskDetectorMethod"),
            "optic_disc_segmentation_model": ctx.get_current_model_name_for_task(self.name)}
    
    def deep_detection(self, ctx):
        try:
            model = ctx.get_current_model_for_task("optic_disc_detection")
        except Exception as e:
            self.logger.error(f"    - Error retrieving optic disc detection model: {e}.")
            raise

        input = model.prepare_input(ctx)
        boxes = model.predict(input)

        target_h, target_w = ctx.get("M0_ff_image").shape
        h, w = model.spec.output_shape
        scale_x, scale_y = target_w / w, target_h / h

        idx = np.argmax(boxes[:, 4, :])  # Assuming the confidence score is in the 5th column
        bestbox = boxes[:, :, idx].flatten()

                # Keep detections above confidence threshold
        if bestbox[4] < 0.05:
            self.logger.warning(
                "Optic disc detection: no confident bounding box found."
            )
            return np.zeros((target_h, target_w), dtype=bool)

        x_center = bestbox[0] * scale_x
        y_center = bestbox[1] * scale_y
        diameter_x = bestbox[2] * scale_x
        diameter_y = bestbox[3] * scale_y

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
    
    def YOLO_segmentation(self, ctx):
        M0 = ctx.get("M0_ff_image")
        model = ctx.get_current_model_for_task(self.name)

        input_tensor = model.prepare_input(ctx)
        pred = model.predict(input_tensor)

        predictions = np.squeeze(pred[0]).T
        target_h, target_w = M0.shape
        h, w = model.spec.output_shape

        scores = predictions[:, 4]

        best_idx = np.argmax(scores)

        # Keep detections above confidence threshold
        if scores[best_idx] < 0.05:
            self.logger.warning(
                "Optic disc segmentation: no confident detections found."
            )
            return np.zeros((target_h, target_w), dtype=bool)

        best_box = predictions[best_idx, :4]

        scaleX = target_w / w
        scaleY = target_h / h

        cx = best_box[0] * scaleX
        cy = best_box[1] * scaleY
        width = best_box[2] * scaleX
        height = best_box[3] * scaleY

        rx = width / 2
        ry = height / 2

        if rx <= 0 or ry <= 0:
            self.logger.warning(
                "Optic disc segmentation: invalid ellipse dimensions."
            )
            return np.zeros((target_h, target_w), dtype=bool)

        Y, X = np.indices((target_h, target_w))

        optic_disk_mask = (
            ((X + 1 - cx) ** 2) / (rx ** 2)
            + ((Y + 1 - cy) ** 2) / (ry ** 2)
        ) <= 1

        return optic_disk_mask
    
    def deep_segmentation(self, ctx):
        model_name = ctx.get_current_model_name_for_task(self.name)
        if "yolo" in model_name.lower():
            self.logger.info("    - Using YOLO-based optic disc segmentation.")
            mask = self.YOLO_segmentation(ctx)
            return mask
        exception_message = f"Unsupported optic disc segmentation model: {model_name}"
        self.logger.error(exception_message)
        raise ValueError(exception_message)

    def run(self, ctx):
        optic_disc_detection_method = ctx.dopplerview_config.get("Mask", {}).get("OpticDiskDetectorMethod", "deep")

        M0_shape = ctx.require("M0_ff_image").shape

        if optic_disc_detection_method == "deep":
            try: 
                optic_disc_mask = self.deep_segmentation(ctx)
                x_min, y_min, x_max, y_max = process_masks.mask_to_bbox(optic_disc_mask)
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                width = x_max - x_min
                height = y_max - y_min
            except Exception as e:
                self.logger.error(f"    - Error occurred during deep optic disc segmentation: {e}. Falling back to mask generation from detected center and diameter.")
                try:
                    center, width, height = self.deep_detection(ctx)
                    optic_disc_mask = process_masks.bbox_to_mask(center, width, height, M0_shape)
                except Exception as e:
                    self.logger.error(f"    - Error occurred during deep optic disc detection: {e}. Falling back to non-deep approach.")
                    optic_disc_detection_method = "moment1"  # Fallback to moment1 detection if deep segmentation fails

        if optic_disc_detection_method == "moment1":
            center, width, height = self.moment1_detection(ctx)
            optic_disc_mask = process_masks.bbox_to_mask(center, width, height, M0_shape)
        
        if optic_disc_detection_method not in ["deep", "moment1"]:
            center, width, height = self.return_image_center(ctx)  # Fallback to image center if no model is used
            optic_disc_mask = process_masks.bbox_to_mask(center, width, height, M0_shape)

        ctx.set("optic_disc_mask", optic_disc_mask)
        ctx.set("optic_disc_center", center)
        ctx.set("optic_disc_width", width)
        ctx.set("optic_disc_height", height)

