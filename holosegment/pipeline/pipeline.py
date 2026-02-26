from holosegment.models.manager import ModelManager
from holosegment.models.builder import build_model_wrapper
from holosegment.input_output.read_moments import Moments
from holosegment.preprocessing.preprocessing import Preprocessor
from holosegment.segmentation import artery_vein_segmentation
from holosegment.segmentation import binary_segmentation
import holosegment.segmentation.pulse_analysis as pulse_analysis
from holosegment.pipeline.output_manager import OutputManager, save_bounding_box
import holosegment.segmentation.process_masks as process_masks
import numpy as np
import torch

class Pipeline:
    def __init__(self, config, model_registry, output_dir=None, debug=False):
        self.config = config
        self.cache = {}
        self.model_registry = model_registry
        self.model_manager = ModelManager(model_registry)
        self.model_instances = {}
        self.output_manager = OutputManager(
            output_dir=output_dir,
            enabled=debug
        )

        # Register steps
        self.steps = {
            "load_moments": LoadMomentsStep(self, "load_moments"),
            "preprocess": PreprocessStep(self, "preprocess"),
            "optic_disc_detection": OpticDiscDetectionStep(self, "optic_disc_detection"),
            "binary_segmentation": BinarySegmentationStep(self, "binary_segmentation"),
            "pulse_analysis": PulseAnalysisStep(self, "pulse_analysis"),
            "av_segmentation": AVSegmentationStep(self, "av_segmentation"),
        }

    # ------------------------------
    # MODEL HANDLING
    # ------------------------------

    def get_model(self, model_name):
        if model_name not in self.model_instances:
            spec, path = self.model_manager.resolve(model_name)
            model = build_model_wrapper(spec, path)
            self.model_instances[model_name] = model

        return self.model_instances[model_name]

    # ------------------------------
    # EXECUTION CONTROL
    # ------------------------------

    def run_all(self, input_path):
        self.cache["input_path"] = input_path

        for name in self.steps:
            self.run_step(name)

        return (
            self.cache.get("artery_mask"),
            self.cache.get("vein_mask"),
        )

    def run_step(self, step_name):
        step = self.steps[step_name]

        # Check dependencies
        for dep in getattr(step, "requires", []):
            if dep not in self.cache:
                raise RuntimeError(
                    f"Step '{step_name}' requires '{dep}' but it is missing."
                )

        print(f"Running step: {step_name}")
        step.run()

    def run_from(self, step_name):
        run = False
        for name in self.steps:
            if name == step_name:
                run = True
            if run:
                self.run_step(name)

class BaseStep:
    name = None
    requires = []     # list of cache keys required
    produces = []     # list of cache keys produced

    def __init__(self, pipeline, name):
        self.pipeline = pipeline
        self.name = name

    # def check_requirements(self):
    #     for key in self.requires:
    #         if key not in self.pipeline.cache:
    #             raise RuntimeError(
    #                 f"Missing dependency '{key}' for step '{self.name}'"
    #             )

    def run(self):
        raise NotImplementedError

class NestedStep(BaseStep):
    def __init__(self, pipeline, name, substeps):
        super().__init__(pipeline, name)
        self.substeps = substeps

    def run(self):
        for step in self.substeps:
            step.run()

class LoadMomentsStep(BaseStep):
    name = "load_moments"
    produces = ["moments"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        input_path = self.pipeline.cache["input_path"]
        reader = Moments(input_path)
        reader.read_moments()
        self.pipeline.cache["moments"] = reader

class PreprocessStep(BaseStep):
    requires = ["moments"]
    produces = ["M0_ff_video", "M0_ff_image"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        moments = self.pipeline.cache["moments"]
        self.pipeline.output_manager.save(self.name, "M0_video", moments.M0, "avi")

        pre = Preprocessor(self.pipeline.config, moments)
        pre.preprocess()

        self.pipeline.cache["M0_ff_video"] = pre.M0_ff_video
        self.pipeline.output_manager.save(self.name, "M0_ff_video", pre.M0_ff_video, "avi")
        
        self.pipeline.cache["M0_ff_image"] = pre.M0_ff_image
        self.pipeline.output_manager.save(self.name, "M0_ff", pre.M0_ff_image, "png")
        print(self.pipeline.cache["M0_ff_image"] is None)


class BinarySegmentationStep(BaseStep):
    requires = ["M0_ff_image", "optic_disc_center"]
    produces = ["vessel_mask"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        method = self.pipeline.config.get("BinarySegmentationMethod", "AI")
        image = self.pipeline.cache["M0_ff_image"]

        if method == "AI":
            # model_name = self.pipeline.config["Mask"]["VesselSegmentationMethod"]
            model_name = "iternet5_vesselness"
            model = self.pipeline.get_model(model_name)
            logits = np.squeeze(model.predict(image))
            mask = logits > 0.5  # Remove channel dimension if present

        else:
            raise NotImplementedError

        self.pipeline.output_manager.save(self.name, "vessel_logits", logits, "png")
        self.pipeline.output_manager.save(self.name, "vessel_mask", mask, "png")

        diaphragm_radius = self.pipeline.config["Mask"]["DiaphragmRadius"]
        crop_chororoid_radius = self.pipeline.config["Mask"]["CropChoroidRadius"]

        width, height = image.shape[0], image.shape[1]
        mask_diaphragm = process_masks.disk_mask(height, width, R1 = diaphragm_radius)
        mask_center = process_masks.disk_mask(height, width, R1 = crop_chororoid_radius, center=self.pipeline.cache["optic_disc_center"])

        largest_connected_components = process_masks.bwareafilt_largest(
            mask & ~mask_center,
            connectivity=2  # 8-connectivity
        )

        self.pipeline.output_manager.save(self.name, "largest_connected_components", largest_connected_components, "png")

        mask_vessel_clean = mask & largest_connected_components & mask_diaphragm

        self.pipeline.cache["vessel_mask"] = mask_vessel_clean
        self.pipeline.output_manager.save(self.name, "vessel_mask_clean", mask_vessel_clean, "png")
        self.pipeline.output_manager.save(self.name, "vessel_mask_diaphragm", mask & mask_diaphragm, "png")
        self.pipeline.output_manager.save(self.name, "vessel_mask_components", mask & largest_connected_components, "png")
        self.pipeline.output_manager.save(self.name, "mask_diaphragm", mask_diaphragm, "png")

class OpticDiscDetectionStep(BaseStep):
    requires = ["M0_ff_image"]
    produces = ["optic_disc_center"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        use_optic_disc_detector = self.pipeline.config.get("OpticDiskDetectorNet", True)
        M0 = self.pipeline.cache["M0_ff_image"]

        if use_optic_disc_detector:
            # model_name = self.pipeline.config["Mask"]["OpticDiscDetectionMethod"]
            model_name = "optic_disc_detector"
            model = self.pipeline.get_model(model_name)
            print(np.min(M0), np.max(M0))
            boxes = model.predict(M0)

            idx = np.argmax(boxes[:, 4, :])  # Assuming the confidence score is in the 5th column
            bestbox = boxes[:, :, idx].flatten()
            # x_center, y_center, diameter_x, diameter_y = bestbox[:4]  # Get center coordinates (x, y) and diameters of the detected optic disc
            
            x_center = bestbox[0]
            y_center = bestbox[1]
            diameter_x = bestbox[2]
            diameter_y = bestbox[3]
            
            center = (int(x_center), int(y_center))

            print(f"Optic disc center detected at: {center}")

            save_bounding_box(M0, x_center, y_center, diameter_x, diameter_y, self.pipeline.output_manager.output_dir / f"{self.name}_optic_disc_detection.png")
        else:
            raise NotImplementedError

        # self.pipeline.output_manager.save(self.name, "optic_disc_logits", logits, "png")
        # self.pipeline.output_manager.save(self.name, "optic_disc_center", center, title="Optic Disc Center")
        self.pipeline.cache["optic_disc_center"] = center

class PreArteryMaskStep(BaseStep):
    requires = ["M0_ff_video", "vessel_mask", "optic_disc_center"]
    produces = ["pre_artery_mask"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        vessel_mask = self.pipeline.cache["vessel_mask"]

        sampling_frequency = 37.037e3

        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask, self.pipeline.cache["optic_disc_center"], sampling_frequency, self.pipeline.output_manager)
        self.pipeline.output_manager.save(self.name, "pre_artery_mask", pre_artery_mask, "png")
        self.pipeline.cache["pre_artery_mask"] = pre_artery_mask
        self.pipeline.output_manager.save(self.name, "pre_vein_mask", pre_vein_mask, "png")
        self.pipeline.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = ["M0_ff_video", "pre_artery_mask"]
    produces = ["temporal_cues"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        pre_artery_mask = self.pipeline.cache["pre_artery_mask"]

        cues_requested = self.pipeline.config.get("TemporalCues", ["correlation", "diasys"])

        temporal_cues = {}

        if "correlation" in cues_requested:
            temporal_cues["correlation"] = pulse_analysis.compute_correlation(video, pre_artery_mask)
            self.pipeline.output_manager.save(self.name, "correlation_map", temporal_cues["correlation"], "png")

        if "diasys" in cues_requested:
            diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, pre_artery_mask)
            self.pipeline.output_manager.save(self.name, "diasys_image", diasys, "png")
            self.pipeline.output_manager.save(self.name, "M0_Systole_img", M0_Systole_img, "png")
            self.pipeline.output_manager.save(self.name, "M0_Diastole_img", M0_Diastole_img, "png")
            self.pipeline.output_manager.save_plot(self.name, "fullPulse", fullPulse, title = "Full Pulse Analysis (mean intensity over time)")

            temporal_cues["diasys"] = diasys

        self.pipeline.cache["temporal_cues"] = temporal_cues

class PulseAnalysisStep(NestedStep):
    requires = ["M0_ff_video", "vessel_mask"]
    produces = ["temporal_cues"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name, [
            PreArteryMaskStep(pipeline, f"{name}_pre"),
            ComputeTemporalCuesStep(pipeline, f"{name}_compute")
        ])

    def run(self):
        for step in self.substeps:
            step.run()
        # video = self.pipeline.cache["M0_ff_video"]
        # vessel_mask = self.pipeline.cache["vessel_mask"]

        # pre_artery_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask)
        # self.pipeline.output_manager.save(self.name, "pre_artery_mask", pre_artery_mask, "png")
        # self.pipeline.cache["pre_artery_mask"] = pre_artery_mask

        # cues_requested = self.pipeline.config.get("TemporalCues", ["correlation", "diasys"])

        # temporal_cues = {}

        # if "correlation" in cues_requested:
        #     temporal_cues["correlation"] = pulse_analysis.compute_correlation(video, pre_artery_mask)
        #     self.pipeline.output_manager.save(self.name, "correlation_map", temporal_cues["correlation"], "png")

        # if "diasys" in cues_requested:
        #     diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, pre_artery_mask)
        #     self.pipeline.output_manager.save(self.name, "diasys_image", diasys, "png")
        #     self.pipeline.output_manager.save(self.name, "M0_Systole_img", M0_Systole_img, "png")
        #     self.pipeline.output_manager.save(self.name, "M0_Diastole_img", M0_Diastole_img, "png")
        #     self.pipeline.output_manager.save_plot(self.name, "fullPulse", fullPulse, title = "Full Pulse Analysis (mean intensity over time)")

        #     temporal_cues["diasys"] = diasys

        # self.pipeline.cache["temporal_cues"] = temporal_cues

class AVSegmentationStep(BaseStep):
    requires = ["M0_ff_video", "M0_ff_image", "temporal_cues"]
    produces = ["artery_mask", "vein_mask"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        M0 = self.pipeline.cache["M0_ff_image"]
        cues = self.pipeline.cache["temporal_cues"]

        if self.pipeline.config.get("AVSegmentationMethod", "AI") == "AI":

            # model_name = self.pipeline.config["models"]["av"]
            model_name = "nnwnet_av_corr_diasys"
            model = self.pipeline.get_model(model_name)

            print(M0.shape, cues["correlation"].shape, cues["diasys"].shape)

            
            input = np.stack([M0, cues["correlation"], cues["diasys"]], axis=0)  # shape (3, H, W)

            print(input.shape)

            mask = model.predict(input)
            mask = np.squeeze(mask)  # Remove channel dimension if present

            if model.spec.output_activation == "argmax":
                self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = np.stack([np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)], axis=0)
            else:
                self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = mask[0], mask[1]

        else:
            self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = artery_vein_segmentation.handmade_segmentation(
                video, M0, cues
            )
        
        self.pipeline.output_manager.save(self.name, "artery_mask", self.pipeline.cache["artery_mask"], "png")
        self.pipeline.output_manager.save(self.name, "vein_mask", self.pipeline.cache["vein_mask"], "png")