from holosegment.models.manager import ModelManager
from holosegment.models.builder import build_model_wrapper
from holosegment.input_output.read_moments import Moments
from holosegment.preprocessing.preprocessing import Preprocessor
from holosegment.segmentation import artery_vein_segmentation
from holosegment.segmentation import binary_segmentation
import holosegment.segmentation.pulse_analysis as pulse_analysis
from holosegment.pipeline.output_manager import OutputManager
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
    requires = ["M0_ff_image"]
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
        self.pipeline.cache["vessel_mask"] = mask
    

class PulseAnalysisStep(BaseStep):
    requires = ["M0_ff_video", "vessel_mask"]
    produces = ["temporal_cues"]

    def __init__(self, pipeline, name):
        super().__init__(pipeline, name)

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        vessel_mask = self.pipeline.cache["vessel_mask"]

        # pre_artery_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask)
        # self.pipeline.output_manager.save(self.name, "pre_artery_mask", pre_artery_mask, "png")

        cues_requested = self.pipeline.config.get("TemporalCues", ["correlation", "diasys"])

        temporal_cues = {}

        if "correlation" in cues_requested:
            temporal_cues["correlation"] = pulse_analysis.compute_correlation(video, vessel_mask)
            self.pipeline.output_manager.save(self.name, "correlation_map", temporal_cues["correlation"], "png")

        if "diasys" in cues_requested:
            diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, vessel_mask)
            self.pipeline.output_manager.save(self.name, "diasys_image", diasys, "png")
            self.pipeline.output_manager.save(self.name, "M0_Systole_img", M0_Systole_img, "png")
            self.pipeline.output_manager.save(self.name, "M0_Diastole_img", M0_Diastole_img, "png")
            self.pipeline.output_manager.save_plot(self.name, "fullPulse", fullPulse, title = "Full Pulse Analysis (mean intensity over time)")

            temporal_cues["diasys"] = diasys

        self.pipeline.cache["temporal_cues"] = temporal_cues

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