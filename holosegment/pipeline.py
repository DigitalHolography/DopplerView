from holosegment.models.manager import ModelManager
from holosegment.models.builder import build_model_wrapper
from holosegment.input_output.read_moments import Moments
from holosegment.preprocessing.preprocessing import Preprocessor
from holosegment.segmentation import artery_vein_segmentation
from holosegment.segmentation import binary_segmentation
from holosegment.segmentation.pulse_analysis import compute_correlation, compute_diasys_image
import numpy as np
import torch

class Pipeline:
    def __init__(self, config, model_registry):
        self.config = config
        self.cache = {}
        self.model_registry = model_registry
        self.model_manager = ModelManager(model_registry)
        self.model_instances = {}

        # Register steps
        self.steps = {
            "load_moments": LoadMomentsStep(self),
            "preprocess": PreprocessStep(self),
            "binary_segmentation": BinarySegmentationStep(self),
            "pulse_analysis": PulseAnalysisStep(self),
            "av_segmentation": AVSegmentationStep(self),
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

    def __init__(self, context):
        self.ctx = context

    def run(self):
        raise NotImplementedError

class LoadMomentsStep:
    name = "load_moments"
    produces = ["moments"]

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        input_path = self.pipeline.cache["input_path"]
        reader = Moments(input_path)
        reader.read_moments()
        self.pipeline.cache["moments"] = reader

class PreprocessStep:
    requires = ["moments"]
    produces = ["M0_ff_video", "M0_ff_image"]

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        moments = self.pipeline.cache["moments"]
        pre = Preprocessor(self.pipeline.config, moments)
        pre.preprocess()

        self.pipeline.cache["M0_ff_video"] = pre.M0_ff_video
        self.pipeline.cache["M0_ff_image"] = pre.M0_ff_image
        print(self.pipeline.cache["M0_ff_image"] is None)

class BinarySegmentationStep:
    requires = ["M0_ff_image"]
    produces = ["vessel_mask"]

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        method = self.pipeline.config.get("BinarySegmentationMethod", "AI")
        image = self.pipeline.cache["M0_ff_image"]

        if method == "AI":
            # model_name = self.pipeline.config["Mask"]["VesselSegmentationMethod"]
            model_name = "iternet5_vesselness"
            model = self.pipeline.get_model(model_name)
            mask = model.predict(image)
            mask = np.squeeze(mask)  # Remove channel dimension if present

        else:
            raise NotImplementedError

        self.pipeline.cache["vessel_mask"] = mask

class PulseAnalysisStep:
    requires = ["M0_ff_video", "vessel_mask"]
    produces = ["temporal_cues"]

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        vessel_mask = self.pipeline.cache["vessel_mask"]

        cues_requested = self.pipeline.config.get("TemporalCues", ["correlation", "diasys"])

        temporal_cues = {}

        if "correlation" in cues_requested:
            temporal_cues["correlation"] = compute_correlation(video, vessel_mask)

        if "diasys" in cues_requested:
            temporal_cues["diasys"] = compute_diasys_image(video, vessel_mask)

        self.pipeline.cache["temporal_cues"] = temporal_cues

class AVSegmentationStep:
    requires = ["M0_ff_video", "M0_ff_image", "temporal_cues"]
    produces = ["artery_mask", "vein_mask"]

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        video = self.pipeline.cache["M0_ff_video"]
        image = self.pipeline.cache["M0_ff_image"]
        cues = self.pipeline.cache["temporal_cues"]

        if self.pipeline.config.get("AVSegmentationMethod", "AI") == "AI":

            # model_name = self.pipeline.config["models"]["av"]
            model_name = "nnwnet_av_corr_diasys"
            model = self.pipeline.get_model(model_name)

            print(image.shape, cues["correlation"].shape, cues["diasys"].shape)

            input = np.stack([image, cues["correlation"], cues["diasys"]], axis=0)  # shape (3, H, W)

            print(input.shape)

            mask = model.predict(input)
            mask = np.squeeze(mask)  # Remove channel dimension if present

            if model.spec.output_activation == "argmax":
                self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = np.stack([np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)], axis=0)
            else:
                self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = mask[0], mask[1]

        else:
            self.pipeline.cache["artery_mask"], self.pipeline.cache["vein_mask"] = artery_vein_segmentation.handmade_segmentation(
                video, image, cues
            )