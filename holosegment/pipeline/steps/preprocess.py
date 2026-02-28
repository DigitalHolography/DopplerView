from holosegment.pipeline.step import BaseStep

from holosegment.preprocessing.registration import register_video
from holosegment.preprocessing import normalization, resize
from utils import image_utils
import numpy as np

class Preprocessor:
    def __init__(self, config, moments):
        self.config = config
        self.M0 = moments.M0
        self.M1 = moments.M1
        self.M2 = None
        self.SH = moments.SH

        self.M0_ff_video = None  # Cache for flatfield-corrected M0 video
        self.M0_ff_image = None  # Cache for flatfield-corrected M0

        self.M1_ff_video = None  # Cache for flatfield-corrected M1 video
        self.M1_ff_image = None  # Cache for flatfield-corrected M1

        self.M2_ff_video = None  # Cache for flatfield-corrected M2 video
        self.M2_ff_image = None  # Cache for flatfield-corrected M2

    def register(self, reference_idx=0):
        firstFrame = self.config['Preprocess']['Register']['StartFrame']
        endFrame = self.config['Preprocess']['Register']['EndFrame']
        enable = self.config['Preprocess']['Register']['Enable']

        if not enable:
            return
        if self.M0 is not None:
            self.M0 = register_video(self.M0, firstFrame, endFrame, reference_idx)
        if self.M1 is not None:
            self.M1 = register_video(self.M1, firstFrame, endFrame, reference_idx)
        if self.M2 is not None:
            self.M2 = register_video(self.M2, firstFrame, endFrame, reference_idx)
        if self.SH is not None:
            self.SH = register_video(self.SH, firstFrame, endFrame, reference_idx)

    def nonrigid_register(self):
        # Implement non-rigid registration logic based on self.config
        return
    
    def crop(self):
        firstFrame = self.config['Preprocess']['Crop']['StartFrame']
        endFrame = self.config['Preprocess']['Crop']['EndFrame']
        if firstFrame == 1 and endFrame == -1:
            return
        print(f"Cropping frames from {firstFrame} to {endFrame}")
        # Implement cropping logic based on self.config
        if self.M0 is not None:
            self.M0 = resize.crop_video(self.M0, first_frame=firstFrame, end_frame=endFrame)
        if self.M1 is not None:
            self.M1 = resize.crop_video(self.M1, first_frame=firstFrame, end_frame=endFrame)
        if self.M2 is not None:
            self.M2 = resize.crop_video(self.M2, first_frame=firstFrame, end_frame=endFrame)
        if self.SH is not None:
            self.SH = resize.crop_video(self.SH, first_frame=firstFrame, end_frame=endFrame)

    def normalize(self):
        # Implement normalization logic based on self.config
        gaussian_blur_ratio = self.config['FlatFieldCorrection']['GWRatio']

        if self.M0 is not None:
            self.M0_ff_video = normalization.compute_moment_ff(self.M0, gaussian_blur_ratio)

        if self.M1 is not None:
            self.M1_ff_video = normalization.compute_moment_ff(self.M1, gaussian_blur_ratio)

        if self.M2 is not None:
            self.M2_ff_video = normalization.compute_moment_ff(self.M2, gaussian_blur_ratio)

        return
    
    def resize(self):
        # Implement resizing logic based on self.config
        return
    
    def remove_outliers(self):
        # Implement outlier removal logic based on self.config
        return
    
    def interpolate(self):
        # Implement interpolation logic based on self.config
        return

    def preprocess(self):
        # # Step 1: Register
        # self.register()

        # # Step 2: Crop frames
        # self.crop()

        # Step 3: Normalize 
        self.normalize()

        # # Step 4: Resize
        # self.resize()

        # # Step 5: Non-rigid registration
        # self.nonrigid_register()

        # # Step 6: Interpolate
        # self.interpolate()

        # # Step 7: Remove outliers 
        # self.remove_outliers()

        self.M0_ff_image = image_utils.normalize_to_uint8(np.mean(self.M0_ff_video, axis=0)) if self.M0_ff_video is not None else None
        self.M1_ff_image = image_utils.normalize_to_uint8(np.mean(self.M1_ff_video, axis=0)) if self.M1_ff_video is not None else None
        self.M2_ff_image = image_utils.normalize_to_uint8(np.mean(self.M2_ff_video, axis=0)) if self.M2_ff_video is not None else None
        return 

class PreprocessStep(BaseStep):
    requires = ["moments"]
    produces = ["M0_ff_video", "M0_ff_image", "M1_ff_image"]
    name = "preprocess"

    def _relevant_config(self, ctx):
        return {
            "Preprocess": {
                "Register": ctx.config.get("Preprocess", {}).get("Register", {}),
                "Crop": ctx.config.get("Preprocess", {}).get("Crop", {})
            },
            "FlatFieldCorrection": {
                "GWRatio": ctx.config.get("FlatFieldCorrection", {}).get("GWRatio", 0.07)
            }
        }

    def run(self, ctx):
        moments = ctx.cache["moments"]
        ctx.output_manager.save(self.name, "M0_video", moments.M0, format="avi")

        pre = Preprocessor(ctx.config, moments)
        pre.preprocess()

        if pre.M0_ff_image is not None:
            ctx.cache["M0_ff_video"] = pre.M0_ff_video
            # ctx.output_manager.save(self.name, "M0_ff_video", pre.M0_ff_video, format="avi")
            ctx.cache["M0_ff_image"] = pre.M0_ff_image
            ctx.output_manager.save(self.name, "M0_ff_image", pre.M0_ff_image, format="png")

        if pre.M1_ff_image is not None:
            ctx.cache["M1_ff_video"] = pre.M1_ff_video
            # ctx.output_manager.save(self.name, "M1_ff_video", pre.M1_ff_video, format="avi")
            ctx.cache["M1_ff_image"] = pre.M1_ff_image
            ctx.output_manager.save(self.name, "M1_ff_image", pre.M1_ff_image, format="png")

        if pre.M2_ff_image is not None:
            ctx.cache["M2_ff_video"] = pre.M2_ff_video
            # ctx.output_manager.save(self.name, "M2_ff_video", pre.M2_ff_video, format="avi")
            ctx.cache["M2_ff_image"] = pre.M2_ff_image
            ctx.output_manager.save(self.name, "M2_ff_image", pre.M2_ff_image, format="png")