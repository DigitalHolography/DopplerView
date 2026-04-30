from dopplerview.pipeline.step import BaseStep

from dopplerview.preprocessing.registration import register_video
from dopplerview.preprocessing import normalization, resize
from dopplerview.utils import image_utils
from dopplerview.utils.parallelization_utils import run_in_parallel

from functools import partial
import numpy as np

class Preprocessor:
    def __init__(self, config, M0, M1, M2=None, SH=None):
        self.dopplerview_config = config
        self.M0 = M0
        self.M1 = M1
        self.M2 = M2
        self.SH = SH

        self.M0_ff_video = None  # Cache for flatfield-corrected M0 video
        self.M0_ff_image = None  # Cache for flatfield-corrected M0

        self.M1_ff_video = None  # Cache for flatfield-corrected M1 video
        self.M1_ff_image = None  # Cache for flatfield-corrected M1

        self.M2_ff_video = None  # Cache for flatfield-corrected M2 video
        self.M2_ff_image = None  # Cache for flatfield-corrected M2

    def register(self, reference_idx=0):
        firstFrame = self.dopplerview_config['Preprocess']['Register']['StartFrame']
        endFrame = self.dopplerview_config['Preprocess']['Register']['EndFrame']
        enable = self.dopplerview_config['Preprocess']['Register']['Enable']

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
        # Implement non-rigid registration logic based on self.dopplerview_config
        return
    
    def crop(self):
        firstFrame = self.dopplerview_config['Preprocess']['Crop']['StartFrame']
        endFrame = self.dopplerview_config['Preprocess']['Crop']['EndFrame']
        if firstFrame == 1 and endFrame == -1:
            return
        print(f"Cropping frames from {firstFrame} to {endFrame}")
        # Implement cropping logic based on self.dopplerview_config
        if self.M0 is not None:
            self.M0 = resize.crop_video(self.M0, first_frame=firstFrame, end_frame=endFrame)
        if self.M1 is not None:
            self.M1 = resize.crop_video(self.M1, first_frame=firstFrame, end_frame=endFrame)
        if self.M2 is not None:
            self.M2 = resize.crop_video(self.M2, first_frame=firstFrame, end_frame=endFrame)
        if self.SH is not None:
            self.SH = resize.crop_video(self.SH, first_frame=firstFrame, end_frame=endFrame)

    def normalize(self):
        # Implement normalization logic based on self.dopplerview_config
        # print(self.dopplerview_config)
        gw = self.dopplerview_config['FlatFieldCorrection']['GWRatio']

        if self.M0 is not None:
            numx = self.M0.shape[2]
            self.M0_ff_video = normalization.flat_field_correction_3d(self.M0, gw * numx, parallel=True) # TODO: add parameter for parallelization 

        if self.M1 is not None:
            self.M1_ff_video = normalization.flat_field_correction_3d(self.M1, gw * numx, parallel=True) # TODO: add parameter for parallelization 

        if self.M2 is not None:
            self.M2_ff_video = normalization.flat_field_correction_3d(self.M2, gw * numx, parallel=True) # TODO: add parameter for parallelization 

        return
    
    def resize(self):
        # Implement resizing logic based on self.dopplerview_config
        return
    
    def remove_outliers(self):
        # Implement outlier removal logic based on self.dopplerview_config
        return
    
    def interpolate(self):
        # Implement interpolation logic based on self.dopplerview_config
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
    requires = {"moment0", "moment1", "moment2"}
    produces = {"M0_ff_video", "M0_ff_image", "M1_ff_video", "M1_ff_image", "M2_ff_video", "M2_ff_image"}
    name = "preprocess"

    def _relevant_config(self, ctx):
        return {
            # "Preprocess": {
            #     "Register": ctx.dopplerview_config["Preprocess"]["Register"],
            #     "Crop": ctx.dopplerview_config["Preprocess"]["Crop"]
            # },
            "NumberOfWorkers": ctx.dopplerview_config["NumberOfWorkers"],
            "FlatFieldCorrection": {
                "GWRatio": ctx.dopplerview_config["FlatFieldCorrection"]["GWRatio"]
            }
        }

    def normalize(self, gaussian_std, M0, M1, M2, n_jobs=-1):
        # Implement normalization logic based on self.dopplerview_config
        # print(self.dopplerview_config)

        numx = M0.shape[2]
        M0_ff_video = normalization.flat_field_correction_3d(M0, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M1_ff_video = normalization.flat_field_correction_3d(M1, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M2_ff_video = normalization.flat_field_correction_3d(M2, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        return M0_ff_video, M1_ff_video, M2_ff_video
    
    def resize(self):
        # Implement resizing logic based on self.dopplerview_config
        return
    
    def remove_outliers(self):
        # Implement outlier removal logic based on self.dopplerview_config
        return
    
    def interpolate(self):
        # Implement interpolation logic based on self.dopplerview_config
        return

    def run(self, ctx):

        moment0 = ctx.require("moment0")
        moment1 = ctx.require("moment1")
        moment2 = ctx.require("moment2")

        # Step 1: Normalize 
        print("    - Applying flat field correction to the moments")
        gaussian_std = ctx.dopplerview_config['FlatFieldCorrection']['GWRatio']
        n_jobs = ctx.dopplerview_config["NumberOfWorkers"]
        M0_ff_video, M1_ff_video, M2_ff_video = self.normalize(gaussian_std, moment0, moment1, moment2, n_jobs=n_jobs)

        # # Step 2: Resize
        # self.resize()

        # # Step 3: Interpolate
        # self.interpolate()

        # # Step 4: Remove outliers 
        # self.remove_outliers()
        ctx.set("M0_ff_video", M0_ff_video)
        ctx.set("M1_ff_video", M1_ff_video)
        ctx.set("M2_ff_video", M2_ff_video)
        ctx.set("M0_ff_image", image_utils.normalize_to_uint8(np.mean(M0_ff_video, axis=0)) if M0_ff_video is not None else None)
        ctx.set("M1_ff_image", image_utils.normalize_to_uint8(np.mean(M1_ff_video, axis=0)) if M1_ff_video is not None else None)
        ctx.set("M2_ff_image", image_utils.normalize_to_uint8(np.mean(M2_ff_video, axis=0)) if M2_ff_video is not None else None)