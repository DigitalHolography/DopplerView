"""
Preprocessing class
"""

import numpy as np
from remove_outliers import remove_outliers
from registration import register_video
from resize import resize_video, crop_video, interpolate_video
from normalization import normalize_video, flat_field_correction

class Preprocessor:
    def __init__(self, config, M0=None, M1=None, M2=None, SH=None):
        self.config = config
        self.M0 = M0
        self.M1 = M1
        self.M2 = M2
        self.SH = SH

        self.M0_ff_video = None  # Cache for flatfield-corrected M0 video
        self.M0_ff_image = None  # Cache for flatfield-corrected M0

    def register(self, reference_idx=0):
        firstFrame = self.config['Preprocess', 'Register', 'FirstFrame']
        endFrame = self.config['Preprocess', 'Register', 'EndFrame']
        enable = self.config['Preprocess', 'Register', 'Enable']

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
    
    def crop(self, crop_size):
        firstFrame = self.config['Preprocess', 'Crop', 'FirstFrame']
        endFrame = self.config['Preprocess', 'Crop', 'EndFrame']
        if firstFrame == 0 and endFrame == -1:
            return
        # Implement cropping logic based on self.config
        if self.M0 is not None:
            self.M0 = crop_video(self.M0, crop_size)
        if self.M1 is not None:
            self.M1 = crop_video(self.M1, crop_size)
        if self.M2 is not None:
            self.M2 = crop_video(self.M2, crop_size)
        if self.SH is not None:
            self.SH = crop_video(self.SH, crop_size)

    def normalize(self):
        # Implement normalization logic based on self.config
        gaussian_blur_ratio = self.config['FlatFieldCorrection', 'GWRatio']
        self.M0_ff_video = flat_field_correction(self.M0, gaussian_blur_ratio)

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
        # Step 1: Register
        self.register()

        # Step 2: Crop frames
        self.crop()

        # Step 3: Normalize 
        self.normalize()

        # Step 4: Resize
        self.resize()

        # Step 5: Non-rigid registration
        self.nonrigid_register()

        # Step 6: Interpolate
        self.interpolate()

        # Step 7: Remove outliers 
        self.remove_outliers()

        return 