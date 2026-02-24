"""
Cache for storing intermediate results during segmentation to speed up processing and avoid redundant computations. This includes:
- Preprocessed frames
- Segmentation models loaded in memory
- Intermediate segmentation masks (e.g., vessel masks, artery masks)
"""

import numpy as np
import torch
import holosegment.models.wrapper as wrapper

class SegmentationCache:
    def __init__(self):
        self.M0 = None  # Cache for preprocessed M0 data
        self.M0_ff_video = None  # Cache for flatfield-corrected M0 video
        self.M0_ff_image = None  # Cache for flatfield-corrected M0 image

        self.M1 = None  # Cache for M1 data if needed
        self.M2 = None  # Cache for M2 data if needed

        self.segmentation_models = {}    # Cache for loaded segmentation models
        self.intermediate_masks = {}     # Cache for intermediate segmentation masks (e.g., vessel masks, artery masks)

    def get_preprocessed_frames(self, raw_frames):
        if self.preprocessed_frames is None:
            self.preprocessed_frames = self.preprocess_frames(raw_frames)
        return self.preprocessed_frames

    def preprocess_frames(self, raw_frames):
        # Implement preprocessing steps (e.g., normalization, resizing) as needed
        preprocessed = raw_frames.astype(np.float32) / 255.0  # Example normalization
        return preprocessed

    def get_av_segmentation_model(self, config):
        use_correlation = config.get('AVCorrelationSegmentationNet', True)
        use_diasys = config.get('AVDiasysSegmentationNet', True)

        model_name_dict = {
            (True, True): 'AVCombinedSegmentationNet',
            (True, False): 'AVCorrelationSegmentationNet',
            (False, True): 'AVDiasysSegmentationNet',
        }

        model_name = model_name_dict.get((use_correlation, use_diasys))
        if model_name not in self.segmentation_models:
            self.segmentation_models[model_name] = wrapper.load_pt_model(model_name)
        return self.segmentation_models[model_name]

    def cache_intermediate_mask(self, mask_name, mask):
        self.intermediate_masks[mask_name] = mask

    def get_intermediate_mask(self, mask_name):
        return self.intermediate_masks.get(mask_name, None)