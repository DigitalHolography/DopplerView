"""
Segmentation module for semantic segmentation
"""

import numpy as np
from skimage import filters, morphology, measure
import holosegment.segmentation.pulse_analysis as pulse_analysis
import holosegment.utils.model_utils as model_utils
  

def handmade_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache):
    pass


def artery_vein_segmentation(M0_ff_video, M0_ff_image, vessel_mask, config, cache):
    """
    Perform artery vein segmentation, using the binary vessel mask
    
    Args:
        M0_ff_video: preprocessed M0_flatfield video of shape (num_frames, height, width)
        M0_ff_image: preprocessed M0_flatfield image of shape (height, width)
        vessel_mask: binary vessel mask of shape (height, width)
        config: artery mask segmentation configuration dict
    
    Returns:
        Refined artery mask of shape (height, width)
    """
   
    # Compute pre-artery mask using pulse analysis
    pre_artery_mask = pulse_analysis.compute_pre_artery_mask(M0_ff_video, vessel_mask)

    if config['AVCorrelationSegmentationNet'] or config['AVDiasysSegmentationNet']:
        print("Using deep segmentation model for artery vein segmentation.")
        artery_mask, vein_mask = deep_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    else:
        print("Use hand-made heuristics for artery vein segmentation.")
        artery_mask, vein_mask = handmade_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    
    return artery_mask, vein_mask