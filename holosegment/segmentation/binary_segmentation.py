"""
Binary segmentation of retinal vessels, using deep learning models or traditional methods.
"""

import holosegment.utils.model_utils as model_utils
import numpy as np

def deep_segmentation(M0_ff_image, config, cache):
    vessel_mask = model_utils.run_model(M0_ff_image, cache.get_av_segmentation_model(config), preprocess=True)

    return vessel_mask

def binary_segmentation(M0_ff_image, config, cache):
    """
    Perform binary vessel segmentation, using the M0_ff image
    
    Args:
        M0_ff_image: preprocessed M0_flatfield image of shape (height, width)
        config: artery mask segmentation configuration dict
        cache: cache object for storing/loading models and intermediate results
    Returns:
        Refined artery mask of shape (height, width)
    """
   
    method = config.get('BinarySegmentationMethod', 'AI')
    if method == 'AI':
        return deep_segmentation(M0_ff_image, config, cache)[0]  # Return artery mask

    if config['AVCorrelationSegmentationNet'] or config['AVDiasysSegmentationNet']:
        print("Using deep segmentation model for artery vein segmentation.")
        artery_mask, vein_mask = deep_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    else:
        print("Use hand-made heuristics for artery vein segmentation.")
        artery_mask, vein_mask = handmade_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    
    return artery_mask, vein_mask