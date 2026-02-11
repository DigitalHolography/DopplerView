"""
Segmentation module for binary and semantic segmentation
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure


def binary_segmentation(frames, config):
    """
    Perform binary segmentation to extract vessel mask from spectral moment data
    
    Args:
        frames: preprocessed frames of shape (num_frames, height, width)
        config: binary segmentation configuration dict
    
    Returns:
        Binary vessel mask of shape (height, width)
    """
    # Extract configuration
    threshold_method = config.get('threshold_method', 'otsu')
    min_vessel_size = config.get('min_vessel_size', 100)
    use_temporal_variance = config.get('use_temporal_variance', True)
    
    # Compute temporal statistics
    if use_temporal_variance:
        # Use temporal variance as feature for segmentation
        # Vessels typically show higher temporal variation due to blood flow
        feature_map = np.var(frames, axis=0)
    else:
        # Use mean intensity
        feature_map = np.mean(frames, axis=0)
    
    # Threshold to get binary mask
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(feature_map)
        vessel_mask = feature_map > threshold
    elif threshold_method == 'adaptive':
        # Adaptive thresholding
        threshold = filters.threshold_local(feature_map, block_size=35)
        vessel_mask = feature_map > threshold
    elif threshold_method == 'percentile':
        percentile = config.get('percentile', 75)
        threshold = np.percentile(feature_map, percentile)
        vessel_mask = feature_map > threshold
    else:
        # Default to Otsu
        threshold = filters.threshold_otsu(feature_map)
        vessel_mask = feature_map > threshold
    
    # Morphological operations to clean up mask
    # Suppress deprecation warnings for scikit-image version compatibility
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        vessel_mask = morphology.remove_small_objects(vessel_mask, min_size=min_vessel_size)
        vessel_mask = morphology.binary_closing(vessel_mask, morphology.disk(3))
        vessel_mask = morphology.binary_opening(vessel_mask, morphology.disk(2))
    
    return vessel_mask.astype(np.uint8)


def semantic_segmentation(frames, vessel_mask, pulse_results, config):
    """
    Perform semantic segmentation to classify vessels as arteries or veins
    
    Args:
        frames: preprocessed frames of shape (num_frames, height, width)
        vessel_mask: binary vessel mask of shape (height, width)
        pulse_results: pulse analysis results dict
        config: semantic segmentation configuration dict
    
    Returns:
        Semantic mask of shape (height, width) where:
        - 0: background
        - 1: vein
        - 2: artery
    """
    # Extract configuration
    pulsatility_threshold = config.get('pulsatility_threshold', 0.5)
    
    # Initialize semantic mask
    semantic_mask = np.zeros_like(vessel_mask, dtype=np.uint8)
    
    # Label connected components in vessel mask
    labeled_vessels = measure.label(vessel_mask, connectivity=2)
    num_vessels = labeled_vessels.max()
    
    # Get pulsatility map from pulse results
    pulsatility_map = pulse_results.get('pulsatility_map', np.zeros_like(vessel_mask, dtype=np.float32))
    
    # Classify each vessel
    for vessel_id in range(1, num_vessels + 1):
        vessel_region = labeled_vessels == vessel_id
        
        # Calculate mean pulsatility for this vessel
        mean_pulsatility = np.mean(pulsatility_map[vessel_region])
        
        # Classify based on pulsatility
        # Arteries typically have higher pulsatility than veins
        if mean_pulsatility > pulsatility_threshold:
            semantic_mask[vessel_region] = 2  # Artery
        else:
            semantic_mask[vessel_region] = 1  # Vein
    
    return semantic_mask
