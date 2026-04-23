"""
Registration of frames to correct for motion artifacts
"""

import numpy as np
from scipy import ndimage

def register_video(frames, firstFrame, endFrame, reference_idx=0):
    """
    Register frames to a reference frame using phase correlation
    
    Args:
        frames: numpy array of shape (num_frames, height, width)
        reference_idx: index of reference frame
    
    Returns:
        Registered frames
    """
    num_frames = frames.shape[0]
    registered = np.zeros_like(frames)
    
    # Use first frame or specified frame as reference
    reference = frames[reference_idx]
    registered[reference_idx] = reference
    
    for i in range(num_frames):
        if i == reference_idx:
            continue
            
        # Calculate shift using phase correlation
        shift = estimate_shift(reference, frames[i])
        
        # Apply shift
        registered[i] = ndimage.shift(frames[i], shift, mode='nearest')
    
    return registered


def estimate_shift(reference, target):
    """
    Estimate shift between two images using phase correlation
    
    Args:
        reference: reference image
        target: target image to align
    
    Returns:
        (dy, dx) shift values
    """
    # Compute FFT
    ref_fft = np.fft.fft2(reference)
    target_fft = np.fft.fft2(target)
    
    # Compute cross-power spectrum
    cross_power = (ref_fft * np.conj(target_fft)) / (np.abs(ref_fft * np.conj(target_fft)) + 1e-8)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Convert to shift values (handle wraparound)
    shift_y = peak[0] if peak[0] < correlation.shape[0] // 2 else peak[0] - correlation.shape[0]
    shift_x = peak[1] if peak[1] < correlation.shape[1] // 2 else peak[1] - correlation.shape[1]
    
    return (shift_y, shift_x)