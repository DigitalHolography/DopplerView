#!/usr/bin/env python3
"""
Script to create a test .holo file for testing the HoloSegment application
"""

import struct
import numpy as np


def create_test_holo_file(filepath, num_frames=10, width=256, height=256):
    """
    Create a synthetic .holo file with simulated vessel data
    
    Args:
        filepath: Path to output .holo file
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height
    """
    # Generate synthetic data with simulated vessels
    frames = generate_synthetic_vessels(num_frames, height, width)
    
    # Write to .holo file
    with open(filepath, 'wb') as f:
        # Write header
        # Magic number
        f.write(b'HOLO')
        
        # Version, width, height, num_frames, data_type, header_size
        version = 1
        data_type = 2  # float32
        header_size = 28  # 4 + 6*4 bytes
        
        header_data = struct.pack('<6I', version, width, height, num_frames, data_type, header_size)
        f.write(header_data)
        
        # Write frame data
        frames.astype(np.float32).tofile(f)
        
        # Write footer
        footer_size = 16
        timestamp = 1234567890.0
        checksum = 0x12345678
        
        footer_data = struct.pack('<IdI', footer_size, timestamp, checksum)
        f.write(footer_data)
    
    print(f"Created test .holo file: {filepath}")
    print(f"  Frames: {num_frames}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Data type: float32")


def generate_synthetic_vessels(num_frames, height, width):
    """
    Generate synthetic hologram data with vessel-like structures
    
    Args:
        num_frames: Number of frames
        height: Frame height
        width: Frame width
    
    Returns:
        Frames array of shape (num_frames, height, width)
    """
    frames = np.zeros((num_frames, height, width), dtype=np.float32)
    
    # Create background noise
    for i in range(num_frames):
        frames[i] = np.random.randn(height, width) * 0.1 + 0.5
    
    # Add synthetic vessels with pulsatile behavior
    # Artery 1: high pulsatility
    add_vessel(frames, center=(height//3, width//3), radius=15, 
               pulsatility=0.8, frequency=1.2)
    
    # Artery 2: high pulsatility
    add_vessel(frames, center=(2*height//3, width//4), radius=12, 
               pulsatility=0.7, frequency=1.1)
    
    # Vein 1: low pulsatility
    add_vessel(frames, center=(height//2, 2*width//3), radius=18, 
               pulsatility=0.2, frequency=0.8)
    
    # Vein 2: low pulsatility
    add_vessel(frames, center=(3*height//4, width//2), radius=14, 
               pulsatility=0.15, frequency=0.7)
    
    return frames


def add_vessel(frames, center, radius, pulsatility, frequency):
    """
    Add a synthetic vessel to frames with temporal pulsatile behavior
    
    Args:
        frames: Frames array to modify
        center: (y, x) center of vessel
        radius: Vessel radius
        pulsatility: Pulsatility index (0-1)
        frequency: Dominant frequency in Hz
    """
    num_frames, height, width = frames.shape
    cy, cx = center
    
    # Create vessel mask
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    vessel_mask = dist <= radius
    
    # Generate temporal signal with pulsatility
    t = np.arange(num_frames)
    base_intensity = 1.0
    temporal_signal = base_intensity * (1 + pulsatility * np.sin(2 * np.pi * frequency * t / num_frames))
    
    # Add to frames
    for i in range(num_frames):
        frames[i][vessel_mask] += temporal_signal[i]


if __name__ == '__main__':
    # Create test .holo file
    create_test_holo_file('/tmp/test_data.holo', num_frames=30, width=256, height=256)
    print("\nTest .holo file created successfully!")
