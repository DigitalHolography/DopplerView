#!/usr/bin/env python3
"""
Script to verify the output of HoloSegment processing
"""

import numpy as np
import json
from pathlib import Path


def verify_output(output_dir):
    """Verify the output files from HoloSegment"""
    output_path = Path(output_dir)
    
    print("Verifying HoloSegment output...")
    print(f"Output directory: {output_path}\n")
    
    # Check vessel mask
    vessel_mask_path = output_path / "vessel_mask.npy"
    if vessel_mask_path.exists():
        vessel_mask = np.load(vessel_mask_path)
        print(f"✓ Vessel mask found")
        print(f"  Shape: {vessel_mask.shape}")
        print(f"  Number of vessel pixels: {np.sum(vessel_mask)}")
        print(f"  Data type: {vessel_mask.dtype}\n")
    else:
        print(f"✗ Vessel mask not found\n")
        return False
    
    # Check pulse results
    pulse_results_path = output_path / "pulse_results.json"
    if pulse_results_path.exists():
        with open(pulse_results_path, 'r') as f:
            pulse_results = json.load(f)
        print(f"✓ Pulse results found")
        num_vessels = len(pulse_results.get('vessel_metrics', []))
        print(f"  Number of vessels detected: {num_vessels}")
        for i, vessel in enumerate(pulse_results.get('vessel_metrics', [])):
            print(f"  Vessel {i+1}:")
            print(f"    Area: {vessel['area']:.0f} pixels")
            print(f"    Pulsatility: {vessel['mean_pulsatility']:.3f}")
            print(f"    Frequency: {vessel['mean_frequency']:.3f} Hz")
        print()
    else:
        print(f"✗ Pulse results not found\n")
        return False
    
    # Check semantic segmentation
    semantic_mask_path = output_path / "artery_vein_mask.npy"
    if semantic_mask_path.exists():
        semantic_mask = np.load(semantic_mask_path)
        print(f"✓ Artery/vein segmentation found")
        print(f"  Shape: {semantic_mask.shape}")
        print(f"  Background pixels (0): {np.sum(semantic_mask == 0)}")
        print(f"  Vein pixels (1): {np.sum(semantic_mask == 1)}")
        print(f"  Artery pixels (2): {np.sum(semantic_mask == 2)}")
        print(f"  Data type: {semantic_mask.dtype}\n")
    else:
        print(f"✗ Artery/vein segmentation not found\n")
        return False
    
    print("✓ All output files verified successfully!")
    return True


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/test_output'
    
    success = verify_output(output_dir)
    sys.exit(0 if success else 1)
