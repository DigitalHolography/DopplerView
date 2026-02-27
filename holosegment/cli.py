"""
CLI interface for HoloSegment application
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from holosegment.pipeline.pipeline import Pipeline
from holosegment.models.registry import ModelRegistryConfig


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='HoloSegment - Artery/vein segmentation from doppler holograms'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        'h5_file',
        type=str,
        help='Path to .h5 input file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    config_path = Path(args.config)
    h5_path = Path(args.h5_file)
    output_dir = Path(args.output)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    if not h5_path.exists():
        print(f"Error: h5 file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.verbose:
        print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    registry = ModelRegistryConfig(Path("models.yaml"))
    pipeline = Pipeline(config, registry, output_dir=output_dir, debug=args.verbose)

    

    pipeline.run_all(h5_path)
    
    # # Step 2: Preprocessing (normalization and registration)
    # if args.verbose:
    #     print("Preprocessing frames (normalization and registration)")
    # preprocessed_frames = preprocess_frames(reader.M0, config.get('preprocessing', {}))
    
    # # Step 3: Binary segmentation
    # if args.verbose:
    #     print("Performing binary segmentation")
    # vessel_mask = binary_segmentation(preprocessed_frames, config.get('binary_segmentation', {}))
    
    # # Save binary segmentation results
    # binary_output_path = output_dir / "vessel_mask.npy"
    
    # np.save(binary_output_path, vessel_mask)
    # if args.verbose:
    #     print(f"Saved vessel mask to {binary_output_path}")
    
    # # Step 4: Pulse analysis using vessel mask
    # if args.verbose:
    #     print("Performing pulse analysis")
    # pulse_results = analyze_pulse(preprocessed_frames, vessel_mask, config.get('pulse_analysis', {}))
    
    # # Save pulse analysis results
    # # Extract only JSON-serializable data
    # pulse_output_data = {
    #     'vessel_metrics': pulse_results.get('vessel_metrics', [])
    # }
    # pulse_output_path = output_dir / "pulse_results.json"
    # with open(pulse_output_path, 'w') as f:
    #     json.dump(pulse_output_data, f, indent=2)
    # if args.verbose:
    #     print(f"Saved pulse analysis results to {pulse_output_path}")
    
    # # Step 5: Semantic segmentation (artery/vein)
    # if args.verbose:
    #     print("Performing artery / vein segmentation (artery/vein)")
    # semantic_mask = artery_vein_segmentation(preprocessed_frames, vessel_mask, pulse_results, config.get('semantic_segmentation', {}))
    
    # # Save semantic segmentation results
    # semantic_output_path = output_dir / "artery_vein_mask.npy"
    # np.save(semantic_output_path, semantic_mask)
    # if args.verbose:
    #     print(f"Saved artery/vein segmentation to {semantic_output_path}")
    
    # print(f"Processing complete. Results saved to {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
