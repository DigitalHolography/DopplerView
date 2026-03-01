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


def load_eyeflow_config(config_path):
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
    config_path = Path(args.eyeflow_config)
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
    config = load_eyeflow_config(config_path)

    registry = ModelRegistryConfig(Path("models.yaml"))
    pipeline = Pipeline(config, registry, output_dir=output_dir, debug=args.verbose)

    pipeline.run(h5_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
