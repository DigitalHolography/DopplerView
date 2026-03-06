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
        'holodoppler_folder',
        type=str,
        help='Path to holodoppler folder'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    input_folder = Path(args.holodoppler_folder)

    debug = args.verbose is not None
    
    if not input_folder.exists():
        print(f"Error: holodoppler folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    registry = ModelRegistryConfig(Path("models.yaml"))
    pipeline = Pipeline(registry, h5_schema=json.load(open("h5_schema.json")), debug_config=json.load(open("debug_config.json")))
    if args.config:
        pipeline.load_eyeflow_config(args.config)
    pipeline.load_input(input_folder)

    pipeline.run(debug=debug)

    return 0


if __name__ == '__main__':
    sys.exit(main())
