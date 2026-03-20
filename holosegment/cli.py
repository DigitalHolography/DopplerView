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

    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Process multiple folders. Folders are either listed in a text file (one folder path per line) or provided as subfolders of the specified path.'
    )

    parser.add_argument(
        '-t', '--targets',
        nargs='+',
        help='List of target steps to run'
    )

    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode. In this mode, steps outputs are read from the .h5, and only targeted steps are re-run. This is useful for debugging specific steps without having to re-run the entire pipeline.'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    input_folder = Path(args.holodoppler_folder)

    debug = args.debug
    
    if not input_folder.exists():
        print(f"Error: holodoppler folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    registry = ModelRegistryConfig(Path("models.yaml"))
    pipeline = Pipeline(registry, h5_schema=json.load(open("h5_schema.json")), output_config=json.load(open("output_config.json")), debug_mode=debug)
    if args.config:
        pipeline.load_eyeflow_config(args.config)

    targets = args.targets if args.targets else None

    if args.batch:
        pipeline.load_folder_list(input_folder)
        pipeline.run_batch(targets=targets)
    else:
        pipeline.load_input(input_folder)
        pipeline.run(targets=targets)

    return 0


if __name__ == '__main__':
    sys.exit(main())
