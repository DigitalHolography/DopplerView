"""
CLI interface for DopplerView application
"""

import argparse
import json
import sys
from pathlib import Path
from dopplerview.input_output import log_config, user_config
import numpy as np
import matplotlib

from dopplerview.pipeline.pipeline import Pipeline
from dopplerview.models.registry import ModelRegistryConfig
import dopplerview.input_output.user_config as user_config

import logging
logger = logging.getLogger(__name__)

def load_dopplerview_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    matplotlib.use("Agg")
    log_config.setup_logging()
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='DopplerView - Artery/vein segmentation from doppler holograms',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input',
        type=str,
        help="""Path to either:
    - measure.holo file: It must have a corresponding measure/measure_HD folder with the hologram data. The measure.holo file is used to determine the input folder for the pipeline.
    - batch folder: A folder containing multiple measure.holo files, each with a corresponding measure/measure_HD folder.
    - a .txt file: Contains a list of paths to measure.holo files, one per line."""
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to JSON configuration file'
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
    input = Path(args.input)

    if not input.exists():
        logger.info(f"Error: input not found: {input}", file=sys.stderr)
        sys.exit(1)

    debug = args.debug
    
    pipeline = Pipeline(debug_mode=debug)

    if args.config:
        pipeline.load_dopplerview_config(args.config)

    targets = args.targets if args.targets else None

    pipeline.load_input(input)
    pipeline.run_batch(targets=targets)

    return 0


if __name__ == '__main__':
    sys.exit(main())
