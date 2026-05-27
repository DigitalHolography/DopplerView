"""
CLI interface for DopplerView application
"""

import argparse
import json
import sys
from pathlib import Path
from dopplerview.pipeline.pipeline import Pipeline
from dopplerview.input_output import log_config, user_config
import matplotlib


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
        '-p', '--params',
        type=str,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '-c', '--config_mode',
        type=str,
        help='Configuration mode to use : either "default" or "local" . If no config is provided using --params argument, "default" will use the default configuration file included in the user\'s config directory, while "local" will use a configuration file located in the processed DopplerView directory.'
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

    if args.params:
        pipeline.load_dopplerview_config(args.params)
    else:
        config_mode = args.config_mode if args.config_mode else "default"
        pipeline.set_config_mode(config_mode)
        logger.info(f"No configuration file provided. Using {config_mode} configuration.")
        if config_mode == "default":
            config_path = user_config.ensure_config_file("default_DV_params.json")
            pipeline.load_dopplerview_config(config_path)

    targets = args.targets if args.targets else None

    pipeline.load_input(input)
    pipeline.run_batch(targets=targets)

    return 0


if __name__ == '__main__':
    sys.exit(main())
