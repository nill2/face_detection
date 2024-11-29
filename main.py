"""
Main entry point for the image processing service.

This script sets up the necessary logging, handles termination signals (e.g., Ctrl+C or SIGTERM),
and runs the `process_photos` function periodically to process images. The processing loop runs
until a termination signal is received.

Modules:
    sys: For manipulating the Python runtime environment.
    os: For interacting with the operating system, including file and path management.
    time: For pausing execution between processing runs.
    logging: For logging application events and errors.
    signal: For handling system signals (e.g., termination signals).

Functions:
    signal_handler(signum, frame): Handles termination signals and gracefully shuts down the program.
    main_loop(interval=30): Runs the photo processing function at regular intervals, logging status
                             and waiting between runs.

Global Variables:
    should_run (bool): Flag to control whether the main loop should continue running.
"""

import sys
import os
import time
import logging
import signal

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # pylint: disable=C0413

from image_processor.processor import PhotoProcessor  # noqa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to indicate whether the program should continue running
should_run = [True]


def signal_handler(_, __, should_run_flag):
    """
    Handles termination signals to gracefully shut down the program.
    """
    logger.info("Termination signal received. Shutting down...")
    should_run_flag[0] = False


def main_loop(interval=30):
    """
    Runs the process_photos function periodically.

    Args:
        interval (int): The interval in seconds between processing runs (default is 30 seconds).
    """
    # Create an instance of PhotoProcessor
    processor = PhotoProcessor()

    while should_run[0]:
        logger.info("Running photo processing...")
        try:
            processor.process_photos()  # Call the process_photos method on the instance
        except (ValueError, TypeError, IOError) as e:
            logger.error("An error occurred during photo processing: %s", e)
        if should_run[0]:
            logger.info("Waiting for %d seconds before the next run...", interval)
            time.sleep(interval)


if __name__ == "__main__":
    # Register the signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, should_run))
    signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, should_run))

    logger.info("Starting the image processing service...")
    main_loop()
    logger.info("Image processing service stopped.")
