"""
Main entry point for the image processing service.

Adds health monitoring and resource logging.
"""

import sys
import os
import time
import logging
import signal
import psutil
from pathlib import Path

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # pylint: disable=C0413
from image_processor.processor import PhotoProcessor  # noqa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global control flag
should_run = [True]
HEALTH_FILE = Path("/tmp/healthcheck.txt")
HEALTH_INTERVAL = 30  # seconds


def signal_handler(_, __, should_run_flag):
    """Handle termination signals to gracefully shut down the program."""
    logger.info("Termination signal received. Shutting down...")
    should_run_flag[0] = False


def log_resource_usage():
    """Log container resource usage (memory, CPU) every interval."""
    process = psutil.Process(os.getpid())
    with process.oneshot():
        cpu = process.cpu_percent(interval=None)
        mem = process.memory_info().rss / (1024 * 1024)
        logger.info("Resource usage — CPU: %.1f%%, Memory: %.1f MB", cpu, mem)


def main_loop(interval=30):
    """Run the process_photos function periodically with health check and resource logging."""
    processor = PhotoProcessor()
    last_health_update = 0

    while should_run[0]:
        try:
            logger.info("Running photo processing...")
            processor.process_photos()

            # Update health check file every HEALTH_INTERVAL
            now = time.time()
            if now - last_health_update > HEALTH_INTERVAL:
                HEALTH_FILE.write_text(str(now), encoding="utf-8")
                last_health_update = now

            # Log CPU/memory usage
            log_resource_usage()

        except (ValueError, TypeError, IOError) as e:
            logger.error("An error occurred during photo processing: %s", e)
        except Exception as e:
            logger.exception("Unexpected error in main loop: %s", e)

        if should_run[0]:
            logger.info("Waiting for %d seconds before the next run...", interval)
            time.sleep(interval)

    logger.info("Exiting main loop.")


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, should_run))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, should_run))

    logger.info("Starting the image processing service...")
    HEALTH_FILE.write_text(str(time.time()), encoding="utf-8")
    main_loop()
    logger.info("Image processing service stopped.")
