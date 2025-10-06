"""
Main entry point for the image processing service.

This script sets up logging, signal handling, a lightweight health-check server,
and runs the image processing loop until a termination signal is received.

Health endpoint:
    GET /health  → returns 200 OK if the process is alive
"""

import sys
import os
import time
import logging
import signal
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # pylint: disable=C0413

from image_processor.processor import PhotoProcessor  # noqa: E402

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Global control flag
# ----------------------------------------------------------------------
should_run = [True]


def signal_handler(_, __, should_run_flag):
    """Handle termination signals to gracefully shut down the program."""
    logger.info("Termination signal received. Shutting down...")
    should_run_flag[0] = False


# ----------------------------------------------------------------------
# Health check HTTP server
# ----------------------------------------------------------------------
def start_health_server(port=5000):
    """Start a tiny HTTP server providing a /health endpoint."""

    class HealthHandler(BaseHTTPRequestHandler):
        """Simple HTTP handler for /health endpoint."""

        def do_GET(self):
            """Return OK for /health requests."""
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            """Suppress default HTTP access logs."""
            return

    def run_server():
        """Run the health check server in a separate thread."""
        try:
            server = HTTPServer(("0.0.0.0", port), HealthHandler)
            logger.info("Health check server running on port %d", port)
            server.serve_forever()
        except Exception as e:
            logger.error("Health server failed: %s", e)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main_loop(interval=30):
    """
    Run the process_photos function periodically.

    Args:
        interval (int): Interval in seconds between processing runs.
    """
    processor = PhotoProcessor()

    while should_run[0]:
        logger.info("Running photo processing...")
        try:
            processor.process_photos()
        except Exception as e:  # noqa: BLE001
            logger.error("An error occurred during photo processing: %s", e)
        if should_run[0]:
            logger.info("Waiting for %d seconds before the next run...", interval)
            time.sleep(interval)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, should_run))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, should_run))

    logger.info("Starting the image processing service...")
    start_health_server(port=5000)  # lightweight /health endpoint
    main_loop()
    logger.info("Image processing service stopped.")
