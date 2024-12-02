"""
This module initializes the package and provides access to its main functionality.
"""

from .image_processor.processor import PhotoProcessor
from .main import main_loop

# Optionally, define `__all__` to specify what is publicly available
__all__ = ["PhotoProcessor", "main_loop"]
