# face_detection/__init__.py

from .image_processor import PhotoProcessor, Config
from .main import run_application

# Optionally, define `__all__` to specify what is publicly available
__all__ = ["PhotoProcessor", "Config", "run_application"]
