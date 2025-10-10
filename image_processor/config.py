"""Configuration module for face detection service."""

import os

# MongoDB configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB = os.getenv("MONGO_DB", "nill-home")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "nill-home-photos")
FACE_COLLECTION = os.getenv("FACE_COLLECTION", "nill-home-faces")
KNOWN_FACES_COLLECTION = os.getenv("KNOWN_FACES_COLLECTION", "nill-known-faces")

# Logging / error level
ERROR_LVL = os.getenv("ERROR_LVL", "info")

# History retention (days)
FACES_HISTORY_DAYS = int(os.getenv("FACES_HISTORY_DAYS", "30"))

# Face detection model (YOLOv8n-face by default)
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "yolov8n-face.pt")

# Whether to run OpenCV face detection before embeddings
# True = use OpenCV prefilter; False = embeddings handle all images directly
USE_OPENCV_PREFILTER = True

# Minimum confidence threshold for embeddings-based detection (if used)
EMBEDDING_CONFIDENCE_THRESHOLD = 0.6

# === Face Detection Settings ===
# Path to OpenCV Haar Cascade XML file (used for face detection)
OPENCV_CASCADE_PATH = "haarcascade_frontalface_default.xml"
