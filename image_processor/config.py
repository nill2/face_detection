"""Configuration module for face detection service."""

import os

# MongoDB configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB = os.getenv("MONGO_DB", "photos")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "images")
FACE_COLLECTION = os.getenv("FACE_COLLECTION", "nill-home-faces")

# Logging / error level
ERROR_LVL = os.getenv("ERROR_LVL", "info")

# History retention (days)
FACES_HISTORY_DAYS = int(os.getenv("FACES_HISTORY_DAYS", "30"))

# Face detection model (YOLOv8n-face by default)
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "yolov8n-face.pt")
