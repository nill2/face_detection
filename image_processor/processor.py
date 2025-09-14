"""Module for processing photos, detecting faces, and storing results in MongoDB."""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Union

import requests
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
import numpy as np
import cv2

from .config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_DB,
    MONGO_COLLECTION,
    FACE_COLLECTION,
    ERROR_LVL,
    FACES_HISTORY_DAYS,
    FACE_DETECTION_MODEL,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhotoProcessor:
    """Process photos, detect faces using YOLO or OpenCV, and store results in MongoDB."""

    def __init__(self) -> None:
        """Initialize the photo processor with YOLO or OpenCV."""
        self.latest_processed_date: int = 0
        self.face_model = None

        if FACE_DETECTION_MODEL:
            try:
                models_dir = Path(__file__).resolve().parent / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = models_dir / Path(FACE_DETECTION_MODEL).name

                if not model_path.exists():
                    logger.info(
                        "YOLO model '%s' not found locally. Downloading...",
                        FACE_DETECTION_MODEL,
                    )
                    url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info("YOLO model downloaded to %s", model_path)

                self.face_model = YOLO(str(model_path))
                logger.info("YOLO model loaded successfully from %s", model_path)

            except Exception as error:
                logger.error(
                    "Failed to load YOLO model %s: %s", FACE_DETECTION_MODEL, error
                )
                self.face_model = None
        else:
            logger.info("Using default OpenCV HaarCascade face detection.")

    def connect_to_mongodb(self, db_name: str, collection_name: str) -> Collection:
        """Connect to a MongoDB collection."""
        try:
            client = MongoClient(MONGO_HOST, MONGO_PORT)
            collection = client[db_name][collection_name]
            if ERROR_LVL == "debug":
                logger.info(
                    "Connected to MongoDB: %s:%s/%s/%s",
                    MONGO_HOST,
                    MONGO_PORT,
                    db_name,
                    collection_name,
                )
            return collection
        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            return None

    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old photos from the face collection based on FACES_HISTORY_DAYS."""
        try:
            time_threshold = time.time() - (FACES_HISTORY_DAYS * 86400)
            result = face_collection.delete_many({"date": {"$lt": time_threshold}})
            logger.info("Deleted old photos from MongoDB: %s", result.deleted_count)
        except PyMongoError as e:
            logger.error("Error deleting old photos from MongoDB: %s", e)

    def detect_faces(
        self, image_data: bytes
    ) -> Union[List[Results], List[Tuple[int, int, int, int]]]:
        """Detect faces in the image and return detections."""
        try:
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return []

            detections = []

            if self.face_model:
                results = self.face_model(image)
                for box in results[0].boxes:
                    detections.append(
                        {
                            "bbox": box.xyxy[0].tolist(),
                            "conf": float(box.conf[0]),
                            "class": self.face_model.names[int(box.cls[0])],
                        }
                    )
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3
                )
                for x, y, w, h in faces:
                    detections.append(
                        {
                            "bbox": [int(x), int(y), int(x + w), int(y + h)],
                            "conf": None,
                            "class": "face",
                        }
                    )

            return detections

        except cv2.error as e:
            logger.error("OpenCV error during face detection: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error in face detection: %s", e)
            return []

    def process_photos(self) -> None:
        """Process new photos from MongoDB and store face detections."""
        if self.latest_processed_date is None:
            self.latest_processed_date = 0

        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

        if main_collection is None or face_collection is None:
            logger.error("Failed to connect to MongoDB collections.")
            return

        query = {"date": {"$gt": self.latest_processed_date}}
        photos = list(main_collection.find(query).sort("date", DESCENDING))

        for photo in photos:
            try:
                image_data = photo["data"]
                detections = self.detect_faces(image_data)
                if detections:
                    face_collection.insert_one(
                        {
                            "filename": photo["filename"],
                            "date": photo["date"],
                            "bsonTime": photo["bsonTime"],
                            "num_faces": len(detections),
                            "faces": detections,
                            "s3_file_url": photo.get("s3_file_url", ""),
                            "size": photo["size"],
                        }
                    )
                self.latest_processed_date = max(
                    photo["date"], self.latest_processed_date
                )
            except Exception as e:
                logger.error("Error processing photo %s: %s", photo["filename"], e)

        self.delete_old_faces(face_collection)
