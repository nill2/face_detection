"""Module for processing photos, detecting faces, storing results in MongoDB, and saving embeddings for RAG queries."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from ultralytics import YOLO
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
from .embeddings import EmbeddingEngine, EfficientEmbeddingEngine

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhotoProcessor:
    """Process photos, detect faces, store results in MongoDB, and save embeddings for RAG."""

    def __init__(self) -> None:
        """Initialize the photo processor and embedding engines."""
        self.latest_processed_date: int = 0
        self.face_model: Optional[YOLO] = None
        self.embedding_engine = EmbeddingEngine()  # Legacy text embeddings
        self.efficient_engine = EfficientEmbeddingEngine()  # Enhanced search embeddings

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
                    with open(model_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
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

    def connect_to_mongodb(
        self, db_name: str, collection_name: str
    ) -> Optional[Collection]:
        """Connect to MongoDB and return the collection, or None on failure."""
        try:
            if MONGO_HOST.startswith("mongodb://") or MONGO_HOST.startswith(
                "mongodb+srv://"
            ):
                client = MongoClient(MONGO_HOST)
            else:
                client = MongoClient(MONGO_HOST, MONGO_PORT)
            mongo_db = client[db_name]
            collection = mongo_db[collection_name]
            if ERROR_LVL == "debug":
                logger.info(
                    "Connected to MongoDB: %s:%s/%s/%s",
                    MONGO_HOST,
                    MONGO_PORT,
                    db_name,
                    collection_name,
                )
            return collection
        except ConnectionFailure as connection_error:
            logger.error("Failed to connect to MongoDB: %s", connection_error)
            return None

    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old photos from the specified collection based on FACES_HISTORY_DAYS."""
        try:
            time_threshold = time.time() - (FACES_HISTORY_DAYS * 86400)
            query = {"date": {"$lt": time_threshold}}
            result = face_collection.delete_many(query)
            logger.info("Deleted old photos from MongoDB: %s", result.deleted_count)
        except PyMongoError as error:
            logger.error("Error deleting old photos from MongoDB: %s", error)

    def detect_faces(self, image_data: bytes) -> bool:
        """Detect faces in the given image data using either YOLO or OpenCV HaarCascade."""
        try:
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None or len(image.shape) != 3:
                logger.error("Invalid image data provided for face detection.")
                return False

            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                enhanced_image, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            detected = len(faces) > 0
            logger.info("Face detection method: OpenCV, detected=%s", detected)

            if detected and self.face_model:
                results = self.face_model(image)
                yolo_detected = any(len(r.boxes) > 0 for r in results)
                logger.info("Face detection method: YOLO, detected=%s", yolo_detected)
                return yolo_detected

            return detected

        except cv2.error as error:
            logger.error("Error during face detection: %s", error)
            return False
        except Exception as error:
            logger.error("Unexpected error in face detection: %s", error)
            return False

    def process_photos(self) -> None:
        """Process photos from MongoDB, detect faces, and compute embeddings."""
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
                logger.info(
                    "Processing photo: %s with date %s",
                    photo.get("filename", "<unknown>"),
                    photo.get("bsonTime", "<unknown>"),
                )
                image_data = photo["data"]

                if self.detect_faces(image_data):
                    search_embeddings = (
                        self.efficient_engine.generate_search_embeddings(
                            image_data=image_data,
                            timestamp=photo["date"],
                            filename=photo.get("filename", ""),
                            camera_location=photo.get("camera_location", ""),
                        )
                    )

                    text_embedding = self.embedding_engine.encode(
                        photo.get("filename", "")
                    )

                    document = {
                        "filename": photo.get("filename", ""),
                        "data": image_data,
                        "embedding": text_embedding,
                        "search_embeddings": search_embeddings,
                        "s3_file_url": photo.get("s3_file_url", ""),
                        "size": photo.get("size", 0),
                        "date": photo.get("date", 0),
                        "bsonTime": photo.get("bsonTime", ""),
                        "camera_location": photo.get("camera_location", ""),
                        "has_faces": True,
                        "face_count": self._extract_face_count(search_embeddings),
                        "vehicle_detected": self._extract_vehicle_score(
                            search_embeddings
                        ),
                        "processing_timestamp": time.time(),
                    }

                    face_collection.insert_one(document)
                    logger.info(
                        "Face detected in %s. Photo copied to face collection with embedding.",
                        photo.get("filename", "<unknown>"),
                    )

                self.latest_processed_date = max(
                    photo.get("date", self.latest_processed_date),
                    self.latest_processed_date,
                )

            except cv2.error as error:
                logger.error(
                    "OpenCV error while processing photo %s: %s",
                    photo.get("filename", "<unknown>"),
                    error,
                )
            except PyMongoError as error:
                logger.error(
                    "MongoDB error while processing photo %s: %s",
                    photo.get("filename", "<unknown>"),
                    error,
                )
            except Exception as error:
                logger.error(
                    "Unexpected error processing photo %s: %s",
                    photo.get("filename", "<unknown>"),
                    error,
                )

        self.delete_old_faces(face_collection)

    def _extract_face_count(self, search_embeddings: Dict[str, bytes]) -> int:
        """Extract face count from search embeddings."""
        try:
            if (
                isinstance(search_embeddings, dict)
                and "face_count" in search_embeddings
            ):
                face_count_bytes = search_embeddings["face_count"]
                face_count = np.frombuffer(face_count_bytes, dtype=np.float32)[0]
                return int(face_count)
            return 0
        except Exception:
            return 0

    def _extract_vehicle_score(self, search_embeddings: Dict[str, bytes]) -> float:
        """Extract vehicle detection score from search embeddings."""
        try:
            if (
                isinstance(search_embeddings, dict)
                and "vehicle_score" in search_embeddings
            ):
                vehicle_bytes = search_embeddings["vehicle_score"]
                vehicle_score = np.frombuffer(vehicle_bytes, dtype=np.float32)[0]
                return float(vehicle_score)
            return 0.0
        except Exception:
            return 0.0
