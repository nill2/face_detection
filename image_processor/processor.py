"""Photo processing service with OpenCV prefilter, YOLO-based face detection, and annotated face storage."""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, Response
import cv2
import warnings

from .config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_DB,
    MONGO_COLLECTION,
    FACE_COLLECTION,
    FACES_HISTORY_DAYS,
    USE_OPENCV_PREFILTER,
)
from .embeddings import EmbeddingEngine

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy libjpeg warnings
warnings.filterwarnings("ignore", message="Corrupt JPEG data")

app = Flask(__name__)


def detect_faces_opencv(image_data: bytes) -> bool:
    """Quick prefilter using OpenCV to detect if faces are present."""
    try:
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        return len(faces) > 0
    except Exception as e:
        logger.error("OpenCV face detection failed: %s", e)
        return False


class PhotoProcessor:
    """Unified photo processor using OpenCV prefilter + YOLO embeddings for faces."""

    def __init__(self) -> None:
        """Initialize the PhotoProcessor, load metadata and known faces."""
        self.latest_processed_date: Optional[datetime] = None
        self.embedding_engine = EmbeddingEngine()

        # Use dedicated metadata collection
        self.meta_collection: Optional[Collection] = self.connect_to_mongodb(
            MONGO_DB, "nill-home-metadata"
        )
        self.load_last_processed_date()

        # Known faces
        self.known_faces_collection = self.connect_to_mongodb(
            MONGO_DB, "nill-known-faces"
        )
        self.known_faces = self.load_known_faces()

    # --- MongoDB utilities ---
    def connect_to_mongodb(
        self, db_name: str, collection_name: str
    ) -> Optional[Collection]:
        """Connect to MongoDB and return a collection handle."""
        try:
            if MONGO_HOST.startswith("mongodb://") or MONGO_HOST.startswith(
                "mongodb+srv://"
            ):
                client = MongoClient(MONGO_HOST)
            else:
                client = MongoClient(MONGO_HOST, MONGO_PORT)
            return client[db_name][collection_name]
        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            return None

    # --- Known face loading ---
    def load_known_faces(self) -> Dict[str, np.ndarray]:
        """Load all known face embeddings from MongoDB."""
        known: Dict[str, np.ndarray] = {}
        if self.known_faces_collection is None:
            logger.warning("No known_faces_collection available.")
            return {}

        try:
            for doc in self.known_faces_collection.find({}):
                if "embedding" in doc and isinstance(doc["embedding"], list):
                    known[doc["name"]] = np.array(doc["embedding"], dtype=np.float32)
            logger.info(f"✅ Loaded {len(known)} known faces: {list(known.keys())}")
        except PyMongoError as e:
            logger.error("Failed to load known faces: %s", e)
        return known

    # --- Meta state handling ---
    def load_last_processed_date(self) -> None:
        """Load the last processed timestamp from MongoDB."""
        if self.meta_collection is None:
            logger.warning("Meta collection unavailable, starting from scratch.")
            return

        try:
            state = self.meta_collection.find_one({"_id": "latest_processed_state"})
            if state and "latest_date" in state:
                self.latest_processed_date = state["latest_date"]
                logger.info(
                    "Resuming from last processed date: %s", self.latest_processed_date
                )
            else:
                self.latest_processed_date = datetime.fromtimestamp(0, tz=timezone.utc)
                logger.info("No previous state found, starting fresh.")
        except PyMongoError as e:
            logger.error("Failed to load meta state: %s", e)
            self.latest_processed_date = datetime.fromtimestamp(0, tz=timezone.utc)

    def save_last_processed_date(self) -> None:
        """Save the last processed timestamp to MongoDB."""
        if self.meta_collection is None or self.latest_processed_date is None:
            return
        try:
            self.meta_collection.update_one(
                {"_id": "latest_processed_state"},
                {"$set": {"latest_date": self.latest_processed_date}},
                upsert=True,
            )
            logger.info(
                "💾 Updated last processed date to: %s", self.latest_processed_date
            )
        except PyMongoError as e:
            logger.error("Failed to update meta state: %s", e)

    # --- Main processing ---
    def process_photos(self) -> None:
        """Process new photos, extract embeddings, match known faces, and store results."""
        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

        if main_collection is None or face_collection is None:
            logger.error("MongoDB collections unavailable.")
            return

        # --- Fetch filenames already processed ---
        try:
            processed_files = {
                doc["filename"] for doc in face_collection.find({}, {"filename": 1})
            }
        except PyMongoError:
            processed_files = set()

        # --- Fetch new unprocessed photos ---
        query: Dict[str, Any] = {"filename": {"$nin": list(processed_files)}}
        if self.latest_processed_date:
            query["date"] = {"$gt": self.latest_processed_date}

        photos = list(main_collection.find(query).sort("date", ASCENDING))
        logger.info(
            "Found %d new unprocessed photos since %s",
            len(photos),
            self.latest_processed_date,
        )

        newest_date_seen = self.latest_processed_date

        for photo in photos:
            filename = photo.get("filename", "unknown")
            image_data = photo.get("data")
            if not image_data:
                logger.warning(f"⚠️ No image data in '{filename}', skipping.")
                continue

            # Step 1 — Optional OpenCV prefilter
            if USE_OPENCV_PREFILTER:
                if not detect_faces_opencv(image_data):
                    logger.info(
                        f"No faces detected by OpenCV in '{filename}', skipping YOLO."
                    )
                    continue

            # Step 2 — YOLO embeddings
            embeddings = self.embedding_engine.generate_face_embeddings(image_data)
            if not embeddings or embeddings.get("face_count", 0) == 0:
                logger.info(f"No faces confirmed by YOLO in '{filename}', skipping.")
                continue

            # Step 3 — Known face matching
            matched_names: List[str] = []
            if "face_embedding" in embeddings and self.known_faces:
                current_emb = np.array(
                    embeddings["face_embedding"], dtype=np.float32
                ).reshape(1, -1)
                for name, known_emb in self.known_faces.items():
                    similarity = cosine_similarity(
                        current_emb, known_emb.reshape(1, -1)
                    )[0][0]
                    if similarity > 0.85:
                        matched_names.append(name)
                        logger.info(
                            f"✅ Match found in '{filename}': {name} (similarity={similarity:.3f})"
                        )

            # Step 4 — Build document
            annotated_bytes = embeddings.get("annotated_bytes")
            document = {
                "filename": filename,
                "data": annotated_bytes if annotated_bytes else image_data,
                "search_embeddings": embeddings,
                "s3_file_url": photo.get("s3_file_url", ""),
                "size": photo.get("size", 0),
                "date": photo.get("date", 0),
                "camera_location": photo.get("camera_location", ""),
                "has_faces": True,
                "face_count": embeddings.get("face_count", 0),
                "matched_persons": matched_names,
                "processing_timestamp": datetime.now(timezone.utc),
            }

            try:
                face_collection.insert_one(document)
                logger.info(
                    f"Stored '{filename}' with {document['face_count']} faces. Matches: {matched_names or 'None'}"
                )
            except PyMongoError as e:
                logger.error(f"MongoDB error while inserting '{filename}': {e}")
                continue

            # Update latest processed timestamp
            photo_date = photo.get("date")
            if isinstance(photo_date, (int, float)):
                photo_date = datetime.fromtimestamp(photo_date, tz=timezone.utc)

            if isinstance(photo_date, datetime):
                if newest_date_seen is None or photo_date > newest_date_seen:
                    newest_date_seen = photo_date

        # After processing all photos, persist watermark
        if newest_date_seen and newest_date_seen != self.latest_processed_date:
            self.latest_processed_date = newest_date_seen
            self.save_last_processed_date()

        self.delete_old_faces(face_collection)

    # --- Cleanup ---
    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old face documents from the collection."""
        try:
            threshold = datetime.now(timezone.utc).timestamp() - (
                FACES_HISTORY_DAYS * 86400
            )
            result = face_collection.delete_many({"date": {"$lt": threshold}})
            logger.info("🧹 Deleted %d old records.", result.deleted_count)
        except PyMongoError as e:
            logger.error("Error deleting old records: %s", e)


@app.route("/health", methods=["GET"])  # type: ignore[misc]
def health() -> Tuple[Response, int]:
    """Health check endpoint."""
    payload: Dict[str, Any] = {"status": "ok", "timestamp": time.time()}
    return jsonify(payload), 200


if __name__ == "__main__":
    logger.info("Starting face detection processor service...")
    processor = PhotoProcessor()
    while True:
        logger.info("Running photo processing...")
        processor.process_photos()
        logger.info("Waiting 30 seconds before next run...")
        time.sleep(30)
