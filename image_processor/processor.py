"""Photo processing service with OpenCV prefilter, YOLO-based face detection, and annotated face storage."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
import numpy as np

from flask import Flask, Response, jsonify
import cv2
import warnings

from .config import (
    FACE_COLLECTION,
    FACES_HISTORY_DAYS,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_HOST,
    MONGO_PORT,
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
    """Quick prefilter using OpenCV to detect if faces are present.

    Returns True if OpenCV Haar cascade finds at least one face-like object.
    """
    try:
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        # Pass the image as first argument (previous bug: gray was not passed)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40)
        )
        return len(faces) > 0
    except Exception as e:
        logger.error("OpenCV face detection failed: %s", e)
        return False


class PhotoProcessor:
    """Unified photo processor using OpenCV prefilter + YOLO embeddings for faces."""

    def __init__(self) -> None:
        """Initialize the PhotoProcessor, load metadata and known faces."""
        # numeric seconds-since-epoch
        self.latest_processed_date: float = 0.0
        self.embedding_engine = EmbeddingEngine()

        # Metadata collection for tracking last processed timestamp
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
            if MONGO_HOST.startswith(("mongodb://", "mongodb+srv://")):
                client = MongoClient(MONGO_HOST)
            else:
                client = MongoClient(MONGO_HOST, MONGO_PORT)
            coll = client[db_name][collection_name]
            logger.debug("Connected to collection: %s.%s", db_name, collection_name)
            return coll
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
    def _parse_meta_value(self, val: Any) -> float:
        """Robustly parse a stored latest_date value into a float timestamp (seconds)."""
        if val is None:
            return 0.0
        # numeric
        if isinstance(val, (int, float)):
            return float(val)
        # datetimes
        if isinstance(val, datetime):
            return float(val.replace(tzinfo=timezone.utc).timestamp())
        # bson-like dict with $date or ISO string
        if isinstance(val, dict) and ("$date" in val):
            d = val["$date"]
            if isinstance(d, (int, float)):
                # milliseconds epoch?
                # if it's obviously > 1e12 we assume ms
                if d > 1e12:
                    return float(d) / 1000.0
                return float(d)
            if isinstance(d, str):
                try:
                    # ISO string with Z
                    iso = d.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    return float(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    return 0.0
        # string numeric or ISO
        if isinstance(val, str):
            try:
                return float(val)
            except Exception:
                try:
                    iso = val.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    return float(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    return 0.0
        return 0.0

    def load_last_processed_date(self) -> None:
        """Load the last processed timestamp (float) from MongoDB."""
        if self.meta_collection is None:
            logger.warning("Meta collection unavailable, starting from scratch.")
            self.latest_processed_date = 0.0
            return

        try:
            state = self.meta_collection.find_one({"_id": "latest_processed_state"})
            if state and "latest_date" in state:
                self.latest_processed_date = self._parse_meta_value(
                    state["latest_date"]
                )
                logger.info(
                    "Resuming from last processed timestamp: %s (%s)",
                    self.latest_processed_date,
                    datetime.fromtimestamp(
                        self.latest_processed_date, tz=timezone.utc
                    ).isoformat(),
                )
            else:
                self.latest_processed_date = 0.0
                logger.info("No previous state found, starting fresh.")
        except PyMongoError as e:
            logger.error("Failed to load meta state: %s", e)
            self.latest_processed_date = 0.0

    def save_last_processed_date(self) -> None:
        """Save the last processed timestamp (float) to MongoDB."""
        if self.meta_collection is None:
            logger.debug("Meta collection missing; not saving last processed date.")
            return
        try:
            self.meta_collection.update_one(
                {"_id": "latest_processed_state"},
                {"$set": {"latest_date": float(self.latest_processed_date)}},
                upsert=True,
            )
            logger.info(
                "💾 Updated last processed date to: %s (%s)",
                self.latest_processed_date,
                datetime.fromtimestamp(
                    self.latest_processed_date, tz=timezone.utc
                ).isoformat(),
            )
        except PyMongoError as e:
            logger.error("Failed to update meta state: %s", e)

    # --- Old photo cleanup ---
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

    # --- Helpers ---
    def _extract_photo_timestamp(self, photo: Dict[str, Any]) -> float:
        """
        Try to extract a numeric timestamp (seconds since epoch) from the photo document.
        Prefers numeric 'date' field, falls back to 'bsonTime' if present.
        """
        # 1) numeric 'date'
        d = photo.get("date")
        if isinstance(d, (int, float)):
            return float(d)
        if isinstance(d, datetime):
            return float(d.replace(tzinfo=timezone.utc).timestamp())
        # 2) bsonTime style: {"$date": "2025-10-10T12:17:32.584Z"} or {"$date": 163...}
        bt = photo.get("bsonTime") or photo.get("bson_date") or photo.get("bson")
        if isinstance(bt, dict) and "$date" in bt:
            v = bt["$date"]
            if isinstance(v, (int, float)):
                # could be ms
                if v > 1e12:
                    return float(v) / 1000.0
                return float(v)
            if isinstance(v, str):
                try:
                    iso = v.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    return float(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    return 0.0
        # 3) maybe direct datetime stored
        if isinstance(bt, datetime):
            return float(bt.replace(tzinfo=timezone.utc).timestamp())
        # nothing found
        return 0.0

    # --- Main processing ---
    def process_photos(self) -> None:
        """Process only new photos, update last processed timestamp."""
        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

        if main_collection is None or face_collection is None:
            logger.error("MongoDB collections unavailable.")
            return

        logger.info(
            "Processing collection: %s.%s",
            main_collection.database.name,
            main_collection.name,
        )
        logger.info(
            "Storing faces into: %s.%s",
            face_collection.database.name,
            face_collection.name,
        )

        last_date = float(self.latest_processed_date or 0.0)
        logger.info(
            "Last processed timestamp (numeric): %s (%s)",
            last_date,
            (
                datetime.fromtimestamp(last_date, tz=timezone.utc).isoformat()
                if last_date > 0
                else "epoch"
            ),
        )

        # Build robust query: either numeric 'date' > last_date or bsonTime > last_date
        bson_cutoff = datetime.fromtimestamp(last_date, tz=timezone.utc)
        query = {
            "$or": [
                {"date": {"$gt": last_date}},
                {"bsonTime": {"$gt": bson_cutoff}},
            ]
        }

        # fetch new photos sorted by date (fallback to bsonTime order)
        try:
            photos_cursor = main_collection.find(query).sort(
                [("date", ASCENDING), ("bsonTime", ASCENDING)]
            )
        except PyMongoError as e:
            logger.error("MongoDB query failed: %s", e)
            return

        photos = list(photos_cursor)
        logger.info("Found %d new photos since %s", len(photos), last_date)

        if not photos:
            return

        for photo in photos:
            filename = photo.get("filename", "unknown")
            photo_ts = self._extract_photo_timestamp(photo)

            # Immediately advance latest_processed_date so we don't reprocess this doc again
            if photo_ts > self.latest_processed_date:
                self.latest_processed_date = photo_ts

            logger.info(
                "Processing photo '%s' (photo_ts=%s -> %s)",
                filename,
                photo_ts,
                (
                    datetime.fromtimestamp(photo_ts, tz=timezone.utc).isoformat()
                    if photo_ts
                    else "unknown"
                ),
            )

            image_data = photo.get("data")
            if not image_data:
                logger.warning("⚠️ No image data in '%s', skipping.", filename)
                continue

            # Optional OpenCV prefilter
            if USE_OPENCV_PREFILTER and not detect_faces_opencv(image_data):
                logger.info(
                    "No faces detected by OpenCV in '%s', skipping YOLO.", filename
                )
                # Note: latest_processed_date already advanced for this photo
                continue

            # YOLO embeddings
            embeddings = self.embedding_engine.generate_face_embeddings(image_data)
            if not embeddings or embeddings.get("face_count", 0) == 0:
                logger.info("No faces confirmed by YOLO in '%s', skipping.", filename)
                continue

            # Known face matching
            matched_names = []
            if "face_embedding" in embeddings and self.known_faces:
                current_emb = np.array(
                    embeddings["face_embedding"], dtype=np.float32
                ).reshape(1, -1)
                for name, known_emb in self.known_faces.items():
                    # Normalize both embeddings
                    current_norm = current_emb / np.linalg.norm(current_emb)
                    known_norm = known_emb / np.linalg.norm(known_emb)
                    similarity = float(np.dot(current_norm, known_norm.T))
                    logger.debug("Sim(%s, %s) = %.3f", filename, name, similarity)

                    if (
                        similarity > 0.51
                    ):  # manual testing indicates its the optimal threshold
                        matched_names.append(name)
                        logger.info(
                            "✅ Match found in '%s': %s (sim=%.3f)",
                            filename,
                            name,
                            similarity,
                        )

            annotated_bytes = embeddings.get("annotated_bytes")
            document = {
                "filename": filename,
                "data": annotated_bytes if annotated_bytes else image_data,
                "search_embeddings": embeddings,
                "s3_file_url": photo.get("s3_file_url", ""),
                "size": photo.get("size", 0),
                "date": photo.get("date", 0),
                "bsonTime": photo.get("bsonTime", None),
                "camera_location": photo.get("camera_location", ""),
                "has_faces": True,
                "face_count": embeddings.get("face_count", 0),
                "matched_persons": matched_names,
                "processing_timestamp": time.time(),
            }

            try:
                face_collection.insert_one(document)
                logger.info(
                    "Stored '%s' with %d faces. Matches: %s",
                    filename,
                    document["face_count"],
                    matched_names or "None",
                )
            except PyMongoError as e:
                logger.error("MongoDB insert error for '%s': %s", filename, e)
                # on insert failure we still keep the latest_processed_date advanced to avoid repeat
                continue

        # Save the latest_processed_date after processing all photos
        self.save_last_processed_date()
        # cleanup old
        self.delete_old_faces(face_collection)


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
