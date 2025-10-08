"""Photo processing service with YOLO-based face detection and MongoDB integration."""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
import numpy as np
from flask import Flask, jsonify, Response

from .config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_DB,
    MONGO_COLLECTION,
    FACE_COLLECTION,
    FACES_HISTORY_DAYS,
)
from .embeddings import EmbeddingEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class PhotoProcessor:
    """Unified photo processor using YOLO embeddings for faces."""

    def __init__(self) -> None:
        """Initialize the PhotoProcessor."""
        self.latest_processed_date: int = 0
        self.embedding_engine = EmbeddingEngine()

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

    def process_photos(self) -> None:
        """Process new photos from the main collection and store embeddings."""
        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)
        if not main_collection or not face_collection:
            logger.error("MongoDB collections unavailable.")
            return

        query = {"date": {"$gt": self.latest_processed_date}}
        photos = list(main_collection.find(query).sort("date", DESCENDING))
        logger.info("Found %d new photos to process.", len(photos))

        for photo in photos:
            try:
                image_data = photo.get("data")
                if not image_data:
                    continue

                embeddings = self.embedding_engine.generate_face_embeddings(image_data)
                face_count = int(
                    np.frombuffer(
                        embeddings.get("face_count", b"\x00\x00\x00\x00"), np.float32
                    )[0]
                )
                if face_count == 0:
                    continue

                document = {
                    "filename": photo.get("filename", ""),
                    "data": image_data,
                    "search_embeddings": embeddings,
                    "s3_file_url": photo.get("s3_file_url", ""),
                    "size": photo.get("size", 0),
                    "date": photo.get("date", 0),
                    "camera_location": photo.get("camera_location", ""),
                    "has_faces": True,
                    "face_count": face_count,
                    "processing_timestamp": time.time(),
                }

                face_collection.insert_one(document)
                logger.info(
                    "Processed and stored photo %s with %d faces.",
                    photo.get("filename"),
                    face_count,
                )
                self.latest_processed_date = max(
                    photo.get("date", 0), self.latest_processed_date
                )

            except PyMongoError as e:
                logger.error("MongoDB error: %s", e)
            except Exception as e:
                logger.error("Error processing photo %s: %s", photo.get("filename"), e)

        # cleanup
        self.delete_old_faces(face_collection)

    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old faces."""
        try:
            threshold = time.time() - (FACES_HISTORY_DAYS * 86400)
            result = face_collection.delete_many({"date": {"$lt": threshold}})
            logger.info("Deleted %d old records.", result.deleted_count)
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
        processor.process_photos()
        time.sleep(60)
