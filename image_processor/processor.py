"""Process photos from MongoDB, detect faces, and save embeddings for search queries."""

import logging
import time
from typing import Optional
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
from .embeddings import FaceEmbeddingEngine
from .config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_DB,
    MONGO_COLLECTION,
    FACE_COLLECTION,
    FACES_HISTORY_DAYS,
)

logger = logging.getLogger(__name__)


class PhotoProcessor:
    """Handles photo ingestion, face detection, and embedding storage."""

    def __init__(self) -> None:
        """Initialize the PhotoProcessor."""
        self.latest_processed_date: int = 0
        self.engine = FaceEmbeddingEngine()

    def connect_to_mongodb(
        self, db_name: str, collection_name: str
    ) -> Optional[Collection]:
        """Connect to MongoDB and return the collection."""
        try:
            if MONGO_HOST.startswith("mongodb://") or MONGO_HOST.startswith(
                "mongodb+srv://"
            ):
                client = MongoClient(MONGO_HOST)
            else:
                client = MongoClient(MONGO_HOST, MONGO_PORT)
            db = client[db_name]
            return db[collection_name]
        except ConnectionFailure as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return None

    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old face records."""
        try:
            time_threshold = time.time() - (FACES_HISTORY_DAYS * 86400)
            result = face_collection.delete_many({"date": {"$lt": time_threshold}})
            logger.info("Deleted %s old face records.", result.deleted_count)
        except PyMongoError as error:
            logger.error("Error deleting old face records: %s", error)

    def process_photos(self) -> None:
        """Process new photos: detect faces, generate embeddings, and store results."""
        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

        if not main_collection or not face_collection:
            logger.error("MongoDB connection failed.")
            return

        query = {"date": {"$gt": self.latest_processed_date}}
        photos = list(main_collection.find(query).sort("date", DESCENDING))

        for photo in photos:
            try:
                filename = photo.get("filename", "<unknown>")
                logger.info("Processing photo: %s", filename)
                image_data = photo["data"]

                face_embeddings = self.engine.generate_embeddings(image_data)
                if not face_embeddings:
                    logger.debug("No faces detected in %s.", filename)
                    continue

                for face_doc in face_embeddings:
                    document = {
                        "filename": filename,
                        "date": photo.get("date", 0),
                        "bsonTime": photo.get("bsonTime", ""),
                        "camera_location": photo.get("camera_location", ""),
                        "embedding": face_doc["embedding"],
                        "bbox": face_doc["bbox"],
                        "face_index": face_doc["face_index"],
                        "processing_timestamp": time.time(),
                    }
                    face_collection.insert_one(document)

                self.latest_processed_date = max(
                    photo.get("date", self.latest_processed_date),
                    self.latest_processed_date,
                )

                logger.info(
                    "Processed %d faces in photo %s",
                    len(face_embeddings),
                    filename,
                )

            except PyMongoError as error:
                logger.error("MongoDB error while processing %s: %s", filename, error)
            except Exception as error:
                logger.error("Unexpected error in %s: %s", filename, error)

        self.delete_old_faces(face_collection)
