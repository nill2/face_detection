"""Photo processing service with YOLO-based face detection, embedding storage, and known-face matching."""

import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.collection import Collection
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class PhotoProcessor:
    """Unified photo processor using YOLO embeddings for faces and known-face matching."""

    def __init__(self) -> None:
        """Initialize the PhotoProcessor."""
        self.latest_processed_date: int = 0
        self.embedding_engine = EmbeddingEngine()

        # Meta collection for storing last processed timestamp
        self.meta_collection: Optional[Collection] = self.connect_to_mongodb(
            MONGO_DB, FACE_COLLECTION + "_meta"
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
            if state:
                self.latest_processed_date = int(state.get("latest_date", 0))
                logger.info(
                    "Resuming from last processed timestamp: %d",
                    self.latest_processed_date,
                )
            else:
                logger.info("No previous state found, starting fresh.")
        except PyMongoError as e:
            logger.error("Failed to load meta state: %s", e)

    def save_last_processed_date(self) -> None:
        """Save the last processed timestamp to MongoDB."""
        if self.meta_collection is None:
            return
        try:
            self.meta_collection.update_one(
                {"_id": "latest_processed_state"},
                {"$set": {"latest_date": self.latest_processed_date}},
                upsert=True,
            )
            logger.debug(
                "Saved last processed timestamp: %d", self.latest_processed_date
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

        # Query only new photos
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

                # --- Match against known faces ---
                matched_names: List[str] = []
                if "face_embedding" in embeddings and self.known_faces:
                    current_emb = np.frombuffer(
                        embeddings["face_embedding"], np.float32
                    ).reshape(1, -1)
                    for name, known_emb in self.known_faces.items():
                        similarity = cosine_similarity(
                            current_emb, known_emb.reshape(1, -1)
                        )[0][0]
                        if similarity > 0.85:  # adjust threshold
                            matched_names.append(name)
                            logger.info(
                                f"✅ Match found in '{photo.get('filename')}': {name} (similarity={similarity:.3f})"
                            )

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
                    "matched_persons": matched_names,
                    "processing_timestamp": time.time(),
                }

                face_collection.insert_one(document)
                logger.info(
                    "Processed and stored photo '%s' with %d faces. Matches: %s",
                    photo.get("filename"),
                    face_count,
                    matched_names or "None",
                )

                # Update latest processed timestamp
                self.latest_processed_date = max(
                    photo.get("date", 0), self.latest_processed_date
                )

            except PyMongoError as e:
                logger.error(
                    "MongoDB error while inserting photo '%s': %s",
                    photo.get("filename"),
                    e,
                )
            except Exception as e:
                logger.error(
                    "Unexpected error processing '%s': %s", photo.get("filename"), e
                )

        # Persist last processed state
        self.save_last_processed_date()
        # Cleanup old data
        self.delete_old_faces(face_collection)

    # --- Cleanup ---
    def delete_old_faces(self, face_collection: Collection) -> None:
        """Delete old face documents from the collection."""
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
