"""Module for processing photos, detecting faces, and storing the results in MongoDB."""

import logging
import time
from pymongo import MongoClient, DESCENDING  # pylint: disable=E0401
from pymongo.errors import ConnectionFailure, PyMongoError  # pylint: disable=E0401
from pymongo.collection import Collection
import numpy as np  # pylint: disable=E0401
import cv2  # pylint: disable=E0401
from .config import (  # pylint: disable=E0402
    MONGO_HOST,
    MONGO_PORT,
    MONGO_DB,
    MONGO_COLLECTION,
    FACE_COLLECTION,
    ERROR_LVL,
    FACES_HISTORY_DAYS,
)

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhotoProcessor:
    """Process photos, detect faces using OpenCV, and store results in MongoDB."""

    def __init__(self) -> None:
        """Initialize the photo processor."""
        self.latest_processed_date: int = 0

    def connect_to_mongodb(self, db_name: str, collection_name: str) -> Collection:
        """
        Connect to MongoDB and return the collection.

        Arguments:
            db_name: The name of the database.
            collection_name: The name of the collection.

        Returns:
            The MongoDB collection object.
        """
        try:
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
        """
        Delete old photos from the specified collection based on the FACES_HISTORY_DAYS setting.

        Arguments:
            face_collection: The collection to delete photos from.
        """
        try:
            # Convert FACES_HISTORY_DAYS to seconds
            time_threshold = time.time() - (
                FACES_HISTORY_DAYS * 86400
            )  # Convert days to seconds

            # Construct query to delete documents with a date field older than the threshold
            query = {"date": {"$lt": time_threshold}}

            # Perform the deletion
            result = face_collection.delete_many(query)

            # Log the number of deleted documents
            logger.info("Deleted old photos from MongoDB: %s", result.deleted_count)

        except PyMongoError as error:
            logger.error("Error deleting old photos from MongoDB: %s", error)

    def detect_faces(self, image_data: bytes) -> bool:
        """
        Detect faces in the given image data using OpenCV.

        Arguments:
            image_data: The image data in byte format.

        Returns:
            True if faces are detected, False otherwise.
        """
        try:
            # Convert the image data to an image array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Ensure the image is valid
            if image is None or len(image.shape) != 3:
                logger.error("Invalid image data provided for face detection.")
                return False

            # Apply denoising and grayscale conversion
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            # Convert the image to grayscale before applying CLAHE
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                enhanced_image, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )

            return len(faces) > 0
        except cv2.error as error:
            logger.error("Error during face detection: %s", error)
            return False

    def process_photos(self) -> None:
        """Process photos from MongoDB, check for faces, and copy photos with faces to a separate collection."""
        if self.latest_processed_date is None:
            self.latest_processed_date = 0
            # Logic to set latest processed date from MongoDB

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
                    photo["filename"],
                    photo["bsonTime"],
                )
                image_data = photo["data"]

                if self.detect_faces(image_data):
                    face_collection.insert_one(
                        {
                            "filename": photo["filename"],
                            "data": image_data,
                            "s3_file_url": photo.get("s3_file_url", ""),
                            "size": photo["size"],
                            "date": photo["date"],
                            "bsonTime": photo["bsonTime"],
                        }
                    )
                    logger.info(
                        "Face detected in %s. Photo copied to face collection.",
                        photo["filename"],
                    )

                self.latest_processed_date = max(
                    photo["date"], self.latest_processed_date
                )

            except cv2.error as error:
                logger.error(
                    "Error with OpenCV while processing photo %s: %s",
                    photo["filename"],
                    error,
                )
            except (
                PyMongoError
            ) as error:  # Using the imported PyMongoError for MongoDB-related exceptions
                logger.error(
                    "MongoDB error while processing photo %s: %s",
                    photo["filename"],
                    error,
                )
            except Exception as error:  # Handle any other exceptions
                logger.error(
                    "Unexpected error processing photo %s: %s", photo["filename"], error
                )

        self.delete_old_faces(face_collection)
