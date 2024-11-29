"""
Module for processing photos, detecting faces, and storing the results in MongoDB.
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import numpy as np  # pylint: disable=E0401
import cv2  # pylint: disable=E0401
from .config import MONGO_HOST, MONGO_PORT, MONGO_DB, MONGO_COLLECTION, FACE_COLLECTION, ERROR_LVL

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PhotoProcessor:
    """
    Class responsible for processing photos, detecting faces using OpenCV,
    and storing the results in a MongoDB database.
    """

    def __init__(self):
        self.latest_processed_date = None

    def connect_to_mongodb(self, db_name, collection_name):
        """
        Connects to a MongoDB instance and returns a collection object.
        """
        try:
            client = MongoClient(MONGO_HOST, MONGO_PORT)
            db = client[db_name]
            collection = db[collection_name]
            if ERROR_LVL == "debug":
                logger.info("Connected to MongoDB: %s:%s/%s/%s", MONGO_HOST, MONGO_PORT, db_name, collection_name)
            return collection
        except ConnectionFailure as connection_error:
            logger.error("Failed to connect to MongoDB: %s", connection_error)
            return None

    def detect_faces(self, image_data):
        """
        Detects faces in the given image data using OpenCV.
        Returns True if faces are detected, False otherwise.
        """
        try:
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            return len(faces) > 0
        except cv2.error as e:
            logger.error("Error during face detection: %s", e)
            return False

    def process_photos(self):
        """
        Processes photos from the main MongoDB collection, checks for faces,
        and copies photos with faces to a separate MongoDB collection.
        """
        if self.latest_processed_date is None:
            self.latest_processed_date = 0
            # Logic to set latest processed date from MongoDB

        main_collection = self.connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
        face_collection = self.connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

        if main_collection is None or face_collection is None:
            logger.error("Failed to connect to MongoDB collections.")
            return

        query = {"date": {"$gt": self.latest_processed_date}}
        photos = list(main_collection.find(query).sort("date", -1))

        for photo in photos:
            try:
                logger.info("Processing photo: %s with date %s", photo["filename"], photo["date"])
                image_data = photo["data"]

                if self.detect_faces(image_data):
                    face_collection.insert_one({
                        "filename": photo["filename"],
                        "data": image_data,
                        "s3_file_url": photo.get("s3_file_url", ""),
                        "size": photo["size"],
                        "date": photo["date"],
                        "bsonTime": photo["bsonTime"]
                    })
                    logger.info("Face detected in %s. Photo copied to face collection.", photo["filename"])

                self.latest_processed_date = max(photo["date"], self.latest_processed_date)

            except cv2.error as e:
                logger.error("Error with OpenCV while processing photo %s: %s", photo["filename"], e)
            except PyMongoError as e:  # Using the imported PyMongoError for MongoDB-related exceptions
                logger.error("MongoDB error while processing photo %s: %s", photo["filename"], e)
            except Exception as e:  # pylint: disable=W0718
                logger.error("Unexpected error processing photo %s: %s", photo["filename"], e)
