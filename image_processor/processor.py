from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import numpy as np
import cv2  # OpenCV for face detection
import logging
from .config import FTP_USER, FTP_ROOT, FTP_PORT, MONGO_HOST, HOURS_KEEP  # pylint: disable=import-error
from .config import MONGO_PORT, MONGO_DB, MONGO_COLLECTION, FACE_COLLECTION , FTP_PASSWORD  # pylint: disable=import-error
from .config import ERROR_LVL, FTP_HOST, FTP_PASSIVE_PORT_FROM, FTP_PASSIVE_PORT_TO  # pylint: disable=import-error
from .config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME, USE_S3  # pylint: disable=import-error


# Configure the logger
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connect_to_mongodb(db_name, collection_name):
    """
    Connects to a MongoDB instance and returns a collection object.

    Args:
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        pymongo.collection.Collection: A MongoDB collection object.
    """
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT)
        db = client[db_name]
        collection = db[collection_name]
        if ERROR_LVL == "debug":
            logger.info(f"Connected to MongoDB: {MONGO_HOST}:{MONGO_PORT}/{db_name}/{collection_name}")
        return collection
    except ConnectionFailure as connection_error:
        logger.error(f"Failed to connect to MongoDB: {connection_error}")
        return None


def detect_faces(image_data):
    """
    Detects faces in the given image data using OpenCV.

    Args:
        image_data (bytes): The image data in bytes format.

    Returns:
        bool: True if at least one face is detected, False otherwise.
    """
    try:
        # Convert image data to a NumPy array and decode it
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Load the pre-trained Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Detect faces
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return len(faces) > 0  # Return True if at least one face is detected
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        return False


def process_photos():
    """
    Processes photos from the main MongoDB collection, checks for faces,
    and copies photos with faces to a separate MongoDB collection.
    """
    # Connect to the main and face collections
    main_collection = connect_to_mongodb(MONGO_DB, MONGO_COLLECTION)
    face_collection = connect_to_mongodb(MONGO_DB, FACE_COLLECTION)

    if main_collection is None or face_collection is None:
        logger.error("Failed to connect to MongoDB collections.")
        return

    # Retrieve photos from the main collection
    photos = main_collection.find({"data": {"$ne": ""}})
    logger.info("Processing photos from MongoDB...")

    for photo in photos:
        try:
            image_data = photo["data"]  # Image data in bytes

            # Check for faces in the image
            if detect_faces(image_data):
                # Copy the document to the face collection
                face_collection.insert_one({
                    "filename": photo["filename"],
                    "data": image_data,
                    "s3_file_url": photo.get("s3_file_url", ""),
                    "size": photo["size"],
                    "date": photo["date"],
                    "bsonTime": photo["bsonTime"]
                })
                logger.info(f"Face detected in {photo['filename']}. Photo copied to face collection.")
        except Exception as e:
            logger.error(f"Error processing photo {photo['filename']}: {e}")
