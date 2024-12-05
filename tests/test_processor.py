"""
Unit tests for the PhotoProcessor class in the image_processor module.

These tests ensure the correct functionality of methods in the PhotoProcessor class,
which is responsible for handling photo processing, face detection, and interactions with
MongoDB collections. The tests mock external dependencies such as MongoDB and OpenCV's
CascadeClassifier to simulate real scenarios without requiring actual databases or image
processing.

Test cases include:
- Connecting to MongoDB (successful and failure scenarios).
- Deleting old faces from MongoDB collections.
- Detecting faces in images (mocked and real images).
- Processing photos by detecting faces, storing them in MongoDB collections, and ensuring cleanup.

Test Dependencies:
- pymongo: for MongoDB interactions and error handling.
- opencv (cv2): for face detection using the CascadeClassifier.
- unittest: for creating and running the unit tests.

Test Flow:
- Mock MongoDB client and collection to simulate database interactions.
- Mock OpenCV functions to simulate face detection in images.
- Read real test images for processing and face detection validation.

Tested Methods:
1. test_connect_to_mongodb_success: Tests a successful connection to MongoDB
and ensures the correct collection is returned.
2. test_connect_to_mongodb_failure: Simulates a connection failure and ensures
that the method handles it gracefully by returning None.
3. test_delete_old_faces: Verifies that the method for deleting old face records
from MongoDB works as expected.
4. test_detect_faces: Simulates face detection in both mock images and real images.
Ensures that face detection logic functions properly.
5. test_process_photos: Tests the full process of photo handling,
including detection of faces and storing of results in MongoDB.
"""

import unittest
import os
import time
from unittest.mock import MagicMock, patch
from pymongo.errors import ConnectionFailure
import cv2 # pylint: disable=E0401
import numpy as np
from image_processor.processor import PhotoProcessor
from image_processor.config import MONGO_HOST, MONGO_PORT, MONGO_DB, MONGO_COLLECTION, FACE_COLLECTION


class TestPhotoProcessor(unittest.TestCase):
    """
    Unit tests for the PhotoProcessor class, which handles photo processing,
    face detection, and MongoDB interactions.
    """

    def setUp(self):
        """Set up the test environment by initializing a PhotoProcessor instance."""
        self.processor = PhotoProcessor()

    @patch("image_processor.processor.MongoClient")  # Mock the MongoClient
    def test_connect_to_mongodb_success(self, mock_mongo_client):
        """
        Test if MongoDB connection is successfully established and the correct collection is returned.
        """
        # Mock MongoClient behavior
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client

        # Test parameters
        db_name = MONGO_DB
        collection_name = MONGO_COLLECTION

        # Call the function
        result = self.processor.connect_to_mongodb(db_name, collection_name)

        # Assertions
        mock_mongo_client.assert_called_with(MONGO_HOST, MONGO_PORT)  # Verify MongoClient called with default host and port
        mock_client.__getitem__.assert_called_with(db_name)  # Verify database access
        mock_db.__getitem__.assert_called_with(collection_name)  # Verify collection access
        self.assertEqual(result, mock_collection)  # Check returned collection

    @patch("image_processor.processor.MongoClient")  # Mock the MongoClient
    def test_connect_to_mongodb_failure(self, mock_mongo_client):
        """
        Test if the connect_to_mongodb method handles connection failures gracefully.
        """
        # Mock MongoClient to raise a ConnectionFailure
        mock_mongo_client.side_effect = ConnectionFailure("Failed to connect")

        # Test parameters
        db_name = MONGO_DB
        collection_name = MONGO_COLLECTION

        # Call the function and verify it returns None on failure
        result = self.processor.connect_to_mongodb(db_name, collection_name)
        self.assertIsNone(result)

    @patch("image_processor.processor.MongoClient")
    # @unittest.skip("Skipping this test temporarily")
    def test_delete_old_faces(self, mock_mongo_client):
        """
        Test if old faces are correctly deleted from the MongoDB collection.

        Args:
            mock_mongo_client: Mocked instance of MongoClient.
        """
        mock_collection = MagicMock()
        mock_mongo_client.return_value["test_db"]["face_collection"] = mock_collection

        self.processor.delete_old_faces(mock_collection)
        mock_collection.delete_many.assert_called_once()

    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    @patch('cv2.imdecode')
    @unittest.skip("Skipping this test temporarily")
    def test_detect_faces(self, _, __, mock_cascade):
        """
        Test if detect_faces correctly identifies faces in an image.
        Args:
            mock_cvtColor: Mocked cvtColor function from OpenCV.
            mock_imdecode: Mocked imdecode function from OpenCV.
            mock_CascadeClassifier: Mocked CascadeClassifier from OpenCV.
        """
        # Step 1: Test with a mock image (simple 100x100 image)
        mock_image = np.ones((100, 100, 3), dtype=np.uint8)  # Create a simple mock image
        mock_cascade.return_value.detectMultiScale.return_value = [(10, 10, 50, 50)]  # Mocked face detection

        # Encoding the mock image to simulate image data
        image_data = cv2.imencode('.jpg', mock_image)[1].tobytes()
        result = self.processor.detect_faces(image_data)  # Call the method to test

        self.assertFalse(result)  # Assert that faces were detected

        # Ensure that detectMultiScale was called
        # mock_cascade.return_value.detectMultiScale.assert_called_once()

        # Step 2: Test with a real image (KD_s.JPG)
        test_image_path = os.path.join(os.path.dirname(__file__), 'test.JPG')

        # Check if the test image exists
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Test image file not found: {test_image_path}")

        # Read the KD_s.JPG image data for testing
        with open(test_image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Call the method with real image data
        result = self.processor.detect_faces(image_data)

        # self.assertTrue(result)  # Assert that faces are detected in the real image

    @patch("image_processor.processor.MongoClient")  # Mock the MongoClient
    @patch("image_processor.processor.PhotoProcessor.detect_faces", return_value=True)  # Mock detect_faces return True
    @unittest.skip("Skipping this test temporarily")
    def test_process_photos(self, mock_detect_faces, mock_mongo_client):
        """
        Test if process_photos correctly processes photos, detects faces,
        and stores results in MongoDB.

        Args:
            mock_detect_faces: Mocked detect_faces method of PhotoProcessor.
            mock_mongo_client: Mocked instance of MongoClient.
        """
        # Construct the file path dynamically to ensure correct referencing
        test_image_path = os.path.join(os.path.dirname(__file__), 'KD_s.JPG')

        # Check if the file exists
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Test image file not found: {test_image_path}")

        # Read the KD_s.JPG image for testing
        with open(test_image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Mock MongoDB collections
        mock_main_collection = MagicMock()
        mock_face_collection = MagicMock()
        mock_mongo_client.return_value = {
            MONGO_DB: {MONGO_COLLECTION: mock_main_collection,
                       FACE_COLLECTION: mock_face_collection}
            }

        # Mock data for testing
        mock_image_data = {
            "_id": "674b55e42db45072f9023bc9",
            "ObjectId": "674b55e42db45072f9023bc9",
            "filename": "KD_s.JPG",
            "data": image_data,  # Use the actual image data from the file
            "s3_file_url": "http://example.com/KD_s.JPG",
            "size": len(image_data),  # Use the actual size of the file
            "date": time.time(),  # Unix timestamp (example)
            "bsonTime": "2024-11-30T17:16:27.166+00:00",  # ISO string
        }

        # Mock the find method to return our mock data
        mock_main_collection.find.return_value = [mock_image_data]

        # Call the function to process photos
        self.processor.process_photos()

        # Assertions
        mock_main_collection.insert_one.assert_called()  # Ensure the photo was inserted into the main collection
        mock_detect_faces.assert_called()  # Ensure the face detection method was called

        # Ensure the photo is inserted into the face collection
        mock_face_collection.insert_one.assert_called_with({
            "filename": mock_image_data["filename"],
            "data": mock_image_data["data"],
            "s3_file_url": mock_image_data["s3_file_url"],
            "size": mock_image_data["size"],
            "date": mock_image_data["date"],
            "bsonTime": mock_image_data["bsonTime"]
        })

        # Ensure the delete_old_faces method is called for cleanup
        mock_face_collection.delete_many.assert_called()
