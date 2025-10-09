"""Register a known face in MongoDB."""

import os
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
from image_processor.embeddings import EmbeddingEngine

# --- Setup logger ---
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
logger = logging.getLogger(__name__)

# --- Load environment ---
load_dotenv()

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_DB = os.getenv("MONGO_DB")
KNOWN_FACES_COLLECTION = os.getenv("KNOWN_FACES_COLLECTION", "nill-known-faces")

# --- Connect to MongoDB ---
client = MongoClient(MONGO_HOST)
db = client[MONGO_DB]
collection = db[KNOWN_FACES_COLLECTION]

# --- Prepare known face ---
engine = EmbeddingEngine()
logger.info("YOLOv8-face model loaded successfully.")

TEST_IMAGE_PATH = "tests/test.jpg"  # adjust if needed

# Read file as bytes
if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"❌ Test image not found: {TEST_IMAGE_PATH}")

with open(TEST_IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# Generate embeddings using your existing API
embeddings = engine.generate_face_embeddings(image_bytes)

# --- Debug output ---
logger.info(f"🔍 Raw embeddings keys: {list(embeddings.keys())}")

# Handle either string or bytes keys
KEY_EMBEDDING = None
if "face_embedding" in embeddings:
    KEY_EMBEDDING = "face_embedding"
elif b"face_embedding" in embeddings:
    KEY_EMBEDDING = b"face_embedding"

if KEY_EMBEDDING is None:
    raise ValueError(
        "No face embedding found in the test image. (Check path or YOLO output)"
    )

embedding_bytes = embeddings[KEY_EMBEDDING]
face_embedding = np.frombuffer(embedding_bytes, dtype=np.float32).tolist()


embedding_bytes = (
    embeddings[b"face_embedding"]
    if b"face_embedding" in embeddings
    else embeddings["face_embedding"]
)
face_embedding = np.frombuffer(embedding_bytes, dtype=np.float32).tolist()

logger.info("✅ Generated embeddings for 1 face(s).")
logger.info("✅ Found 3-D face embedding.")

# --- Prepare and insert document ---
document = {
    "name": "Danil",
    "embedding": face_embedding,
}

logger.info(f"Writing known face to db={db.name}, collection={collection.name}")

insert_result = collection.insert_one(document)
logger.info(f"✅ Inserted document with _id={insert_result.inserted_id}")
logger.info(
    f"✅ Registered known face for 'Danil' in collection '{KNOWN_FACES_COLLECTION}'."
)

# --- Optional verification ---
count = collection.count_documents({})
logger.info(f"📊 Collection now contains {count} documents.")
