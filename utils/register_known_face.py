"""Register or update a known face in MongoDB using the same embedding engine as runtime."""

import os
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
from image_processor.embeddings import EmbeddingEngine

# --- Setup logger ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
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

logger.info(f"🔗 Connected to MongoDB host: {MONGO_HOST}")
logger.info(f"📂 Using database: {db.name}")
logger.info(f"🧩 Using collection: {collection.name}")

# --- Initialize embedding engine ---
engine = EmbeddingEngine()
logger.info("YOLOv8-face model loaded successfully.")

# --- Path to known face image ---
NAME = "Danil"
IMAGE_PATH = "tests/test.jpg"  # adjust
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Test image not found: {IMAGE_PATH}")

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# --- Generate embedding ---
embeddings = engine.generate_face_embeddings(image_bytes)
if not embeddings or embeddings.get("face_count", 0) == 0:
    raise ValueError("No face detected in image. Check input photo.")

face_embedding = embeddings.get("face_embedding")
if not isinstance(face_embedding, (list, np.ndarray)):
    raise TypeError("Invalid embedding format returned from engine.")

embedding_vector = (
    np.array(face_embedding, dtype=np.float32) / np.linalg.norm(face_embedding)
).tolist()

logger.info(f"✅ Generated embedding vector of length {len(embedding_vector)}")

# --- Upsert (update or insert) ---
result = collection.update_one(
    {"name": NAME},
    {"$set": {"embedding": embedding_vector}},
    upsert=True,
)

if result.matched_count:
    logger.info(f"🔁 Updated existing face for '{NAME}'.")
else:
    logger.info(f"✅ Registered new known face for '{NAME}'.")

logger.info(f"📊 Collection now contains {collection.count_documents({})} known faces.")
