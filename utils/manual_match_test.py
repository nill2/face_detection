"""Manually compare a specific CCTV face from MongoDB with the known face embedding."""

import logging
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from image_processor.embeddings import EmbeddingEngine
import base64
import os

# --- Setup logger ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("manual_match_test")

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_HOST", "localhost")
DB_NAME = "nill-home"
KNOWN_COLLECTION = "nill-known-faces"
CCTV_COLLECTION = "nill-home-faces"
PERSON_NAME = "Danil"
TARGET_FILENAME = "000DC5DB8A94()_0_20251011081632_8691.jpg"  # "000DC5DB8A94()_0_20251011081652_8692.jpg" # "000DC5DB8A94()_0_20251011081712_8693.jpg"

# --- Connect to MongoDB ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
known_col = db[KNOWN_COLLECTION]
cctv_col = db[CCTV_COLLECTION]

logger.info("Connected to MongoDB database: %s", DB_NAME)

# --- Load specific CCTV document by filename ---
cctv_doc = cctv_col.find_one({"filename": TARGET_FILENAME})
if not cctv_doc:
    raise ValueError(
        f"❌ No CCTV document found for filename '{TARGET_FILENAME}' in collection '{CCTV_COLLECTION}'."
    )

logger.info(
    "Loaded CCTV image: %s (%.1f KB)",
    cctv_doc.get("filename"),
    cctv_doc.get("size", 0) / 1024,
)

# --- Extract raw image bytes ---
binary_data = cctv_doc["data"]
if isinstance(binary_data, dict) and "$binary" in binary_data:
    image_bytes = base64.b64decode(binary_data["$binary"]["base64"])
else:
    image_bytes = binary_data

# --- Generate embedding from CCTV image ---
engine = EmbeddingEngine()
cctv_embeddings = engine.generate_face_embeddings(image_bytes)

if not cctv_embeddings or not cctv_embeddings.get("face_embedding"):
    raise ValueError("❌ No faces detected in CCTV image.")

cctv_embedding = np.array(cctv_embeddings["face_embedding"], dtype=np.float32)
logger.info(
    "✅ CCTV embedding generated: len=%d mean=%.4f",
    len(cctv_embedding),
    cctv_embedding.mean(),
)

# --- Load known face from MongoDB ---
known_doc = known_col.find_one({"name": PERSON_NAME})
if not known_doc:
    raise ValueError(
        f"❌ No known face found for '{PERSON_NAME}' in {KNOWN_COLLECTION}."
    )

known_embedding = np.array(known_doc["embedding"], dtype=np.float32)
logger.info(
    "✅ Known embedding loaded: len=%d mean=%.4f",
    len(known_embedding),
    known_embedding.mean(),
)


# --- Normalize & Compare ---
def normalize(vec):
    """Manual normalizaion"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


sim = cosine_similarity([normalize(known_embedding)], [normalize(cctv_embedding)])[0][0]
logger.info("📈 Cosine similarity: %.4f", sim)

# --- Match Decision ---
THRESHOLD = 0.6
if sim >= THRESHOLD:
    logger.info("✅ MATCHED: %s (similarity=%.4f ≥ %.2f)", PERSON_NAME, sim, THRESHOLD)
else:
    logger.warning("❌ NOT MATCHED (similarity=%.4f < %.2f)", sim, THRESHOLD)
