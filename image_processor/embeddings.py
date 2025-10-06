"""Embedding engines for generating face and text embeddings for photo analysis."""

import logging
import numpy as np
import torch
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Legacy text embedding engine using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded SentenceTransformer model: %s", model_name)
        except Exception as error:
            logger.error("Failed to load SentenceTransformer model: %s", error)
            self.model = None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into embeddings."""
        if not self.model:
            logger.warning("SentenceTransformer model not loaded.")
            return None
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as error:
            logger.error("Text embedding failed: %s", error)
            return None


class EfficientEmbeddingEngine:
    """Enhanced embedding engine for efficient image and metadata processing."""

    def __init__(self) -> None:
        """Initialize FaceNet and preprocessing pipeline."""
        try:
            self.model = InceptionResnetV1(pretrained="vggface2").eval()
            logger.info("Loaded InceptionResnetV1 model for embeddings.")
        except Exception as error:
            logger.error("Failed to load InceptionResnetV1: %s", error)
            self.model = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def generate_search_embeddings(
        self,
        image_data: bytes,
        timestamp: int,
        filename: str,
        camera_location: str,
    ) -> Dict[str, bytes]:
        """Generate embeddings for face, time, and metadata to support RAG queries."""
        embeddings: Dict[str, bytes] = {}
        try:
            image = Image.open(np.frombuffer(image_data, np.uint8)).convert("RGB")
            tensor_image = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                embedding_tensor = self.model(tensor_image)
                face_embedding = embedding_tensor.squeeze().numpy().astype(np.float32)
                embeddings["face_embedding"] = face_embedding.tobytes()
                embeddings["face_count"] = np.array([1.0], dtype=np.float32).tobytes()
                embeddings["vehicle_score"] = np.array(
                    [0.0], dtype=np.float32
                ).tobytes()

            metadata_text = f"{filename} {camera_location} {timestamp}"
            metadata_embedding = np.mean([ord(c) for c in metadata_text]).astype(
                np.float32
            )
            embeddings["metadata_embedding"] = np.array(
                [metadata_embedding], dtype=np.float32
            ).tobytes()

            logger.debug("Generated search embeddings for %s", filename)
            return embeddings

        except Exception as error:
            logger.error("Failed to generate embeddings for %s: %s", filename, error)
            return {}
