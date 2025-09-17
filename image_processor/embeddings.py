"""
Embedding utilities for semantic search.

This module provides a wrapper around sentence-transformers
to compute embeddings and similarity for natural language queries.
"""

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Wrapper for encoding text and computing similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding engine.

        Args:
            model_name (str): Pretrained model from HuggingFace.

        Raises:
            Exception: If model loading fails.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as error:
            logger.error(f"Failed to load model {model_name}: {error}")
            raise

    def encode(self, text: Union[str, list[str]]) -> Union[bytes, list[bytes]]:
        """
        Encode text into vector embedding(s).

        Args:
            text: Input text or list of texts.

        Returns:
            bytes or list[bytes]: Serialized numpy array(s) for storage.

        Raises:
            ValueError: If text is empty or invalid.
        """
        if not text or (isinstance(text, str) and not text.strip()):
            raise ValueError("Text input cannot be empty")

        if isinstance(text, str):
            # Single text encoding
            vector = np.asarray(self.model.encode(text), dtype=np.float32)
            return bytes(vector.tobytes())

        # Batch encoding
        return self.batch_encode(text)

    def decode(self, blob: bytes) -> np.ndarray:
        """
        Decode a BLOB back into a numpy array.

        Args:
            blob (bytes): Stored BLOB.

        Returns:
            np.ndarray: Reconstructed embedding.

        Raises:
            ValueError: If blob is invalid or corrupted.
        """
        if not blob:
            raise ValueError("Blob cannot be empty")

        try:
            return np.frombuffer(blob, dtype=np.float32)
        except Exception as error:
            raise ValueError(f"Failed to decode blob: {error}") from error

    def similarity(self, vec1_blob: bytes, vec2_blob: bytes) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            vec1_blob (bytes): First embedding (BLOB).
            vec2_blob (bytes): Second embedding (BLOB).

        Returns:
            float: Cosine similarity score.
        """
        vec1 = self.decode(vec1_blob)
        vec2 = self.decode(vec2_blob)
        return float(util.cos_sim(vec1, vec2))

    def batch_encode(self, texts: list[str], chunk_size: int = 100) -> list[bytes]:
        """
        Encode multiple texts efficiently in batches.

        Args:
            texts: List of input texts.
            chunk_size: Process texts in chunks for memory efficiency.

        Returns:
            list[bytes]: List of serialized embeddings.

        Raises:
            ValueError: If texts list is empty or contains invalid items.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Validate all texts are non-empty strings
        invalid_texts = [
            i
            for i, text in enumerate(texts)
            if not isinstance(text, str) or not text.strip()
        ]
        if invalid_texts:
            raise ValueError(f"Invalid or empty texts at indices: {invalid_texts}")

        all_embeddings = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            vectors = self.model.encode(chunk, convert_to_numpy=True)
            chunk_embeddings = [
                bytes(np.asarray(vector, dtype=np.float32).tobytes())
                for vector in vectors
            ]
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    def find_most_similar(
        self,
        query_blob: bytes,
        candidate_blobs: list[bytes],
        top_k: int = 1,
    ) -> Union[tuple[int, float], list[tuple[int, float]]]:
        """
        Find the most similar embedding(s) from a list of candidates.

        Args:
            query_blob: Query embedding as bytes.
            candidate_blobs: List of candidate embeddings as bytes.
            top_k: Number of top matches to return.

        Returns:
            tuple[int, float] or list[tuple[int, float]]:
            Index and similarity score(s) of best match(es).

        Raises:
            ValueError: If inputs are invalid or empty.
        """
        if not candidate_blobs:
            raise ValueError("Candidate blobs list cannot be empty")

        if top_k < 1 or top_k > len(candidate_blobs):
            raise ValueError(f"top_k must be between 1 and {len(candidate_blobs)}")

        query_vec = self.decode(query_blob)
        candidate_vecs = [self.decode(blob) for blob in candidate_blobs]

        similarities = util.cos_sim(query_vec, candidate_vecs)[0]

        if top_k == 1:
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            return best_idx, best_score

        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
