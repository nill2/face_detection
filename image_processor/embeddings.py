"""
Efficient embedding utilities for CCTV photo analysis and search.

This module provides optimized embeddings for:
- Face detection and recognition
- Vehicle detection (cars, motorcycles)
- Temporal context (when photos were taken)
- Scene analysis (indoor/outdoor, lighting)
- Search optimization
"""

import logging
import time
from datetime import datetime
from typing import Union, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class EfficientEmbeddingEngine:
    """Efficient embedding engine optimized for CCTV photo search."""
    
    def __init__(self, 
                 text_model: str = "all-MiniLM-L6-v2",
                 clip_model: str = "openai/clip-vit-base-patch32") -> None:
        """
        Initialize the efficient embedding engine.
        
        Args:
            text_model: Text embedding model for semantic search
            clip_model: CLIP model for visual understanding
        """
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # Load text model
        try:
            self.text_model = SentenceTransformer(text_model)
            logger.info(f"Loaded text model: {text_model}")
        except Exception as error:
            logger.error(f"Failed to load text model {text_model}: {error}")
            
        # Load CLIP model for visual understanding
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained(clip_model)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            logger.info(f"Loaded CLIP model: {clip_model}")
        except Exception as error:
            logger.warning(f"Failed to load CLIP model {clip_model}: {error}")

    def generate_search_embeddings(self, 
                                 image_data: bytes, 
                                 timestamp: float,
                                 filename: str = "",
                                 camera_location: str = "") -> Dict[str, bytes]:
        """
        Generate optimized embeddings for efficient search.
        
        Args:
            image_data: Raw image bytes
            timestamp: Unix timestamp
            filename: Image filename
            camera_location: Camera location identifier
            
        Returns:
            Dict containing optimized embeddings
        """
        embeddings = {}
        
        try:
            # Convert bytes to OpenCV image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return {}
            
            # 1. Visual content embedding (CLIP-based)
            visual_emb = self._encode_visual_content(image)
            if visual_emb is not None:
                embeddings['visual'] = visual_emb
            
            # 2. Temporal embedding (time-based features)
            temporal_emb = self._encode_temporal_context(timestamp, filename)
            embeddings['temporal'] = temporal_emb
            
            # 3. Scene embedding (lighting, indoor/outdoor, quality)
            scene_emb = self._encode_scene_context(image, camera_location)
            embeddings['scene'] = scene_emb
            
            # 4. Object detection embedding (faces, vehicles)
            object_emb = self._encode_objects(image)
            if object_emb:
                embeddings.update(object_emb)
            
            # 5. Search-optimized text embedding
            search_text = self._generate_search_text(image, timestamp, filename, camera_location)
            if self.text_model and search_text:
                text_emb = self.text_model.encode(search_text)
                embeddings['search_text'] = bytes(text_emb.tobytes())
            
            logger.info(f"Generated {len(embeddings)} embedding types for {filename}")
            return embeddings
            
        except Exception as error:
            logger.error(f"Failed to generate embeddings: {error}")
            return {}

    def _encode_visual_content(self, image: np.ndarray) -> Optional[bytes]:
        """Generate visual content embedding using CLIP."""
        if not self.clip_model:
            return None
            
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Generate CLIP embedding
            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            # Convert to numpy and serialize
            vector = np.asarray(image_features.squeeze(), dtype=np.float32)
            return bytes(vector.tobytes())
            
        except Exception as error:
            logger.error(f"Failed to encode visual content: {error}")
            return None

    def _encode_temporal_context(self, timestamp: float, filename: str = "") -> bytes:
        """Generate efficient temporal context embedding."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            
            # Extract key temporal features
            features = [
                dt.hour / 24.0,  # Hour of day (0-1)
                dt.weekday() / 7.0,  # Day of week (0-1)
                dt.day / 31.0,  # Day of month (0-1)
                dt.month / 12.0,  # Month (0-1)
                # Time periods (binary flags)
                int(dt.hour < 6 or dt.hour > 22),  # Night
                int(6 <= dt.hour < 12),  # Morning
                int(12 <= dt.hour < 18),  # Afternoon
                int(18 <= dt.hour < 22),  # Evening
                int(dt.weekday() < 5),  # Weekday
            ]
            
            # Add filename-based context if available
            if filename and "_" in filename:
                try:
                    parts = filename.split("_")
                    for part in parts:
                        if part.isdigit() and len(part) >= 8:
                            file_timestamp = int(part[:10])
                            file_dt = datetime.fromtimestamp(file_timestamp)
                            features.extend([
                                file_dt.hour / 24.0,
                                file_dt.weekday() / 7.0,
                            ])
                            break
                except:
                    pass
            
            vector = np.asarray(features, dtype=np.float32)
            return bytes(vector.tobytes())
            
        except Exception as error:
            logger.error(f"Failed to encode temporal context: {error}")
            # Return default temporal embedding
            default_features = [0.5] * 9
            vector = np.asarray(default_features, dtype=np.float32)
            return bytes(vector.tobytes())

    def _encode_scene_context(self, image: np.ndarray, camera_location: str = "") -> bytes:
        """Generate scene context embedding."""
        try:
            features = []
            
            # Lighting analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            features.extend([brightness, contrast])
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            dominant_hue = np.mean(hsv[:, :, 0]) / 180.0
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            features.extend([dominant_hue, saturation])
            
            # Image quality
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = 1.0 / (1.0 + laplacian_var / 1000.0)
            features.append(blur_score)
            
            # Indoor/outdoor detection
            blue_ratio = np.mean(image[:, :, 0]) / (np.mean(image) + 1e-6)
            is_outdoor = 1.0 if blue_ratio > 0.4 else 0.0
            features.append(is_outdoor)
            
            # Camera location encoding
            if camera_location:
                location_hash = hash(camera_location) % 1000 / 1000.0
                features.append(location_hash)
            else:
                features.append(0.0)
            
            vector = np.asarray(features, dtype=np.float32)
            return bytes(vector.tobytes())
            
        except Exception as error:
            logger.error(f"Failed to encode scene context: {error}")
            # Return default scene embedding
            default_features = [0.5] * 7
            vector = np.asarray(default_features, dtype=np.float32)
            return bytes(vector.tobytes())

    def _encode_objects(self, image: np.ndarray) -> Dict[str, bytes]:
        """Generate object detection embeddings (faces, vehicles)."""
        embeddings = {}
        
        try:
            # Face detection using OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Face embeddings
            if len(faces) > 0:
                embeddings['face_count'] = bytes(np.array([len(faces)], dtype=np.float32).tobytes())
                
                # Generate face region embeddings
                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = image[y:y+h, x:x+w]
                    
                    # Simple face feature extraction
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (64, 64))
                    face_features = face_resized.flatten() / 255.0
                    
                    embeddings[f'face_{i}'] = bytes(face_features.astype(np.float32).tobytes())
                    embeddings[f'face_{i}_bbox'] = bytes(np.array([x, y, w, h], dtype=np.float32).tobytes())
            else:
                embeddings['face_count'] = bytes(np.array([0], dtype=np.float32).tobytes())
            
            # Vehicle detection (simplified using color and shape analysis)
            vehicle_score = self._detect_vehicles(image)
            embeddings['vehicle_score'] = bytes(np.array([vehicle_score], dtype=np.float32).tobytes())
            
            return embeddings
            
        except Exception as error:
            logger.error(f"Failed to encode objects: {error}")
            return {}

    def _detect_vehicles(self, image: np.ndarray) -> float:
        """Simple vehicle detection based on color and shape analysis."""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Look for car-like colors (white, black, gray, blue, red)
            car_colors = [
                # White cars
                ([0, 0, 200], [180, 30, 255]),
                # Black cars
                ([0, 0, 0], [180, 255, 50]),
                # Gray cars
                ([0, 0, 50], [180, 30, 200]),
                # Blue cars
                ([100, 50, 50], [130, 255, 255]),
                # Red cars
                ([0, 50, 50], [10, 255, 255]),
            ]
            
            vehicle_pixels = 0
            total_pixels = image.shape[0] * image.shape[1]
            
            for lower, upper in car_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                vehicle_pixels += np.sum(mask > 0)
            
            # Calculate vehicle score
            vehicle_ratio = vehicle_pixels / total_pixels
            return min(vehicle_ratio * 10, 1.0)  # Normalize to 0-1
            
        except Exception as error:
            logger.error(f"Failed to detect vehicles: {error}")
            return 0.0

    def _generate_search_text(self, image: np.ndarray, timestamp: float, 
                            filename: str, camera_location: str) -> str:
        """Generate search-optimized text description."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            
            # Basic scene analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            # Generate descriptive text
            text_parts = []
            
            # Time context
            if dt.hour < 6 or dt.hour > 22:
                text_parts.append("night time")
            elif 6 <= dt.hour < 12:
                text_parts.append("morning")
            elif 12 <= dt.hour < 18:
                text_parts.append("afternoon")
            else:
                text_parts.append("evening")
            
            # Lighting context
            if brightness < 0.3:
                text_parts.append("dark")
            elif brightness > 0.7:
                text_parts.append("bright")
            else:
                text_parts.append("normal lighting")
            
            # Location context
            if camera_location:
                text_parts.append(f"camera {camera_location}")
            
            # Day context
            if dt.weekday() < 5:
                text_parts.append("weekday")
            else:
                text_parts.append("weekend")
            
            return " ".join(text_parts)
            
        except Exception as error:
            logger.error(f"Failed to generate search text: {error}")
            return ""

    def similarity(self, vec1_blob: bytes, vec2_blob: bytes) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.frombuffer(vec1_blob, dtype=np.float32)
            vec2 = np.frombuffer(vec2_blob, dtype=np.float32)
            return float(util.cos_sim(vec1, vec2))
        except Exception as error:
            logger.error(f"Error computing similarity: {error}")
            return 0.0

    def find_similar_photos(self, query_embeddings: Dict[str, bytes], 
                          candidate_embeddings: List[Dict[str, bytes]], 
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """Find similar photos based on embeddings."""
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                total_score = 0.0
                weight_sum = 0.0
                
                # Compare each embedding type
                for emb_type, query_emb in query_embeddings.items():
                    if emb_type in candidate:
                        similarity = self.similarity(query_emb, candidate[emb_type])
                        
                        # Weight different embedding types
                        if emb_type == 'visual':
                            weight = 0.4
                        elif emb_type == 'temporal':
                            weight = 0.3
                        elif emb_type == 'scene':
                            weight = 0.2
                        else:
                            weight = 0.1
                        
                        total_score += similarity * weight
                        weight_sum += weight
                
                if weight_sum > 0:
                    final_score = total_score / weight_sum
                    similarities.append((i, final_score))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as error:
            logger.error(f"Error finding similar photos: {error}")
            return []


class EmbeddingEngine:
    """Legacy wrapper for text-based semantic search (maintained for compatibility)."""

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
