"""
Embedding utilities for CCTV photo search.

This module provides embedding generation and similarity search
for the search service.
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


class VisualEmbeddingEngine:
    """Enhanced embedding engine for CCTV photo analysis with multiple embedding types."""
    
    def __init__(self, 
                 text_model: str = "all-MiniLM-L6-v2",
                 clip_model: str = "openai/clip-vit-base-patch32") -> None:
        """
        Initialize the multi-modal embedding engine.
        
        Args:
            text_model: Text embedding model for semantic search
            clip_model: CLIP model for visual-text understanding
        """
        self.text_model = None
        self.clip_model = None
        
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

    def encode_visual_content(self, image_data: bytes) -> Optional[bytes]:
        """Generate visual content embedding using CLIP."""
        if not self.clip_model:
            logger.warning("CLIP model not available for visual encoding")
            return None
            
        try:
            # Convert bytes to PIL Image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return None
                
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

    def encode_face_embeddings(self, image_data: bytes) -> Dict[str, bytes]:
        """Generate face recognition embeddings for known faces."""
        try:
            # Convert bytes to OpenCV image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return {}
                
            # Use OpenCV for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            face_embeddings = {}
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Generate face embedding using CLIP (simplified approach)
                if self.clip_model:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    inputs = self.clip_processor(images=face_pil, return_tensors="pt")
                    with torch.no_grad():
                        face_features = self.clip_model.get_image_features(**inputs)
                        face_features = face_features / face_features.norm(dim=-1, keepdim=True)
                    
                    # Store face embedding
                    face_vector = np.asarray(face_features.squeeze(), dtype=np.float32)
                    face_embeddings[f'face_{i}'] = bytes(face_vector.tobytes())
                    
                    # Store face metadata
                    face_embeddings[f'face_{i}_bbox'] = bytes(np.array([x, y, w, h], dtype=np.float32).tobytes())
                    face_embeddings[f'face_{i}_confidence'] = bytes(np.array([1.0], dtype=np.float32).tobytes())
            
            return face_embeddings
            
        except Exception as error:
            logger.error(f"Failed to encode face embeddings: {error}")
            return {}

    def identify_faces(self, image_data: bytes, known_faces_db: Dict[str, bytes]) -> Dict[str, str]:
        """Identify faces in image against known faces database."""
        try:
            # Get face embeddings from current image
            current_faces = self.encode_face_embeddings(image_data)
            
            identifications = {}
            threshold = 0.7  # Similarity threshold for face recognition
            
            for face_id, face_embedding in current_faces.items():
                if not face_id.startswith('face_') or face_id.endswith('_bbox') or face_id.endswith('_confidence'):
                    continue
                    
                best_match = None
                best_similarity = 0.0
                
                # Compare with known faces
                for person_name, known_embedding in known_faces_db.items():
                    if self.text_model:
                        similarity = self.text_model.similarity(face_embedding, known_embedding)
                        if similarity > best_similarity and similarity > threshold:
                            best_similarity = similarity
                            best_match = person_name
                
                identifications[face_id] = best_match if best_match else 'unknown'
            
            return identifications
            
        except Exception as error:
            logger.error(f"Failed to identify faces: {error}")
            return {}

    def similarity(self, vec1_blob: bytes, vec2_blob: bytes) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.frombuffer(vec1_blob, dtype=np.float32)
            vec2 = np.frombuffer(vec2_blob, dtype=np.float32)
            return float(util.cos_sim(vec1, vec2))
        except Exception as error:
            logger.error(f"Error computing similarity: {error}")
            return 0.0


class FaceDatabaseManager:
    """Manage known faces database for recognition."""
    
    def __init__(self, visual_engine: VisualEmbeddingEngine):
        """Initialize face database manager."""
        self.visual_engine = visual_engine
        self.known_faces = {}
    
    def add_person(self, name: str, image_data: bytes) -> bool:
        """Add a person to the known faces database."""
        try:
            # Generate face embedding
            face_embeddings = self.visual_engine.encode_face_embeddings(image_data)
            
            if face_embeddings:
                # Use the first detected face
                for face_id, face_embedding in face_embeddings.items():
                    if face_id.startswith('face_') and not face_id.endswith('_bbox') and not face_id.endswith('_confidence'):
                        self.known_faces[name] = face_embedding
                        logger.info(f"Added {name} to known faces database")
                        return True
            
            logger.warning(f"No face detected in image for {name}")
            return False
            
        except Exception as error:
            logger.error(f"Failed to add {name} to face database: {error}")
            return False
    
    def get_known_faces(self) -> Dict[str, bytes]:
        """Get the known faces database."""
        return self.known_faces.copy()
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the known faces database."""
        if name in self.known_faces:
            del self.known_faces[name]
            logger.info(f"Removed {name} from known faces database")
            return True
        return False
