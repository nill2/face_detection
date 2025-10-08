"""Unified embedding engine for detecting faces and generating embeddings efficiently."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class FaceEmbeddingEngine:
    """Unified engine combining YOLO face detection and FaceNet embeddings."""

    def __init__(self, model_path: str = "yolov8n-face.pt") -> None:
        """Initialize YOLO and FaceNet models."""
        try:
            self.face_detector = YOLO(model_path)
            logger.info("Loaded YOLO face detector: %s", model_path)
        except Exception as error:
            logger.error("Failed to load YOLO model: %s", error)
            self.face_detector = None

        try:
            self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()
            logger.info("Loaded FaceNet embedding model.")
        except Exception as error:
            logger.error("Failed to load FaceNet model: %s", error)
            self.embedding_model = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def generate_embeddings(self, image_data: bytes) -> Optional[List[Dict[str, Any]]]:
        """Detect faces and generate embeddings for each face."""
        if self.face_detector is None or self.embedding_model is None:
            logger.error("Models not initialized properly.")
            return None

        try:
            # Decode image
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Invalid image data.")
                return None

            # Run YOLO detection
            results = self.face_detector(image)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

            if len(boxes) == 0:
                logger.debug("No faces detected.")
                return None

            face_embeddings: List[Dict[str, Any]] = []

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                # Crop and preprocess face
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cropped_face = image[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                tensor_image = self.transform(pil_image).unsqueeze(0)

                with torch.no_grad():
                    embedding_tensor = self.embedding_model(tensor_image)
                    face_embedding = (
                        embedding_tensor.squeeze().numpy().astype(np.float32)
                    )

                face_embeddings.append(
                    {
                        "face_index": i,
                        "bbox": [x1, y1, x2, y2],
                        "embedding": face_embedding.tobytes(),
                    }
                )

            logger.info("Generated %d face embeddings.", len(face_embeddings))
            return face_embeddings

        except Exception as error:
            logger.error("Failed to generate embeddings: %s", error)
            return None
