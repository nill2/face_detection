"""Lightweight embedding engine using YOLO for face detection and a pretrained model for face embeddings."""

import logging
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
from typing import Any, Dict
from facenet_pytorch import InceptionResnetV1
import torch

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Face embedding engine using YOLOv8-face for detection and Facenet for embeddings."""

    def __init__(self, model_path: str = "yolov8n-face.pt") -> None:
        """Load YOLOv8-face and Facenet models."""
        try:
            models_dir = Path(__file__).resolve().parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = models_dir / Path(model_path).name

            if not local_model_path.exists():
                import requests

                logger.info("Downloading YOLO face model...")
                url = (
                    "https://github.com/akanametov/yolov8-face/releases/download/"
                    "v0.0.0/yolov8n-face.pt"
                )
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(local_model_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                logger.info("Model downloaded to %s", local_model_path)

            # YOLO for detection
            self.model = YOLO(str(local_model_path))
            # Facenet for embeddings
            self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()

            logger.info("YOLOv8-face + Facenet models loaded successfully.")
        except Exception as error:
            logger.error("Failed to load models: %s", error)
            self.model = None
            self.embedding_model = None

    def generate_face_embeddings(self, image_data: bytes) -> Dict[str, Any]:
        """Detect faces and generate 512-dim embeddings."""
        embeddings: Dict[str, Any] = {}

        if self.model is None or self.embedding_model is None:
            logger.error("Models not available.")
            return embeddings

        try:
            np_img = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Invalid image data.")
                return embeddings

            h_img, w_img = img.shape[:2]
            results = self.model(img, verbose=False)

            if not results or not results[0].boxes:
                embeddings["face_count"] = 0
                return embeddings

            faces = []
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                face_crop = img[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    continue

                resized = cv2.resize(face_crop, (160, 160))
                # Preprocess for Facenet
                tensor = (
                    torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                )
                emb = self.embedding_model(tensor).detach().numpy()[0]
                faces.append(emb)

            if not faces:
                embeddings["face_count"] = 0
                return embeddings

            embeddings["face_embedding"] = (
                np.mean(faces, axis=0).astype(np.float32).tolist()
            )
            embeddings["face_count"] = len(faces)

            _, annotated_bytes = cv2.imencode(".jpg", img)
            embeddings["annotated_bytes"] = annotated_bytes.tobytes()

            logger.info(
                "✅ Generated %d embedding(s) of length %d.",
                len(faces),
                len(embeddings["face_embedding"]),
            )
            return embeddings

        except Exception as error:
            logger.error("Error generating face embeddings: %s", error)
            return {}
