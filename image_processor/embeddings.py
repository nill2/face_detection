"""Lightweight embedding engine using YOLO for face detection and embedding extraction."""

import logging
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
from typing import Dict

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified face embedding engine using YOLOv8-face."""

    def __init__(self, model_path: str = "yolov8n-face.pt") -> None:
        """Load YOLOv8-face model immediately."""
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

            self.model = YOLO(str(local_model_path))
            logger.info("YOLOv8-face model loaded successfully.")

        except Exception as error:
            logger.error("Failed to load YOLO model: %s", error)
            self.model = None

    def generate_face_embeddings(self, image_data: bytes) -> Dict[str, bytes]:
        """Detect faces and generate compact embeddings from YOLO face boxes."""
        embeddings: Dict[str, bytes] = {}

        if self.model is None:
            logger.error("YOLO model not available.")
            return embeddings

        try:
            np_img = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Invalid image data.")
                return embeddings

            results = self.model(img, verbose=False)
            if not results or not results[0].boxes:
                embeddings["face_count"] = np.array([0], dtype=np.float32).tobytes()
                return embeddings

            boxes = results[0].boxes.xywh.cpu().numpy()
            faces = []
            for x, y, w, h in boxes:
                x1, y1, x2, y2 = (
                    int(x - w / 2),
                    int(y - h / 2),
                    int(x + w / 2),
                    int(y + h / 2),
                )
                face = img[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(cv2.resize(face, (64, 64)))

            if not faces:
                embeddings["face_count"] = np.array([0], dtype=np.float32).tobytes()
                return embeddings

            faces_np = np.stack(faces)
            face_embedding = faces_np.mean(axis=(0, 1, 2)).astype(np.float32)
            embeddings["face_embedding"] = face_embedding.tobytes()
            embeddings["face_count"] = np.array(
                [len(faces_np)], dtype=np.float32
            ).tobytes()

            return embeddings

        except Exception as error:
            logger.error("Error generating face embeddings: %s", error)
            return {}
