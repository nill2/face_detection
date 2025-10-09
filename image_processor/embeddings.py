"""Lightweight embedding engine using YOLO for face detection and embedding extraction with annotated output."""

import logging
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
from typing import Dict
import uuid

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified face embedding engine using YOLOv8-face."""

    def __init__(self, model_path: str = "yolov8n-face.pt") -> None:
        """Load YOLOv8-face model."""
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
        """Detect faces, generate embeddings, and save annotated image with rectangles."""
        embeddings: Dict[str, bytes] = {}

        if self.model is None:
            logger.error("YOLO model not available.")
            return embeddings

        try:
            # Decode image from bytes
            np_img = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Invalid image data.")
                return embeddings

            h_img, w_img = img.shape[:2]

            # Run YOLO inference
            results = self.model(img, verbose=False)
            if not results or not results[0].boxes:
                embeddings["face_count"] = np.array([0], dtype=np.float32).tobytes()
                return embeddings

            faces = []
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw red rectangle for visualization
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                face_crop = img[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    continue

                resized = cv2.resize(face_crop, (64, 64))
                faces.append(resized)

            if not faces:
                embeddings["face_count"] = np.array([0], dtype=np.float32).tobytes()
                return embeddings

            # Compute lightweight embedding
            faces_np = np.stack(faces)
            face_embedding = faces_np.mean(axis=(0, 1, 2)).astype(np.float32)
            embeddings["face_embedding"] = face_embedding.tobytes()
            embeddings["face_count"] = np.array(
                [len(faces_np)], dtype=np.float32
            ).tobytes()

            # Save annotated image for reference
            annotated_dir = Path("/app/processed_faces")
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = annotated_dir / f"faces_{uuid.uuid4().hex}.jpg"

            success = cv2.imwrite(str(annotated_path), img)
            if success:
                logger.info(
                    "✅ Generated embeddings for %d face(s). Annotated image saved to: %s",
                    len(faces_np),
                    annotated_path,
                )
            else:
                logger.warning("⚠️ Failed to save annotated image for this photo.")

            return embeddings

        except Exception as error:
            logger.error("Error generating face embeddings: %s", error)
            return {}
