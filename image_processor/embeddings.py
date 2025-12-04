"""Improved face embedding engine with higher-accuracy face detection (YOLOv8-s-face)."""

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Face embedding engine using YOLOv8-s-face for detection and Facenet for embeddings.

    This class provides:
    - model loading (YOLOv8-s-face + Facenet)
    - detection with confidence & size filtering
    - non-max suppression (NMS)
    - batched embedding extraction
    """

    def __init__(
        self,
        model_path: str = "yolov8s-face.pt",
        conf_threshold: float = 0.60,
        min_face_size: int = 50,
        iou_threshold: float = 0.4,
    ) -> None:
        """Initialize models and thresholds.

        Args:
            model_path: local filename for the YOLO model (downloaded to ./models if missing)
            conf_threshold: minimum detection confidence to accept a box
            min_face_size: minimum width/height (px) for a face box
            iou_threshold: NMS IOU threshold
        """
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self.iou_threshold = iou_threshold

        try:
            models_dir = Path(__file__).resolve().parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            # Correct filename from user.
            local_model_path = models_dir / "yolov11s-face.pt"

            # Correct download URL.
            download_url = (
                "https://github.com/YapaLab/yolo-face/releases/download/"
                "v0.0.0/yolov11s-face.pt"
            )

            if not local_model_path.exists():
                import requests

                logger.info("Downloading YOLOv11s-face model...")
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()

                with open(local_model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info("YOLOv11s-face model downloaded.")

            # Load YOLO model (this must succeed now)
            self.model = YOLO(str(local_model_path))

            # Load Facenet
            self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval()

            logger.info("YOLOv11s-face + Facenet loaded successfully.")

        except Exception as error:
            logger.error("Failed to load models: %s", error)
            self.model = None
            self.embedding_model = None

    # -----------------------------
    #       Non-max suppression
    # -----------------------------
    def non_max_suppression(self, boxes: List[List[float]]) -> List[List[float]]:
        """Apply non-max suppression on bounding boxes.

        Args:
            boxes: List of [x1, y1, x2, y2, score].

        Returns:
            List of kept boxes as lists of floats.
        """
        if len(boxes) == 0:
            return []

        np_boxes = np.array(boxes, dtype=float)
        x1, y1, x2, y2 = np_boxes[:, 0], np_boxes[:, 1], np_boxes[:, 2], np_boxes[:, 3]
        scores = np_boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union

            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        result = np_boxes[keep].astype(float).tolist()
        return cast(List[List[float]], result)

    # -----------------------------
    #       Main method
    # -----------------------------
    def generate_face_embeddings(self, image_data: bytes) -> Dict[str, Any]:
        """Detect faces in the provided JPEG bytes and return embeddings + annotated image.

        Returns dict with keys:
            - face_count: int
            - face_embedding: averaged embedding (list[float]) if any
            - annotated_bytes: jpeg bytes with drawn boxes
        """
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

            h, w = img.shape[:2]

            # Optional sharpening to help detector on some inputs
            blur = cv2.GaussianBlur(img, (0, 0), 3)
            img_sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

            results = self.model(img_sharp, verbose=False)

            if not results or not results[0].boxes:
                embeddings["face_count"] = 0
                return embeddings

            raw_boxes: List[List[float]] = []
            # results[0].boxes.data is an ndarray Nx6 (x1,y1,x2,y2,score,class)
            for b in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, score = (
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]),
                    float(b[4]),
                )

                if score < float(self.conf_threshold):
                    continue

                width = x2 - x1
                height = y2 - y1
                if width < float(self.min_face_size) or height < float(
                    self.min_face_size
                ):
                    continue

                raw_boxes.append([x1, y1, x2, y2, score])

            # Apply NMS
            boxes = self.non_max_suppression(raw_boxes)

            if not boxes:
                embeddings["face_count"] = 0
                return embeddings

            face_tensors: List[torch.Tensor] = []
            valid_boxes: List[List[int]] = []

            for bx in boxes:
                # bx is [x1, y1, x2, y2, score]
                x1, y1, x2, y2, _score = bx  # _score intentionally unused afterwards
                x1_i, y1_i, x2_i, y2_i = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )

                face_crop = img[
                    max(0, y1_i) : min(h, y2_i), max(0, x1_i) : min(w, x2_i)
                ]
                if face_crop.size == 0:
                    continue

                resized = cv2.resize(face_crop, (160, 160))
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                tensor = torch.tensor(resized).permute(2, 0, 1).float() / 255.0
                face_tensors.append(tensor)
                valid_boxes.append([x1_i, y1_i, x2_i, y2_i])

            if not face_tensors:
                embeddings["face_count"] = 0
                return embeddings

            batch = torch.stack(face_tensors)
            embs = self.embedding_model(batch).detach().numpy()

            embeddings["face_embedding"] = (
                np.mean(embs, axis=0).astype(np.float32).tolist()
            )
            embeddings["face_count"] = int(len(embs))

            # Draw detected faces on original image for annotated_bytes
            for x1_i, y1_i, x2_i, y2_i in valid_boxes:
                cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), (0, 0, 255), 2)

            _, annotated_bytes = cv2.imencode(".jpg", img)
            embeddings["annotated_bytes"] = annotated_bytes.tobytes()

            logger.info("Generated %d face embeddings.", embeddings["face_count"])
            return embeddings

        except Exception as error:
            logger.exception("Error generating face embeddings: %s", error)
            return {}
