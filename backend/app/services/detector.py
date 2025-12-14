import cv2
import os
import numpy as np
from ultralytics import YOLO
from typing import List
import logging

from app.models.schemas import Detection
from app.config import settings

logger = logging.getLogger(__name__)

USE_YOLO = os.getenv("USE_YOLO", "false").lower() == "true"


class YOLODetector:
    def __init__(self):
        self.model = None
        self.class_names = {}

        if USE_YOLO:
            logger.info("🧠 YOLO ENABLED → Loading model")
            self.model = YOLO(settings.YOLO_MODEL)
            self.class_names = self.model.names
            logger.info(f"✅ YOLO model loaded: {settings.YOLO_MODEL}")
        else:
            logger.warning("⚠️ YOLO DISABLED → Using dummy detections (cloud-safe)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        If YOLO is enabled → real detections
        If YOLO is disabled → fake boxes for demo (cloud-safe)
        """

        # 🟡 CLOUD MODE — FAKE DETECTIONS
        if not USE_YOLO:
            h, w, _ = frame.shape
            fake_detections = []

            for i in range(np.random.randint(2, 5)):
                x1 = np.random.randint(0, w // 2)
                y1 = np.random.randint(0, h // 2)
                x2 = x1 + np.random.randint(80, 160)
                y2 = y1 + np.random.randint(60, 140)

                fake_detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=round(np.random.uniform(0.4, 0.9), 2),
                        class_id=2,
                        class_name="car",
                        track_id=i
                    )
                )

            return fake_detections

        # 🟢 LOCAL MODE — REAL YOLO
        try:
            results = self.model(
                frame,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                classes=settings.VEHICLE_CLASSES,
                verbose=False
            )

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]

                        detections.append(
                            Detection(
                                bbox=[x1, y1, x2, y2],
                                confidence=float(conf),
                                class_id=class_id,
                                class_name=class_name
                            )
                        )

            return detections

        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            return []


class OpenCVDetector:
    """Fallback detector using OpenCV DNN"""

    def __init__(self):
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []
