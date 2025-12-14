import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
from app.models.schemas import Detection
from app.config import settings

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self):
        self.model = YOLO(settings.YOLO_MODEL)
        self.class_names = self.model.names
        logger.info(f"Loaded YOLO model: {settings.YOLO_MODEL}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in frame"""
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
                        
                        detection = Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=float(conf),
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
