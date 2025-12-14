import time
from flask import Flask, request
from flask_socketio import SocketIO, emit
from threading import Thread, Lock
# from ..services.cycle_timings.timing_optimizer import TrafficTimingOptimizer  # import your optimizer


import os
import cv2
from typing import List, Dict
# from app.config import settings
# from app.models.schemas import Approach, Phase, CyclePlan, Detection
# from app.services.yolo.detector import YOLODetector


from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    # Model settings
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.4
    IOU_THRESHOLD: float = 0.4
    VEHICLE_CLASSES: List[int] = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
    MAX_FPS: int = 10
    
    # Timing parameters
    MIN_GREEN: int = 10
    MAX_GREEN: int = 60
    YELLOW_TIME: int = 3
    ALL_RED_TIME: int = 2
    CYCLE_MIN: int = 40
    CYCLE_MAX: int = 120
    SMOOTH_ALPHA: float = 0.7
    
    # Counting parameters
    COUNTING_LINE_OFFSET: int = 100
    TRACK_BUFFER: int = 30
    MAX_AGE: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()


# app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict
from enum import Enum

class Approach(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

class Phase(BaseModel):
    approach: Approach
    green: int
    yellow: int
    red: int

class CyclePlan(BaseModel):
    cycle_seconds: int
    phases: List[Phase]
    priority_order: List[Approach]
    version: int

class ImageUploadRequest(BaseModel):
    north_image: str
    south_image: str
    east_image: str
    west_image: str

class Detection(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int = None


import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
# from app.models.schemas import Detection
# from app.config import settings

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


class TrafficTimingOptimizer:
    def __init__(self):
        self.saturation_flows = {
            'car': 1.0, 'motorcycle': 0.7, 'bus': 2.0, 'truck': 1.8
        }

    def get_vehicle_counts(self, image_path: str, detector: YOLODetector) -> Dict[str, int]:
        counts = {vt: 0 for vt in self.saturation_flows.keys()}
        if not os.path.isfile(image_path):
            return counts
        img = cv2.imread(image_path)
        if img is None:
            return counts
        detections = detector.detect(img)
        for det in detections:
            if det.class_name in counts:
                counts[det.class_name] += 1
        return counts

    def compute_cycle_plan(
        self, images: Dict[str, str], detector: YOLODetector, cycle_version: int = 1
    ) -> CyclePlan:
        # images: {approach_name: image_path}
        detected_counts = {}
        effective_demands = {}

        # Get vehicle counts for each approach
        for approach in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            counts = self.get_vehicle_counts(images[approach], detector)
            # Weighted demand sum
            effective_demand = sum(counts[k]*self.saturation_flows[k] for k in counts)
            detected_counts[approach] = counts
            effective_demands[approach] = max(1, effective_demand)  # prevent zero

        total_demand = sum(effective_demands.values())
        if total_demand == 0:
            # All MIN_GREEN if no detected vehicles
            green_times = {a: settings.MIN_GREEN for a in effective_demands}
        else:
            available_green = settings.CYCLE_MAX - \
                (settings.YELLOW_TIME + settings.ALL_RED_TIME) * 4
            # Allocate green proportionally to demand
            green_times = {}
            for approach in effective_demands:
                prop = effective_demands[approach] / total_demand
                green = int(prop * available_green)
                green_times[approach] = max(settings.MIN_GREEN, min(settings.MAX_GREEN, green))

        # Sort approaches for priority (descending demand)
        priority_order = sorted(effective_demands.keys(), key=lambda k: -effective_demands[k])

        # Build phases (individual for each approach here)
        phases = []
        for approach in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            phases.append(Phase(
                approach=approach,
                green=green_times[approach],
                yellow=settings.YELLOW_TIME,
                red=settings.CYCLE_MAX - green_times[approach] - settings.YELLOW_TIME
            ))

        cycle_seconds = sum(p.green + p.yellow + settings.ALL_RED_TIME for p in phases)
        return CyclePlan(
            cycle_seconds=cycle_seconds,
            phases=phases,
            priority_order=priority_order,
            version=cycle_version
        )



app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
lock = Lock()

# Current vehicle counts
counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
last_counts = counts.copy()
refresh_flag = False

optimizer = TrafficTimingOptimizer()
cycle_version = 0
cycle_duration = 60   # Example: 60 second cycle for simplicity

def run_cycle_thread():
    global counts, last_counts, cycle_version, refresh_flag
    while True:
        lock.acquire()
        # Use the latest counts to compute this cycle
        plan = optimizer.compute_cycle_plan(counts, cycle_version)
        cycle_version += 1
        last_counts = counts.copy()
        lock.release()

        # Dispatch the plan to all clients
        socketio.emit('cycle_plan', {
            'version': cycle_version,
            'plan': plan.model_dump() if hasattr(plan, 'model_dump') else plan.dict() if hasattr(plan, 'dict') else plan.__dict__
        })


        start_time = time.time()
        # Simulate running signal cycle (with mid-cycle update logic)
        while time.time() - start_time < cycle_duration:
            time.sleep(2)
            # If refresh_flag set (counts updated mid-cycle), only log info
            if refresh_flag:
                socketio.emit('info', {'msg': 'Vehicle counts updated during cycle, will be used next.'})
                refresh_flag = False
        # Cycle ends, loop continues, new counts taken into account for next cycle

@socketio.on('update_counts')
def handle_update_counts(data):
    global counts, refresh_flag
    lock.acquire()
    try:
        if data:
            for key in ['north', 'south', 'east', 'west']:
                if key in data:
                    counts[key] = int(data[key])
            refresh_flag = True
        emit('ack', {'msg': 'Counts updated', 'counts': counts})
    finally:
        lock.release()

@app.route('/')
def index():
    return "WebSocket Traffic Signal Server Running."

if __name__ == '__main__':
    print("Runnign")
    thread = Thread(target=run_cycle_thread)
    thread.daemon = True
    thread.start()
    socketio.run(app, host="0.0.0.0", port=6000)
