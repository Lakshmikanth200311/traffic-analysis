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