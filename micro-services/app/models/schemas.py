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
