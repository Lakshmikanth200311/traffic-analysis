from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum

class Approach(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

class VehicleType(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"

class Point(BaseModel):
    x: int
    y: int

class ROI(BaseModel):
    points: List[Point]
    
class CountingLine(BaseModel):
    start: Point
    end: Point

class CameraConfig(BaseModel):
    approach: Approach
    roi: ROI
    counting_line: CountingLine
    source: str  # file path or stream URL

class UploadRequest(BaseModel):
    north: str
    south: str
    east: str
    west: str

# Alternative: More flexible RunRequest
class RunRequest(BaseModel):
    configs: Dict[str, Any]  # Accept any structure for now
    
    # Or be more specific but flexible:
    # configs: Dict[Approach, Dict[str, Any]]

class Detection(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

class VehicleCounts(BaseModel):
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0
    
    @property
    def total(self) -> int:
        return self.car + self.motorcycle + self.bus + self.truck

class LiveCount(BaseModel):
    approach: Approach
    vehicles: VehicleCounts
    total: int

class Phase(BaseModel):
    approach: Approach
    green: int
    yellow: int = 3
    red: int

class CyclePlan(BaseModel):
    cycle_seconds: int
    phases: List[Phase]
    version: int

class OptimizationDelta(BaseModel):
    approach: Approach
    prev_green: int
    new_green: int
    delta: int

class SystemState(BaseModel):
    running: bool
    cycle_plan: Optional[CyclePlan] = None
    live_counts: List[LiveCount]
    phase_active: Optional[Approach] = None
    remaining_seconds: int = 0

class WSMessage(BaseModel):
    type: Literal[
        "live_counts", 
        "detections_meta", 
        "phase_update", 
        "cycle_plan", 
        "optimization_delta",
        "system_state"
    ]
    data: Dict[str, Any]