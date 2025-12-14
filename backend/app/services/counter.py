import numpy as np
from typing import List, Set, Dict
from app.models.schemas import Detection, Approach, VehicleCounts, CountingLine
from app.services.tracker import CentroidTracker
import logging

logger = logging.getLogger(__name__)

class LineCrossingCounter:
    def __init__(self, counting_line: CountingLine, approach: Approach):
        self.counting_line = counting_line
        self.approach = approach
        self.tracker = CentroidTracker()
        self.counted_tracks: Set[int] = set()
        self.vehicle_counts = VehicleCounts()
        
        # Direction vector for the counting line
        line_vec = np.array([
            counting_line.end.x - counting_line.start.x,
            counting_line.end.y - counting_line.start.y
        ])
        self.line_normal = np.array([-line_vec[1], line_vec[0]])
        
        # FIXED: Handle division by zero for very short lines
        line_normal_norm = np.linalg.norm(self.line_normal)
        if line_normal_norm > 0:
            self.line_normal = self.line_normal / line_normal_norm
        else:
            # Default to horizontal line if line is degenerate
            self.line_normal = np.array([0, 1])
    
    def update(self, detections: List[Detection]) -> VehicleCounts:
        try:
            tracked_detections = self.tracker.update(detections)
            
            for detection in tracked_detections:
                if detection.track_id is None:
                    continue
                    
                if detection.track_id not in self.counted_tracks:
                    if self._has_crossed_line(detection):
                        self.counted_tracks.add(detection.track_id)
                        self._increment_count(detection.class_name)
            
            return self.vehicle_counts
        except Exception as e:
            logger.error(f"Error in LineCrossingCounter.update: {e}")
            return self.vehicle_counts
    
    def _has_crossed_line(self, detection: Detection) -> bool:
        """Check if vehicle centroid has crossed the counting line"""
        try:
            centroid = self._get_centroid(detection.bbox)
            
            # Vector from line start to centroid
            to_centroid = np.array([
                centroid[0] - self.counting_line.start.x,
                centroid[1] - self.counting_line.start.y
            ])
            
            # Dot product with normal indicates which side of the line
            dot_product = np.dot(to_centroid, self.line_normal)
            
            # For different approaches, we expect different crossing directions
            expected_direction = self._get_expected_direction()
            
            return dot_product * expected_direction > 0
        except Exception as e:
            logger.error(f"Error in _has_crossed_line: {e}")
            return False
    
    def _get_expected_direction(self) -> int:
        """Get expected crossing direction based on approach"""
        directions = {
            Approach.NORTH: 1,   # Moving south to north
            Approach.SOUTH: -1,  # Moving north to south  
            Approach.EAST: 1,    # Moving west to east
            Approach.WEST: -1    # Moving east to west
        }
        return directions.get(self.approach, 1)
    
    def _get_centroid(self, bbox: List[float]) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _increment_count(self, class_name: str):
        try:
            if class_name == 'car':
                self.vehicle_counts.car += 1
            elif class_name == 'motorcycle':
                self.vehicle_counts.motorcycle += 1
            elif class_name == 'bus':
                self.vehicle_counts.bus += 1
            elif class_name == 'truck':
                self.vehicle_counts.truck += 1
        except Exception as e:
            logger.error(f"Error incrementing count for {class_name}: {e}")
    
    def reset_counts(self):
        self.vehicle_counts = VehicleCounts()
        self.counted_tracks.clear()