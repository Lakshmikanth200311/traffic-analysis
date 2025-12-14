from collections import defaultdict, deque
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.models.schemas import Detection
import time

class Track:
    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=30)
        self.last_seen = time.time()
        self.add_detection(detection)
    
    def add_detection(self, detection: Detection):
        detection.track_id = self.track_id
        self.detections.append(detection)
        self.last_seen = time.time()
    
    @property
    def current_bbox(self):
        return self.detections[-1].bbox if self.detections else None
    
    @property
    def centroid(self):  # FIXED: singular 'centroid'
        if not self.detections:
            return None
        bbox = self.current_bbox
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

class CentroidTracker:
    def __init__(self, max_age: int = 30, max_distance: float = 50.0):
        self.max_age = max_age
        self.max_distance = max_distance
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.vehicle_weights = {
            'car': 1.0,
            'motorcycle': 0.7,
            'bus': 2.0,
            'truck': 1.8
        }
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        current_time = time.time()
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Match detections to existing tracks
        if self.tracks and detections:
            # FIXED: Use 'centroid' (singular) and filter out None centroids
            track_centroids = [track.centroid for track in self.tracks.values() if track.centroid is not None]
            detection_centroids = [self._get_centroid(det.bbox) for det in detections]
            
            matched_tracks = set()
            matched_detections = set()
            
            # Simple centroid distance matching
            for i, track in enumerate(self.tracks.values()):
                # FIXED: Check if centroid exists and is not None
                if track.centroid is None:
                    continue
                    
                for j, det_centroid in enumerate(detection_centroids):
                    if j in matched_detections:
                        continue
                    
                    distance = np.linalg.norm(
                        np.array(track.centroid) - np.array(det_centroid)  # FIXED: singular 'centroid'
                    )
                    
                    if distance < self.max_distance:
                        track.add_detection(detections[j])
                        matched_tracks.add(i)
                        matched_detections.add(j)
                        break
            
            # Create new tracks for unmatched detections
            for j, detection in enumerate(detections):
                if j not in matched_detections:
                    new_track = Track(self.next_id, detection)
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1
        
        elif detections:
            # First frame or no existing tracks
            for detection in detections:
                new_track = Track(self.next_id, detection)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Return detections with track IDs
        tracked_detections = []
        for track in self.tracks.values():
            if track.detections:
                tracked_detections.append(track.detections[-1])
        
        return tracked_detections
    
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)