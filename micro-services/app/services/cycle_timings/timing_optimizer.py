import os
import cv2
from typing import List, Dict
from app.config import settings
from app.models.schemas import Approach, Phase, CyclePlan, Detection
from app.services.yolo.detector import YOLODetector

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
