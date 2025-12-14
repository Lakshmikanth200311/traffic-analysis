import numpy as np
from typing import Dict, List
from app.models.schemas import Approach, Phase, CyclePlan
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class TrafficTimingOptimizer:
    def __init__(self):
        self.saturation_flows = {
            'car': 1.0,
            'motorcycle': 0.7, 
            'bus': 2.0,
            'truck': 1.8
        }
        self.arrival_rates_ewma = {a: 0.0 for a in Approach}
        self.queue_lengths_ewma = {a: 0 for a in Approach}
    
    def compute_cycle_plan(
        self, 
        current_counts: Dict[Approach, int],
        arrival_rates: Dict[Approach, float],
        queue_lengths: Dict[Approach, int],
        cycle_version: int
    ) -> CyclePlan:
        """Compute optimal cycle plan based on current traffic conditions"""
        
        # Update EWMA values
        for approach in Approach:
            if approach in arrival_rates:
                self.arrival_rates_ewma[approach] = (
                    settings.SMOOTH_ALPHA * arrival_rates[approach] + 
                    (1 - settings.SMOOTH_ALPHA) * self.arrival_rates_ewma[approach]
                )
            
            if approach in queue_lengths:
                self.queue_lengths_ewma[approach] = int(
                    settings.SMOOTH_ALPHA * queue_lengths[approach] + 
                    (1 - settings.SMOOTH_ALPHA) * self.queue_lengths_ewma[approach]
                )
        
        # Calculate effective demand for each approach
        effective_demands = {}
        for approach in Approach:
            queue = max(1, self.queue_lengths_ewma[approach])
            arrival_rate = max(0.1, self.arrival_rates_ewma[approach])
            
            # Effective demand = queue + k * arrival_rate * horizon
            k = 0.8  # Tuning parameter
            horizon = 60  # seconds
            effective_demands[approach] = queue + k * arrival_rate * horizon
        
        # Calculate total demand
        total_demand = sum(effective_demands.values())
        
        if total_demand == 0:
            # Equal distribution if no demand
            green_times = {a: settings.MIN_GREEN for a in Approach}
        else:
            # Allocate green time proportionally to demand
            available_green = (
                settings.CYCLE_MAX - 
                (settings.YELLOW_TIME + settings.ALL_RED_TIME) * len(Approach)
            )
            
            green_times = {}
            for approach in Approach:
                proportion = effective_demands[approach] / total_demand
                green_time = int(proportion * available_green)
                green_times[approach] = max(
                    settings.MIN_GREEN, 
                    min(settings.MAX_GREEN, green_time)
                )
        
        # Build phases (simple two-phase system: N-S and E-W)
        phases = []
        
        # North-South phase
        ns_green = max(green_times[Approach.NORTH], green_times[Approach.SOUTH])
        phases.append(Phase(
            approach=Approach.NORTH,
            green=ns_green,
            yellow=settings.YELLOW_TIME,
            red=settings.CYCLE_MAX - ns_green - settings.YELLOW_TIME
        ))
        
        phases.append(Phase(
            approach=Approach.SOUTH, 
            green=ns_green,
            yellow=settings.YELLOW_TIME,
            red=settings.CYCLE_MAX - ns_green - settings.YELLOW_TIME
        ))
        
        # East-West phase  
        ew_green = max(green_times[Approach.EAST], green_times[Approach.WEST])
        phases.append(Phase(
            approach=Approach.EAST,
            green=ew_green,
            yellow=settings.YELLOW_TIME,
            red=settings.CYCLE_MAX - ew_green - settings.YELLOW_TIME
        ))
        
        phases.append(Phase(
            approach=Approach.WEST,
            green=ew_green, 
            yellow=settings.YELLOW_TIME,
            red=settings.CYCLE_MAX - ew_green - settings.YELLOW_TIME
        ))
        
        cycle_seconds = ns_green + ew_green + 2 * settings.YELLOW_TIME + 2 * settings.ALL_RED_TIME
        
        return CyclePlan(
            cycle_seconds=cycle_seconds,
            phases=phases,
            version=cycle_version + 1
        )
    
    def compute_optimization_deltas(
        self,
        prev_plan: CyclePlan,
        new_plan: CyclePlan
    ) -> List[Dict]:
        """Compute optimization deltas between cycles"""
        deltas = []
        
        prev_greens = {phase.approach: phase.green for phase in prev_plan.phases}
        new_greens = {phase.approach: phase.green for phase in new_plan.phases}
        
        for approach in Approach:
            prev_green = prev_greens.get(approach, settings.MIN_GREEN)
            new_green = new_greens.get(approach, settings.MIN_GREEN)
            
            deltas.append({
                "approach": approach,
                "prev_green": prev_green,
                "new_green": new_green,
                "delta": new_green - prev_green
            })
        
        return deltas