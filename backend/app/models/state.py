import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from app.models.schemas import Approach, LiveCount, CyclePlan, CameraConfig
import json

@dataclass
class TrafficSystemState:
    running: bool = False
    cycle_plan: Optional[CyclePlan] = None
    live_counts: Dict[Approach, LiveCount] = field(default_factory=dict)
    phase_active: Optional[Approach] = None
    remaining_seconds: int = 0
    camera_configs: Dict[Approach, CameraConfig] = field(default_factory=dict)
    cycle_version: int = 0
    
    # Count tracking
    queue_counts: Dict[Approach, int] = field(default_factory=lambda: {a: 0 for a in Approach})
    arrival_rates: Dict[Approach, float] = field(default_factory=lambda: {a: 0.0 for a in Approach})
    
    # Locks
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __post_init__(self):
        # Initialize live counts for all approaches
        for approach in Approach:
            self.live_counts[approach] = LiveCount(
                approach=approach,
                vehicles={},
                total=0
            )
    
    async def update_live_count(self, approach: Approach, vehicles: dict, total: int):
        async with self._lock:
            self.live_counts[approach] = LiveCount(
                approach=approach,
                vehicles=vehicles,
                total=total
            )
    
    async def update_cycle_plan(self, plan: CyclePlan):
        async with self._lock:
            self.cycle_plan = plan
            self.cycle_version = plan.version
    
    async def update_phase(self, phase: Approach, remaining: int):
        async with self._lock:
            self.phase_active = phase
            self.remaining_seconds = remaining
    
    async def get_state_snapshot(self):
        async with self._lock:
            # Convert to serializable format
            cycle_plan_dict = self.cycle_plan.dict() if self.cycle_plan else None
            
            # Convert live_counts to list of dicts
            live_counts_list = [lc.dict() for lc in self.live_counts.values()]
            
            # Convert phase_active to string if it exists
            phase_active_str = self.phase_active.value if self.phase_active else None
            
            return {
                "running": self.running,
                "cycle_plan": cycle_plan_dict,
                "live_counts": live_counts_list,
                "phase_active": phase_active_str,
                "remaining_seconds": self.remaining_seconds
            }
    
    # Add this method for direct dict conversion
    def to_dict(self):
        """Convert state to dictionary for WebSocket broadcasting"""
        return {
            "running": self.running,
            "cycle_plan": self.cycle_plan.dict() if self.cycle_plan else None,
            "live_counts": [lc.dict() for lc in self.live_counts.values()],
            "phase_active": self.phase_active.value if self.phase_active else None,
            "remaining_seconds": self.remaining_seconds
        }