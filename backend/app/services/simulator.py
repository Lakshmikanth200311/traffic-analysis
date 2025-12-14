import asyncio
import time
from typing import Dict, Any
import logging
import random
from app.models.schemas import Approach, CyclePlan, VehicleCounts, LiveCount
from app.models.state import TrafficSystemState
from app.services.timing import TrafficTimingOptimizer
from app.pipelines.camera import CameraPipeline
from app.websocket_manager import broadcaster, websocket_manager


logger = logging.getLogger(__name__)

class TrafficSimulator:
    def __init__(self, system_state: TrafficSystemState):
        self.system_state = system_state
        self.timing_optimizer = TrafficTimingOptimizer()
        self.camera_pipelines: Dict[Approach, CameraPipeline] = {}
        self.scheduler_task: asyncio.Task = None
        self.is_running = False
        
    async def start(self, camera_configs: Dict[str, Any]):
        """Start the traffic simulation"""
        if self.is_running:
            logger.warning("Simulator already running")
            return
        
        self.is_running = True
        self.system_state.running = True
        logger.info(f"🚀 Starting traffic simulator with configs: {list(camera_configs.keys())}")
        
        # FIX: Convert string keys to Approach enum and ensure we have all approaches
        valid_configs = {}
        for approach_str, config in camera_configs.items():
            try:
                # Convert string to Approach enum
                approach = Approach(approach_str)
                valid_configs[approach] = config
                logger.info(f"📹 Creating pipeline for {approach}")
                
                # Create pipeline
                pipeline = CameraPipeline(approach, config)
                self.camera_pipelines[approach] = pipeline
                await pipeline.start()
                
            except ValueError:
                logger.warning(f"❌ Invalid approach name: {approach_str}")
        
        # If no valid configs, create default pipelines for all approaches
        if not valid_configs:
            logger.warning("No valid camera configs found, creating default pipelines")
            for approach in Approach:
                default_config = {
                    "approach": approach.value,
                    "roi": {
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 1280, "y": 0},
                            {"x": 1280, "y": 720},
                            {"x": 0, "y": 720}
                        ]
                    },
                    "counting_line": {
                        "start": {"x": 600, "y": 0},
                        "end": {"x": 600, "y": 720}
                    },
                    "source": f"file:///uploads/{approach.value}.mp4"
                }
                pipeline = CameraPipeline(approach, default_config)
                self.camera_pipelines[approach] = pipeline
                await pipeline.start()
        
        # Start scheduler
        self.scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("✅ Traffic simulator started")
        
        # Broadcast initial state
        await self._broadcast_current_state()
    
    async def stop(self):
        """Stop the traffic simulation"""
        self.is_running = False
        self.system_state.running = False
        
        # Stop camera pipelines
        for pipeline in self.camera_pipelines.values():
            await pipeline.stop()
        
        # Cancel scheduler
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Traffic simulator stopped")
    
    async def _broadcast_current_state(self):
        """Broadcast current system state"""
        state_snapshot = await self.system_state.get_state_snapshot()
        await broadcaster.broadcast_system_state(state_snapshot)
        
        # Also broadcast live counts
        await self._broadcast_live_counts()
    
    async def _broadcast_live_counts(self):
        """Broadcast live vehicle counts"""
        live_counts = []
        for approach in Approach:
            if approach in self.camera_pipelines:
                counts = await self.camera_pipelines[approach].get_current_counts()
                live_count = LiveCount(
                    approach=approach,
                    vehicles=counts,
                    total=counts.total
                )
                live_counts.append(live_count.dict())
            else:
                # Create default counts for missing approaches
                default_counts = VehicleCounts(
                    car=random.randint(3, 8),
                    motorcycle=random.randint(1, 3),
                    bus=random.randint(0, 2),
                    truck=random.randint(0, 2)
                )
                live_count = LiveCount(
                    approach=approach,
                    vehicles=default_counts,
                    total=default_counts.total
                )
                live_counts.append(live_count.dict())
        
        await broadcaster.broadcast_live_counts(live_counts)
    
    async def _run_scheduler(self):
        """Main scheduler loop"""
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                logger.info(f"🔄 Starting cycle {cycle_count}")
                
                # Get current traffic data from pipelines
                current_counts = {}
                arrival_rates = {}
                queue_lengths = {}
                
                for approach, pipeline in self.camera_pipelines.items():
                    counts = await pipeline.get_current_counts()
                    current_counts[approach] = counts.total
                    arrival_rates[approach] = await pipeline.get_arrival_rate()
                    queue_lengths[approach] = await pipeline.get_queue_length()
                
                # Fill in missing approaches with default values
                for approach in Approach:
                    if approach not in current_counts:
                        current_counts[approach] = random.randint(5, 15)
                        arrival_rates[approach] = random.uniform(1.0, 3.0)
                        queue_lengths[approach] = random.randint(3, 8)
                
                logger.info(f"📊 Current counts: {current_counts}")
                
                # Compute new cycle plan
                new_plan = self.timing_optimizer.compute_cycle_plan(
                    current_counts,
                    arrival_rates,
                    queue_lengths,
                    self.system_state.cycle_version
                )
                
                logger.info(f"📋 New cycle plan: {new_plan.dict()}")
                
                # Compute optimization deltas if we have a previous plan
                if self.system_state.cycle_plan:
                    deltas = self.timing_optimizer.compute_optimization_deltas(
                        self.system_state.cycle_plan,
                        new_plan
                    )
                    await broadcaster.broadcast_optimization_delta(deltas)
                    logger.info(f"📈 Optimization deltas: {deltas}")
                
                # Update system state
                await self.system_state.update_cycle_plan(new_plan)
                await broadcaster.broadcast_cycle_plan(new_plan.dict())
                
                # Execute the cycle
                await self._execute_cycle(new_plan)
                
                # Broadcast final state after cycle completion
                await self._broadcast_current_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_cycle(self, cycle_plan: CyclePlan):
        """Execute a traffic light cycle"""
        logger.info(f"🚦 Executing cycle {cycle_plan.version}")
        
        # Update live counts at the start of cycle
        await self._broadcast_live_counts()
        
        for phase in cycle_plan.phases:
            # Set active phase
            await self.system_state.update_phase(phase.approach, phase.green)
            await broadcaster.broadcast_phase_update(phase.approach.value, phase.green)  # FIX: Use .value
            logger.info(f"🟢 {phase.approach.value} phase started: {phase.green}s green")
            
            # Countdown for green phase
            for remaining in range(phase.green, 0, -1):
                if not self.is_running:
                    return
                await self.system_state.update_phase(phase.approach, remaining)
                await broadcaster.broadcast_phase_update(phase.approach.value, remaining)  # FIX: Use .value
                
                # Update live counts every 5 seconds during green phase
                if remaining % 5 == 0:
                    await self._broadcast_live_counts()
                
                await asyncio.sleep(1)
            
            # Yellow phase
            if not self.is_running:
                return
            await self.system_state.update_phase(phase.approach, phase.yellow)
            await broadcaster.broadcast_phase_update(phase.approach.value, phase.yellow)  # FIX: Use .value
            logger.info(f"🟡 {phase.approach.value} yellow phase: {phase.yellow}s")
            await asyncio.sleep(phase.yellow)
            
            # All-red phase (brief)
            if not self.is_running:
                return
            await self.system_state.update_phase(None, 2)  # 2 seconds all-red
            await broadcaster.broadcast_phase_update(None, 2)
            logger.info("🔴 All-red phase: 2s")
            await asyncio.sleep(2)