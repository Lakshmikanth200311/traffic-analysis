import time
import threading
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficManager:
    def __init__(self):
        """Initialize traffic light manager with default settings"""
        self.states = ['RED', 'GREEN', 'YELLOW']
        self.current_state = 'RED'
        self.state_index = 0
        
        # Default timings (in seconds)
        self.default_timings = {
            'RED': 30,
            'GREEN': 25,
            'YELLOW': 5
        }
        
        # Current timings (can be modified based on traffic)
        self.current_timings = self.default_timings.copy()
        
        # Traffic analysis
        self.current_vehicle_count = 0
        self.traffic_intensity = 'LOW'  # LOW, MEDIUM, HIGH
        
        # Simulation control
        self.simulation_active = False
        self.simulation_thread = None
        self.last_change_time = time.time()
        
        
        # Statistics
        self.stats = {
            'total_vehicles_detected': 0,
            'peak_vehicle_count': 0,
            'light_changes': 0,
            'adaptive_adjustments': 0
        }
        
        logger.info("Traffic Manager initialized")
    
    def update_traffic_state(self, vehicle_count):
        """Update traffic state based on detected vehicle count"""
        self.current_vehicle_count = vehicle_count
        self.stats['total_vehicles_detected'] += vehicle_count
        
        if vehicle_count > self.stats['peak_vehicle_count']:
            self.stats['peak_vehicle_count'] = vehicle_count
        
        # Determine traffic intensity
        if vehicle_count <= 3:
            self.traffic_intensity = 'LOW'
        elif vehicle_count <= 8:
            self.traffic_intensity = 'MEDIUM'
        else:
            self.traffic_intensity = 'HIGH'
        
        # Adaptive timing adjustment
        self._adjust_timings_based_on_traffic(vehicle_count)
        
        return self.get_current_state()
    
    def _adjust_timings_based_on_traffic(self, vehicle_count):
        """Adjust traffic light timings based on vehicle count"""
        old_timings = self.current_timings.copy()
        
        if vehicle_count > 10:  # High traffic
            # Increase green time, reduce red time
            self.current_timings['GREEN'] = min(self.default_timings['GREEN'] + 15, 60)
            self.current_timings['RED'] = max(self.default_timings['RED'] - 10, 15)
            self.current_timings['YELLOW'] = 8  # Longer yellow for safety
            
        elif vehicle_count > 6:  # Medium traffic
            # Moderate adjustment
            self.current_timings['GREEN'] = self.default_timings['GREEN'] + 8
            self.current_timings['RED'] = self.default_timings['RED'] - 5
            self.current_timings['YELLOW'] = 6
            
        elif vehicle_count < 2:  # Low traffic
            # Shorter green, longer red to save energy
            self.current_timings['GREEN'] = max(self.default_timings['GREEN'] - 5, 15)
            self.current_timings['RED'] = self.default_timings['RED'] + 5
            self.current_timings['YELLOW'] = 4
            
        else:  # Normal traffic
            self.current_timings = self.default_timings.copy()
        
        # Check if timings changed
        if old_timings != self.current_timings:
            self.stats['adaptive_adjustments'] += 1
            logger.info(f"Adaptive timing adjustment: {self.current_timings}")
    
    def get_current_state(self):
        """Get current traffic light state with all relevant information"""
        time_since_change = time.time() - self.last_change_time
        current_duration = self.current_timings[self.current_state]
        time_remaining = max(0, current_duration - time_since_change)
        
        return {
            'current_light': self.current_state,
            'time_remaining': round(time_remaining, 1),
            'current_timings': self.current_timings,
            'vehicle_count': self.current_vehicle_count,
            'traffic_intensity': self.traffic_intensity,
            'simulation_active': self.simulation_active,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def manual_change_light(self):
        """Manually change traffic light to next state"""
        self._change_to_next_state()
        logger.info(f"Manual light change to {self.current_state}")
        return self.get_current_state()
    
    def _change_to_next_state(self):
        """Change to next traffic light state"""
        self.state_index = (self.state_index + 1) % len(self.states)
        self.current_state = self.states[self.state_index]
        self.last_change_time = time.time()
        self.stats['light_changes'] += 1
    
    def start_simulation(self):
        """Start automatic traffic light simulation"""
        if not self.simulation_active:
            self.simulation_active = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            logger.info("Traffic light simulation started")
    
    def stop_simulation(self):
        """Stop automatic traffic light simulation"""
        self.simulation_active = False
        if self.simulation_thread:
            self.simulation_thread = None
        logger.info("Traffic light simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop for automatic traffic light changes"""
        while self.simulation_active:
            try:
                time_since_change = time.time() - self.last_change_time
                current_duration = self.current_timings[self.current_state]
                
                if time_since_change >= current_duration:
                    self._change_to_next_state()
                    logger.info(f"Auto light change to {self.current_state}")
                    
                    # Emit update via WebSocket if available
                    try:
                        from app import socketio
                        socketio.emit('traffic_update', {
                            'type': 'automatic_change',
                            'traffic_state': self.get_current_state()
                        })
                    except:
                        pass  # SocketIO might not be available in all contexts
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(1)
    
    def reset(self):
        """Reset traffic manager to default state"""
        self.stop_simulation()
        self.current_state = 'RED'
        self.state_index = 0
        self.current_timings = self.default_timings.copy()
        self.current_vehicle_count = 0
        self.traffic_intensity = 'LOW'
        self.last_change_time = time.time()
        
        # Reset stats
        self.stats = {
            'total_vehicles_detected': 0,
            'peak_vehicle_count': 0,
            'light_changes': 0,
            'adaptive_adjustments': 0
        }
        
        logger.info("Traffic manager reset to default state")
    
    def get_current_time(self):
        """Get current timestamp"""
        return time.time()