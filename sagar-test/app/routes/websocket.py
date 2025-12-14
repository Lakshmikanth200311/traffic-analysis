from flask_socketio import emit, disconnect
from app.services.traffic_manager import TrafficManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

traffic_manager = TrafficManager()

def register_socketio_events(socketio):
    """Register all WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info('Client connected')
        # Send current traffic state to newly connected client
        emit('traffic_state', traffic_manager.get_current_state())
        emit('connection_status', {'status': 'connected', 'message': 'Real-time connection established'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('Client disconnected')
    
    @socketio.on('request_traffic_state')
    def handle_traffic_state_request():
        """Handle request for current traffic state"""
        emit('traffic_state', traffic_manager.get_current_state())
    
    @socketio.on('manual_traffic_control')
    def handle_manual_control(data):
        """Handle manual traffic light control"""
        try:
            action = data.get('action')
            if action == 'change_light':
                new_state = traffic_manager.manual_change_light()
                emit('traffic_update', {
                    'type': 'manual_control',
                    'traffic_state': new_state
                })
            elif action == 'reset':
                traffic_manager.reset()
                emit('traffic_update', {
                    'type': 'reset',
                    'traffic_state': traffic_manager.get_current_state()
                })
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('start_simulation')
    def handle_start_simulation():
        """Start traffic light simulation"""
        try:
            traffic_manager.start_simulation()
            emit('simulation_status', {'status': 'started'})
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('stop_simulation')
    def handle_stop_simulation():
        """Stop traffic light simulation"""
        try:
            traffic_manager.stop_simulation()
            emit('simulation_status', {'status': 'stopped'})
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('ping')
    def handle_ping():
        """Handle ping for connection testing"""
        emit('pong', {'timestamp': traffic_manager.get_current_time()})