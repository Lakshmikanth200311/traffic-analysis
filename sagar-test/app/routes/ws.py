from flask import Blueprint
from app import socketio
from flask_socketio import emit
from app.services.traffic_light import simulate_traffic_logic

ws_bp = Blueprint('ws_bp', __name__)

@socketio.on('video_frame')
def handle_frame(data):
    # Process incoming frame for detection
    result = simulate_traffic_logic(data['frame'])
    emit('light_update', result)
