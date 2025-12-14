from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.services.detection_service import DetectionService
from app.services.traffic_manager import TrafficManager
from app import socketio
import threading
import time

api_bp = Blueprint('api', __name__)
detection_service = DetectionService()
traffic_manager = TrafficManager()

@api_bp.route('/upload/image', methods=['POST'])
def upload_image():
    """Handle image upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(current_app.config['RESULT_FOLDER'], f"annotated_{filename}")
        
        file.save(upload_path)

        # Analyze image
        analysis_result = detection_service.analyze_image(upload_path, result_path)
        
        # Update traffic light based on vehicle count
        vehicle_count = analysis_result['vehicle_count']
        traffic_state = traffic_manager.update_traffic_state(vehicle_count)
        
        # Emit real-time update via WebSocket
        socketio.emit('traffic_update', {
            'type': 'image_analysis',
            'vehicle_count': vehicle_count,
            'traffic_state': traffic_state,
            'analysis_result': analysis_result
        })

        return jsonify({
            "success": True,
            "vehicle_count": vehicle_count,
            "detections": analysis_result['detections'],
            "annotated_image_url": f"/static/results/annotated_{filename}",
            "traffic_state": traffic_state
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/upload/video', methods=['POST'])
def upload_video():
    """Handle video upload and start real-time processing"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(current_app.config['RESULT_FOLDER'], f"annotated_{filename}")
        
        file.save(upload_path)

        # Start video processing in background thread
        thread = threading.Thread(
            target=process_video_realtime,
            args=(upload_path, result_path, filename)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "message": "Video processing started",
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_video_realtime(video_path, result_path, filename):
    """Process video frames in real-time and emit updates"""
    try:
        frame_count = 0
        total_vehicles = 0
        
        for frame_data in detection_service.process_video_stream(video_path, result_path):
            frame_count += 1
            vehicle_count = frame_data['vehicle_count']
            total_vehicles += vehicle_count
            
            # Update traffic light state
            traffic_state = traffic_manager.update_traffic_state(vehicle_count)
            
            # Emit real-time frame data
            socketio.emit('video_frame_update', {
                'frame_number': frame_count,
                'vehicle_count': vehicle_count,
                'detections': frame_data['detections'],
                'traffic_state': traffic_state,
                'timestamp': time.time()
            })
            
            # Small delay to prevent overwhelming the client
            time.sleep(0.1)
        
        # Send completion notification
        socketio.emit('video_processing_complete', {
            'filename': filename,
            'total_frames': frame_count,
            'average_vehicles': total_vehicles / frame_count if frame_count > 0 else 0,
            'annotated_video_url': f"/static/results/annotated_{filename}"
        })
        
    except Exception as e:
        socketio.emit('processing_error', {'error': str(e)})

@api_bp.route('/traffic/state', methods=['GET'])
def get_traffic_state():
    """Get current traffic light state"""
    return jsonify(traffic_manager.get_current_state())

@api_bp.route('/traffic/reset', methods=['POST'])
def reset_traffic():
    """Reset traffic light to default state"""
    traffic_manager.reset()
    return jsonify({"success": True, "message": "Traffic light reset"})