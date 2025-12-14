from flask import Blueprint, request, jsonify, current_app, send_file
import os
from werkzeug.utils import secure_filename
from app.services.detection import detect_and_annotate

image_bp = Blueprint('image_bp', __name__)

@image_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(current_app.config['RESULT_FOLDER'], f"annotated_{filename}")

    file.save(upload_path)

    detections = detect_and_annotate(upload_path, result_path)

    return jsonify({
        "vehicle_count": len(detections),
        "detections": detections,
        "annotated_image_url": f"/{result_path}"
    })
