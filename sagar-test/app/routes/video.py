from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.services.video_handler import process_video

video_bp = Blueprint('video_bp', __name__)

@video_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    filename = secure_filename(file.filename)
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(current_app.config['RESULT_FOLDER'], f"annotated_{filename}")

    file.save(upload_path)

    summary = process_video(upload_path, result_path)

    return jsonify({
        "summary": summary,
        "annotated_video_url": f"/{result_path}"
    })
