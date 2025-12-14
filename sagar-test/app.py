from app import create_app, socketio
from flask import render_template
import os

app = create_app()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory('static', filename)

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    
    print("🚦 Traffic Analysis System V2 Starting...")
    print("📡 Real-time WebSocket server enabled")
    print("🎥 Video streaming support enabled")
    print("🚨 Traffic light simulation active")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)