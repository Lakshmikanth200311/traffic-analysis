from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os

# Initialize SocketIO
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

def create_app():
    # Create Flask app with explicit template folder
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static'))
    
    # Enable CORS
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Configuration
    app.config['SECRET_KEY'] = 'traffic_analysis_v2_secret_key'
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULT_FOLDER'] = 'static/results'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from app.routes.api import api_bp
    from app.routes.websocket import register_socketio_events
    
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize SocketIO with app
    socketio.init_app(app)
    
    # Register WebSocket events
    register_socketio_events(socketio)
    
    return app