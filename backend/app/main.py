from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import json
import time

from app.config import settings
from app.routers import api
from app.models.state import TrafficSystemState
from app.services.simulator import TrafficSimulator
from app.websocket_manager import websocket_manager, broadcaster


from app.routers import video  # Create this file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
system_state = TrafficSystemState()
traffic_simulator = TrafficSimulator(system_state)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting Traffic Optimization System")
    import os

    PORT = int(os.getenv("PORT", settings.PORT))

    logger.info(
        f"📡 WebSocket endpoint will be available at: ws://{settings.HOST}:{PORT}/ws"
        )

    yield
    # Shutdown
    logger.info("🛑 Shutting down Traffic Optimization System")
    await traffic_simulator.stop()

app = FastAPI(
    title="Real-time Traffic Optimizer",
    description="AI-powered traffic light optimization using YOLO and adaptive timing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.video_router, prefix="/api/v1")
# Include API routers (but NOT websocket router if it exists there)
app.include_router(api.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Traffic Optimization API", 
        "version": "1.0.0",
        "websocket": f"ws://{settings.HOST}:{settings.PORT}/ws"
    }

@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy", 
        "running": system_state.running,
        "connections": len(websocket_manager.active_connections)
    }


# In main.py - KEEP THIS ONE
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔌 New WebSocket connection attempt")
    await websocket_manager.connect(websocket)
    try:
        # Send immediate welcome message with current state
        welcome_msg = {
            "type": "connection_established",
            "data": {
                "message": "WebSocket connected successfully",
                "system_state": await system_state.get_state_snapshot()
            }
        }
        await websocket_manager.send_personal_message(json.dumps(welcome_msg), websocket)
        logger.info("✅ WebSocket connected successfully")
        
        while True:
            # Keep connection alive - we can receive ping messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.info(f"Received WebSocket message: {message}")
                
                # Handle ping messages
                if message.get("type") == "ping":
                    response = {
                        "type": "pong",
                        "data": {"timestamp": time.time()}
                    }
                    await websocket_manager.send_personal_message(json.dumps(response), websocket)
                    
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON WebSocket message: {data}")
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )