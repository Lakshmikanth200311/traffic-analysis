from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict
import shutil
import os
import asyncio
from app.models.schemas import UploadRequest, RunRequest, SystemState, Approach, CameraConfig
from app.models.state import TrafficSystemState
from app.services.simulator import TrafficSimulator
from app.websocket_manager import broadcaster
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()
system_state = TrafficSystemState()
traffic_simulator = TrafficSimulator(system_state)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_videos(request: UploadRequest):
    """Upload videos or register stream URLs"""
    try:
        configs = {}
        
        for approach, source in request.dict().items():
            approach_enum = Approach(approach)
            
            # If it's a file path, save it
            if source.startswith("file://"):
                file_path = source[7:]
                # TODO: Handle file upload
                pass
            elif os.path.isfile(source):
                file_path = source
            else:
                # Assume it's a stream URL
                file_path = source
            
            # Default ROI and counting line for 1280x720
            default_config = CameraConfig(
                approach=approach_enum,
                roi={
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 1280, "y": 0},
                        {"x": 1280, "y": 720},
                        {"x": 0, "y": 720}
                    ]
                },
                counting_line={
                    "start": {"x": 600, "y": 0},
                    "end": {"x": 600, "y": 720}
                },
                source=file_path
            )
            configs[approach_enum] = default_config
        
        system_state.camera_configs = configs
        
        # Broadcast updated state
        state_snapshot = await system_state.get_state_snapshot()
        await broadcaster.broadcast_system_state(state_snapshot)
        
        return {"message": "Videos registered successfully", "configs": configs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# In your router - fix the file path handling
@router.post("/upload-files")
async def upload_files(
    north: UploadFile = File(...),
    south: UploadFile = File(...), 
    east: UploadFile = File(...),
    west: UploadFile = File(...)
):
    try:
        UPLOAD_DIR = "uploads"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        files = {
            "north": north,
            "south": south,
            "east": east, 
            "west": west
        }
        
        file_paths = {}
        
        for approach, file in files.items():
            # Validate file type
            if not file.content_type.startswith('video/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File for {approach} must be a video"
                )
            
            # Save file with proper extension
            file_extension = os.path.splitext(file.filename)[1] or '.mp4'
            file_path = os.path.join(UPLOAD_DIR, f"{approach}{file_extension}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Use relative path that OpenCV can handle
            file_paths[approach] = f"file://{os.path.abspath(file_path)}"
            logger.info(f"✅ Saved video for {approach}: {file_path}")
        
        return file_paths
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.post("/run")
async def run_model(request: RunRequest, background_tasks: BackgroundTasks):
    """Start the traffic optimization model with REAL video processing"""
    try:
        logger.info(f"📨 Received run request with configs keys: {list(request.configs.keys())}")
        
        configs_to_use = {}
        
        # Validate and prepare camera configs
        for key, value in request.configs.items():
            try:
                approach = Approach(key)
                # Ensure source path is properly formatted
                if 'source' in value and value['source'].startswith('file://'):
                    # Verify file exists
                    file_path = value['source'][7:]
                    if not os.path.exists(file_path):
                        logger.warning(f"⚠️ Video file not found: {file_path}")
                        # Continue anyway - CameraPipeline will handle missing files
                
                configs_to_use[approach] = value
                logger.info(f"✅ Valid approach: {key} -> {approach}")
                
            except ValueError:
                logger.warning(f"❌ Invalid approach key: {key}")
        
        if not configs_to_use:
            logger.warning("🔄 No valid configs found, creating default configs")
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
                    "source": f"file://uploads/{approach.value}.mp4"
                }
                configs_to_use[approach] = default_config
        
        # Start in background with REAL processing
        background_tasks.add_task(traffic_simulator.start, configs_to_use)
        
        return {
            "message": "Model starting with REAL video processing", 
            "running": True,
            "approaches_configured": len(configs_to_use)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model: {str(e)}")

@router.post("/stop")
async def stop_model(background_tasks: BackgroundTasks):
    """Stop the traffic optimization model"""
    try:
        background_tasks.add_task(traffic_simulator.stop)
        return {"message": "Model stopping in background", "running": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop model: {str(e)}")

@router.get("/state")
async def get_state():
    """Get current system state"""
    return await system_state.get_state_snapshot()

@router.get("/plan")
async def get_plan():
    """Get current cycle plan"""
    return {"cycle_plan": system_state.cycle_plan}

@router.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "running": system_state.running}