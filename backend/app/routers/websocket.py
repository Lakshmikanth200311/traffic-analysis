from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from app.models.schemas import WSMessage
from app.models.state import TrafficSystemState


router = APIRouter()
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()
system_state = TrafficSystemState()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            message = json.loads(data)
            logger.info(f"Received WS message: {message}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Utility functions to broadcast different message types
async def broadcast_live_counts(counts):
    message = WSMessage(
        type="live_counts",
        data={"counts": counts}
    )
    await manager.broadcast(message.json())

async def broadcast_phase_update(phase, remaining):
    message = WSMessage(
        type="phase_update", 
        data={"phase": phase, "remaining_seconds": remaining}
    )
    await manager.broadcast(message.json())

async def broadcast_cycle_plan(plan):
    message = WSMessage(
        type="cycle_plan",
        data={"plan": plan.dict()}
    )
    await manager.broadcast(message.json())