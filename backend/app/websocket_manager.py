import json
import logging
from fastapi import WebSocket
from typing import Dict, List, Any
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = "client"
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send immediate welcome message
        welcome_msg = {
            "type": "connection_established",
            "data": {"message": "WebSocket connected successfully"}
        }
        await self.send_personal_message(json.dumps(welcome_msg), websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        if not self.active_connections:
            logger.warning("No active WebSocket connections to broadcast to")
            return
            
        disconnected = []
        for connection in self.active_connections.keys():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

# Global WebSocket manager instance
websocket_manager = ConnectionManager()

class WebSocketBroadcaster:
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
    
    async def broadcast_system_state(self, system_state: Dict[str, Any]):
        """Broadcast system state to all connected clients"""
        message = {
            "type": "system_state",
            "data": system_state
        }
        await self.manager.broadcast(json.dumps(message))
        logger.info("📡 Broadcast system state")

    async def broadcast_live_counts(self, counts: List[Dict[str, Any]]):
        """Broadcast live counts to all connected clients"""
        message = {
            "type": "live_counts", 
            "data": {"counts": counts}
        }
        await self.manager.broadcast(json.dumps(message))
        logger.info("📡 Broadcast live counts")

    async def broadcast_phase_update(self, phase: str, remaining: int):
        """Broadcast phase update to all connected clients"""
        message = {
            "type": "phase_update",
            "data": {"phase": phase, "remaining_seconds": remaining}
        }
        await self.manager.broadcast(json.dumps(message))
        logger.info(f"📡 Broadcast phase update: {phase}, {remaining}s")

    async def broadcast_cycle_plan(self, plan: Dict[str, Any]):
        """Broadcast cycle plan to all connected clients"""
        message = {
            "type": "cycle_plan",
            "data": {"plan": plan}
        }
        await self.manager.broadcast(json.dumps(message))
        logger.info("📡 Broadcast cycle plan")

    async def broadcast_optimization_delta(self, deltas: List[Dict[str, Any]]):
        """Broadcast optimization deltas to all connected clients"""
        message = {
            "type": "optimization_delta", 
            "data": deltas
        }
        await self.manager.broadcast(json.dumps(message))
        logger.info("📡 Broadcast optimization deltas")

# Create global broadcaster instance
broadcaster = WebSocketBroadcaster(websocket_manager)