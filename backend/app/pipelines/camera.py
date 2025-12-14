import cv2
import asyncio
from app.services.detector import YOLODetector
from app.services.counter import LineCrossingCounter
from app.models.schemas import Approach, VehicleCounts, Point, CountingLine
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraPipeline:
    def __init__(self, approach: Approach, config: Dict[str, Any]):
        self.approach = approach
        self.config = config
        self.is_running = False
        self.current_counts = VehicleCounts()
        self.arrival_rate = 0.0
        
        # Initialize real components
        self.detector = YOLODetector()
        
        # Handle dictionary input for counting_line
        counting_line_data = config.get('counting_line', {})
        start_data = counting_line_data.get('start', {'x': 600, 'y': 0})
        end_data = counting_line_data.get('end', {'x': 600, 'y': 720})
        
        counting_line = CountingLine(
            start=Point(x=start_data.get('x', 600), y=start_data.get('y', 0)),
            end=Point(x=end_data.get('x', 600), y=end_data.get('y', 720))
        )
        
        self.counter = LineCrossingCounter(counting_line, approach)
        self.cap = None
        self.processing_task = None
        
    async def start(self):
        """Start the camera pipeline with real video processing"""
        self.is_running = True
        
        # Extract video source from config
        source = self.config.get('source', '')
        if source.startswith('file://'):
            video_path = source[7:]  # Remove 'file://' prefix
        else:
            video_path = source
            
        logger.info(f"📹 Starting video processing for {self.approach}: {video_path}")
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"❌ Failed to open video: {video_path}")
            self.is_running = False
            return
            
        # Start background processing task
        self.processing_task = asyncio.create_task(self._process_video_frames())
        logger.info(f"📹 Started camera pipeline for {self.approach}")
        
    async def stop(self):
        """Stop the camera pipeline"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
        if self.cap:
            self.cap.release()
        logger.info(f"📹 Stopped camera pipeline for {self.approach}")
    
    async def _process_video_frames(self):
        """Process video frames and count vehicles in real-time"""
        frame_skip = 5  # Process every 5th frame for performance
        frame_count = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video when ended
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                # Detect vehicles in frame
                detections = self.detector.detect(frame)
                
                # Count vehicles crossing the line
                current_counts = self.counter.update(detections)
                self.current_counts = current_counts
                
                # Calculate arrival rate (vehicles per minute)
                await self._update_arrival_rate()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing video frames for {self.approach}: {e}")
                await asyncio.sleep(1)
    
    async def _update_arrival_rate(self):
        """Calculate vehicles per minute based on recent counts"""
        # Simple implementation - track counts over time
        # You might want to implement a more sophisticated rate calculation
        total_vehicles = self.current_counts.total
        # This is a simplified rate calculation - adjust based on your needs
        self.arrival_rate = total_vehicles / 10.0  # Rough estimate
    
    async def get_current_counts(self) -> VehicleCounts:
        """Get ACTUAL vehicle counts from video processing"""
        if not self.is_running:
            return VehicleCounts()
        return self.current_counts
    
    async def get_arrival_rate(self) -> float:
        """Get current arrival rate (vehicles per minute)"""
        if not self.is_running:
            return 0.0
        return self.arrival_rate
    
    async def get_queue_length(self) -> int:
        """Estimate queue length based on traffic density"""
        if not self.is_running:
            return 0
            
        # Simple estimation based on total vehicles detected
        counts = await self.get_current_counts()
        queue_length = min(50, counts.total * 2)  # More realistic estimation
        return max(0, queue_length)