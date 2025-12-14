import cv2
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging

from app.services.detector import YOLODetector

logger = logging.getLogger(__name__)

# Create a new router for video streaming
video_router = APIRouter()

class VideoStreamer:
    def __init__(self):
        self.caps = {}  # Store video captures for each approach
        self.detector = None  # will lazily instantiate YOLODetector

    async def get_video_stream(self, approach: str):
        """Generate video stream with YOLO detection overlay"""
        try:
            # Get the video source for this approach
            video_source = await self._get_video_source(approach)
            if not video_source:
                raise HTTPException(status_code=404, detail=f"No video source for {approach}")

            # Open video capture
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail=f"Failed to open video for {approach}")

            self.caps[approach] = cap

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop video when ended
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Process frame with YOLO detection (draw bboxes + labels)
                processed_frame = await self._process_frame(frame, approach)

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Small delay to control frame rate
                await asyncio.sleep(0.03)  # ~30 FPS

        except Exception as e:
            logger.error(f"Video streaming error for {approach}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_video_source(self, approach: str) -> str:
        """Get video source path for the given approach"""
        # You can get this from your system_state or config
        # For now, using the uploaded files path
        import os
        video_path = f"uploads/{approach}.mp4"
        if os.path.exists(video_path):
            return video_path
        return None

    async def _process_frame(self, frame, approach: str):
        """Process frame with vehicle detection overlay (draw bboxes + labels)."""
        try:
            # Lazy-load detector once (so startup cost occurs only when first streaming)
            if self.detector is None:
                try:
                    self.detector = YOLODetector()
                except Exception as e:
                    logger.error(f"Failed to initialize YOLODetector in VideoStreamer: {e}")
                    self.detector = None

            detections = []
            if self.detector:
                try:
                    detections = self.detector.detect(frame)
                except Exception as e:
                    logger.error(f"Detector error in VideoStreamer: {e}")
                    detections = []

            # Draw detections (bbox + label)
            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    label = f"{det.class_name} {det.confidence:.2f}"

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = max(0, y1 - 6)
                    cv2.rectangle(frame, (x1, label_y - h - 2), (x1 + w, label_y + 2), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except Exception as e:
                    logger.debug(f"Error drawing detection: {e}")

            # Add simple overlay with approach and (approx) counts from system_state if available
            try:
                from app.models.state import system_state
                counts = {}
                try:
                    counts = system_state.live_counts.get(approach, {})
                except Exception:
                    counts = {}
                total_vehicles = counts.get('total', 0)
                cv2.putText(frame, f"{approach.upper()} - Vehicles: {total_vehicles}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception:
                # If system_state not accessible, skip overlay
                pass

            return frame

        except Exception as e:
            logger.error(f"Frame processing error in VideoStreamer: {e}")
            return frame

# Create global video streamer instance
video_streamer = VideoStreamer()

@video_router.get("/video_feed/{approach}")
async def video_feed(approach: str):
    """Video streaming route for each approach"""
    return StreamingResponse(
        video_streamer.get_video_stream(approach),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
