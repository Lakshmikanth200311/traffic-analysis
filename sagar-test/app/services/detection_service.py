from ultralytics import YOLO
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self):
        """Initialize YOLO model and vehicle classes"""
        try:
            self.model = YOLO("yolov8n.pt")
            self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def analyze_image(self, image_path, save_path):
        """Analyze single image for vehicle detection"""
        try:
            results = self.model(image_path)
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError("Could not load image")
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = self.model.names[cls_id]
                        
                        if label.lower() in self.vehicle_classes and conf > 0.3:
                            detections.append({
                                "class": label,
                                "confidence": round(conf, 3),
                                "bbox": [x1, y1, x2, y2]
                            })
                            
                            # Draw annotation
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{label} {conf:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 2)
            
            # Save annotated image
            cv2.imwrite(save_path, img)
            
            return {
                "vehicle_count": len(detections),
                "detections": detections,
                "image_path": save_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    def process_video_stream(self, video_path, save_path):
        """Process video frames and yield real-time results"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                results = self.model.predict(source=frame, verbose=False, conf=0.3)
                detections = []
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls)
                            conf = float(box.conf)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = self.model.names[cls_id]
                            
                            if label.lower() in self.vehicle_classes:
                                detections.append({
                                    "class": label,
                                    "confidence": round(conf, 3),
                                    "bbox": [x1, y1, x2, y2]
                                })
                                
                                # Draw annotation
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}", 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, (255, 0, 0), 2)
                
                # Write frame to output video
                out.write(frame)
                
                # Yield frame data for real-time processing
                yield {
                    "frame_number": frame_count,
                    "vehicle_count": len(detections),
                    "detections": detections,
                    "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                }
            
            # Cleanup
            cap.release()
            out.release()
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            raise