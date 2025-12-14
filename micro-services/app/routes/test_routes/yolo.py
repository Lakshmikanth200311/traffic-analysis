from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import cv2
import numpy as np
from app.services.yolo.detector import YOLODetector
import base64

router = APIRouter()
UPLOAD_DIR = "uploads"
RESULT_DIR = "uploads/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

detector = YOLODetector()

def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        label = f"{det.class_name} ({det.confidence:.2f})"
        color = (0, 255, 0)  # Green for all classes; change for per-class coloring if desired
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

@router.post("/yolo-image")
async def yolo_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    ext = os.path.splitext(file.filename)[1].lower()
    detections = []
    result_filename = f"result_{file.filename}"


    if ext in [".jpg", ".jpeg", ".png"]:
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        detections = detector.detect(image)
        # Draw boxes/labels on the image
        result_img = draw_boxes(image.copy(), detections)
        # Save output image
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_DIR, result_filename)
        cv2.imwrite(result_path, result_img)
        # Count vehicles by class
        count_by_class = {}
        for det in detections:
            count_by_class[det.class_name] = count_by_class.get(det.class_name, 0) + 1
        # Encode output image to base64 for direct API response (optional)
        with open(result_path, "rb") as img_f:
            img_bytes = img_f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return {
            "type": "image",
            "filename": file.filename,
            "detections": [d.model_dump() if hasattr(d, "model_dump") else d.__dict__ for d in detections],
            "vehicle_count": count_by_class,
            # "annotated_image_base64": img_b64,
            "result_image_url": f"/api/result-image/{result_filename}"
        }

    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        # --- Video annotation section ---
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(RESULT_DIR, result_filename)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_count = 0
        count_by_class = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            dets = detector.detect(frame)
            # Annotate frame
            annotated_frame = draw_boxes(frame, dets)
            writer.write(annotated_frame)
            # Count vehicles by class (aggregate over video)
            for det in dets:
                count_by_class[det.class_name] = count_by_class.get(det.class_name, 0) + 1
            # Optional: Limit for demo/testing
            if frame_count > 100:  # Remove/adjust for full videos
                break
        cap.release()
        writer.release()
        return {
            "type": "video",
            "filename": file.filename,
            "total_frames": frame_count,
            "vehicle_count": count_by_class,
            "result_video_url": f"/api/result-video/{result_filename}"
        }

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload image or video.")


@router.get("/result-image/{filename}")
async def result_image(filename: str):
    result_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(result_path, media_type="image/jpeg")

@router.get("/result-video/{filename}")
async def result_video(filename: str):
    result_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(result_path, media_type="video/mp4")