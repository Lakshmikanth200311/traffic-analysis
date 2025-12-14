from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.cycle_timings.timing_optimizer import TrafficTimingOptimizer
from app.services.yolo.detector import YOLODetector
from app.models.schemas import CyclePlan
import os
import shutil
import cv2
import numpy as np

router = APIRouter()
UPLOAD_DIR = "uploads/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ANNOT_DIR = "uploads/annotated"
os.makedirs(ANNOT_DIR, exist_ok=True)

def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        label = f"{det.class_name} ({det.confidence:.2f})"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def process_approach_image(image_path: str, approach: str, detector: YOLODetector) -> dict:
    img = cv2.imread(image_path)
    detections = detector.detect(img)
    annotated_img = draw_boxes(img.copy(), detections)
    annot_path = os.path.join(ANNOT_DIR, f"{approach}_annotated_{os.path.basename(image_path)}")
    cv2.imwrite(annot_path, annotated_img)

    count_by_class = {}
    label_set = set()
    confidence_list = []
    for det in detections:
        count_by_class[det.class_name] = count_by_class.get(det.class_name, 0) + 1
        label_set.add(det.class_name)
        confidence_list.append(det.confidence)

    acc = round(float(np.mean(confidence_list)), 3) if confidence_list else 0.0

    return {
        "approach": approach,
        "vehicle_count": count_by_class,
        "labels": list(label_set),
        "average_accuracy": acc,
        "annotated_image_url": f"/api/annotated-image/{approach}/{os.path.basename(annot_path)}"
    }

@router.post("/calc-cycle-plan")
async def calc_cycle_plan(
    north: UploadFile = File(...),
    south: UploadFile = File(...),
    east: UploadFile = File(...),
    west: UploadFile = File(...)
):
    # Save four uploaded images
    paths = {}
    results = {}
    detector = YOLODetector()
    for name, file in zip(['north', 'south', 'east', 'west'], [north, south, east, west]):
        file_path = os.path.join(UPLOAD_DIR, f"{name}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        paths[name] = file_path
        # Annotate and gather stats
        results[name] = process_approach_image(file_path, name, detector)
    # Instantiate optimizer and calculate cycle plan
    optimizer = TrafficTimingOptimizer()
    plan = optimizer.compute_cycle_plan(paths, detector, cycle_version=1)
    # Return detection results and cycle plan in one response
    return {
        "detection_results": results,
        "cycle_plan": plan
    }

@router.get("/annotated-image/{approach}/{filename}")
async def get_annotated_image(approach: str, filename: str):
    full_path = os.path.join(ANNOT_DIR, f"{approach}_annotated_{filename}")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(full_path, media_type="image/jpeg")
