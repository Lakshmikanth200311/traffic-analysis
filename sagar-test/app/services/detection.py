from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

def detect_and_annotate(image_path, save_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    output_data = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls_id]

            if label in VEHICLE_CLASSES:
                output_data.append({
                    "class": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    return output_data
