import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    summary = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]

                if label in VEHICLE_CLASSES:
                    detections.append(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        summary.append({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "detected": detections
        })

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))

        out.write(frame)

    cap.release()
    out.release()
    return summary
