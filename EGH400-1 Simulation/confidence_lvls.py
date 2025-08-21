#!/usr/bin/env python3
"""
detect_with_confidence.py

Real-time detection of people & animals, showing class name + confidence.

1) Install dependencies: ultralytics, opencv-python, torch
2) Run: python detect_with_confidence.py
Press 'q' to quit.
"""

import cv2
import torch
from ultralytics import YOLO

# COCO class IDs for “animal” categories we care about:
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# 0 is “person”
DETECT_CLASSES = [0] + ANIMAL_CLASSES

def main():
    # Load YOLOv8
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("▶️ Detecting persons + animals with confidence labels")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=0.25, classes=DETECT_CLASSES, verbose=False)[0]

        for box in results.boxes:
            # Box coordinates
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            # Class id and name
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = results.names[cls_id]
            # Confidence score
            conf = float(box.conf.cpu().numpy()[0])

            # Choose color: green for person, orange for animals
            color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label with class + confidence
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("YOLOv8: Person+Animals + Confidence", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
