#!/usr/bin/env python3
"""
detect_inverse_distance_with_animals.py

Real‐time detection of people & animals + true distance (m) via inverse‐depth
+ simple facing‐toward/away for people (Haar cascade).

1) Paste your SCALE_CONSTANT from calibrate_inverse.py below.
2) Run: python detect_inverse_distance_with_animals.py
Press 'q' to quit.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ←—— Replace this with your new C from calibrate_inverse.py:
SCALE_CONSTANT = 572.145

# COCO class IDs for “animal” categories we care about:
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# 0 is “person”
DETECT_CLASSES = [0] + ANIMAL_CLASSES

def main():
    # 1) Load YOLO & MiDaS
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small").to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS","transforms")
    midas_transform = transforms.small_transform

    # 2) Haar for face detection (people only)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("❌ Could not load Haar cascade for face detection.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print(f"▶️ Detecting persons + animals; SCALE_CONSTANT={SCALE_CONSTANT:.3f}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # — build inverse‐depth map
        inp = midas_transform(frame).to(device)
        with torch.no_grad():
            pred = midas(inp).unsqueeze(1)
            pred = F.interpolate(
                pred,
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)
        depth_map = pred[0].cpu().numpy()

        # — run YOLO, filtering to person+animals
        results = model(frame, conf=0.5, classes=DETECT_CLASSES, verbose=False)[0]

        for box in results.boxes:
            # get box coords
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            # compute distance
            patch = depth_map[y1:y2, x1:x2]
            if patch.size < 10:
                continue
            d_rel = float(np.median(patch))
            d_m = SCALE_CONSTANT / d_rel

            # identify class
            cls_id = box.cls.cpu().numpy().astype(int)[0]
            cls_name = results.names[cls_id]  # e.g. "person", "dog", "cat", ...

            # if it's a person, check face → orientation
            if cls_id == 0:
                roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30,30)
                )
                if len(faces) > 0:
                    label = "Facing"
                    color = (0,255,0)
                else:
                    label = "Facing away"
                    color = (0,0,255)
                text = f"{d_m:.2f}m | {label}"
            else:
                # it's an animal
                label = cls_name
                color = (255,165,0)  # orange
                text = f"{d_m:.2f}m | {label}"

            # draw and annotate
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Detect: Person+Animals + Distance + Orientation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
