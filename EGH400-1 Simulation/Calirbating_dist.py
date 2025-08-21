#!/usr/bin/env python3
"""
calibrate_inverse.py

Place a person (or flat board) at KNOWN_DISTANCE metres from your camera,
then run once to compute your inverse‐depth scale constant C.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ▶ Set this to the distance you can measure exactly:
KNOWN_DISTANCE   = 1   # metres
NUM_FRAMES       = 50     # how many frames to average

def main():
    yolo = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small").to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS","transforms")
    midas_transform = transforms.small_transform

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam."); return

    readings = []
    print(f"▶️  Hold target at {KNOWN_DISTANCE} m for {NUM_FRAMES} frames...")

    while len(readings) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1) detect the first person
        res = yolo(frame, conf=0.5, classes=[0], verbose=False)[0]
        if not res.boxes:
            cv2.imshow("Calibrating", frame); cv2.waitKey(1)
            continue
        x1,y1,x2,y2 = res.boxes.xyxy.cpu().numpy().astype(int)[0]

        # 2) build the MiDaS depth map
        inp = midas_transform(frame).to(device)
        with torch.no_grad():
            pred = midas(inp).unsqueeze(1)
            pred = F.interpolate(pred,
                                 size=frame.shape[:2],
                                 mode="bicubic",
                                 align_corners=False).squeeze(1)
        depth_map = pred[0].cpu().numpy()

        # 3) sample median depth in that box
        patch = depth_map[y1:y2, x1:x2]
        if patch.size:
            readings.append(float(np.median(patch)))
            print(f"\rReading {len(readings)}/{NUM_FRAMES}: {readings[-1]:.1f}", end="")

        # show progress
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.imshow("Calibrating", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    mean_rel = sum(readings)/len(readings)
    # For inverse‐depth: KNOWN_DISTANCE = C / mean_rel  ⇒  C = KNOWN_DISTANCE * mean_rel
    C = KNOWN_DISTANCE * mean_rel
    print(f"\n🔧 Calibration done. SCALE_CONSTANT = {C:.3f}  (m·unit)")
    print("▶ Copy this value into detect_inverse_distance.py and save.")
    

if __name__ == "__main__":
    main()
