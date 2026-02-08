import cv2
import torch
import numpy as np
import pandas as pd
import time
from collections import defaultdict

# =========================
# LOAD YOLOv5 MODEL
# =========================
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

# Classes
PERSON_CLASS = 'person'
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

# =========================
# VIDEO INPUT
# =========================
import sys
VIDEO_PATH = sys.argv[1]
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# LINE POSITION (CHANGE IF NEEDED)
# =========================
LINE_Y = 300  # horizontal line

# =========================
# COUNTERS
# =========================
footfall_count = 0
vehicle_count = 0
log_data = []
start_time = time.time()
last_logged_time = 0 


counted_ids = set()

# =========================
# SIMPLE ID TRACKING (CENTROID BASED)
# =========================
object_id = 0
objects = {}

def get_centroid(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    current_objects = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]

        if label != PERSON_CLASS and label not in VEHICLE_CLASSES:
            continue

        cx, cy = get_centroid(x1, y1, x2, y2)

        matched_id = None
        for oid, (px, py) in objects.items():
            if abs(cx - px) < 40 and abs(cy - py) < 40:
                matched_id = oid
                break

        if matched_id is None:
            matched_id = object_id
            object_id += 1

        current_objects[matched_id] = (cx, cy)

        # =========================
        # LINE CROSSING LOGIC
        # =========================
        if matched_id not in counted_ids:
        # Crossed line from TOP to BOTTOM
         last_count_time = 0
         COUNT_DELAY = 2  # seconds (jitna bada, utna slow count)

         previous_positions = {}
         cx, cy = get_centroid(x1, y1, x2, y2)
         prev_cy = previous_positions.get(matched_id, cy)
         previous_positions[matched_id] = cy

         if prev_cy < LINE_Y and cy >= LINE_Y:
           footfall_count 

        if label == PERSON_CLASS:
            footfall_count += 1
        else:
            vehicle_count += 1


        # =========================
        # DRAW BOX
        # =========================
        color = (0, 255, 0) if label == PERSON_CLASS else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{label}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    objects = current_objects

    # Log every 1 second
    current_time = round(time.time() - start_time, 2)
    if current_time - last_logged_time >= 1:
        log_data.append({
            "Time_Seconds": current_time,
            "Footfall_Count": footfall_count,
            "Vehicle_Count": vehicle_count
        })
        last_logged_time = current_time

    # =========================
    # DRAW LINE & COUNTS
    # =========================
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)

    cv2.putText(frame, f"Footfall Count: {footfall_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Footfall & Vehicle Count", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE CSV FOR POWER BI
# =========================
df = pd.DataFrame(log_data)
df.to_csv("count_data.csv", index=False)
print("Data saved to count_data.csv")

