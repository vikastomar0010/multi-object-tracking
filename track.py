import cv2
import time
import random
from collections import defaultdict
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")  

# Video input
cap = cv2.VideoCapture("input/video.mp4")

# Video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
out = cv2.VideoWriter("output/outputvideo.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps_video,
                      (width, height))

# Track history for trajectory
track_history = defaultdict(list)

# Unique player IDs
unique_ids = set()

# FPS calculation
prev_time = time.time()

# Color generator
def get_color(id):
    random.seed(id)
    return (random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Tracking (person + ball)
    results = model.track(frame, persist=True, classes=[0, 32])

    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                cls = int(cls)

                # Center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Save trajectory
                track_history[track_id].append((cx, cy))

                # Limit history length
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Ball → RED color
                if cls == 32:
                    color = (0, 0, 255)
                    label = f"Ball ID {track_id}"
                else:
                    color = get_color(track_id)
                    label = f"ID {track_id}"
                    unique_ids.add(track_id)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                cv2.putText(annotated_frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                # Draw trajectory
                for i in range(1, len(track_history[track_id])):
                    cv2.line(annotated_frame,
                             track_history[track_id][i - 1],
                             track_history[track_id][i],
                             color, 2)

    # Person count
    cv2.putText(annotated_frame,
                f"Persons: {len(unique_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(annotated_frame,
                f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2)

    out.write(annotated_frame)

cap.release()
out.release()

print("✅ Final tracking video saved as output_tracked.mp4")