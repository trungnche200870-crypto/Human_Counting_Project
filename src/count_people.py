
import cv2
import time
import torch

from pathlib import Path
from ultralytics import YOLO


# =====================================
# 1. CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data" / "videos" / "people_walking.mp4"
MODEL_PATH = ROOT / "yolo11n.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

DISPLAY_WIDTH = 960

# Counting settings
LINE_POSITION = 0.50
MARGIN_RATIO = 0.02


# =====================================
# 2. LOAD MODEL AND VIDEO
# =====================================

print("Using device:", DEVICE)

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_y = int(height * LINE_POSITION)
margin = max(5, int(height * MARGIN_RATIO))

print("Video resolution:", width, "x", height)
print("Counting line Y:", line_y)


# =====================================
# 3. COUNTING VARIABLES
# =====================================

in_count = 0
out_count = 0

# Last confirmed side for each track
last_side = {}

# IDs already counted
counted_ids = set()


# =====================================
# 4. SIDE DETECTION
# =====================================

def get_side(y):
    if y < line_y - margin:
        return "TOP"

    if y > line_y + margin:
        return "BOTTOM"

    return "MIDDLE"


# =====================================
# 5. VIDEO PROCESSING
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video completed!")
        break

    start_time = time.perf_counter()

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    tracked_people = 0

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        tracked_people = len(ids)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Bottom-center point
            cx = (x1 + x2) // 2
            cy = y2

            side = get_side(cy)

            # Draw tracking point
            cv2.circle(
                annotated,
                (cx, cy),
                5,
                (0, 255, 255),
                -1
            )

            # Ignore uncertain middle zone
            if side == "MIDDLE":
                continue

            if track_id not in last_side:
                last_side[track_id] = side
                continue

            previous_side = last_side[track_id]

            if (
                previous_side != side
                and track_id not in counted_ids
            ):

                if previous_side == "TOP" and side == "BOTTOM":
                    in_count += 1
                    counted_ids.add(track_id)
                    print(f"IN: ID {track_id}")

                elif previous_side == "BOTTOM" and side == "TOP":
                    out_count += 1
                    counted_ids.add(track_id)
                    print(f"OUT: ID {track_id}")

            last_side[track_id] = side


    # =================================
    # 6. DRAW COUNTING LINE
    # =================================

    cv2.line(
        annotated,
        (0, line_y),
        (width, line_y),
        (0, 0, 255),
        3
    )

    # Draw margin boundaries
    cv2.line(
        annotated,
        (0, line_y - margin),
        (width, line_y - margin),
        (100, 100, 100),
        1
    )

    cv2.line(
        annotated,
        (0, line_y + margin),
        (width, line_y + margin),
        (100, 100, 100),
        1
    )


    # =================================
    # 7. DISPLAY STATISTICS
    # =================================

    elapsed = time.perf_counter() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 0

    # Background panel
    cv2.rectangle(
        annotated,
        (10, 10),
        (340, 165),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        annotated,
        f"IN: {in_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"OUT: {out_count}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 180, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Tracking: {tracked_people}",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (25, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =================================
    # 8. DISPLAY VIDEO
    # =================================

    display_height = int(
        height * DISPLAY_WIDTH / width
    )

    display_frame = cv2.resize(
        annotated,
        (DISPLAY_WIDTH, display_height)
    )

    cv2.imshow(
        "Human Counting - IN OUT",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================
# 9. CLEANUP
# =====================================

cap.release()
cv2.destroyAllWindows()

print("==========================")
print("FINAL COUNTING RESULTS")
print("IN:", in_count)
print("OUT:", out_count)
print("TOTAL:", in_count + out_count)
print("==========================")
