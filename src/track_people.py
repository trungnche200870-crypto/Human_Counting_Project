
import cv2
import time
import torch

from pathlib import Path
from ultralytics import YOLO


# ====================================
# 1. PROJECT PATHS
# ====================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT / "data" / "videos" / "people_walking.mp4"
)

MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"


# ====================================
# 2. GPU CHECK
# ====================================

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Running on CPU")


# ====================================
# 3. LOAD YOLO
# ====================================

model = YOLO(str(MODEL_PATH))

print("YOLO loaded successfully!")


# ====================================
# 4. OPEN VIDEO
# ====================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )

print("Video opened successfully!")


# ====================================
# 5. TRACKING LOOP
# ====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed!")
        break

    start_time = time.perf_counter()

    # YOLO + ByteTrack
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,
        imgsz=640,
        device=device,
        verbose=False
    )

    result = results[0]

    # Extract tracking IDs
    track_ids = []

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):
        track_ids = (
            result.boxes.id.int().cpu().tolist()
        )

    # Draw bounding boxes and tracking IDs
    annotated_frame = result.plot()

    # Current tracked people
    tracked_count = len(track_ids)

    # Calculate FPS
    elapsed = time.perf_counter() - start_time

    fps = 1.0 / elapsed if elapsed > 0 else 0

    # Display tracked people count
    cv2.putText(
        annotated_frame,
        f"Tracked People: {tracked_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Resize for display only
    display_frame = cv2.resize(
        annotated_frame,
        (960, 540)
    )

    cv2.imshow(
        "Human Counting - ByteTrack",
        display_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ====================================
# 6. RELEASE RESOURCES
# ====================================

cap.release()
cv2.destroyAllWindows()

print("Tracking finished successfully!")
