
import cv2
import time
import torch

from pathlib import Path
from ultralytics import YOLO


# ====================================
# 1. PROJECT PATHS
# ====================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "people_walking.mp4"

MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"


# ====================================
# 2. CHECK GPU
# ====================================

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if device == "cpu":
    print("WARNING: CUDA unavailable. Running on CPU.")
else:
    print("GPU:", torch.cuda.get_device_name(0))


# ====================================
# 3. LOAD YOLO MODEL
# ====================================

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully!")


# ====================================
# 4. OPEN VIDEO
# ====================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video!")

print("Video opened successfully!")


# ====================================
# 5. PROCESS VIDEO
# ====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed!")
        break

    start_time = time.perf_counter()

    results = model.predict(
        source=frame,
        device=device,
        classes=[0],
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    person_count = len(result.boxes)

    annotated_frame = result.plot()

    elapsed = time.perf_counter() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 0

    # Display number of people
    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # Resize video for display only
    display_width = 960
    display_height = 540

    display_frame = cv2.resize(
        annotated_frame,
        (display_width, display_height)
    )

    cv2.imshow("Human Counting - YOLO Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ====================================
# 6. RELEASE RESOURCES
# ====================================

cap.release()
cv2.destroyAllWindows()

print("Detection finished successfully!")
