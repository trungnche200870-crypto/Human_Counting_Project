
import cv2
import csv
import time
import torch

from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# =====================================
# 1. CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data" / "videos" / "people_walking.mp4"
MODEL_PATH = ROOT / "yolo11n.pt"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "human_counting_result.mp4"
SUMMARY_CSV = OUTPUT_DIR / "counting_summary.csv"
EVENTS_CSV = OUTPUT_DIR / "counting_events.csv"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

LINE_POSITION = 0.50
MARGIN_RATIO = 0.02

DISPLAY_WIDTH = 960


# =====================================
# 2. LOAD MODEL AND VIDEO
# =====================================

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video resolution")

if video_fps <= 0:
    cap.release()
    raise RuntimeError("Invalid source video FPS")

line_y = int(height * LINE_POSITION)
margin = max(5, int(height * MARGIN_RATIO))

print("Resolution:", width, "x", height)
print("Source FPS:", video_fps)
print("Total frames:", total_frames)


# =====================================
# 3. VIDEO WRITER
# =====================================

# MP4 video using MPEG-4 Part 2 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    video_fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot initialize VideoWriter")


# =====================================
# 4. COUNTING VARIABLES
# =====================================

in_count = 0
out_count = 0

last_side = {}
counted_ids = set()

events = []

frame_index = 0
processed_frames = 0

processing_start = time.perf_counter()
completed_normally = False


def get_side(y):
    if y < line_y - margin:
        return "TOP"

    if y > line_y + margin:
        return "BOTTOM"

    return "MIDDLE"


# =====================================
# 5. MAIN PROCESSING LOOP
# =====================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            completed_normally = True
            print("Video processing completed!")
            break

        frame_index += 1

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

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            tracked_people = len(ids)

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center tracking point
                cx = (x1 + x2) // 2
                cy = min(y2, height - 1)

                side = get_side(cy)

                cv2.circle(
                    annotated,
                    (cx, cy),
                    5,
                    (0, 255, 255),
                    -1
                )

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

                    direction = None

                    if previous_side == "TOP" and side == "BOTTOM":
                        in_count += 1
                        direction = "IN"

                    elif previous_side == "BOTTOM" and side == "TOP":
                        out_count += 1
                        direction = "OUT"

                    if direction is not None:

                        counted_ids.add(track_id)

                        event_time = frame_index / video_fps

                        events.append({
                            "frame": frame_index,
                            "video_time_seconds": round(event_time, 3),
                            "track_id": track_id,
                            "direction": direction
                        })

                        print(
                            f"{direction}: ID {track_id} "
                            f"at {event_time:.2f}s"
                        )

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
        # 7. PERFORMANCE AND STATISTICS
        # =================================

        elapsed = time.perf_counter() - start_time
        processing_fps = 1.0 / elapsed if elapsed > 0 else 0

        cv2.rectangle(
            annotated,
            (10, 10),
            (370, 170),
            (20, 20, 20),
            -1
        )

        stats = [
            (f"IN: {in_count}", (0, 255, 0)),
            (f"OUT: {out_count}", (255, 180, 0)),
            (f"Tracking: {tracked_people}", (255, 255, 255)),
            (f"Processing FPS: {processing_fps:.1f}", (255, 255, 255))
        ]

        for i, (text, color) in enumerate(stats):

            cv2.putText(
                annotated,
                text,
                (25, 45 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )


        # =================================
        # 8. EXPORT FRAME
        # =================================

        writer.write(annotated)
        processed_frames += 1


        # =================================
        # 9. PREVIEW
        # =================================

        display_height = int(
            height * DISPLAY_WIDTH / width
        )

        display_frame = cv2.resize(
            annotated,
            (DISPLAY_WIDTH, display_height)
        )

        cv2.imshow(
            "Human Counting - Export",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing stopped by user.")
            break


        # =================================
        # 10. PROGRESS
        # =================================

        if frame_index % 100 == 0:

            if total_frames > 0:
                progress = frame_index / total_frames * 100
                print(
                    f"Processed {frame_index}/{total_frames} "
                    f"frames ({progress:.1f}%)"
                )
            else:
                print(f"Processed {frame_index} frames")

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# =====================================
# 11. EXPORT CSV FILES
# =====================================

total_time = time.perf_counter() - processing_start

average_fps = (
    processed_frames / total_time
    if total_time > 0 else 0
)

with open(
    EVENTS_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "frame",
        "video_time_seconds",
        "track_id",
        "direction"
    ]

    csv_writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    csv_writer.writeheader()
    csv_writer.writerows(events)


summary = {
    "source_video": VIDEO_PATH.name,
    "processed_at": datetime.now().isoformat(timespec="seconds"),
    "status": "completed" if completed_normally else "stopped_early",
    "processed_frames": processed_frames,
    "source_fps": round(video_fps, 2),
    "in_count": in_count,
    "out_count": out_count,
    "total_crossings": in_count + out_count,
    "processing_seconds": round(total_time, 2),
    "average_processing_fps": round(average_fps, 2)
}

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    csv_writer = csv.DictWriter(
        file,
        fieldnames=summary.keys()
    )

    csv_writer.writeheader()
    csv_writer.writerow(summary)


# =====================================
# 12. FINAL REPORT
# =====================================

print("\n==============================")
print("HUMAN COUNTING FINAL REPORT")
print("==============================")

print("Status:", summary["status"])
print("IN:", in_count)
print("OUT:", out_count)
print("TOTAL CROSSINGS:", in_count + out_count)
print("Processed frames:", processed_frames)
print("Average processing FPS:", round(average_fps, 2))

print("\nSaved files:")
print(OUTPUT_VIDEO)
print(SUMMARY_CSV)
print(EVENTS_CSV)

print("==============================")
