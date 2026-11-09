
import csv
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def process_video(
    input_path,
    output_dir,
    confidence=0.25,
    line_position=0.50,
    progress_callback=None,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    model_path = root / "yolo11n.pt"

    output_video = output_dir / "result.mp4"
    summary_csv = output_dir / "summary.csv"
    events_csv = output_dir / "events.csv"

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    line_y = int(height * line_position)
    margin = max(5, int(height * 0.02))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    in_count = 0
    out_count = 0
    last_side = {}
    counted_ids = set()
    events = []

    frame_index = 0
    started = time.perf_counter()

    def get_side(y):
        if y < line_y - margin:
            return "TOP"
        if y > line_y + margin:
            return "BOTTOM"
        return "MIDDLE"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=confidence,
                imgsz=640,
                device=device,
                verbose=False,
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
                    cx = (x1 + x2) // 2
                    cy = min(y2, height - 1)

                    cv2.circle(
                        annotated, (cx, cy), 5,
                        (0, 255, 255), -1
                    )

                    side = get_side(cy)

                    if side == "MIDDLE":
                        continue

                    if track_id not in last_side:
                        last_side[track_id] = side
                        continue

                    previous = last_side[track_id]

                    if (
                        previous != side
                        and track_id not in counted_ids
                    ):
                        direction = None

                        if previous == "TOP" and side == "BOTTOM":
                            in_count += 1
                            direction = "IN"

                        elif previous == "BOTTOM" and side == "TOP":
                            out_count += 1
                            direction = "OUT"

                        if direction:
                            counted_ids.add(track_id)
                            events.append({
                                "frame": frame_index,
                                "video_time_seconds": round(
                                    (frame_index - 1) / fps, 3
                                ),
                                "track_id": track_id,
                                "direction": direction,
                            })

                    last_side[track_id] = side

            cv2.line(
                annotated,
                (0, line_y),
                (width, line_y),
                (0, 0, 255),
                3,
            )

            elapsed = time.perf_counter() - started
            average_fps = frame_index / max(elapsed, 0.001)

            cv2.rectangle(
                annotated, (10, 10), (380, 170),
                (20, 20, 20), -1
            )

            stats = [
                f"IN: {in_count}",
                f"OUT: {out_count}",
                f"Tracking: {tracked_people}",
                f"Processing FPS: {average_fps:.1f}",
            ]

            for i, text in enumerate(stats):
                cv2.putText(
                    annotated,
                    text,
                    (25, 45 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            writer.write(annotated)

            if progress_callback and frame_index % 10 == 0:
                if total_frames > 0:
                    progress_callback(
                        min(frame_index / total_frames, 1.0)
                    )

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    average_fps = frame_index / max(elapsed, 0.001)

    with open(
        events_csv, "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "frame", "video_time_seconds",
            "track_id", "direction"
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(events)

    summary = {
        "source_video": input_path.name,
        "processed_frames": frame_index,
        "in_count": in_count,
        "out_count": out_count,
        "total_crossings": in_count + out_count,
        "processing_seconds": round(elapsed, 2),
        "average_processing_fps": round(average_fps, 2),
    }

    with open(
        summary_csv, "w", newline="", encoding="utf-8-sig"
    ) as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=summary.keys()
        )
        csv_writer.writeheader()
        csv_writer.writerow(summary)

    if progress_callback:
        progress_callback(1.0)

    return {
        "video": output_video,
        "summary_csv": summary_csv,
        "events_csv": events_csv,
        "summary": summary,
    }
