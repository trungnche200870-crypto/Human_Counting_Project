"""Stateful crossing counter and per-camera YOLO processor."""
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2


@dataclass
class TrackState:
    side: str = ""
    candidate: str = ""
    streak: int = 0
    seen: float = 0


class CrossingCounter:
    def __init__(self, confirmations=3, expiry=2.0):
        self.confirmations = confirmations
        self.expiry = expiry
        self.tracks = {}
        self.in_count = self.out_count = 0
        self.reverse = False

    def clear_tracks(self):
        self.tracks.clear()

    def reset(self):
        self.clear_tracks()
        self.in_count = self.out_count = 0

    def update(self, points, height, now=None):
        now = time.monotonic() if now is None else now
        self.tracks = {key: value for key, value in self.tracks.items()
                       if now - value.seen < self.expiry}
        line = height / 2
        margin = max(5, height * .02)
        events = []
        present = set()
        for track_id, y in points:
            present.add(track_id)
            state = self.tracks.setdefault(track_id, TrackState())
            state.seen = now
            side = "TOP" if y < line - margin else "BOTTOM" if y > line + margin else ""
            if not side:
                state.candidate, state.streak = "", 0
                continue
            state.streak = state.streak + 1 if state.candidate == side else 1
            state.candidate = side
            if state.streak < self.confirmations:
                continue
            if state.side and state.side != side:
                direction = "IN" if side == "BOTTOM" else "OUT"
                if self.reverse:
                    direction = "OUT" if direction == "IN" else "IN"
                if direction == "IN":
                    self.in_count += 1
                else:
                    self.out_count += 1
                events.append({"track_id": track_id, "direction": direction})
            state.side = side
        for key, state in self.tracks.items():
            if key not in present:
                state.candidate, state.streak = "", 0
        return events


class LiveProcessor:
    def __init__(self, model_path, image_size=960):
        # Constructed in the inference worker, never on the web event loop.
        from ultralytics import YOLO
        import torch
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.model = YOLO(str(model_path))
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        # Keep weak observations for association, but require the configured
        # confidence to start a track. Filtering them out loses existing IDs.
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.25,
            track_buffer=30, match_thresh=.8, fuse_score=True))
        self.counter = CrossingCounter()
        self.shape = None

    def clear_tracking(self):
        self.counter.clear_tracks()
        self.tracker.reset()

    def process(self, frame, confidence, reverse, reset=False, reconnect=False):
        if reconnect or self.shape != frame.shape:
            self.clear_tracking()
            self.shape = frame.shape
        if reset:
            self.clear_tracking()
            self.counter.reset()
        if reverse != self.counter.reverse:
            self.clear_tracking()
        self.counter.reverse = reverse
        self.tracker.args.track_high_thresh = confidence
        self.tracker.args.new_track_thresh = confidence
        self.tracker.args.track_low_thresh = min(.1, confidence)
        result = self.model.predict(frame, classes=[0], conf=min(.1, confidence),
                                    imgsz=self.image_size, device=self.device, verbose=False)[0]
        tracks = self.tracker.update(result.boxes.cpu().numpy(), frame)
        if len(tracks):
            import torch
            result = result[tracks[:, -1].astype(int)]
            result.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
        else:
            result = result[:0]
        annotated = result.plot()
        points = []
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist()):
                x1, _, x2, y2 = box
                y = min(y2, frame.shape[0] - 1)
                points.append((track_id, y))
                cv2.circle(annotated, (int((x1 + x2) / 2), int(y)), 4, (0, 255, 255), -1)
        events = self.counter.update(points, frame.shape[0])
        h, w = frame.shape[:2]
        for offset, color in [(0, (0, 0, 255)), (-max(5, int(h * .02)), (120, 120, 120)),
                              (max(5, int(h * .02)), (120, 120, 120))]:
            cv2.line(annotated, (0, h // 2 + offset), (w, h // 2 + offset), color, 2)
        label = "TOP -> BOTTOM: OUT" if reverse else "TOP -> BOTTOM: IN"
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("Cannot encode processed frame")
        return jpg.tobytes(), {"in_count": self.counter.in_count, "out_count": self.counter.out_count,
                               "tracked": len(points)}, events
