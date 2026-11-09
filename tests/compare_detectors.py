"""Diagnostic ablation: same JPEG frames, detector/tracker settings and confirmation counts."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from ultralytics import YOLO
from src.live_counter import CrossingCounter


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/accuracy_experiments'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name, image_size, floor in [('yolo11n.pt',640,.25), ('yolo11n.pt',640,.1),
                                         ('yolo11s.pt',640,.1), ('yolo11s.pt',960,.1)]:
        model = YOLO(str(root / model_name))
        counters = {n: CrossingCounter(confirmations=n) for n in (2,3)}
        cap = cv2.VideoCapture(str(root / 'data/public_samples/doorway.mp4'))
        times, traces, index = [], [], -1
        source_fps = cap.get(5)
        while True:
            ok, frame = cap.read()
            index += 1
            if not ok:
                break
            if index % round(source_fps/10):
                continue
            scale = min(1,640/max(frame.shape[:2]))
            frame = cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
            frame = cv2.imdecode(cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,65])[1],cv2.IMREAD_COLOR)
            start = time.perf_counter()
            result = model.track(frame, persist=True, tracker='bytetrack.yaml',classes=[0],conf=floor,
                                 imgsz=image_size,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
            times.append(time.perf_counter()-start)
            points, boxes = [], []
            if result.boxes is not None and result.boxes.id is not None:
                for box, track, score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.id.int().cpu().tolist(),result.boxes.conf.cpu().tolist()):
                    points.append((track,min(box[3],frame.shape[0]-1)))
                    boxes.append({'id':track,'xyxy':box,'score':score})
            events = {}
            for n, counter in counters.items():
                events[n] = counter.update(points,frame.shape[0],now=index/source_fps)
            traces.append({'time':index/source_fps,'boxes':boxes,'events':events})
        cap.release()
        name=f'{Path(model_name).stem}_{image_size}_{floor}'
        report={'model':model_name,'image_size':image_size,'detection_floor':floor,
                'counts':{n:{'in':c.in_count,'out':c.out_count} for n,c in counters.items()},
                'mean_warm_ms':round(sum(times[1:])/len(times[1:])*1000,2)}
        (output/(name+'.json')).write_text(json.dumps({'report':report,'traces':traces}))
        results.append(report)
        print(json.dumps(report),flush=True)
    (output/'comparison.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
