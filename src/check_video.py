
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "people_walking.mp4"

print("Video path:", VIDEO_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open video. Check file path and format.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully!")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", frame_count)

if fps > 0:
    print("Duration (seconds):", round(frame_count / fps, 2))

ret, frame = cap.read()

if ret:
    print("First frame read successfully!")
    cv2.imshow("Human Counting - Test Video", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Cannot read first frame.")

cap.release()
