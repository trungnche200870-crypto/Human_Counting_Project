"""Exercise the actual WebSocket receiver and YOLO worker with the sample video."""
import asyncio
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from aiohttp.test_utils import TestClient, TestServer
from live_server import CameraSession, create_app


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/live_smoke'))
    output = parser.parse_args().output
    session = CameraSession(output_dir=output)
    async with TestClient(TestServer(create_app(session))) as client:
        ws = await client.ws_connect('/ws/camera')
        await ws.send_json({'token': session.camera_token})
        assert (await ws.receive_json())['ready']
        cap = cv2.VideoCapture('data/videos/people_walking.mp4')
        assert cap.isOpened()
        frames, start = 0, time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                scale = min(1, 640 / max(frame.shape[:2]))
                frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
                jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])[1].tobytes()
                await ws.send_bytes(jpg)
                assert (await ws.receive_json())['received']
                frames += 1
                deadline = time.monotonic() + 60
                while session.stats['processed_frames'] < frames:
                    assert not session.error, session.error
                    assert time.monotonic() < deadline, 'Inference timeout'
                    await asyncio.sleep(.005)
        finally:
            cap.release()
        assert frames > 0
        assert session.stats['processed_frames'] == frames
        headers = {'Authorization': 'Bearer ' + session.admin_token}
        response = await client.get('/api/frame', headers=headers)
        jpg = await response.read()
        assert jpg[:2] == b'\xff\xd8'
        (output / 'last_frame.jpg').write_bytes(jpg)
        report = {**session.stats, 'elapsed_seconds': round(time.perf_counter() - start, 2)}
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        await ws.close()
        assert not session.status()['connected']


if __name__ == '__main__':
    asyncio.run(main())
