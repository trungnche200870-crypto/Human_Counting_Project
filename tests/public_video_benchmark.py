"""Replay public clips through an isolated local camera WebSocket at phone settings.

Usage: python tests/public_video_benchmark.py [--fps 10]
Sources must already exist in data/public_samples/sources.json.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from live_server import CameraSession, create_app

ROOT = Path(__file__).resolve().parents[1]


async def run_clip(source, target_fps, output_root='public_video_tests'):
    name = source['name']
    output = ROOT / 'outputs' / output_root / f'{name}_{target_fps:g}fps'
    session = CameraSession(output_dir=output)
    cap = cv2.VideoCapture(str(ROOT / source['path']))
    if not cap.isOpened():
        raise RuntimeError('Cannot open ' + source['path'])
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    scale = min(1, 640 / max(width, height))
    size = (round(width * scale), round(height * scale))
    writer = cv2.VideoWriter(str(output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, size)
    assert writer.isOpened()
    inference_times, sent, peak_tracked, events, overruns = [], 0, 0, [], 0
    observed_crossings = 0
    try:
        async with TestClient(TestServer(create_app(session))) as client:
            # Warm model on a blank image; exclude model initialization from playback timing.
            blank = cv2.imencode('.jpg', np.zeros((size[1], size[0], 3), np.uint8))[1].tobytes()
            await asyncio.get_running_loop().run_in_executor(session.executor, session.infer, blank, .25, False, True, True)
            ws = await client.ws_connect('/ws/camera')
            await ws.send_json({'token': session.camera_token})
            assert (await ws.receive_json())['ready']
            started = time.perf_counter()
            decoded_index = -1
            for index in range(int((count - 1) / fps * target_fps) + 1):
                source_index = min(count - 1, round(index * fps / target_fps))
                ok = True
                while decoded_index < source_index:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    decoded_index += 1
                if not ok:
                    break  # Container metadata can overstate the last decodable frame.
                await asyncio.sleep(max(0, started + index / target_fps - time.perf_counter()))
                frame = cv2.resize(frame, size)
                jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])[1].tobytes()
                previous = session.stats['processed_frames']
                begin = time.perf_counter()
                await ws.send_bytes(jpg)
                assert (await ws.receive_json())['received']
                sent += 1
                deadline = time.monotonic() + 30
                while session.stats['processed_frames'] == previous:
                    assert not session.error, session.error
                    assert time.monotonic() < deadline, 'Frame processing timed out'
                    await asyncio.sleep(.002)
                elapsed = time.perf_counter() - begin
                inference_times.append(elapsed)
                overruns += elapsed > 1 / target_fps
                peak_tracked = max(peak_tracked, session.stats['tracked'])
                annotated = cv2.imdecode(np.frombuffer(session.latest, np.uint8), cv2.IMREAD_COLOR)
                cv2.putText(annotated, f"IN {session.stats['in_count']}  OUT {session.stats['out_count']}  t={source_index/fps:.1f}s",
                            (10, size[1]-12), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
                writer.write(annotated)
                current = session.stats['in_count'] + session.stats['out_count']
                if current > observed_crossings:
                    events.append({'time_seconds': round(source_index / fps, 3),
                                   'in_count': session.stats['in_count'], 'out_count': session.stats['out_count'],
                                   'new_crossings': current - observed_crossings})
                    observed_crossings = current
                    cv2.imwrite(str(output / f'crossing_{current:03d}.jpg'), annotated)
                if index % max(1, int(target_fps * 2)) == 0:
                    cv2.imwrite(str(output / f'frame_{index:04d}.jpg'), annotated)
            wall = time.perf_counter() - started
            assert sent > 0 and session.stats['processed_frames'] == sent
            await ws.close()
            assert not session.status()['connected']
            values = sorted(inference_times)
            report = {'name': name, 'source': source['source'], 'download_url': source['url'],
                      'duration_seconds': round(count/fps, 2), 'source_frames': count,
                      'source_fps': round(fps, 3), 'target_fps': target_fps, 'resolution': list(size),
                      'sent_frames': sent, **session.stats, 'wall_seconds': round(wall, 2),
                      'mean_frame_ms': round(sum(values)/len(values)*1000, 2),
                      'p95_frame_ms': round(values[min(len(values)-1,int(len(values)*.95))]*1000, 2),
                      'frames_over_budget': overruns, 'peak_tracked': peak_tracked,
                      'events_video_time': events, 'error': session.error,
                      'ground_truth': None,
                      'note': 'Local WebSocket replay, same JPEG size/quality as phone. Sequential processing with pacing; network is not simulated.'}
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
            return report
    finally:
        cap.release()
        writer.release()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--only', nargs='*', help='Optional source names')
    parser.add_argument('--output-root', default='public_video_tests')
    args = parser.parse_args()
    assert 0 < args.fps <= 30
    sources = json.loads((ROOT / 'data/public_samples/sources.json').read_text())
    if args.only:
        sources = [source for source in sources if source['name'] in args.only]
    results = []
    for source in sources:
        results.append(await run_clip(source, args.fps, args.output_root))
    output = ROOT / 'outputs' / args.output_root
    (output / f'summary_{args.fps:g}fps.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
