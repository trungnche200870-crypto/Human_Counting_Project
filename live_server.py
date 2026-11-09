"""One-camera phone capture server. Run locally; expose only through HTTPS."""
import argparse
import asyncio
import csv
import io
import json
import logging
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from aiohttp import web, WSMsgType
from PIL import Image

from src.live_counter import LiveProcessor

ROOT = Path(__file__).resolve().parent
FIELDS = ["timestamp_utc", "track_id", "direction", "in_count", "out_count"]
LOG = logging.getLogger(__name__)


class CameraSession:
    def __init__(self, processor_factory=None, output_dir=None, model_path=None, image_size=960):
        self.admin_token = secrets.token_urlsafe(32)
        self.camera_token = secrets.token_urlsafe(32)
        self.public_base = ""
        self.confidence = .25
        self.reverse = False
        self.generation = 0
        self.reset_version = 0
        self.socket = None
        self.queue = asyncio.Queue(maxsize=1)
        self.latest = None
        self.last_received = 0
        self.stats = {"in_count": 0, "out_count": 0, "tracked": 0, "fps": 0, "processed_frames": 0}
        self.error = ""
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processor = None
        self.factory = processor_factory or (lambda: LiveProcessor(model_path or ROOT / 'yolo11s.pt', image_size))
        self.output_dir = Path(output_dir or ROOT / 'outputs' / ('live_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / 'events.csv'
        with self.events_path.open('w', newline='', encoding='utf-8-sig') as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        self.task = None

    def status(self):
        age = time.monotonic() - self.last_received if self.last_received else None
        return {**self.stats, "connected": self.socket is not None and age is not None and age < 3,
                "frame_age_seconds": round(age, 1) if age is not None else None,
                "error": self.error, "reverse": self.reverse, "confidence": self.confidence,
                "public_base": self.public_base,
                "camera_link": self.public_base + '/camera#' + self.camera_token if self.public_base else ""}

    def infer(self, data, confidence, reverse, reset, reconnect):
        with Image.open(io.BytesIO(data)) as header:
            if header.format != 'JPEG' or max(header.size) > 1280:
                raise ValueError('Expected JPEG up to 1280 pixels')
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > 1280:
            raise ValueError("Invalid frame or resolution above 1280 pixels")
        if self.processor is None:
            self.processor = self.factory()
        return self.processor.process(frame, confidence, reverse, reset=reset, reconnect=reconnect)

    async def worker(self):
        last_generation, applied_reset = -1, -1
        loop = asyncio.get_running_loop()
        while True:
            generation, version, received, data = await self.queue.get()
            if generation != self.generation or version != self.reset_version or time.monotonic() - received > 3:
                continue
            start = time.monotonic()
            try:
                jpg, counts, events = await loop.run_in_executor(
                    self.executor, self.infer, data, self.confidence, self.reverse,
                    version != applied_reset, generation != last_generation)
                last_generation, applied_reset = generation, version
                if generation != self.generation or version != self.reset_version:
                    continue
                self.latest = jpg
                self.stats.update(counts)
                self.stats['fps'] = round(1 / max(time.monotonic() - start, .001), 1)
                self.stats['processed_frames'] += 1
                self.error = ""
                with self.events_path.open('a', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    for event in events:
                        writer.writerow({**event, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                                         'in_count': counts['in_count'], 'out_count': counts['out_count']})
            except Exception as exc:
                LOG.exception('Camera processing failed')
                self.error = str(exc)
                last_generation = -1


def create_app(session=None):
    session = session or CameraSession()

    @web.middleware
    async def access(request, handler):
        if request.path.startswith('/api/'):
            if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + session.admin_token):
                raise web.HTTPUnauthorized()
        response = await handler(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    app = web.Application(middlewares=[access], client_max_size=512 * 1024)

    async def page(request):
        name = 'camera.html' if request.path == '/camera' else 'dashboard.html'
        return web.FileResponse(ROOT / 'web' / name)

    async def asset(request):
        name = request.match_info['name']
        if name not in ('camera.js', 'dashboard.js', 'style.css'):
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / 'web' / name)

    async def status(request):
        return web.json_response(session.status())

    async def image(request):
        if session.latest is None:
            return web.Response(status=204)
        return web.Response(body=session.latest, content_type='image/jpeg')

    async def configure(request):
        try:
            data = await request.json()
            confidence = float(data.get('confidence', session.confidence))
            base = str(data.get('public_base', session.public_base)).rstrip('/')
            if not .1 <= confidence <= .9:
                raise ValueError('Confidence must be between 0.1 and 0.9')
            parsed = urlsplit(base)
            if base and (parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username):
                raise ValueError('Enter an HTTPS origin such as https://camera.example.com')
            reverse = data.get('reverse', session.reverse)
            if not isinstance(reverse, bool):
                raise ValueError('Invalid direction')
        except (ValueError, TypeError):
            raise web.HTTPBadRequest(text='Invalid settings: HTTPS origin and confidence 0.1–0.9 required')
        if reverse != session.reverse or confidence != session.confidence:
            session.generation += 1
        session.confidence, session.public_base, session.reverse = confidence, base, reverse
        return web.json_response(session.status())

    async def reset(request):
        session.reset_version += 1
        session.stats.update(in_count=0, out_count=0, tracked=0, processed_frames=0)
        session.latest = None
        return web.json_response(session.status())

    async def disconnect(request):
        session.camera_token = secrets.token_urlsafe(32)
        session.generation += 1
        if session.socket is not None:
            await session.socket.close()
        return web.json_response(session.status())

    async def events(request):
        return web.Response(body=session.events_path.read_bytes(), content_type='text/csv',
                            headers={'Content-Disposition': 'attachment; filename="camera_events.csv"'})

    async def camera(request):
        ws = web.WebSocketResponse(heartbeat=15, max_msg_size=512 * 1024)
        await ws.prepare(request)
        try:
            message = await asyncio.wait_for(ws.receive(), 10)
            auth = json.loads(message.data) if message.type == WSMsgType.TEXT else {}
            if not isinstance(auth, dict) or not isinstance(auth.get('token'), str) or not secrets.compare_digest(auth['token'], session.camera_token):
                await ws.close(code=1008, message=b'Invalid pairing link')
                return ws
            if session.socket is not None:
                await ws.close(code=1008, message=b'Another camera is connected')
                return ws
            session.socket = ws
            session.generation += 1
            await ws.send_json({'ready': True})
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    now = time.monotonic()
                    if now - session.last_received > 3:
                        session.generation += 1
                    session.last_received = now
                    if session.queue.full():
                        session.queue.get_nowait()
                    session.queue.put_nowait((session.generation, session.reset_version, now, message.data))
                    await ws.send_json({'received': True})
                elif message.type == WSMsgType.ERROR:
                    break
        except (TimeoutError, ValueError, TypeError):
            await ws.close(code=1008, message=b'Invalid camera handshake')
        finally:
            if session.socket is ws:
                session.socket = None
                session.generation += 1
        return ws

    async def lifecycle(app):
        session.task = asyncio.create_task(session.worker())
        yield
        if session.socket is not None:
            await session.socket.close()
        session.task.cancel()
        await asyncio.gather(session.task, return_exceptions=True)
        session.executor.shutdown(wait=True, cancel_futures=True)

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/', page)
    app.router.add_get('/camera', page)
    app.router.add_get('/assets/{name}', asset)
    app.router.add_get('/ws/camera', camera)
    app.router.add_get('/api/status', status)
    app.router.add_get('/api/frame', image)
    app.router.add_get('/api/events', events)
    app.router.add_post('/api/config', configure)
    app.router.add_post('/api/reset', reset)
    app.router.add_post('/api/disconnect', disconnect)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--model', type=Path, default=ROOT / 'yolo11s.pt')
    parser.add_argument('--image-size', type=int, choices=[640, 960, 1280], default=960)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f'Model not found: {args.model}. Download the official weights or use --model yolo11n.pt.')
    session = CameraSession(model_path=args.model, image_size=args.image_size)
    print(f'Computer dashboard (keep private): http://127.0.0.1:{args.port}/#{session.admin_token}', flush=True)
    web.run_app(create_app(session), host=args.host, port=args.port, access_log=None)
