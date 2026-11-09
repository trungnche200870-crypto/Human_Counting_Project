import asyncio
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from live_server import CameraSession, create_app
from src.live_counter import CrossingCounter
from src.live_counter import LiveProcessor


class CounterTests(unittest.TestCase):
    def feed(self, counter, y, start, track=1):
        result = []
        for i in range(3):
            result += counter.update([(track, y)], 480, now=start + i * .1)
        return result

    def test_return_trip_and_stationary_person(self):
        counter = CrossingCounter()
        self.assertEqual(self.feed(counter, 100, 0), [])
        self.assertEqual(self.feed(counter, 400, .3)[0]['direction'], 'IN')
        self.assertEqual(self.feed(counter, 400, .6), [])
        self.assertEqual(self.feed(counter, 100, .9)[0]['direction'], 'OUT')
        self.assertEqual((counter.in_count, counter.out_count), (1, 1))

    def test_jitter_and_missing_detections(self):
        counter = CrossingCounter()
        self.feed(counter, 100, 0)
        for i, y in enumerate([239, 241, 255, 239, 255]):
            self.assertEqual(counter.update([(1, y)], 480, .3 + .1 * i), [])
        counter.update([], 480, .8)
        self.assertEqual(counter.update([(1, 400)], 480, .9), [])
        self.assertEqual(counter.in_count, 0)

    def test_reconnect_expiry_reverse_reset(self):
        counter = CrossingCounter()
        self.feed(counter, 100, 0)
        self.assertEqual(self.feed(counter, 400, 5), [])  # expired ID
        counter.clear_tracks()
        self.assertEqual(self.feed(counter, 100, 6), [])  # new baseline
        counter.reverse = True
        self.assertEqual(self.feed(counter, 400, 6.3)[0]['direction'], 'OUT')
        counter.reset()
        self.assertEqual((counter.in_count, counter.out_count, len(counter.tracks)), (0, 0, 0))


class FakeProcessor:
    def __init__(self):
        self.n = 0

    def process(self, frame, confidence, reverse, reset=False, reconnect=False):
        if reset:
            self.n = 0
        self.n += 1
        return cv2.imencode('.jpg', frame)[1].tobytes(), {'in_count': self.n, 'out_count': 0, 'tracked': 1}, [{'track_id': 1, 'direction': 'IN'}]


class TrackingRecoveryTests(unittest.TestCase):
    def test_weak_observations_continue_id_but_cannot_start_new_tracks(self):
        from ultralytics.engine.results import Results
        frame = np.zeros((480, 640, 3), np.uint8)
        model = Mock()
        with patch('ultralytics.YOLO', return_value=model), patch('torch.cuda.is_available', return_value=False):
            processor = LiveProcessor('unused.pt')
        observed_ids = set()
        events = []
        # First establish a strong track, then maintain it through weak detections
        # while an unrelated weak detection must never become a new person.
        for index, y in enumerate(list(range(120, 160, 10)) + list(range(160, 350, 10))):
            confidence = .8 if index < 4 else .15
            boxes = np.array([[50, y-100, 150, y, confidence, 0],
                              [350, 20, 450, 120, .15, 0]], dtype=np.float32)
            model.predict.return_value = [Results(frame, 'synthetic', {0:'person'}, boxes=boxes)]
            _, counts, emitted = processor.process(frame, .25, False)
            events.extend(emitted)
            self.assertEqual(counts['tracked'], 1)
            observed_ids.update(processor.counter.tracks)
        self.assertEqual(len(observed_ids), 1)
        self.assertEqual([event['direction'] for event in events], ['IN'])
        self.assertEqual(model.predict.call_args.kwargs['conf'], .1)
        # Raising confidence must affect new-track admission immediately.
        processor.clear_tracking()
        boxes = np.array([[50, 20, 150, 120, .6, 0]], dtype=np.float32)
        model.predict.return_value = [Results(frame, 'synthetic', {0:'person'}, boxes=boxes)]
        _, counts, _ = processor.process(frame, .7, False)
        self.assertEqual(counts['tracked'], 0)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = CameraSession(FakeProcessor, self.temp.name)
        self.client = TestClient(TestServer(create_app(self.session)))
        await self.client.start_server()
        self.headers = {'Authorization': 'Bearer ' + self.session.admin_token}

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def pair(self):
        ws = await self.client.ws_connect('/ws/camera')
        await ws.send_json({'token': self.session.camera_token})
        self.assertTrue((await ws.receive_json())['ready'])
        return ws

    async def wait_processed(self, n):
        for _ in range(100):
            if self.session.stats['processed_frames'] >= n:
                return
            await asyncio.sleep(.02)
        self.fail('Frame was not processed: ' + self.session.error)

    async def test_access_and_settings(self):
        for path in ['status', 'frame', 'events', 'reset', 'config', 'disconnect']:
            method = self.client.get if path in ('status','frame','events') else self.client.post
            self.assertEqual((await method('/api/' + path)).status, 401)
        response = await self.client.post('/api/config', headers=self.headers, json={'public_base': 'http://bad'})
        self.assertEqual(response.status, 400)
        response = await self.client.post('/api/config', headers=self.headers, json={'public_base': 'https://demo.example.com', 'reverse': True})
        self.assertEqual(response.status, 200)
        self.assertIn('/camera#', (await response.json())['camera_link'])
        self.assertEqual((await self.client.get('/assets/../live_server.py')).status, 404)

    async def test_camera_auth_and_exclusive_sender(self):
        ws = await self.client.ws_connect('/ws/camera')
        await ws.send_json({'token': 'wrong'})
        self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)
        ws = await self.pair()
        other = await self.client.ws_connect('/ws/camera')
        await other.send_json({'token': self.session.camera_token})
        self.assertEqual((await other.receive()).type, WSMsgType.CLOSE)
        await ws.close()

    async def test_frame_results_csv_reset_and_rotation(self):
        ws = await self.pair()
        jpg = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
        await ws.send_bytes(jpg)
        self.assertTrue((await ws.receive_json())['received'])
        await self.wait_processed(1)
        response = await self.client.get('/api/frame', headers=self.headers)
        self.assertEqual(response.content_type, 'image/jpeg')
        csv_data = await (await self.client.get('/api/events', headers=self.headers)).text(encoding='utf-8-sig')
        self.assertIn('IN', csv_data)
        self.assertEqual(self.session.stats['in_count'], 1)
        await self.client.post('/api/reset', headers=self.headers)
        self.assertEqual(self.session.stats['in_count'], 0)
        await ws.send_bytes(jpg)
        await ws.receive_json()
        await self.wait_processed(1)
        self.assertEqual(self.session.stats['in_count'], 1)
        old = self.session.camera_token
        await self.client.post('/api/disconnect', headers=self.headers)
        self.assertNotEqual(old, self.session.camera_token)
        self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)

    async def test_invalid_frame_recovers(self):
        ws = await self.pair()
        await ws.send_bytes(b'not a JPEG')
        await ws.receive_json()
        for _ in range(50):
            if self.session.error:
                break
            await asyncio.sleep(.02)
        self.assertTrue(self.session.error)
        jpg = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
        await ws.send_bytes(jpg)
        await ws.receive_json()
        await self.wait_processed(1)
        self.assertEqual(self.session.error, '')
        await ws.close()


if __name__ == '__main__':
    unittest.main()
