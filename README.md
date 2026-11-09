# Human Counting Project

Count people crossing a virtual line using YOLO11 and ByteTrack, with a Streamlit upload interface and a live phone-camera dashboard.

## VS Code on Windows

See [Hướng dẫn chạy trong VS Code](HUONG_DAN_VSCODE.md). This workspace includes F5 launch configurations for the video interface and both live camera models, plus environment, tunnel, and test tasks. All application configurations explicitly use `.venv/Scripts/python.exe`.

On this development machine, the Python 3.11 environment and NVIDIA CUDA support are already installed. On a new Windows checkout, run `python scripts/setup_environment.py` (with Python 3.11 installed) to create the environment, install dependencies, and download missing model files. Add `--with-tunnel` to download the optional official Cloudflare Tunnel tool.

## Setup

Use Python 3.11. From the project directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For GPU acceleration, install a PyTorch build compatible with your NVIDIA driver. The development machine uses PyTorch 2.11.0 with CUDA 12.8.

## Usage

Open the local Streamlit URL, upload an MP4, AVI, or MOV video, adjust confidence and counting-line position, then select **Run Human Counting**. Download the annotated video, summary CSV, and event CSV from the results.

- Top to bottom: IN.
- Bottom to top: OUT.
- The line defaults to the middle of the frame.
- Each tracked ID is counted at most once per processing run in the current implementation.

The repository includes `yolo11n.pt` and the sample video `data/videos/people_walking.mp4`. Generated results are saved under `outputs/` and excluded from Git.

## Files

- `app.py`: Streamlit interface.
- `src/counting_engine.py`: video processing and CSV export used by the interface.
- `src/check_video.py`: inspect and display the sample video.
- `src/detect_people.py`: standalone detection demo.
- `src/track_people.py`: standalone tracking demo.
- `src/count_people.py`: standalone IN/OUT counting demo.
- `src/export_results.py`: standalone result-export demo.

## Current status

Uploaded-video processing and a separate live phone-camera dashboard are implemented.

## Live iPhone camera over 4G/5G

Start the live server on the computer:

```powershell
.\.venv\Scripts\python.exe live_server.py
```

Open the private dashboard URL printed in the terminal. The fragment in this URL is the administrator credential: do not share it. The server listens on `127.0.0.1:8080` by default.

To reach it from an iPhone on mobile data, expose port 8080 using an HTTPS reverse proxy with WebSocket support. For a temporary development link, install the official [Cloudflare Tunnel client](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) and run in another terminal:

```powershell
cloudflared tunnel --url http://127.0.0.1:8080
```

Paste the resulting HTTPS origin into **Địa chỉ HTTPS** on the computer dashboard and save. Copy the **camera link**, open it directly in Safari on the iPhone, allow camera access and press **Bắt đầu truyền**. Keep Safari visible and the phone fixed. Switching away stops capture; return and press Start again. Audio is never requested.

The computer shows the annotated image, IN/OUT, tracked people, processing FPS, and connection state. The horizontal counting line is fixed at 50% of image height; the direction can be reversed. Reset clears counts while retaining previous CSV rows. Disconnect rotates the camera credential, invalidating the old camera link. Only one phone may send at a time.

Live mode defaults to the included `yolo11s.pt` model at inference size 960. It retains detections down to confidence 0.10 for recovering existing tracks; the dashboard confidence controls the score needed for strong association and starting new tracks. Weak detections alone cannot start a new person. Three observations beyond the dead band are still required, and the line remains fixed in the middle. The uploaded-video mode continues using YOLO11n.

For slower computers, use the lighter model explicitly (counts may be less accurate):

```powershell
.\.venv\Scripts\python.exe live_server.py --model yolo11n.pt --image-size 640
```

If downloading model weights separately, obtain YOLO11s from the [official Ultralytics assets release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt).

### Transport and limits

- JPEG frames over secure WebSocket, capped at 640 pixels on the longest side, JPEG quality 65%, and up to 10 frames/second. This avoids STUN/TURN and works through an HTTPS tunnel across different networks; it uses more bandwidth than an encoded WebRTC video stream.
- Capture skips frames while a previous frame is awaiting receipt. The inference worker retains only the latest pending frame; inference runs outside the HTTP event loop.
- Both directions require three confirmed observations beyond a 2% dead band. A tracked person may cross repeatedly. IDs expire after two seconds without observation.
- Reconnect, orientation changes, and direction changes clear tracking baselines while retaining totals. No frames for three seconds marks the camera offline. Crossings during an outage can be missed. A network stall stops the phone sender after ten seconds without acknowledgment.
- Counts live in memory until the server exits. Events append to `outputs/live_<timestamp>/events.csv` with UTC timestamps. The live mode does not record a full video. Temporary HTTPS tunnel URLs change on restart; Quick Tunnels are for development, not guaranteed uptime. Video passes through the reverse-proxy provider.
- The phone URL grants sending access; the administrator URL grants viewing, reset, export and configuration access. Treat each link as a password. Tokens are kept in URL fragments/session storage rather than HTTP query strings. API routes require an administrator bearer token.

### Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_live_video.py
```

The smoke check sends the bundled video through a real local WebSocket and the actual YOLO worker. This does not replace an iPhone Safari + 4G/5G field test: test IN, OUT, a return trip, standing near the line, disconnect/reconnect, and poor coverage at the real doorway.
