# Human Counting Project

Count people crossing a virtual line in uploaded videos using YOLO11 and ByteTrack, with a Streamlit interface.

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

Uploaded-video processing is implemented. Live phone-camera streaming through a browser is planned and is not implemented yet.
