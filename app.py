
import sys
import uuid
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from counting_engine import process_video


st.set_page_config(
    page_title="Human Counting System",
    page_icon="🎥",
    layout="wide",
)

st.title("Human Counting System")
st.caption("YOLO11 + ByteTrack | Computer Vision Project")

st.divider()

with st.sidebar:
    st.header("Settings")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    line_position = st.slider(
        "Counting line position (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
    )

    st.info(
        "TOP → BOTTOM: IN\n\n"
        "BOTTOM → TOP: OUT"
    )

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"],
)

if "result" not in st.session_state:
    st.session_state.result = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = None

if uploaded_file is not None:

    upload_key = (
        uploaded_file.name,
        uploaded_file.size,
        confidence,
        line_position,
    )

    if st.session_state.upload_key != upload_key:
        st.session_state.result = None
        st.session_state.upload_key = upload_key

    st.subheader("Input Video")
    st.video(uploaded_file)

    if st.button(
        "Run Human Counting",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.result = None

        run_id = uuid.uuid4().hex[:12]
        output_dir = ROOT / "outputs" / run_id

        with tempfile.TemporaryDirectory() as temp_dir:

            suffix = Path(uploaded_file.name).suffix.lower()
            input_path = Path(temp_dir) / f"input{suffix}"

            input_path.write_bytes(uploaded_file.getvalue())

            progress = st.progress(0)

            def update_progress(value):
                progress.progress(
                    min(max(float(value), 0.0), 1.0)
                )

            try:
                with st.spinner(
                    "Processing video with YOLO + ByteTrack..."
                ):
                    result = process_video(
                        input_path=input_path,
                        output_dir=output_dir,
                        confidence=confidence,
                        line_position=line_position / 100,
                        progress_callback=update_progress,
                    )

                st.session_state.result = result
                st.success("Processing completed!")

            except Exception as e:
                st.error(f"Processing failed: {e}")

result = st.session_state.result

if result is not None:

    summary = result["summary"]

    st.divider()
    st.subheader("Counting Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("IN", summary["in_count"])
    col2.metric("OUT", summary["out_count"])
    col3.metric("Total Crossings", summary["total_crossings"])
    col4.metric(
        "Average FPS",
        summary["average_processing_fps"],
    )

    st.subheader("Processed Video")

    video_path = Path(result["video"])

    if video_path.exists():
        st.video(str(video_path))

        st.download_button(
            "Download Processed MP4",
            data=video_path.read_bytes(),
            file_name="human_counting_result.mp4",
            mime="video/mp4",
        )

    st.subheader("Counting Events")

    events_path = Path(result["events_csv"])
    events_df = pd.read_csv(events_path)

    st.dataframe(
        events_df,
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "Download Summary CSV",
            data=Path(result["summary_csv"]).read_bytes(),
            file_name="counting_summary.csv",
            mime="text/csv",
        )

    with col_b:
        st.download_button(
            "Download Events CSV",
            data=events_path.read_bytes(),
            file_name="counting_events.csv",
            mime="text/csv",
        )
