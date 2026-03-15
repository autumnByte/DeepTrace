import streamlit as st
import tempfile
from main import run_pipeline

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

st.title("🎥 Deepfake Detection & Manipulation Timestamp Localization")

st.markdown("Upload a video to detect **deepfake manipulation** and identify suspicious timestamps.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

if uploaded_video is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Uploaded Video")
        st.video(video_path)

    with col2:
        st.subheader("⚙ Detection Control")

        if st.button("Run Deepfake Detection"):

            with st.spinner("Analyzing video for manipulation..."):
                results = run_pipeline(video_path)

            st.success("Detection Completed")

            segments = results["segments"]
            faces = results["faces"]
            fake_scores = results["fake_scores"]

            threshold = 0.55

            st.subheader("⚠ Manipulated Video Segments")

            if len(segments) == 0:
               st.info("No continuous manipulation segments detected, but some frames may appear suspicious.")
            else:
                for seg in segments:
                    start = seg["start"]
                    end = seg["end"]
                    st.warning(f"Manipulation detected from {start:.2f}s to {end:.2f}s")

            st.subheader("📷 Suspicious Frames")

            if len(faces) == 0:
                st.info("No suspicious frames available.")
            else:

                suspicious = []

                for frame, data in faces.items():
                    score = fake_scores.get(frame, 0)

                    if score >= threshold:
                        suspicious.append((frame, score, data["face_path"]))

                if len(suspicious) == 0:
                    st.info("No suspicious frames detected.")
                else:

                    # sort frames by highest fake score
                    suspicious = sorted(suspicious, key=lambda x: x[1], reverse=True)

                    # show only top 10 suspicious frames
                    for frame, score, path in suspicious[:10]:
                        st.image(path, caption=f"{frame} | Score: {score:.2f}")