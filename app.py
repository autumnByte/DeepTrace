import streamlit as st
import tempfile
# from main import run_pipeline   # keep commented until backend is ready

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

st.title("🎥 Deepfake Detection & Manipulation Timestamp Localization")

st.markdown("Upload a video to detect **deepfake manipulation** and identify suspicious timestamps.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

if uploaded_video is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Uploaded Video")
        st.video(temp_file.name)

    with col2:
        st.subheader("⚙ Detection Control")

        if st.button("Run Deepfake Detection"):

            with st.spinner("Analyzing video for manipulation..."):

                # Temporary results until backend is ready
                results = {
                    "timestamps": [
                        ("00:04","00:07"),
                        ("00:15","00:18")
                    ],
                    "frames": []
                }

            st.success("Detection Completed")

            st.subheader("⚠ Manipulated Video Segments")

            if len(results["timestamps"]) == 0:
                st.success("No manipulation detected.")
            else:
                for start, end in results["timestamps"]:
                    st.warning(f"Manipulation detected from {start} to {end}")

            st.subheader("📷 Suspicious Frames")

            if len(results["frames"]) == 0:
                st.info("No suspicious frames available yet.")
            else:
                for frame in results["frames"]:
                    st.image(frame)