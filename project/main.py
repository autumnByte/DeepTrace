from modules.video_processing import extract_frames
from modules.deepfake_detection import detect_deepfake
from modules.face_analysis import analyze_faces
from modules.timestamp_logic import localize_timestamps

def run_pipeline(video_path):
    frames, timestamps = extract_frames(video_path)
    fake_scores = detect_deepfake(frames)
    face_scores = analyze_faces(frames)
    segments = localize_timestamps(fake_scores, timestamps)
    return segments

if __name__ == "__main__":
    print("Pipeline ready")
