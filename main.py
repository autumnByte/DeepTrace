from modules.video_processing import extract_frames
from modules.face_analysis import analyze_faces
from modules.deepfake_detection import detect_deepfake
from modules.timestamp_logic import localize_timestamps


def run_pipeline(video_path):

    print("Step 1: Extracting frames from video...")
    frames, timestamps = extract_frames(video_path)

    print(f"{len(frames)} frames extracted.")

    print("Step 2: Running face analysis...")
    face_data = analyze_faces(frames)

    print(f"{len(face_data)} faces detected.")

    print("Step 3: Running deepfake detection...")
    fake_scores = detect_deepfake(frames)

    print("Step 4: Localizing manipulated timestamps...")
    segments = localize_timestamps(fake_scores, timestamps)

    return {
        "segments": segments,
        "faces": face_data,
        "fake_scores": fake_scores
    }


if __name__ == "__main__":

    video_path = "test/test_video.mp4"

    results = run_pipeline(video_path)

    print("\nDetected Manipulated Segments:")
    print(results["segments"])
