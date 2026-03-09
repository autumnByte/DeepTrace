from modules.video_processing import extract_frames

video_path = "test/test_video.mp4"

frames, timestamps = extract_frames(video_path)

print("Total frames extracted:", len(frames))
print("Frames saved in data/frames")
print("Timestamps saved in data/timestamps.csv")