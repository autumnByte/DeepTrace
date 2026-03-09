from modules.video_processing import extract_frames

video_path = "test_video.mp4"

total_frames = extract_frames(video_path)

print("Total frames extracted:", total_frames)
print("Frames saved in 'frames' folder")
print("Timestamps saved in timestamps.csv")