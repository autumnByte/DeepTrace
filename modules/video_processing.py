# Description: Extracts frames from video and maps each frame to timestamp

import cv2
import os
import csv


def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback fps

    frame_id = 0

    frames = []
    timestamps = {}

    # create frames folder
    os.makedirs("data/frames", exist_ok=True)

    # save timestamps inside data folder
    with open("data/timestamps.csv", "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp"])

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            timestamp = frame_id / fps
            frame_name = f"frame_{frame_id}.jpg"

            frame_path = f"data/frames/{frame_name}"

            # save frame
            cv2.imwrite(frame_path, frame)

            writer.writerow([frame_name, timestamp])

            # store data in memory
            frames.append(frame_name)
            timestamps[frame_name] = timestamp

            frame_id += 1

    cap.release()

    return frames, timestamps