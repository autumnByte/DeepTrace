# By: HansikaaBanick
# Description: Extracts frames from video and maps each frame to timestamp

import cv2
import os
import csv

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0

    os.makedirs("frames", exist_ok=True)

    with open("timestamps.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp"])

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = frame_id / fps

            frame_name = f"frame_{frame_id}.jpg"

            cv2.imwrite(f"frames/{frame_name}", frame)

            writer.writerow([frame_name, timestamp])

            frame_id += 1

    cap.release()

    return frame_id