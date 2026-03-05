import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core


def analyze_faces(frames_folder, faces_folder):

    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    # MediaPipe face detector
    base_options = core.BaseOptions(
        model_asset_path=None
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options
    )

    detector = vision.FaceDetector.create_from_options(options)

    face_data = []

    for frame_file in os.listdir(frames_folder):

        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect(mp_image)

        for detection in results.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            face = image[y:y+height, x:x+width]

            face_path = os.path.join(
                faces_folder,
                "face_" + frame_file
            )

            cv2.imwrite(face_path, face)

            face_data.append({
                "frame": frame_file,
                "face_path": face_path,
                "bbox": (x, y, width, height)
            })

    print("Face analysis completed")

    return face_data


if __name__ == "__main__":
    analyze_faces("../frames", "../faces")