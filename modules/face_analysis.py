import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def analyze_faces(frames):

    frames_folder = "data/frames"
    faces_folder = "data/faces"

    os.makedirs(faces_folder, exist_ok=True)

    model_path = "face_detector.tflite"

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceDetectorOptions(base_options=base_options)

    detector = vision.FaceDetector.create_from_options(options)

    face_data = {}

    for frame_file in frames:

        frame_path = os.path.join(frames_folder, frame_file)

        image = cv2.imread(frame_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = detector.detect(mp_image)

        if not results.detections:
            continue

        for detection in results.detections:

            bbox = detection.bounding_box

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            h_img, w_img, _ = image.shape

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_path = os.path.join(faces_folder, f"face_{frame_file}")

            cv2.imwrite(face_path, face)

            face_data[frame_file] = {
                "face_path": face_path,
                "bbox": (x, y, w, h)
            }

    print("Face analysis completed")

    return face_data