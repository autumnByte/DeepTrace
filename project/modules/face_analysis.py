import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core


def analyze_faces(frames_folder, faces_folder):

    # FIX 1: Use exist_ok=True instead of checking manually
    # This prevents errors if the folder already exists
    os.makedirs(faces_folder, exist_ok=True)

    # MediaPipe face detector configuration
    # FIX 2: Previously the model path was None.
    # MediaPipe requires a valid .tflite model file.
    base_options = core.BaseOptions(
        model_asset_path="models/blaze_face_short_range.tflite"
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options
    )

    detector = vision.FaceDetector.create_from_options(options)

    face_data = []

    # FIX 3: sorted() ensures frames are processed in order
    # os.listdir() alone does not guarantee correct ordering
    for frame_file in sorted(os.listdir(frames_folder)):

        # FIX 4: Process only image files
        # Prevents crashes if other files exist in the folder
        if not frame_file.endswith(".jpg"):
            continue

        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)

        # FIX 5: Skip invalid images
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect(mp_image)

        # FIX 6: Skip frames where no faces are detected
        if not results.detections:
            continue

        for detection in results.detections:

            bbox = detection.bounding_box

            # FIX 7: Convert bounding box values to integers
            # Some detectors return float values
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)

            h, w, _ = image.shape

            # FIX 8: Ensure bounding box stays within image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            face = image[y1:y2, x1:x2]

            # FIX 9: Skip empty crops
            # Prevents saving invalid images
            if face.size == 0:
                continue

            face_path = os.path.join(
                faces_folder,
                f"face_{frame_file}"
            )

            cv2.imwrite(face_path, face)

            # FIX 10: face_data.append must be inside the detection loop
            # Previously it was outside which could cause errors
            face_data.append({
                "frame": frame_file,
                "face_path": face_path,
                "bbox": (x, y, width, height)
            })

    print("Face analysis completed")

    return face_data


if __name__ == "__main__":
    analyze_faces("../frames", "../faces")