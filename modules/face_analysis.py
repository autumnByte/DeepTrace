"""
Face detection and extraction module using MediaPipe
"""
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FRAMES_FOLDER = Path("data/frames")
FACES_FOLDER = Path("data/faces")
MODEL_PATH = Path("models/face_detector.tflite")

# Ensure directories exist
FACES_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize face detector
detector = None
try:
    if MODEL_PATH.exists():
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)
        logger.info("Face detector initialized successfully")
    else:
        logger.error(f"Face detector model not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize face detector: {e}")

def analyze_faces(frames: List[str]) -> Dict[str, dict]:
    """
    Detect and extract faces from frames
    
    Args:
        frames: List of frame filenames
        
    Returns:
        Dictionary mapping frame names to face data
    """
    if detector is None:
        logger.error("Face detector not available")
        return {}
    
    face_data = {}
    
    try:
        for i, frame_file in enumerate(frames):
            frame_path = FRAMES_FOLDER / frame_file
            
            if not frame_path.exists():
                logger.warning(f"Frame not found: {frame_path}")
                continue
            
            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning(f"Failed to read image: {frame_path}")
                continue
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect faces
            detection_result = detector.detect(mp_image)
            
            if detection_result.detections:
                # Process each detected face (take the first one for now)
                detection = detection_result.detections[0]
                bbox = detection.bounding_box
                
                # Get confidence score - FIX HERE
                # The score might be in different locations depending on MediaPipe version
                confidence = 1.0  # default
                if hasattr(detection, 'score') and detection.score:
                    confidence = detection.score[0] if isinstance(detection.score, list) else detection.score
                elif hasattr(detection, 'confidence') and detection.confidence:
                    confidence = detection.confidence[0] if isinstance(detection.confidence, list) else detection.confidence
                elif hasattr(detection, 'scores') and detection.scores:
                    confidence = detection.scores[0]
                
                # Extract face region
                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                w = min(bbox.width, image.shape[1] - x)
                h = min(bbox.height, image.shape[0] - y)
                
                if w > 0 and h > 0:
                    face_roi = image[y:y+h, x:x+w]
                    
                    # Save face image
                    face_filename = f"face_{frame_file}"
                    face_path = FACES_FOLDER / face_filename
                    cv2.imwrite(str(face_path), face_roi)
                    
                    # Store face data
                    face_data[frame_file] = {
                        "face_path": str(face_path),
                        "bbox": [x, y, w, h],
                        "confidence": float(confidence)
                    }
                    
                    logger.debug(f"Face detected in {frame_file}")
            
            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(frames)} frames")
    
    except Exception as e:
        logger.error(f"Error in face analysis: {e}")
        import traceback
        traceback.print_exc()
    logger.info(f"Faces detected in {len(face_data)} frames")
    return face_data

def extract_all_faces(frames: List[str]) -> Tuple[Dict[str, dict], int]:
    """
    Extract faces and return data with count
    
    Returns:
        Tuple of (face_data dict, number of faces detected)
    """
    face_data = analyze_faces(frames)
    return face_data, len(face_data)

def get_face_path(frame_name: str) -> Optional[str]:
    """Get the path to the face image for a given frame"""
    face_path = FACES_FOLDER / f"face_{frame_name}"
    return str(face_path) if face_path.exists() else None