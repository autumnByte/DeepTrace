"""
Video processing module for frame extraction
"""
import cv2
import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FRAMES_FOLDER = Path("data/frames")
DEFAULT_FPS = 30

def extract_frames(
    video_path: str, 
    frame_interval: int = 1,
    max_frames: int = None
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract frames from video and map to timestamps
    
    Args:
        video_path: Path to video file
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        
    Returns:
        Tuple of (list of frame filenames, dict mapping filenames to timestamps)
    """
    # Create frames directory
    FRAMES_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Clear existing frames (optional)
    for f in FRAMES_FOLDER.glob("*.jpg"):
        f.unlink()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = DEFAULT_FPS
        logger.warning(f"Could not detect FPS, using default: {fps}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    logger.info(f"Video: {video_path}")
    logger.info(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    
    frames = []
    timestamps = {}
    frame_id = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every frame_interval frame
            if frame_id % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_id / fps
                
                # Save frame
                frame_filename = f"frame_{saved_count:04d}.jpg"
                frame_path = FRAMES_FOLDER / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                frames.append(frame_filename)
                timestamps[frame_filename] = round(timestamp, 3)
                
                saved_count += 1
                
                # Check max frames limit
                if max_frames and saved_count >= max_frames:
                    logger.info(f"Reached max frames limit: {max_frames}")
                    break
            
            frame_id += 1
            
            # Log progress
            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id}/{total_frames} frames")
    
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
    
    finally:
        cap.release()
    
    logger.info(f"Extracted {len(frames)} frames")
    return frames, timestamps

def get_frame_timestamp(frame_filename: str, fps: float) -> float:
    """Calculate timestamp from frame filename"""
    try:
        frame_num = int(frame_filename.split('_')[1].split('.')[0])
        return frame_num / fps
    except:
        return 0.0

def get_video_info(video_path: str) -> dict:
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": 0
    }
    
    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    
    cap.release()
    return info

def frames_to_video(frame_folder: str, output_path: str, fps: int = 30):
    """Convert frames back to video (useful for testing)"""
    frames = sorted(Path(frame_folder).glob("*.jpg"))
    if not frames:
        logger.error("No frames found")
        return
    
    # Get frame size from first frame
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        out.write(frame)
    
    out.release()
    logger.info(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Test the module
    test_video = "test/test_video.mp4"
    if Path(test_video).exists():
        frames, timestamps = extract_frames(test_video, max_frames=10)
        print(f"Extracted {len(frames)} frames")
        for frame, ts in list(timestamps.items())[:5]:
            print(f"  {frame}: {ts}s")