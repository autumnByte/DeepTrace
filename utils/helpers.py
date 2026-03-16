"""
Helper utilities for the deepfake detection system
"""
import logging
import json
from pathlib import Path
from typing import Any, Dict
import time
from functools import wraps

logger = logging.getLogger(__name__)

def timer_decorator(func):
    """Decorator to time functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def load_results(input_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(input_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return {}

def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def clean_frames():
    """Clean up frames directory"""
    import shutil
    frames_dir = Path("data/frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
        frames_dir.mkdir()
        logger.info("Cleaned frames directory")

def clean_faces():
    """Clean up faces directory"""
    import shutil
    faces_dir = Path("data/faces")
    if faces_dir.exists():
        shutil.rmtree(faces_dir)
        faces_dir.mkdir()
        logger.info("Cleaned faces directory")