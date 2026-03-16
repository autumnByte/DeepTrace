"""
Modules package
"""
from .video_processing import extract_frames
from .face_analysis import analyze_faces
from .deepfake_detection import detect_deepfake
from .timestamp_logic import localize_timestamps

__all__ = [
    'extract_frames',
    'analyze_faces', 
    'detect_deepfake',
    'localize_timestamps'
]