"""
Timestamp localization logic for deepfake detection
"""
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_THRESHOLD = 0.5
GAP_TOLERANCE = 1.5  # seconds
MIN_SEGMENT_DURATION = 0.5  # seconds

def localize_timestamps(
    fake_scores: Dict[str, float], 
    timestamps: Dict[str, float], 
    threshold: float = DEFAULT_THRESHOLD
) -> List[Dict[str, float]]:
    """
    Localize manipulated segments based on fake scores
    
    Args:
        fake_scores: Dictionary mapping frame paths to fake scores
        timestamps: Dictionary mapping frame paths to timestamps
        threshold: Score threshold for considering a frame as fake
        
    Returns:
        List of segments with start and end times
    """
    if not fake_scores or not timestamps:
        logger.warning("No fake scores or timestamps provided")
        return []
    
    # Filter frames above threshold
    fake_frames = []
    for frame_path, score in fake_scores.items():
        if score >= threshold:
            time_sec = timestamps.get(frame_path)
            if time_sec is not None:
                fake_frames.append((frame_path, time_sec, score))
    
    # Sort by timestamp
    fake_frames.sort(key=lambda x: x[1])
    
    if not fake_frames:
        logger.info(f"No fake frames detected above threshold {threshold}")
        return []
    
    logger.info(f"Found {len(fake_frames)} fake frames above threshold")
    
    # Group into segments
    segments = []
    seg_start = fake_frames[0][1]
    seg_end = fake_frames[0][1]
    
    for i in range(1, len(fake_frames)):
        current_time = fake_frames[i][1]
        
        if current_time - seg_end <= GAP_TOLERANCE:
            # Continue current segment
            seg_end = current_time
        else:
            # End current segment and start new one
            segments.append({
                "start": round(seg_start, 2),
                "end": round(seg_end, 2)
            })
            seg_start = current_time
            seg_end = current_time
    
    # Add the last segment
    segments.append({
        "start": round(seg_start, 2),
        "end": round(seg_end, 2)
    })
    
    # Filter by minimum duration
    segments = [
        s for s in segments 
        if (s["end"] - s["start"]) >= MIN_SEGMENT_DURATION
    ]
    
    # Filter by valid timestamps
    if timestamps:
        max_time = max(timestamps.values())
        segments = [
            s for s in segments
            if s["start"] <= max_time and s["end"] <= max_time
        ]
    
    logger.info(f"Found {len(segments)} manipulated segment(s)")
    return segments

def format_segments(segments: List[Dict[str, float]]) -> str:
    """Format segments for display"""
    if not segments:
        return "No manipulated segments detected."
    
    result = []
    for i, seg in enumerate(segments, 1):
        start = seg["start"]
        end = seg["end"]
        duration = round(end - start, 2)
        result.append(f"  Segment {i}: {start}s - {end}s (duration: {duration}s)")
    
    return "\n".join(result)

def get_overall_confidence(
    fake_scores: Dict[str, float], 
    threshold: float = 0.7
) -> float:
    """
    Calculate overall confidence score
    
    Args:
        fake_scores: Dictionary of fake scores
        threshold: Threshold for considering a frame as fake
        
    Returns:
        Overall confidence percentage
    """
    if not fake_scores:
        return 0.0
    
    fake_count = sum(1 for score in fake_scores.values() if score >= threshold)
    confidence = (fake_count / len(fake_scores)) * 100
    return round(confidence, 2)

def merge_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Merge overlapping or adjacent segments"""
    if len(segments) <= 1:
        return segments
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    merged = []
    current = sorted_segments[0]
    
    for next_seg in sorted_segments[1:]:
        if next_seg["start"] <= current["end"] + GAP_TOLERANCE:
            # Merge segments
            current["end"] = max(current["end"], next_seg["end"])
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged

if __name__ == "__main__":
    # Test data
    test_fake_scores = {
        "frame_150.jpg": 0.85,
        "frame_151.jpg": 0.79,
        "frame_152.jpg": 0.91,
        "frame_153.jpg": 0.88,
        "frame_154.jpg": 0.20,
        "frame_155.jpg": 0.15,
        "frame_200.jpg": 0.93,
        "frame_201.jpg": 0.87,
        "frame_202.jpg": 0.76,
        "frame_250.jpg": 0.60,
    }
    
    test_timestamps = {
        "frame_150.jpg": 5.0,
        "frame_151.jpg": 5.04,
        "frame_152.jpg": 5.08,
        "frame_153.jpg": 5.12,
        "frame_154.jpg": 5.16,
        "frame_155.jpg": 5.20,
        "frame_200.jpg": 8.0,
        "frame_201.jpg": 8.04,
        "frame_202.jpg": 8.08,
        "frame_250.jpg": 20.1,
    }
    
    print("=" * 50)
    print(" DEEPFAKE TIMESTAMP LOCALIZATION — TEST RUN")
    print("=" * 50)
    
    segments = localize_timestamps(test_fake_scores, test_timestamps, threshold=0.7)
    print("\n📍 Manipulated Segments:")
    print(format_segments(segments))
    
    confidence = get_overall_confidence(test_fake_scores, threshold=0.7)
    print(f"\n🔍 Overall Fake Confidence: {confidence}%")