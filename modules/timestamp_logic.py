"""
Timestamp localization logic for deepfake detection
"""
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_THRESHOLD = 0.65
GAP_TOLERANCE = 0.3 # seconds
MIN_SEGMENT_DURATION = 0.5  # seconds

def localize_timestamps(
    fake_scores: Dict[str, float], 
    timestamps: Dict[str, float], 
    threshold: float = DEFAULT_THRESHOLD
) -> List[Dict[str, float]]:

    if not fake_scores or not timestamps:
        logger.warning("No fake scores or timestamps provided")
        return []

    # 🔧 NEW PARAMETERS
    MIN_CONSECUTIVE_FRAMES = 5
    MAX_GAP = 0.3  # was 1.5 → too large
    MIN_AVG_SCORE = threshold + 0.05  # avoid weak detections

    # Step 1: Collect candidate fake frames
    frames = []
    for frame_path, score in fake_scores.items():
        time_sec = timestamps.get(frame_path)
        if time_sec is not None:
            frames.append((frame_path, time_sec, score))

    # Sort all frames by time
    frames.sort(key=lambda x: x[1])

    segments = []
    temp_segment = []

    # Step 2: Build segments with consecutive logic
    for i in range(len(frames)):
        _, time_sec, score = frames[i]

        if score >= threshold:
            if not temp_segment:
                temp_segment.append(frames[i])
            else:
                prev_time = temp_segment[-1][1]

                if time_sec - prev_time <= MAX_GAP:
                    temp_segment.append(frames[i])
                else:
                    # finalize previous segment
                    if len(temp_segment) >= MIN_CONSECUTIVE_FRAMES:
                        segments.append(temp_segment)
                    temp_segment = [frames[i]]
        else:
            # break segment
            if len(temp_segment) >= MIN_CONSECUTIVE_FRAMES:
                segments.append(temp_segment)
            temp_segment = []

    # last segment
    if len(temp_segment) >= MIN_CONSECUTIVE_FRAMES:
        segments.append(temp_segment)

    # Step 3: Convert to timestamp format with score filtering
    final_segments = []

    for seg in segments:
        scores = [x[2] for x in seg]
        avg_score = sum(scores) / len(scores)

        # 🔧 Filter weak segments
        if avg_score < MIN_AVG_SCORE:
            continue

        start = seg[0][1]
        end = seg[-1][1]

        if (end - start) >= MIN_SEGMENT_DURATION:
            final_segments.append({
                "start": round(start, 2),
                "end": round(end, 2)
            })

    logger.info(f"Found {len(final_segments)} manipulated segment(s)")
    return final_segments

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

def get_overall_confidence(fake_scores: Dict[str, float], threshold: float = 0.55) -> float:
    if not fake_scores:
        return 0.0

    scores = list(fake_scores.values())

    # Count how many frames are confidently fake
    confident_fake = sum(1 for s in scores if s >= threshold)

    # Confidence = % of frames classified as fake
    confidence = (confident_fake / len(scores)) * 100

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
# Add to modules/timestamp_logic.py
def get_manipulation_confidence(segments, fake_scores, timestamps):
    """
    Calculate overall confidence that video is manipulated
    Using multiple signals:
    1. Percentage of fake frames
    2. Duration of fake segments
    3. Consistency of fake scores
    """
    if not fake_scores:
        return 0.0
    
    # Signal 1: What % of frames are fake?
    fake_frames = sum(1 for s in fake_scores.values() if s > 0.5)
    frame_ratio = fake_frames / len(fake_scores)
    
    # Signal 2: What % of video duration is manipulated?
    if timestamps:
        video_duration = max(timestamps.values())
        manipulated_duration = sum(seg["end"] - seg["start"] for seg in segments)
        duration_ratio = manipulated_duration / video_duration if video_duration > 0 else 0
    else:
        duration_ratio = 0
    
    # Signal 3: How consistent are the fake scores?
    scores = list(fake_scores.values())
    avg_score = sum(scores) / len(scores)
    score_consistency = avg_score  # Higher avg = more confidence
    
    # Combined score (weighted)
    final_confidence = (
        frame_ratio * 0.4 +      # 40% weight
        duration_ratio * 0.4 +   # 40% weight  
        score_consistency * 0.2   # 20% weight
    ) * 100
    
    return min(100, final_confidence)
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