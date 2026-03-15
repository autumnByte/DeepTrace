def localize_timestamps(fake_scores, timestamps, threshold=0.5):
    fake_frames = []
    for frame_path, score in fake_scores.items():
        if score >= threshold:
            time_sec = timestamps.get(frame_path)
            if time_sec is not None:
                fake_frames.append((frame_path, time_sec, score))
    fake_frames.sort(key=lambda x: x[1])
    if not fake_frames:
        print("[Timestamp Logic] No fake frames detected above threshold.")
        return []
    GAP_TOLERANCE = 1.5
    segments = []
    seg_start = fake_frames[0][1]
    seg_end = fake_frames[0][1]
    for i in range(1, len(fake_frames)):
        current_time = fake_frames[i][1]
        if current_time - seg_end <= GAP_TOLERANCE:
            seg_end = current_time
        else:
            segments.append({"start": round(seg_start, 2), "end": round(seg_end, 2)})
            seg_start = current_time
            seg_end = current_time
    segments.append({"start": round(seg_start, 2), "end": round(seg_end, 2)})
    MIN_DURATION = 0.5
    segments = [s for s in segments if (s["end"] - s["start"]) >= MIN_DURATION]
    max_time = max(timestamps.values())
    segments =  [
        s for s in segments
        if s["start"] <= max_time and s["end"] <= max_time
]
    print(f"[Timestamp Logic] {len(segments)} manipulated segment(s) found.")
    return segments


def format_segments(segments):
    if not segments:
        return "No manipulated segments detected."
    result = []
    for i, seg in enumerate(segments, 1):
        start = seg["start"]
        end = seg["end"]
        duration = round(end - start, 2)
        result.append(f"  Segment {i}: {start}s → {end}s  (duration: {duration}s)")
    return "\n".join(result)


def get_overall_confidence(fake_scores, threshold=0.7):
    if not fake_scores:
        return 0.0
    fake_count = sum(1 for score in fake_scores.values() if score >= threshold)
    confidence = (fake_count / len(fake_scores)) * 100
    return round(confidence, 2)


if __name__ == "__main__":
    fake_scores = {
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
    timestamps = {
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
    print("  DEEPFAKE TIMESTAMP LOCALIZATION — TEST RUN")
    print("=" * 50)
    segments = localize_timestamps(fake_scores, timestamps, threshold=0.7)
    print("\n📍 Manipulated Segments:")
    print(format_segments(segments))
    confidence = get_overall_confidence(fake_scores, threshold=0.7)
    print(f"\n🔍 Overall Fake Confidence: {confidence}%")
    print("\n✅ Raw Output (for main.py integration):")
    print(segments)