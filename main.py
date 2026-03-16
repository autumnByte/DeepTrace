"""
Main pipeline for deepfake detection
"""
import logging
from pathlib import Path
from typing import Dict, List, Any

# Import modules
from modules.video_processing import extract_frames
from modules.face_analysis import analyze_faces
from modules.deepfake_detection import detect_deepfake
from modules.timestamp_logic import localize_timestamps, get_overall_confidence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(video_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run the complete deepfake detection pipeline
    
    Args:
        video_path: Path to input video
        threshold: Threshold for fake detection
        
    Returns:
        Dictionary containing results
    """
    results = {
        "segments": [],
        "faces": {},
        "fake_scores": {},
        "timestamps": {},
        "confidence": 0.0
    }
    
    try:
        # Step 1: Extract frames
        logger.info("Step 1: Extracting frames from video...")
        frames, timestamps = extract_frames(video_path, max_frames=500)  # Limit frames for performance
        logger.info(f"{len(frames)} frames extracted")
        results["timestamps"] = timestamps
        
        if not frames:
            logger.warning("No frames extracted")
            return results
        
        # Step 2: Face analysis
        logger.info("Step 2: Running face analysis...")
        face_data = analyze_faces(frames)
        logger.info(f"Faces detected in {len(face_data)} frames")
        results["faces"] = face_data
        
        # Step 3: Deepfake detection
        logger.info("Step 3: Running deepfake detection...")
        fake_scores = detect_deepfake(frames)
        logger.info(f"Processed {len(fake_scores)} frames")
        results["fake_scores"] = fake_scores
        
        # Step 4: Localize timestamps
        logger.info("Step 4: Localizing manipulated timestamps...")
        segments = localize_timestamps(fake_scores, timestamps, threshold)
        results["segments"] = segments
        
        # Step 5: Calculate overall confidence
        results["confidence"] = get_overall_confidence(fake_scores, threshold)
        
        logger.info(f"Pipeline completed. Found {len(segments)} manipulated segments")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    return results

def run_pipeline_batch(video_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Run pipeline on multiple videos
    
    Args:
        video_paths: List of video paths
        
    Returns:
        List of results dictionaries
    """
    results = []
    for video_path in video_paths:
        logger.info(f"Processing: {video_path}")
        result = run_pipeline(video_path)
        results.append(result)
    return results

def print_summary(results: Dict[str, Any]):
    """Print a summary of results"""
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION SUMMARY")
    print("="*50)
    
    print(f"Frames analyzed: {len(results.get('fake_scores', {}))}")
    print(f"Faces detected: {len(results.get('faces', {}))}")
    print(f"Manipulated segments: {len(results.get('segments', []))}")
    print(f"Overall confidence: {results.get('confidence', 0)}%")
    
    if results['segments']:
        print("\nManipulated segments:")
        for seg in results['segments']:
            print(f"  {seg['start']:.2f}s → {seg['end']:.2f}s")
    
    print("="*50)

if __name__ == "__main__":
    # Test the pipeline
    test_video = Path("test/test_video.mp4")
    
    if test_video.exists():
        logger.info(f"Testing pipeline with {test_video}")
        results = run_pipeline(str(test_video))
        print_summary(results)
    else:
        logger.warning(f"Test video not found: {test_video}")
        print("Please add a test video at: test/test_video.mp4")