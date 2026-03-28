"""
Main pipeline for deepfake detection with model comparison
"""
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Import modules
from modules.video_processing import extract_frames
from modules.face_analysis import analyze_faces
from modules.deepfake_detection import detect_deepfake_comparison, get_comparison_report
from modules.timestamp_logic import localize_timestamps, get_overall_confidence, get_manipulation_confidence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(video_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run the complete deepfake detection pipeline with model comparison
    
    Args:
        video_path: Path to input video
        threshold: Threshold for fake detection
        
    Returns:
        Dictionary containing results with model comparison
    """
    results = {
        "segments": [],
        "faces": {},
        "fake_scores": {},
        "timestamps": {},
        "confidence": 0.0,
        "manipulation_confidence": 0.0,
        "comparison_report": None,
        "comparison_data": {},
        "ensemble_scores": {},
        "model_agreement": 0.0
    }
    
    try:
        # Step 1: Extract frames
        logger.info("Step 1: Extracting frames from video...")
        frames, timestamps = extract_frames(video_path)
        logger.info(f"✅ {len(frames)} frames extracted")
        results["timestamps"] = timestamps
        
        if not frames:
            logger.warning("No frames extracted")
            return results
        
        # Step 2: Face analysis
        logger.info("Step 2: Running face analysis...")
        face_data = analyze_faces(frames)  # This creates face files
        logger.info(f"✅ Faces detected in {len(face_data)} frames")
        results["faces"] = face_data
    
    # Get all frames that have faces
        frames_with_faces = list(face_data.keys())
        logger.info(f"Frames with faces: {len(frames_with_faces)}")
    
    # Step 3: Deepfake detection with model comparison
        logger.info("Step 3: Running deepfake detection (CNN vs Random Forest)...")
    
    # Pass ALL frames that have faces
        comparison_results = detect_deepfake_comparison(frames_with_faces)
        
        # Extract scores for compatibility
        fake_scores = {}
        ensemble_scores = {}
        for frame, scores in comparison_results.items():
            if scores["cnn_score"] is not None:
                fake_scores[frame] = scores["cnn_score"]
            if scores["ensemble_score"] is not None:
                ensemble_scores[frame] = scores["ensemble_score"]
        
        results["fake_scores"] = fake_scores
        results["ensemble_scores"] = ensemble_scores
        results["comparison_data"] = comparison_results
        
        # Generate comparison report
        results["comparison_report"] = get_comparison_report(comparison_results)
        
        # Calculate model agreement
        if results["comparison_report"] is not None and not results["comparison_report"].empty:
            agreement = (results["comparison_report"]["Agreement"] == "Yes").sum()
            total = len(results["comparison_report"])
            results["model_agreement"] = (agreement / total) * 100 if total > 0 else 0
            logger.info(f"Model agreement: {results['model_agreement']:.1f}%")
        
        logger.info(f"✅ Processed {len(fake_scores)} frames with both models")
        
        if not fake_scores:
            logger.warning("No fake scores generated")
            return results
        
        # Step 4: Localize timestamps (using CNN scores)
        logger.info("Step 4: Localizing manipulated timestamps...")
        segments = localize_timestamps(fake_scores, timestamps, threshold)
        results["segments"] = segments
        
        # Step 5: Calculate confidence scores
        # Basic confidence from frame ratio
        results["confidence"] = get_overall_confidence(fake_scores, threshold)
        
        # Advanced manipulation confidence (multiple signals)
        results["manipulation_confidence"] = get_manipulation_confidence(
            segments, fake_scores, timestamps
        )
        
        logger.info(f"✅ Pipeline completed.")
        logger.info(f"   - Found {len(segments)} manipulated segments")
        logger.info(f"   - CNN Confidence: {results['confidence']:.1f}%")
        logger.info(f"   - Manipulation Confidence: {results['manipulation_confidence']:.1f}%")
        logger.info(f"   - Model Agreement: {results['model_agreement']:.1f}%")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
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
    for i, video_path in enumerate(video_paths):
        logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        result = run_pipeline(video_path)
        results.append(result)
    
    return results

def print_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of results"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SUMMARY")
    print("="*60)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"   Frames analyzed: {len(results.get('fake_scores', {}))}")
    print(f"   Faces detected: {len(results.get('faces', {}))}")
    print(f"   Manipulated segments: {len(results.get('segments', []))}")
    
    # Confidence scores
    print(f"\n🎯 Confidence Scores:")
    print(f"   CNN-based Confidence: {results.get('confidence', 0):.1f}%")
    print(f"   Manipulation Confidence: {results.get('manipulation_confidence', 0):.1f}%")
    
    # Model comparison
    if results.get('comparison_report') is not None:
        df = results['comparison_report']
        if not df.empty:
            print(f"\n🤖 Model Comparison (CNN vs Random Forest):")
            print(f"   Model Agreement: {results.get('model_agreement', 0):.1f}%")
            print(f"   Avg CNN Score: {df['CNN Score'].mean():.3f}")
            print(f"   Avg RF Score: {df['Random Forest Score'].mean():.3f}")
            print(f"   Frames with disagreement: {(df['Agreement'] == 'No').sum()}")
    
    # Timestamp segments
    if results['segments']:
        print(f"\n⏱️ Manipulated Segments:")
        total_duration = 0
        for i, seg in enumerate(results['segments'], 1):
            duration = seg['end'] - seg['start']
            total_duration += duration
            print(f"   Segment {i}: {seg['start']:.2f}s → {seg['end']:.2f}s (duration: {duration:.2f}s)")
        print(f"   Total Manipulated Time: {total_duration:.2f}s")
    
    # Decision
    print(f"\n🔍 Final Verdict:")
    if results.get('manipulation_confidence', 0) > 70:
        print("   ⚠️  HIGH PROBABILITY OF DEEPFAKE MANIPULATION")
        print("   Multiple frames and segments indicate manipulation")
    elif results.get('manipulation_confidence', 0) > 50:
        print("   ⚠️  SUSPICIOUS - Possible manipulation detected")
        print("   Further analysis recommended")
    else:
        print("   ✅ LIKELY AUTHENTIC - No strong manipulation signals detected")
    
    print("\n" + "="*60)

def generate_detailed_report(results: Dict[str, Any], output_path: str = None):
    """
    Generate detailed HTML/CSV report of results
    
    Args:
        results: Results dictionary from run_pipeline
        output_path: Optional path to save report
    """
    if not results.get('comparison_report') is not None:
        print("No comparison data available")
        return
    
    df = results['comparison_report']
    
    # Add timestamp column
    timestamps = results.get('timestamps', {})
    df['Timestamp (s)'] = df['Frame'].map(lambda x: timestamps.get(x, 0))
    
    # Add classification
    threshold = 0.55
    df['CNN Classification'] = df['CNN Score'].apply(lambda x: 'FAKE' if x > threshold else 'REAL')
    df['RF Classification'] = df['Random Forest Score'].apply(lambda x: 'FAKE' if x > threshold else 'REAL')
    
    # Reorder columns
    df = df[['Frame', 'Timestamp (s)', 'CNN Score', 'Random Forest Score', 
             'CNN Classification', 'RF Classification', 'Difference', 'Agreement']]
    
    if output_path:
        # Save to CSV
        csv_path = output_path.replace('.html', '.csv') if output_path else 'detection_report.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV report saved to: {csv_path}")
        
        # Simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .fake {{ color: red; font-weight: bold; }}
                .real {{ color: green; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Deepfake Detection Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Frames Analyzed:</strong> {len(df)}</p>
                <p><strong>Fake Frames (CNN):</strong> {(df['CNN Classification'] == 'FAKE').sum()}</p>
                <p><strong>Fake Frames (RF):</strong> {(df['RF Classification'] == 'FAKE').sum()}</p>
                <p><strong>Model Agreement:</strong> {results.get('model_agreement', 0):.1f}%</p>
                <p><strong>Manipulation Confidence:</strong> {results.get('manipulation_confidence', 0):.1f}%</p>
            </div>
            
            <h2>Detailed Frame Analysis</h2>
            {df.to_html(classes='dataframe', escape=False)}
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"✅ HTML report saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Test the pipeline
    test_video = Path("test/test_video.mp4")
    
    if test_video.exists():
        logger.info(f"Testing pipeline with {test_video}")
        
        # Run pipeline
        results = run_pipeline(str(test_video), threshold=0.55)
        
        # Print summary
        print_summary(results)
        
        # Generate detailed report
        generate_detailed_report(results, "detection_report.html")
        
    else:
        logger.warning(f"Test video not found: {test_video}")
        print("\n" + "="*60)
        print("Please add a test video at: test/test_video.mp4")
        print("\nOr train Random Forest model first:")
        print("  python train_rf.py")
        print("="*60)