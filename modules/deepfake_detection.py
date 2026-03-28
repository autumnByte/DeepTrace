"""
Deepfake detection module with model comparison
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import timm
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Import Random Forest detector
from modules.rf_detector import rf_detector

logger = logging.getLogger(__name__)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
MODEL_PATH = Path("models/deepfake_model_best.pth")

# Initialize CNN model
cnn_model = None
try:
    cnn_model = timm.create_model("efficientnet_b0", pretrained=False)
    cnn_model.classifier = torch.nn.Linear(cnn_model.classifier.in_features, 1)
    
    if MODEL_PATH.exists():
        cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        cnn_model = cnn_model.to(DEVICE)
        cnn_model.eval()
        logger.info(f"✅ CNN model loaded from {MODEL_PATH}")
    else:
        # Try original model
        original_path = Path("models/deepfake_model.pth")
        if original_path.exists():
            cnn_model.load_state_dict(torch.load(original_path, map_location=DEVICE))
            cnn_model = cnn_model.to(DEVICE)
            cnn_model.eval()
            logger.info(f"✅ CNN model loaded from {original_path}")
        else:
            logger.warning(f"⚠️ CNN model not found")
            cnn_model = None
except Exception as e:
    logger.error(f"Failed to load CNN model: {e}")
    cnn_model = None

# Load Random Forest model
rf_loaded = False
try:
    if rf_detector.load_model():
        logger.info("✅ Random Forest model loaded")
        rf_loaded = True
    else:
        logger.warning("⚠️ Random Forest model not found. Run train_rf.py first")
except Exception as e:
    logger.error(f"Failed to load Random Forest: {e}")

# Image transforms
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_deepfake_comparison(frames: List[str]) -> Dict[str, Dict]:
    """
    Detect deepfake using both CNN and Random Forest
    Returns comparison results for ALL frames
    """
    results = {}
    
    processed_count = 0
    cnn_success = 0
    rf_success = 0
    both_success = 0
    
    # Get list of available face files
    faces_dir = Path("data/faces")
    available_faces = set()
    if faces_dir.exists():
        for f in faces_dir.glob("face_*.jpg"):
            # Extract frame name from face filename
            face_name = f.name.replace("face_", "")
            available_faces.add(face_name)
    
    for idx, frame in enumerate(frames):
        # Check if face exists
        if frame not in available_faces:
            continue
        
        face_path = os.path.join("data/faces", f"face_{frame}")
        processed_count += 1
        
        # Initialize scores
        cnn_score = None
        rf_score = None
        
        # CNN prediction
        if cnn_model:
            try:
                image = Image.open(face_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = cnn_model(tensor)
                    cnn_score = torch.sigmoid(output).item()
                    cnn_success += 1
            except Exception as e:
                logger.error(f"CNN error for {frame}: {e}")
        
        # Random Forest prediction
        if rf_loaded:
            try:
                rf_score = rf_detector.predict(face_path)
                if rf_score is not None:
                    rf_success += 1
            except Exception as e:
                logger.error(f"RF error for {frame}: {e}")
        
        # Count both
        if cnn_score is not None and rf_score is not None:
            both_success += 1
        
        # Store results
        results[frame] = {
            "cnn_score": cnn_score,
            "rf_score": rf_score,
            "ensemble_score": (cnn_score + rf_score) / 2 if (cnn_score is not None and rf_score is not None) else None
        }
    
    logger.info(f"Comparison complete: {processed_count} frames with faces, {both_success} with both models")
    
    return results

def detect_deepfake(frames: List[str]) -> Dict[str, float]:
    """
    Original function - returns only CNN scores for compatibility
    """
    results = detect_deepfake_comparison(frames)
    return {frame: data["cnn_score"] for frame, data in results.items() 
            if data["cnn_score"] is not None}

def get_comparison_report(comparison_results: Dict) -> pd.DataFrame:
    """
    Generate comparison report from all comparison results
    """
    data = []
    
    for frame, scores in comparison_results.items():
        cnn_score = scores.get("cnn_score")
        rf_score = scores.get("rf_score")
        
        if cnn_score is not None and rf_score is not None:
            diff = abs(cnn_score - rf_score)
            agreement = "Yes" if diff < 0.2 else "No"
            
            data.append({
                "Frame": frame,
                "CNN Score": round(cnn_score, 3),
                "Random Forest Score": round(rf_score, 3),
                "Difference": round(diff, 3),
                "Agreement": agreement
            })
    
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()