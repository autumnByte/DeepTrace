"""
Deepfake detection module using EfficientNet
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
MODEL_PATH = Path("models/deepfake_model.pth")

# Initialize model
try:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    # Replace classifier for binary classification
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    
    # Load trained weights
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Using untrained model.")
    
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Image transforms
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_deepfake(frames: List[str]) -> Dict[str, float]:
    """
    Detect deepfake in frames using batch processing
    
    Args:
        frames: List of frame identifiers
        
    Returns:
        Dictionary mapping frame paths to fake scores
    """
    fake_scores = {}
    
    try:
        # Process in batches
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i:i+BATCH_SIZE]
            batch_tensors = []
            valid_frames = []
            
            # Load and preprocess images
            for frame in batch_frames:
                face_path = os.path.join("data/faces", f"face_{frame}")
                if not os.path.exists(face_path):
                    continue
                    
                try:
                    image = Image.open(face_path).convert("RGB")
                    tensor = transform(image)
                    batch_tensors.append(tensor)
                    valid_frames.append(frame)
                except Exception as e:
                    logger.warning(f"Failed to process {face_path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Batch inference
            batch_input = torch.stack(batch_tensors).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(batch_input)
                scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            # Store results
            for frame, score in zip(valid_frames, scores):
                fake_scores[frame] = float(score)
            
            logger.info(f"Processed batch {i//BATCH_SIZE + 1}/{(len(frames)-1)//BATCH_SIZE + 1}")
    
    except Exception as e:
        logger.error(f"Error in deepfake detection: {e}")
        raise
    
    return fake_scores

def detect_deepfake_single(face_path: str) -> Optional[float]:
    """
    Detect deepfake in a single face image
    
    Args:
        face_path: Path to face image
        
    Returns:
        Fake score or None if error
    """
    try:
        if not os.path.exists(face_path):
            return None
            
        image = Image.open(face_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(tensor)
            score = torch.sigmoid(output).item()
        
        return score
    
    except Exception as e:
        logger.error(f"Error processing {face_path}: {e}")
        return None

if __name__ == "__main__":
    # Test the module
    test_frames = ["frame_001.jpg", "frame_002.jpg"]
    scores = detect_deepfake(test_frames)
    print(f"Test results: {scores}")