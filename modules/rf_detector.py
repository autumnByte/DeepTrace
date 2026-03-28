"""
Improved Random Forest Detector with better feature extraction
"""
import numpy as np
import pickle
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import logging
from typing import Dict, List
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RandomForestDeepfakeDetector:
    def __init__(self, model_path="models/rf_model.pkl"):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = Path(model_path)
        self.cache = {}  # Add prediction cache
        self.feature_cache = {}  # Add feature cache
        
    def extract_features_improved(self, face_image):
        """
        Extract better features from face image for deepfake detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. GLCM-like texture features (more comprehensive)
        for i in range(0, gray.shape[0], 16):
            for j in range(0, gray.shape[1], 16):
                block = gray[i:min(i+16, gray.shape[0]), 
                            j:min(j+16, gray.shape[1])]
                if block.size > 0:
                    features.append(np.mean(block))
                    features.append(np.std(block))
                    features.append(np.max(block) - np.min(block))
        
        # 2. Edge features with different thresholds
        for threshold in [50, 100, 150]:
            edges = cv2.Canny(gray, threshold, threshold*2)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
        
        # 3. Color histograms (more detailed)
        if len(face_image.shape) == 3:
            for channel in range(3):
                hist = cv2.calcHist([face_image], [channel], None, [32], [0, 256])
                hist = hist.flatten() / np.sum(hist)
                features.extend(hist[:16])  # Take first 16 bins
        
        # 4. Frequency domain features (more detailed)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Extract frequency rings
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        for radius in [10, 20, 30, 40, 50]:
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            mask_area = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            ring_energy = np.mean(magnitude[mask_area])
            features.append(ring_energy)
        
        # 5. Facial landmark features (using simple geometric properties)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            features.append(np.mean(face_region))
            features.append(np.std(face_region))
            features.append(h / w)  # Aspect ratio
        else:
            features.extend([0, 0, 0])
        
        # 6. Noise estimation (deepfakes often have different noise patterns)
        noise = cv2.GaussianBlur(gray, (5,5), 0) - gray
        features.append(np.std(noise))
        features.append(np.mean(np.abs(noise)))
        
        return np.array(features)
    
    def train(self, dataset_path="dataset"):
        """Train Random Forest with improved features"""
        print("=" * 60)
        print("Training Improved Random Forest Model")
        print("=" * 60)
        
        X = []
        y = []
        
        # Load real images
        real_path = Path(dataset_path) / "real"
        if real_path.exists():
            real_images = list(real_path.glob("*.jpg")) + list(real_path.glob("*.png"))
            print(f"\nLoading {len(real_images)} real images...")
            
            for img_path in tqdm(real_images, desc="Processing real images"):
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    try:
                        # Resize to consistent size
                        img = cv2.resize(img, (224, 224))
                        features = self.extract_features_improved(img)
                        X.append(features)
                        y.append(0)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Load fake images
        fake_path = Path(dataset_path) / "fake"
        if fake_path.exists():
            fake_images = list(fake_path.glob("*.jpg")) + list(fake_path.glob("*.png"))
            print(f"\nLoading {len(fake_images)} fake images...")
            
            for img_path in tqdm(fake_images, desc="Processing fake images"):
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    try:
                        # Resize to consistent size
                        img = cv2.resize(img, (224, 224))
                        features = self.extract_features_improved(img)
                        X.append(features)
                        y.append(1)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        if len(X) == 0:
            print("No images found in dataset!")
            return 0.0
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset Summary:")
        print(f"  Total samples: {len(X)}")
        print(f"  Real samples: {sum(y == 0)}")
        print(f"  Fake samples: {sum(y == 1)}")
        print(f"  Feature dimension: {X.shape[1]}")
        
        # Normalize features
        print("\nNormalizing features...")
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Compute class weights for imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights: Real={class_weights[0]:.2f}, Fake={class_weights[1]:.2f}")
        
        # Train Random Forest with better parameters
        print("\nTraining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=20,      # Deeper trees
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"\nResults:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        print(f"\nTop 10 most important features: {top_features}")
        
        # Save model
        self.model_path.parent.mkdir(exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importances': importances
            }, f)
        print(f"\n✅ Model saved to: {self.model_path}")
        
        return test_acc
    
    def predict(self, face_path):
        """Predict if face is fake (with caching)"""
        # Check cache first
        if face_path in self.cache:
            return self.cache[face_path]
        
        try:
            img = cv2.imread(face_path)
            if img is None:
                return 0.5
            
            # Resize to expected size
            img = cv2.resize(img, (224, 224))
            
            # Check feature cache
            if face_path in self.feature_cache:
                features = self.feature_cache[face_path]
            else:
                features = self.extract_features_improved(img)
                self.feature_cache[face_path] = features
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            if self.model:
                proba = self.model.predict_proba(features_scaled)[0]
                score = proba[1]  # Probability of being fake
                self.cache[face_path] = score
                return score
            else:
                return 0.5
                
        except Exception as e:
            # Silent fail to avoid spam
            return 0.5
    
    def predict_batch(self, face_paths):
        """Predict batch of faces"""
        results = {}
        for path in face_paths:
            results[path] = self.predict(path)
        return results
    
    def load_model(self):
        """Load pre-trained model"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    # Clear caches when loading new model
                    self.cache = {}
                    self.feature_cache = {}
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.cache = {}
        self.feature_cache = {}

# Global instance
rf_detector = RandomForestDeepfakeDetector()