import os
from modules.rf_detector import rf_detector
from pathlib import Path

# Load model
if rf_detector.load_model():
    print("✅ Random Forest model loaded")
else:
    print("❌ Failed to load model")
    exit()

# Test on a sample face
face_files = list(Path("data/faces").glob("*.jpg"))
print(f"\nFound {len(face_files)} face files")

if face_files:
    test_face = face_files[0]
    print(f"\nTesting on: {test_face}")
    
    score = rf_detector.predict(str(test_face))
    print(f"RF Score: {score:.3f}")
    
    # Test first 10 faces
    print("\nTesting first 10 faces:")
    for i, face in enumerate(face_files[:10]):
        score = rf_detector.predict(str(face))
        print(f"  {face.name}: {score:.3f}")
else:
    print("No face files found. Run main.py first to detect faces.")