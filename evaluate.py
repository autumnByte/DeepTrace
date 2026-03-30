import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import YOUR pipeline and model (correct way for your project)
from main import run_pipeline
from modules.deepfake_detection import cnn_model

# ⚙️ CONFIG
VIDEO_PATH = "test/test_video.mp4"   # change if needed
THRESHOLD = 0.55

# 🚨 Check model loaded
if cnn_model is None:
    raise Exception("❌ CNN model not loaded. Check models folder.")

print("🚀 Running evaluation...")

# ▶️ Run full pipeline
results = run_pipeline(VIDEO_PATH, THRESHOLD)

# Get predictions
fake_scores = results.get("fake_scores", {})

if not fake_scores:
    raise Exception("❌ No predictions found. Check pipeline output.")

# Convert scores → predictions
y_pred = [1 if score >= THRESHOLD else 0 for score in fake_scores.values()]

# ⚠️ IMPORTANT: YOU MUST DEFINE GROUND TRUTH
# 👉 If your test video is FAKE:
y_true = [1 for _ in y_pred]

# 👉 If your test video is REAL:
# y_true = [0 for _ in y_pred]

# 🔢 Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# 📊 Save metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 🖨️ Print results
print("\n📊 Evaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n✅ Metrics saved to metrics.json")