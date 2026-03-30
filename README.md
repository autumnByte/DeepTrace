# DeepTrace - AI-Powered Deepfake Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

A deep learning-based system for detecting deepfake content in videos using a hybrid approach combining a Convolutional Neural Network (CNN) and a Random Forest classifier.

---

## 🚀 Overview

This project detects manipulated (deepfake) video frames using:

- **CNN (EfficientNet-B0)** → Primary decision maker  
- **Random Forest** → Confidence booster (validation layer)

The system performs frame-by-frame analysis and provides confidence-based classification along with visual insights.

---

## 🧠 Key Features

- 🎥 Frame-level deepfake detection
- 🧩 Hybrid model architecture (CNN + ML ensemble)
- 📊 Confidence-based classification (High / Uncertain / Low)
- 📈 Interactive dashboard with:
  - Confidence distributions
  - Model comparison (CNN vs RF)
  - Frame-wise predictions
- ⚠️ Built-in uncertainty handling (0.4–0.65 range)
- 📉 Confusion matrix & performance metrics

---

## 🏗️ Architecture
Input Video → Frame Extraction → Face Detection → Feature Extraction → Dual Model Analysis → Temporal Localization → Results Visualization

---

## 📊 Model Details

### 🔹 CNN (Primary Detector)
- Architecture: EfficientNet-B0  
- Accuracy: ~89%  
- Precision: 100%  
- Recall: ~89%  
- F1 Score: ~94%  
- Role: Final decision maker  

---

### 🔹 Random Forest (Confidence Booster)
- Ensemble: 200 trees  
- Features: Texture, edges, color, frequency  
- Role: Supports CNN predictions to improve reliability  

---

## 🎯 Classification Logic

- **Score > 0.65 → High Confidence (Likely Fake)**
- **0.4 – 0.65 → Uncertain**
- **< 0.4 → Likely Real**

Random Forest agreement increases confidence but does not override CNN decisions.

---

## 📦 Dataset

- Total images: ~1,480  
  - Real: ~1,000  
  - Fake: ~480  

> ⚠️ Dataset is limited and may not generalize to all deepfake types.

---

## 📉 Performance Evaluation

Confusion Matrix (Test Set):

|                | Predicted Fake | Predicted Real |
|----------------|---------------|---------------|
| Actual Fake    | 84            | 11            |
| Actual Real    | 0             | 180           |

---

## ⚠️ Limitations

- Trained on a relatively small dataset  
- May not generalize to unseen deepfake techniques  
- Sensitive to:
  - Video quality
  - Compression artifacts
  - Lighting conditions  
- Thresholds are heuristic and not fully optimized  

---

## 🧪 Use Cases

- Educational demonstration of deepfake detection  
- Research prototype for ensemble-based detection  
- Video screening tool (not for forensic-level decisions)

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Scikit-learn  
- OpenCV  
- Streamlit  

---

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/your-username/deepfake-detector.git

# Navigate
cd deepfake-detector

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

