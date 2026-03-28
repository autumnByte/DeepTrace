"""
Train Random Forest model for comparison
"""
from modules.rf_detector import rf_detector

print("=" * 50)
print("Training Random Forest Model")
print("=" * 50)

# Train the model
accuracy = rf_detector.train("dataset")

print("\n" + "=" * 50)
print(f"✅ Random Forest trained with accuracy: {accuracy:.3f}")
print("Model saved to: models/rf_model.pkl")
print("=" * 50)