# train_model_improved.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from pathlib import Path

# Create models directory
Path("models").mkdir(exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
print("Creating EfficientNet model...")
model = timm.create_model("efficientnet_b0", pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(device)

# Transform with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Loading dataset...")
dataset = datasets.ImageFolder("dataset", transform=transform)

# Calculate class counts
class_counts = [0, 0]  # [real, fake]
for _, label in dataset.samples:
    class_counts[label] += 1

print(f"Dataset: Real={class_counts[0]} images, Fake={class_counts[1]} images")

# Create weighted sampler for imbalanced dataset
weights = []
for _, label in dataset.samples:
    weights.append(1.0 / class_counts[label])

sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# DataLoader
loader = DataLoader(dataset, batch_size=16, sampler=sampler)

# Loss with class weights
pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

# Training
print("\nStarting training...")
best_loss = float('inf')

for epoch in range(20):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 9:
            print(f"Epoch {epoch}, Batch {batch_idx+1}: loss = {total_loss/(batch_idx+1):.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "models/deepfake_model_best.pth")
        print(f"  → Saved best model (loss: {best_loss:.4f})")

print("\n✅ Training complete!")
print(f"Best model saved to: models/deepfake_model_best.pth")
print(f"Final loss: {best_loss:.4f}")