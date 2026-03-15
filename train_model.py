import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm

device = "cpu"

# CREATE MODEL FIRST
model = timm.create_model("efficientnet_b0", pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Linear(model.classifier.in_features, 1)

# THEN move to device
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

for epoch in range(10):

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)

torch.save(model.state_dict(), "deepfake_model.pth")