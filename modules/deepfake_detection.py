import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import timm

# Load same architecture
model = timm.create_model("efficientnet_b0", pretrained=False)

# Replace classifier
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)

# Load trained weights
model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


def detect_deepfake(frames):
    fake_scores = {}

    for frame in frames:
        face_path = os.path.join("data/faces", f"face_{frame}")

        if not os.path.exists(face_path):
            continue

        image = Image.open(face_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            score = torch.sigmoid(output).item()

        print(f"[Model Output] Frame: {frame}  Score: {score}")

        fake_scores[frame] = score
    return fake_scores
