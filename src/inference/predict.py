import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ---------------------------
# CONFIG
# ---------------------------
NUM_CLASSES = 7
THRESHOLD = 0.85   # 🔥 confidence threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_LABEL = {
    0: 10,
    1: 20,
    2: 50,
    3: 100,
    4: 200,
    5: 500,
    6: 2000
}


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict_currency(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)

        max_prob, pred_idx = torch.max(probs, dim=1)

    confidence = max_prob.item()

    if confidence < THRESHOLD:
        return {
            "prediction": "UNKNOWN",
            "confidence": round(confidence, 4)
        }

    return {
        "prediction": IDX_TO_LABEL[pred_idx.item()],
        "confidence": round(confidence, 4)
    }
