import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report

from src.training.dataloader import create_dataloaders
from src.utils.logger import logging

# -------------------------
# CONFIG
# -------------------------
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_LABEL = {
    0: "10",
    1: "20",
    2: "50",
    3: "100",
    4: "200",
    5: "500",
    6: "2000"
}

# -------------------------
# LOAD MODEL
# -------------------------
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def evaluate():
    # -------------------------
    # DATALOADER (TEST ONLY)
    # -------------------------
    _, _, test_loader = create_dataloaders(
        train_csv="data/processed/splits/train.csv",
        val_csv="data/processed/splits/val.csv",
        test_csv="data/processed/splits/test.csv",
        image_root="data/raw",
        batch_size=BATCH_SIZE
    )

    model = load_model("best_currency_model.pth")
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    y_true = []
    y_pred = []

    # -------------------------
    # TEST LOOP
    # -------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )

    print("\n TEST SET CLASSIFICATION REPORT")
    print(report)

    logging.info(f"Test Loss: {avg_test_loss:.4f}")
    logging.info(f"\n{report}")


if __name__ == "__main__":
    evaluate()
