import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.training.dataloader import create_dataloaders
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys


def train_model():
    try:
        # -------------------------------
        # CONFIG
        # -------------------------------
        NUM_CLASSES = 7
        EPOCHS = 15
        BATCH_SIZE = 32
        LEARNING_RATE = 0.0001
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {DEVICE}")

        # -------------------------------
        # DATALOADERS
        # -------------------------------
        train_loader, val_loader, _ = create_dataloaders(
            train_csv="data/processed/splits/train.csv",
            val_csv="data/processed/splits/val.csv",
            test_csv="data/processed/splits/test.csv",
            image_root="data/raw",
            batch_size=BATCH_SIZE
        )


        # -------------------------------
        # MODEL (Transfer Learning)
        # -------------------------------
        model = models.resnet18(pretrained=True)

        model.fc = nn.Sequential(
    nn.Dropout(p=0.5),   
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

        model = model.to(DEVICE)

        # -------------------------------
        # LOSS & OPTIMIZER
        # -------------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # -------------------------------
        # TRAINING LOOP
        # -------------------------------
        for epoch in range(1, EPOCHS + 1):
            print(f"\n🚀 Epoch {epoch} started")
            logging.info(f"Epoch {epoch} started")

            # ==========================
            # TRAINING
            # ==========================
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            print(f"⏳ Training epoch {epoch} is ongoing...")

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                train_total += labels.size(0)
                train_correct += (preds == labels).sum().item()

            train_acc = 100 * train_correct / train_total
            train_loss /= len(train_loader)

            print(
                f"✅ Epoch {epoch} TRAIN completed | "
                f"Loss: {train_loss:.4f} | "
                f"Accuracy: {train_acc:.2f}%"
            )

            # ==========================
            # VALIDATION
            # ==========================
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)

                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss /= len(val_loader)

            print(
                f" Epoch {epoch} VALIDATION | "
                f"Loss: {val_loss:.4f} | "
                f"Accuracy: {val_acc:.2f}%"
            )

            logging.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        # -------------------------------
        # SAVE MODEL
        # -------------------------------
        torch.save(model.state_dict(), "best_currency_model.pth")
        logging.info("Model saved as best_currency_model.pth")

        print("\n Training finished successfully!")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    train_model()
