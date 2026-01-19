from torch.utils.data import DataLoader
from torchvision import transforms

import os

NUM_WORKERS = 0 if os.path.exists("/.dockerenv") else 2


from src.training.dataset import CurrencyDataset

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    image_root,
    batch_size=32,
    num_workers=2
):
    train_dataset = CurrencyDataset(
        csv_file=train_csv,
        image_root=image_root,
        transform=get_transforms(train=True)
    )

    val_dataset = CurrencyDataset(
        csv_file=val_csv,
        image_root=image_root,
        transform=get_transforms(train=False)
    )

    test_dataset = CurrencyDataset(
        csv_file=test_csv,
        image_root=image_root,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
