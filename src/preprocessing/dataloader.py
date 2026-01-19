from torch.utils.data import DataLoader
from torchvision import transforms
from src.preprocessing.dataset import CurrencyDataset
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0
):
    try:
        logger.info("Creating DataLoaders")

        # ImageNet normalization (standard practice)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),   # RESIZE HERE
            transforms.ToTensor(),
            transforms.Normalize(mean, std)                # NORMALIZE HERE
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = CurrencyDataset(train_csv, transform=train_transform)
        val_dataset = CurrencyDataset(val_csv, transform=val_test_transform)
        test_dataset = CurrencyDataset(test_csv, transform=val_test_transform)

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

        logger.info("DataLoaders created successfully")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error("Error creating DataLoaders", exc_info=True)
        raise CustomException(e)
