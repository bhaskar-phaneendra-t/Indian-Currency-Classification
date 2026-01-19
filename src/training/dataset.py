import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform

        # 🔥 LABEL MAPPING (CRITICAL)
        self.label_map = {
            10: 0,
            20: 1,
            50: 2,
            100: 3,
            200: 4,
            500: 5,
            2000: 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_root, row["image_path"])
        raw_label = int(row["label"])

        # 🔥 ENCODE LABEL
        label = self.label_map[raw_label]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
