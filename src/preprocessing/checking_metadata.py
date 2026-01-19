import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from PIL import Image
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

def visual_image_check(metadata_csv: str, sample_size: int = 9) -> None:
    try:
        logger.info("starting visual check")

        dataframe = pd.read_csv(metadata_csv)
        if dataframe.empty:
            raise ValueError("metadata is empty")

        sample_dataframe = dataframe.sample(n=min(sample_size, len(dataframe)))

        cols = int(sample_size ** 0.5)
        rows = (sample_size + cols - 1) // cols

        plt.figure(figsize=(10, 10))

        for index, (_, row) in enumerate(sample_dataframe.iterrows()):
            image = Image.open(row["image_path"]).convert("RGB")

            plt.subplot(rows, cols, index + 1)
            plt.imshow(image)
            plt.title(f"Label: {row['label']}")
            plt.axis("off")

        plt.suptitle("Visual image Check", fontsize=16)
        plt.tight_layout()
        plt.show()

        logger.info("Visual image check completed successfully")

    except Exception as e:
        logger.error("Error during visual image check", exc_info=True)
        raise CustomException(e)


def main():
    try:
        root_dir = os.getcwd()
        metadata_csv = os.path.join(
            root_dir, "data", "processed", "metadata", "metadata.csv"
        )

        visual_image_check(metadata_csv, sample_size=9)

    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    main()
