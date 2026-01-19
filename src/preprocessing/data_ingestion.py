import os
import pandas as pd
from PIL import Image
import sys
from typing import List, Dict
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def create_metadata(raw_data_dir: str) -> pd.DataFrame:
    logger.info("Creating metadata dataframe")
    records: List[Dict] = []

    for label in os.listdir(raw_data_dir):
        label_path = os.path.join(raw_data_dir, label)

        if not os.path.isdir(label_path):
            continue

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)

            records.append({
                "image_name": image_name,
                "image_path": image_path,
                "label": int(label)
            })

    df = pd.DataFrame(records)
    df.drop_duplicates(inplace=True)
    logger.info(f"Metadata created with {len(df)} records")

    return df


def remove_corrupted_images(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Checking for corrupted images")
    valid_rows = []

    for _, row in df.iterrows():
        try:
            with Image.open(row["image_path"]) as img:
                img.verify()
            valid_rows.append(row)
        except Exception:
            logger.warning(f"Corrupted image removed: {row['image_path']}")

    clean_df = pd.DataFrame(valid_rows)
    logger.info(f"Remaining images after cleanup: {len(clean_df)}")

    return clean_df


def save_metadata(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Metadata saved at {output_path}")


def main():
    try:
        root_dir = os.getcwd()
        raw_data_dir = os.path.join(root_dir, "data", "raw")
        metadata_dir = os.path.join(root_dir, "data", "processed", "metadata")

        df = create_metadata(raw_data_dir)
        df = remove_corrupted_images(df)
        save_metadata(df, metadata_dir)

        print(f"Metadata created with {len(df)} samples")

    except Exception as e:
        logger.error("Error in data ingestion", exc_info=True)
        raise CustomException(e)


if __name__ == "__main__":
    main()
