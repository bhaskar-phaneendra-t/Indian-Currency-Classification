import os
from src.utils.dir_paths import processed_dir
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def split_data(
    metadata_csv: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> None:
    try:
        logger.info("starting data splitting")

        dataframe = pd.read_csv(metadata_csv)

        if dataframe.empty:
            raise ValueError("dataframe is empty")

        logger.info(f"total length of the dataframe: {len(dataframe)}")

        # Train + Temp split
        train_dataframe, temp_dataframe = train_test_split(
            dataframe,
            test_size=test_size + val_size,
            stratify=dataframe["label"],
            random_state=random_state
        )

        # Validation + Test split
        val_dataframe, test_dataframe = train_test_split(
            temp_dataframe,
            test_size=test_size / (test_size + val_size),
            stratify=temp_dataframe["label"],
            random_state=random_state
        )

        os.makedirs(output_dir, exist_ok=True)

        train_dataframe.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_dataframe.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_dataframe.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        logger.info(f"Train samples: {len(train_dataframe)}")
        logger.info(f"Validation samples: {len(val_dataframe)}")
        logger.info(f"Test samples: {len(test_dataframe)}")

        logger.info("Data splitting completed successfully")

    except Exception as e:
        logger.error("Error during data splitting", exc_info=True)
        raise CustomException(e)


def main():
    try:
        root_dir = os.getcwd()

        metadata_csv = os.path.join(
            root_dir, "data", "processed", "metadata", "metadata.csv"
        )

        split_output_dir = os.path.join(
            processed_dir, "splits"
        )

        split_data(
            metadata_csv=metadata_csv,
            output_dir=split_output_dir
        )

    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    main()
