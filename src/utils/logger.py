import logging
from datetime import datetime
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

    return logger
