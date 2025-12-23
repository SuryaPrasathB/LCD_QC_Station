import cv2
import shutil
import glob
import os
from pathlib import Path
from datetime import datetime
from src.core.types import RawImage
from src.core.config import settings
from src.infra.logging import get_logger

logger = get_logger(__name__)

class ImageStore:
    """
    Manages filesystem storage for images.
    """
    @staticmethod
    def save_image(raw_image: RawImage) -> str:
        """
        Saves the raw image to disk.
        Returns the relative file path.
        """
        # Structure: YYYY/MM/DD/UUID.jpg
        now = raw_image.timestamp
        date_folder = settings.IMAGE_STORE_PATH / now.strftime("%Y/%m/%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        filename = f"{raw_image.id}.jpg"
        filepath = date_folder / filename

        cv2.imwrite(str(filepath), raw_image.data)

        # Return path relative to data dir for portability
        # e.g. "images/2023/10/01/uuid.jpg"
        relative_path = filepath.relative_to(settings.DATA_DIR)
        logger.info(f"Saved image to {relative_path}")
        return str(relative_path)

    @staticmethod
    def get_image_path(relative_path: str) -> Path:
        return settings.DATA_DIR / relative_path

    @staticmethod
    def cleanup_old_images(max_count: int = 10000):
        # Implementation of retention policy
        # For MVP, we'll just log
        logger.info("Cleanup check skipped (Not fully implemented in MVP)")
