import cv2
import glob
import time
from pathlib import Path
from typing import List, Optional
from src.hal.base import ImageSource
from src.core.types import RawImage
from src.infra.logging import get_logger

logger = get_logger(__name__)

class FileImageSource(ImageSource):
    """
    Developer Mode Image Source.
    Reads images from a folder sequentially or by ID.
    """
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.image_files: List[Path] = []
        self.current_index = 0

    def initialize(self) -> None:
        if not self.folder_path.exists():
            logger.warning(f"Image source folder {self.folder_path} does not exist. Creating it.")
            self.folder_path.mkdir(parents=True, exist_ok=True)

        self._scan_folder()
        logger.info(f"FileImageSource initialized. Found {len(self.image_files)} images in {self.folder_path}")

    def _scan_folder(self):
        # Support common formats
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        files = []
        for ext in extensions:
            files.extend(self.folder_path.glob(ext))
        self.image_files = sorted(files)

    def capture(self) -> RawImage:
        """
        In Dev Mode, 'capture' returns the next image in the folder.
        If empty, raises an error or returns None (handled by caller).
        """
        self._scan_folder() # Refresh to catch new drops

        if not self.image_files:
            raise RuntimeError("No images found in input folder.")

        # Cycle through images
        if self.current_index >= len(self.image_files):
            self.current_index = 0

        file_path = self.image_files[self.current_index]
        self.current_index += 1

        logger.info(f"Simulating capture from file: {file_path.name}")

        # Load Image
        img_data = cv2.imread(str(file_path))
        if img_data is None:
             raise RuntimeError(f"Failed to load image: {file_path}")

        return RawImage(
            data=img_data,
            metadata={
                "source_file": file_path.name,
                "mode": "developer_simulation"
            }
        )

    def release(self) -> None:
        pass
