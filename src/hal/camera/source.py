import subprocess
import cv2
import time
import numpy as np
from pathlib import Path
from src.hal.base import ImageSource
from src.core.types import RawImage
from src.core.config import settings
from src.infra.logging import get_logger

logger = get_logger(__name__)

class PiCameraSource(ImageSource):
    """
    Production Mode Camera Source using `rpicam-still`.
    """
    def __init__(self):
        self.camera_ready = False

    def initialize(self) -> None:
        """
        Check if `rpicam-still` is available.
        """
        try:
            # Check version or help to verify presence
            subprocess.run(["rpicam-still", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            self.camera_ready = True
            logger.info("PiCameraSource: rpicam-still detected.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("PiCameraSource: rpicam-still NOT found. Camera mode will fail.")
            self.camera_ready = False

    def capture(self) -> RawImage:
        if not self.camera_ready:
             raise RuntimeError("Camera is not available.")

        # We capture to a temporary file (stdout piping is faster but can be tricky with rpicam-still formatting)
        # Using stdout with encoding is better for latency if possible, but temp file is safer for MVP.
        # Let's try temp file for robustness.
        temp_path = "/tmp/capture.jpg"

        cmd = [
            "rpicam-still",
            "-o", temp_path,
            "-t", "10", # Short timeout (10ms) - assumes camera is running or cold start
            "--immediate", # Capture immediately
            "--width", "4624", # 64MP is huge (9248x6944), maybe scale down?
            # Request said "High-resolution still capture (64MP)".
            # But 64MP is slow to process on Pi 3B.
            # We will capture full res but maybe the user wants full res saved.
            # Warning: 64MP decode on Pi 3B takes seconds.
            # Let's stick to full res as requested, but user might want to reconsider.
            "--nopreview"
        ]

        # Optimization: If startup latency is high, we might want "rpicam-still -t 0 -s" (signal mode)
        # But for "Deterministic", single shot is simpler.

        start_time = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=settings.CAMERA_TIMEOUT)
        except subprocess.TimeoutError:
            raise RuntimeError("Camera capture timed out.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Camera capture failed: {e}")

        capture_duration = time.time() - start_time
        logger.info(f"Camera capture took {capture_duration:.2f}s")

        # Read back
        img_data = cv2.imread(temp_path)
        if img_data is None:
            raise RuntimeError("Failed to read captured image.")

        return RawImage(
            data=img_data,
            metadata={
                "capture_duration_ms": int(capture_duration * 1000),
                "mode": "production_camera"
            }
        )

    def release(self) -> None:
        pass
