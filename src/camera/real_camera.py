import shutil
import subprocess
import cv2
import numpy as np
import threading
import time
from .interface import CameraInterface

# Import Picamera2.
# Note: This will fail in environments where picamera2 is not installed unless mocked.
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

class RealCamera(CameraInterface):
    """
    Implementation of CameraInterface using Picamera2.
    Configures a dual-stream pipeline:
      - 'lores': 640x480 YUV420 for live preview (fast)
      - 'main':  3280x2464 RGB888 for still capture (high res)
    Ensures identical framing via locked ScalerCrop.
    """

    def __init__(self):
        """
        Initialize the camera instance.
        """
        if Picamera2 is None:
            raise ImportError("Picamera2 library is not installed.")

        self.picam2 = Picamera2()
        self.lock = threading.Lock()
        self._running = False

        # Canonical configuration values
        self.main_config = {"size": (3280, 2464), "format": "RGB888"}
        self.lores_config = {"size": (640, 480), "format": "YUV420"}

    @staticmethod
    def check_camera_availability() -> bool:
        """
        Checks if a camera is available using `libcamera-hello --list-cameras`.
        Returns True if a camera is found, False otherwise.
        """
        if not shutil.which("libcamera-hello"):
            # Fallback for systems where libcamera-hello might be missing but rpicam-still exists
            # (though we are removing rpicam usage, checking existence is harmless)
             if not shutil.which("rpicam-still"):
                 return False
             cmd_base = "rpicam-still"
        else:
            cmd_base = "libcamera-hello"

        try:
            # Run the list command
            result = subprocess.run(
                [cmd_base, "--list-cameras"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and "Available cameras" in result.stdout:
                return True
            return False
        except Exception as e:
            print(f"[RealCamera] Error checking camera: {e}")
            return False

    def start(self) -> None:
        """
        Configures and starts the Picamera2 instance with dual streams.
        Locks the geometry to the full sensor FOV.
        """
        if self._running:
            return

        print("[RealCamera] Configuring dual streams...")

        # Configure the camera with two streams
        config = self.picam2.create_still_configuration(
            main=self.main_config,
            lores=self.lores_config,
            display="lores"
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Lock the ScalarCrop to the full sensor resolution to ensure identical FOV
        # for both streams (preventing digital zoom or aspect ratio crop mismatches).
        full_res = self.picam2.camera_properties["PixelArraySize"]
        self.picam2.set_controls({
            "ScalerCrop": [0, 0, full_res[0], full_res[1]],
            "AfMode": 2  # Continuous autofocus
        })

        self._running = True
        print("[RealCamera] Started.")

    def stop(self) -> None:
        """
        Stops the camera and releases resources.
        """
        if not self._running:
            return

        print("[RealCamera] Stopping...")
        self.picam2.stop()
        # self.picam2.close() # Picamera2 usually stays open, but stop is enough.
        self._running = False
        print("[RealCamera] Stopped.")

    def get_frame(self) -> np.ndarray:
        """
        Retrieves the latest frame from the 'lores' stream for preview.
        Converts YUV420 -> BGR.
        """
        if not self._running:
            return None

        try:
            with self.lock:
                # Capture from the low-res preview stream (fast)
                # YUV420 comes as I420 (Planar Y, U, V)
                yuv_frame = self.picam2.capture_array("lores")

            if yuv_frame is not None:
                # Convert I420 YUV to BGR
                return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        except Exception as e:
            print(f"[RealCamera] get_frame error: {e}")

        return None

    def is_running(self) -> bool:
        return self._running

    def capture_still(self, output_path: str) -> None:
        """
        Captures a high-resolution still from the 'main' stream.
        Converts RGB888 -> BGR and saves to disk.
        Does NOT stop the live stream.
        """
        if not self._running:
            raise RuntimeError("Camera is not running.")

        print(f"[RealCamera] Capturing still to {output_path}...")

        try:
            with self.lock:
                # Capture from the high-res main stream
                # Picamera2 returns RGB888 for 'main' as configured
                rgb_frame = self.picam2.capture_array("main")

            if rgb_frame is not None:
                # Convert RGB to BGR for OpenCV saving
                #bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Save to disk
                success = cv2.imwrite(output_path, rgb_frame)
                if not success:
                    raise IOError(f"Failed to write image to {output_path}")
                print("[RealCamera] Capture successful.")
            else:
                raise RuntimeError("Captured frame was None")

        except Exception as e:
            print(f"[RealCamera] Capture failed: {e}")
            raise e
