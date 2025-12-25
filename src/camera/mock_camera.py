import time
import numpy as np
import cv2
from .interface import CameraInterface

class MockCamera(CameraInterface):
    """
    Simulates a camera for development/testing when hardware is unavailable.
    Generates synthetic noise frames.
    """

    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.interval = 1.0 / fps
        self._running = False
        self.last_frame_time = 0

    def start(self) -> None:
        self._running = True
        print("[MockCamera] Started.")

    def stop(self) -> None:
        self._running = False
        print("[MockCamera] Stopped.")

    def get_frame(self) -> np.ndarray:
        if not self._running:
            return None

        # Simulate frame rate limiting
        current_time = time.time()
        if current_time - self.last_frame_time < self.interval:
            time.sleep(0.01) # Sleep briefly to avoid busy loop
            return None

        self.last_frame_time = current_time

        # Generate a random noise frame (Gray or Color)
        # Creating a BGR image
        frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)

        # Add some text to indicate it's a mock
        cv2.putText(frame, "MOCK CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {current_time:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        return frame

    def is_running(self) -> bool:
        return self._running

    def capture_still(self, output_path: str) -> None:
        """Simulates capturing a high-resolution image."""
        print(f"[MockCamera] Capturing still to {output_path}...")

        # Simulate shutter delay (1 second)
        time.sleep(1.0)

        # Create a higher resolution dummy image (e.g. 1920x1080)
        h, w = 1080, 1920
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Add details
        cv2.putText(img, "MOCK STILL CAPTURE", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4)
        cv2.putText(img, f"Timestamp: {time.time()}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # Save to disk
        success = cv2.imwrite(output_path, img)
        if not success:
            raise RuntimeError(f"Failed to save mock capture to {output_path}")

        print(f"[MockCamera] Saved mock capture.")
