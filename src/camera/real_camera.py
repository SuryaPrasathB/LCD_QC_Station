import subprocess
import shutil
import cv2
import numpy as np
import threading
import os
from .interface import CameraInterface

class RealCamera(CameraInterface):
    """
    Implementation of CameraInterface using rpicam-vid (libcamera-vid).
    Captures MJPEG stream from stdout.
    """

    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self.process = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = None

    @staticmethod
    def check_camera_availability() -> bool:
        """
        Checks if a camera is available using `rpicam-still --list-cameras`.
        Returns True if a camera is found, False otherwise.
        """
        if not shutil.which("rpicam-still"):
            return False

        try:
            # Run the list command
            result = subprocess.run(
                ["rpicam-still", "--list-cameras"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )

            # Check for success and keywords in output
            # Usually output contains "Available cameras" or lists index 0, 1 etc.
            # If no cameras, it usually prints "No cameras available" or returns non-zero.
            if result.returncode == 0 and "Available cameras" in result.stdout:
                return True
            return False
        except Exception as e:
            print(f"Error checking camera: {e}")
            return False

    def start(self) -> None:
        if self._running:
            return

        cmd = [
            "rpicam-vid",
            "--inline", # Headers with every I-frame
            "--nopreview",
            "--codec", "mjpeg",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--timeout", "0", # Run indefinitely
            "--output", "-"   # Output to stdout
        ]

        print(f"[RealCamera] Starting: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # Avoid deadlock if buffer fills
                bufsize=10**6 # Large buffer
            )
            self._running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[RealCamera] Failed to start process: {e}")
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        print("[RealCamera] Stopped.")

    def _update(self):
        """
        Internal loop to read from stdout and decode MJPEG frames.
        """
        buffer = b""
        chunk_size = 4096

        while self._running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk

                while True:
                    # Find JPEG Start of Image (SOI)
                    a = buffer.find(b'\xff\xd8')
                    if a == -1:
                        # Keep the last few bytes just in case split happens exactly at marker
                        if len(buffer) > 4:
                            buffer = buffer[-4:]
                        break

                    # Find JPEG End of Image (EOI) starting after SOI
                    b = buffer.find(b'\xff\xd9', a)
                    if b == -1:
                        # Not yet received the end
                        break

                    # Extract the full JPEG frame
                    jpg_data = buffer[a:b+2]
                    buffer = buffer[b+2:] # Move past this frame

                    # Decode to OpenCV image
                    # Use numpy frombuffer -> cv2.imdecode for speed
                    try:
                        data = np.frombuffer(jpg_data, dtype=np.uint8)
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame
                    except Exception as e:
                        print(f"[RealCamera] Decode error: {e}")

            except Exception as e:
                print(f"[RealCamera] Stream error: {e}")
                break

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def is_running(self) -> bool:
        return self._running

    def capture_still(self, output_path: str) -> None:
        """
        Captures a still image using rpicam-still.
        Stops the live stream if it is running to free the camera resource.
        """
        # Ensure live stream is stopped to release camera
        if self._running:
            print("[RealCamera] Stopping live stream before capture...")
            self.stop()

        # rpicam-still capture command
        # -t 2000: 2 second warmup for AWB/AE convergence
        # --width/height: 8MP limit (safe for Pi 3B)
        # --autofocus-mode auto: Ensure AF runs
        cmd = [
            "rpicam-still",
            "-o", output_path,
            "--autofocus-mode", "auto",
            "--autofocus-range", "normal",
            "--autofocus-speed", "normal",
            "-t", "2000",
            "--width", "3280",
            "--height", "2464",
            "--nopreview"
        ]

        print(f"[RealCamera] Running capture: {' '.join(cmd)}")

        try:
            # Run blocking command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("[RealCamera] Capture successful.")
        except subprocess.CalledProcessError as e:
            print(f"[RealCamera] Capture failed: {e.stderr}")
            raise RuntimeError(f"rpicam-still failed: {e.stderr}")
        except Exception as e:
            print(f"[RealCamera] Capture error: {e}")
            raise e
