from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage
import cv2
import numpy as np
import time

class CameraWorker(QThread):
    """
    Worker thread that continuously fetches frames from the camera interface
    and emits them as QImages for the GUI.
    """
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._running = True

    def run(self):
        """
        Main loop of the thread.
        """
        self.camera.start()

        while self._running:
            try:
                if not self.camera.is_running():
                    # If camera stopped unexpectedly
                    time.sleep(0.1)
                    continue

                frame = self.camera.get_frame()

                if frame is not None:
                    # Convert BGR (OpenCV) to RGB (Qt)
                    # frame is numpy array
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width

                    # Convert to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Create QImage
                    q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                    # Copy to ensure data persistence after loop iteration
                    self.frame_ready.emit(q_img.copy())
                else:
                    # If no frame, sleep briefly to avoid 100% CPU
                    time.sleep(0.01)

            except Exception as e:
                self.error_occurred.emit(str(e))
                time.sleep(1)

    def stop(self):
        self._running = False
        self.quit()
        self.wait()
        self.camera.stop()
