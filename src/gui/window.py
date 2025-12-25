from enum import Enum, auto
import os
import sys

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QRect
from PyQt5.QtGui import QPixmap, QImage

from camera.worker import CameraWorker, CaptureWorker
from .video_label import VideoLabel
from roi import ROIManager

class AppState(Enum):
    LIVE_VIEW = auto()
    CAPTURED = auto()

class MainWindow(QMainWindow):
    def __init__(self, camera_impl):
        super().__init__()
        self.camera_impl = camera_impl
        self.worker = None
        self.capture_worker = None
        self.current_state = None

        # ROI Manager
        # We assume main.py is in src/, but project root is parent of src?
        # Let's check where main.py is running from.
        # Usually standard is execution from root.
        # But based on file list, main.py is in src/main.py or root?
        # List files showed main.py in src/.
        # The requirements said "roi.json should live in the project root directory, alongside main.py."
        # If main.py is in src/, then root is src/.
        # However, typically project root is one level up.
        # Let's assume '.' works relative to CWD.
        self.roi_manager = ROIManager(".")

        # Temp storage for ROI selection before saving
        self.temp_roi_rect = None

        # Cache for ROI data to avoid reading disk in Live View loop
        self.cached_roi_data = None

        # Initial load of ROI
        self.refresh_roi_cache()

        self.init_ui()

        # Start in LIVE_VIEW
        self.set_state(AppState.LIVE_VIEW)

    def init_ui(self):
        self.setWindowTitle("Camera Live View")
        self.resize(800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label for Camera Feed
        self.image_label = VideoLabel("Initializing Camera...")
        self.image_label.selection_changed.connect(self.handle_selection_change)
        layout.addWidget(self.image_label)

        # Buttons Layout (Capture / Resume)
        btn_layout = QHBoxLayout()

        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(self.handle_capture_click)
        btn_layout.addWidget(self.btn_capture)

        self.btn_resume = QPushButton("Resume Live")
        self.btn_resume.clicked.connect(self.handle_resume_click)
        btn_layout.addWidget(self.btn_resume)

        layout.addLayout(btn_layout)

        # ROI Buttons Layout
        roi_layout = QHBoxLayout()

        self.btn_set_roi = QPushButton("Set ROI")
        self.btn_set_roi.clicked.connect(self.handle_set_roi_click)
        roi_layout.addWidget(self.btn_set_roi)

        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.handle_clear_roi_click)
        roi_layout.addWidget(self.btn_clear_roi)

        layout.addLayout(roi_layout)

    def set_state(self, state: AppState):
        self.current_state = state

        if state == AppState.LIVE_VIEW:
            self.btn_capture.setEnabled(True)
            self.btn_resume.setEnabled(False)

            # ROI Buttons disabled in Live View
            self.btn_set_roi.setEnabled(False)
            self.btn_clear_roi.setEnabled(False)

            # Disable selection interactivity
            self.image_label.set_selection_enabled(False)

            self.start_live_view()

        elif state == AppState.CAPTURED:
            self.btn_capture.setEnabled(False)
            self.btn_resume.setEnabled(True)

            # ROI Buttons enabled in Captured View
            self.btn_set_roi.setEnabled(True)
            self.btn_clear_roi.setEnabled(True)

            # Enable selection interactivity
            self.image_label.set_selection_enabled(True)

            # Live view is stopped in the transition logic before reaching here

    def start_live_view(self):
        # Stop existing worker if any (sanity check)
        if self.worker:
            self.worker.stop()

        self.image_label.setText("Starting Live View...")
        self.worker = CameraWorker(self.camera_impl)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

        # When starting live view, refresh ROI cache to ensure we show latest
        self.refresh_roi_cache()

    def handle_capture_click(self):
        """
        Transition from LIVE_VIEW to CAPTURED (via async capture).
        """
        if self.current_state != AppState.LIVE_VIEW:
            return

        # 1. Disable buttons to prevent re-entry
        self.btn_capture.setEnabled(False)
        self.btn_resume.setEnabled(False)

        # 2. Stop Live View
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.image_label.setText("Capturing...")

        # 3. Start Capture Worker
        self.capture_worker = CaptureWorker(self.camera_impl, "/tmp/capture.jpg")
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.error_occurred.connect(self.on_capture_error)
        self.capture_worker.start()

    def on_capture_finished(self):
        # 4. Load and display image
        pixmap = QPixmap("/tmp/capture.jpg")
        if not pixmap.isNull():
            self.image_label.set_frame(pixmap)

            # Make sure cache is fresh (e.g. if modified elsewhere, though unlikely)
            self.refresh_roi_cache()

            # Apply stored ROI overlay if exists (scaled to this capture)
            self.apply_stored_roi_overlay(pixmap.width(), pixmap.height())

            # 5. Update State
            self.set_state(AppState.CAPTURED)
        else:
            self.on_capture_error("Failed to load captured image.")

    def on_capture_error(self, error_msg):
        # Show error
        QMessageBox.critical(self, "Capture Error", f"Capture failed:\n{error_msg}")
        # Revert to Live View
        self.set_state(AppState.LIVE_VIEW)

    def handle_resume_click(self):
        if self.current_state != AppState.CAPTURED:
            return

        # Discard unsaved changes is handled implicitly by reloading from disk next time
        self.temp_roi_rect = None

        self.image_label.setText("Resuming...")
        self.set_state(AppState.LIVE_VIEW)

    @pyqtSlot(QImage)
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.set_frame(pixmap)

        # Use cached ROI data to avoid disk I/O in loop
        self.apply_stored_roi_overlay(pixmap.width(), pixmap.height())

    def refresh_roi_cache(self):
        """Load ROI from disk into memory."""
        self.cached_roi_data = self.roi_manager.load_roi()

    def apply_stored_roi_overlay(self, current_w, current_h):
        """
        Uses cached ROI data, scales it to current_w/h, and sets it on VideoLabel.
        """
        roi_data = self.cached_roi_data
        if roi_data:
            orig_w = roi_data['image_width']
            orig_h = roi_data['image_height']

            if orig_w == 0 or orig_h == 0:
                return

            # Scale
            scale_x = current_w / orig_w
            scale_y = current_h / orig_h

            x = int(roi_data['x'] * scale_x)
            y = int(roi_data['y'] * scale_y)
            w = int(roi_data['width'] * scale_x)
            h = int(roi_data['height'] * scale_y)

            self.image_label.set_roi_overlay(QRect(x, y, w, h))
        else:
            self.image_label.set_roi_overlay(QRect())

    def handle_selection_change(self, rect: QRect):
        """Called when user drags a box in VideoLabel."""
        self.temp_roi_rect = rect

    def handle_set_roi_click(self):
        """Commit the current temporary ROI to disk."""
        if not self.temp_roi_rect:
            QMessageBox.information(self, "ROI Info", "Please draw a region first.")
            return

        # We need to save it relative to the CURRENT image resolution
        # because the user drew it on the current image.
        if not self.image_label.pixmap_frame:
            return

        current_w = self.image_label.pixmap_frame.width()
        current_h = self.image_label.pixmap_frame.height()

        self.roi_manager.save_roi(
            self.temp_roi_rect.x(),
            self.temp_roi_rect.y(),
            self.temp_roi_rect.width(),
            self.temp_roi_rect.height(),
            current_w,
            current_h
        )

        # Clear temp
        self.temp_roi_rect = None

        QMessageBox.information(self, "ROI Saved", "Region of Interest saved successfully.")

        # Reload to cache and apply
        self.refresh_roi_cache()
        self.apply_stored_roi_overlay(current_w, current_h)

    def handle_clear_roi_click(self):
        self.roi_manager.clear_roi()
        self.image_label.set_roi_overlay(QRect())
        self.temp_roi_rect = None
        self.cached_roi_data = None # Clear cache
        # QMessageBox.information(self, "ROI Cleared", "Region of Interest cleared.")

    @pyqtSlot(str)
    def handle_error(self, error_msg):
        print(f"Camera Error: {error_msg}")
        # Could show a message box if critical, but for live stream usually just log

    def closeEvent(self, event):
        if self.worker:
            self.image_label.setText("Stopping Camera...")
            self.worker.stop()
        if self.capture_worker and self.capture_worker.isRunning():
            self.capture_worker.quit()
            self.capture_worker.wait()
        event.accept()
