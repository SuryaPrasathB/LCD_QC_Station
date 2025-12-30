from enum import Enum, auto
import os
import sys

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox, QLabel
from PyQt5.QtCore import Qt, pyqtSlot, QRect
from PyQt5.QtGui import QPixmap, QImage

import cv2
import numpy as np

from camera.worker import CameraWorker, CaptureWorker
from .video_label import VideoLabel
from roi import ROIManager
from core.inspection import perform_inspection
from core.dataset import DatasetManager, OverrideRecord
import uuid
from datetime import datetime

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

        # Dataset Manager
        self.dataset_manager = DatasetManager(".")
        self.dataset_manager.initialize()

        # ROI Manager
        self.roi_manager = ROIManager(self.dataset_manager)

        # Temp storage for ROI selection before saving
        self.temp_roi_rect = None

        # Cache for ROI data to avoid reading disk in Live View loop
        self.cached_roi_data = None

        # Inspection Context
        self.last_inspection_context = None

        # Initial load of ROI
        self.refresh_roi_cache()

        self.init_ui()

        # Start in LIVE_VIEW
        self.set_state(AppState.LIVE_VIEW)

    def init_ui(self):
        self.setWindowTitle("Camera Live View")
        self.resize(800, 750)

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

        # Inspection Status Panel
        status_layout = QHBoxLayout()

        self.lbl_inspection_result = QLabel("Result: N/A")
        self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_layout.addWidget(self.lbl_inspection_result)

        self.lbl_inspection_score = QLabel("Score: -")
        status_layout.addWidget(self.lbl_inspection_score)

        # Override Buttons
        self.btn_mark_pass = QPushButton("Mark as PASS")
        self.btn_mark_pass.clicked.connect(lambda: self.handle_override(True))
        self.btn_mark_pass.setEnabled(False)
        status_layout.addWidget(self.btn_mark_pass)

        self.btn_mark_fail = QPushButton("Mark as FAIL")
        self.btn_mark_fail.clicked.connect(lambda: self.handle_override(False))
        self.btn_mark_fail.setEnabled(False)
        status_layout.addWidget(self.btn_mark_fail)

        status_layout.addStretch() # Push labels to left
        layout.addLayout(status_layout)

        # Learning Panel
        learning_layout = QHBoxLayout()

        self.lbl_pending_count = QLabel("Pending Overrides: 0")
        learning_layout.addWidget(self.lbl_pending_count)

        self.btn_commit_learning = QPushButton("Commit Learning")
        self.btn_commit_learning.clicked.connect(self.handle_commit_learning)
        learning_layout.addWidget(self.btn_commit_learning)

        learning_layout.addStretch()
        layout.addLayout(learning_layout)

        # Refresh pending count
        self.update_pending_count()

    def set_state(self, state: AppState):
        self.current_state = state

        if state == AppState.LIVE_VIEW:
            self.btn_capture.setEnabled(True)
            self.btn_resume.setEnabled(False)

            # ROI Buttons disabled in Live View
            self.btn_set_roi.setEnabled(False)
            self.btn_clear_roi.setEnabled(False)

            # Disable Override Buttons
            self.btn_mark_pass.setEnabled(False)
            self.btn_mark_fail.setEnabled(False)

            # Disable selection interactivity
            self.image_label.set_selection_enabled(False)

            # Clear inspection status in live view
            self.lbl_inspection_result.setText("Result: N/A")
            self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: black;")
            self.lbl_inspection_score.setText("Score: -")

            self.last_inspection_context = None

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
        capture_path = "/tmp/capture.jpg"
        pixmap = QPixmap(capture_path)
        if not pixmap.isNull():
            self.image_label.set_frame(pixmap)

            # Make sure cache is fresh (e.g. if modified elsewhere, though unlikely)
            self.refresh_roi_cache()

            # Apply stored ROI overlay if exists (scaled to this capture)
            self.apply_stored_roi_overlay(pixmap.width(), pixmap.height())

            # 5. Update State
            self.set_state(AppState.CAPTURED)

            # 6. Perform Inspection
            self.run_inspection(capture_path)
        else:
            self.on_capture_error("Failed to load captured image.")

    def run_inspection(self, capture_path: str):
        """
        Runs the deterministic inspection if valid ROI and Reference exist.
        """
        roi_data = self.cached_roi_data
        ref_paths = self.dataset_manager.get_active_references()

        if not roi_data or not ref_paths:
            self.lbl_inspection_result.setText("Setup Required")
            self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: orange;")
            self.lbl_inspection_score.setText("Score: -")
            return

        # Load images
        try:
            # cv2.imread loads as BGR, which is what we want for inspection core
            captured_img = cv2.imread(capture_path)
            if captured_img is None:
                print(f"Error loading captured image: {capture_path}")
                return

            # Load all reference images
            reference_images = {}
            for rp in ref_paths:
                ref_img = cv2.imread(rp)
                if ref_img is not None:
                    reference_images[os.path.basename(rp)] = ref_img
                else:
                    print(f"Warning: Failed to load reference {rp}")

            if not reference_images:
                raise ValueError("No valid reference images loaded.")

            dataset_version = self.dataset_manager.active_version

            # Feature-based threshold (0.25) as per Step 7 requirements
            # Step 8: Default threshold for embedding is 0.35 distance
            # If we are using ORB (explicitly or fallback), we might want 0.25 score.
            # But the single threshold config variable is tricky.
            # Let's set default threshold to 0.35 for embedding.
            # Ideally this should be in config.
            threshold = 0.35

            # Method is now "embedding" by default for Step 8
            # STEP 9 Update: Use default "hybrid" (unset) or respect env var.
            # We omit the method argument so it defaults to hybrid or env logic.
            result = perform_inspection(
                captured_img,
                reference_images,
                roi_data,
                threshold,
                dataset_version
            )

            # Generate ID and Record
            insp_id = str(uuid.uuid4())

            # Log data
            log_data = {
                "passed": result.passed,
                "score": result.best_score,
                "best_reference": result.best_reference_id,
                "all_scores": result.all_scores,
                "roi": roi_data,
                "threshold": threshold,
                "dataset_version": dataset_version,
                "orb_passed": result.orb_passed,
                "embedding_passed": result.embedding_passed,
                "decision_path": result.decision_path
            }

            # Save to dataset (persisted inspection)
            # We assume capture_path is the source
            final_path = self.dataset_manager.record_inspection(
                insp_id,
                capture_path,
                log_data
            )

            # Store context for overrides
            self.last_inspection_context = {
                "id": insp_id,
                "passed": result.passed,
                "score": result.best_score,
                "image_path": final_path,
                "roi": [roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height']]
            }

            # Update UI
            if result.passed:
                self.lbl_inspection_result.setText(f"PASS")
                self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
                self.btn_mark_fail.setEnabled(True) # Can override to FAIL
                self.btn_mark_pass.setEnabled(False)
            else:
                self.lbl_inspection_result.setText(f"FAIL")
                self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
                self.btn_mark_pass.setEnabled(True) # Can override to PASS
                self.btn_mark_fail.setEnabled(False)

            self.lbl_inspection_score.setText(f"Score: {result.best_score:.2f}\nRef: {result.best_reference_id}\nVer: {dataset_version}")

        except Exception as e:
            print(f"Inspection error: {e}")
            import traceback
            traceback.print_exc()
            self.lbl_inspection_result.setText("Error")
            self.lbl_inspection_score.setText(f"Err: {str(e)}")

    def handle_override(self, new_result: bool):
        if not self.last_inspection_context:
            return

        ctx = self.last_inspection_context

        # Prevent double override? No, user might change mind.
        # But we create a new record each time?
        # Requirement: "Append-only". Yes.

        record = OverrideRecord(
            inspection_id=ctx["id"],
            original_result=ctx["passed"],
            overridden_result=new_result,
            score=ctx["score"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            image_path=ctx["image_path"],
            roi=ctx["roi"]
        )

        self.dataset_manager.save_override(record)
        self.update_pending_count()

        # Visual feedback
        if new_result:
             self.lbl_inspection_result.setText("PASS (Overridden)")
             self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
             self.btn_mark_pass.setEnabled(False)
             self.btn_mark_fail.setEnabled(True)
        else:
             self.lbl_inspection_result.setText("FAIL (Overridden)")
             self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
             self.btn_mark_fail.setEnabled(False)
             self.btn_mark_pass.setEnabled(True)

        QMessageBox.information(self, "Override Saved", f"Result marked as {'PASS' if new_result else 'FAIL'}.")

    def handle_commit_learning(self):
        count = self.dataset_manager.get_pending_count()
        if count == 0:
            QMessageBox.information(self, "Learning", "No pending items to commit.")
            return

        success, msg = self.dataset_manager.commit_learning()
        if success:
            QMessageBox.information(self, "Learning Committed", msg)
            self.update_pending_count()
            # Reload ROI manager since active version changed?
            # ROIManager delegates to DatasetManager which tracks active version.
            # But we might want to refresh cache.
            self.refresh_roi_cache()
        else:
            QMessageBox.warning(self, "Learning Error", msg)

    def update_pending_count(self):
        count = self.dataset_manager.get_pending_count()
        self.lbl_pending_count.setText(f"Pending Overrides: {count}")

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

        # Save current capture as reference
        # We are in CAPTURED state, so the image is at /tmp/capture.jpg
        if self.roi_manager.save_reference("/tmp/capture.jpg"):
            msg = "Region of Interest and Reference Image saved successfully."
        else:
            msg = "Region of Interest saved, but failed to save Reference Image."

        # Clear temp
        self.temp_roi_rect = None

        QMessageBox.information(self, "ROI Saved", msg)

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
        # Stop worker threads
        if self.worker:
            self.image_label.setText("Stopping Preview...")
            self.worker.stop()
        if self.capture_worker and self.capture_worker.isRunning():
            self.capture_worker.quit()
            self.capture_worker.wait()

        # Explicitly stop the camera hardware as per lifecycle requirements
        # (Though main.py also has a stop, doing it here ensures it stops when window closes)
        if self.camera_impl and self.camera_impl.is_running():
            self.camera_impl.stop()

        event.accept()
