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

        # Cache for ROI data to avoid reading disk in Live View loop
        # Format: {'rois': [{'id':..., 'x':..., 'y':..., 'w':..., 'h':...}], ...}
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
        # Note: We rely on `rois_changed` if we want to track changes, but `handle_set_roi_click` reads state directly usually.
        # But VideoLabel keeps state now.
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

        self.btn_set_roi = QPushButton("Save ROIs")
        self.btn_set_roi.clicked.connect(self.handle_set_roi_click)
        roi_layout.addWidget(self.btn_set_roi)

        self.btn_clear_roi = QPushButton("Clear ROIs")
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
        # Reset visual status of ROIs to 'none' (blue)
        self.reset_roi_visuals()

    def handle_error(self, error_msg):
        """
        Handles errors reported by the CameraWorker.
        """
        self.image_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Camera Error", f"An error occurred:\n{error_msg}")
        # Stop worker if running
        if self.worker:
            self.worker.stop()
            self.worker = None

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

            # Make sure cache is fresh
            self.refresh_roi_cache()

            # Apply stored ROI overlay (scaled)
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

        # roi_data structure: {'rois': [..], 'image_width': .., ...}

        if not roi_data or not roi_data.get("rois"):
            self.lbl_inspection_result.setText("Setup Required")
            self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: orange;")
            self.lbl_inspection_score.setText("Score: -")
            return

        # Fetch references from DatasetManager
        # Now returns nested dict: {roi_id: {ref_id: path}}
        ref_paths_map = self.dataset_manager.get_active_references()
        if not ref_paths_map:
             print("DEBUG: No references found in dataset manager.")
             self.lbl_inspection_result.setText("No References")
             return

        # Load images
        try:
            captured_img = cv2.imread(capture_path)
            if captured_img is None:
                print(f"Error loading captured image: {capture_path}")
                return

            # Load all reference images per ROI
            reference_images_nested = {}
            for roi_id, ref_dict in ref_paths_map.items():
                print(f"DEBUG: Loading refs for ROI {roi_id}: {list(ref_dict.keys())}")
                roi_refs = {}
                for r_id, r_path in ref_dict.items():
                    r_img = cv2.imread(r_path)
                    if r_img is not None:
                         roi_refs[r_id] = r_img
                if roi_refs:
                    reference_images_nested[roi_id] = roi_refs

            if not reference_images_nested:
                print("No valid reference images loaded.")
                self.lbl_inspection_result.setText("Ref Error")
                return

            dataset_version = self.dataset_manager.active_version
            threshold = 0.35 # Default for embedding

            result = perform_inspection(
                captured_img,
                reference_images_nested,
                roi_data,
                threshold,
                dataset_version
            )

            # Generate ID and Record
            insp_id = str(uuid.uuid4())

            # Log data
            # Convert InspectionResult to dict
            roi_results_dict = {}
            for rid, res in result.roi_results.items():
                roi_results_dict[rid] = {
                    "passed": res.passed,
                    "score": res.best_score,
                    "ref": res.best_reference_id,
                    "orb": res.orb_passed,
                    "emb": res.embedding_passed
                }

            log_data = {
                "passed": result.passed,
                "score": result.best_score,
                "roi_results": roi_results_dict,
                "dataset_version": dataset_version,
                "decision_path": result.decision_path
            }

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
                "roi": roi_data # Store full ROI structure for multi-roi learning
            }

            # Update UI Status
            if result.passed:
                self.lbl_inspection_result.setText(f"PASS")
                self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
                self.btn_mark_fail.setEnabled(True)
                self.btn_mark_pass.setEnabled(False)
            else:
                self.lbl_inspection_result.setText(f"FAIL")
                self.lbl_inspection_result.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
                self.btn_mark_pass.setEnabled(True)
                self.btn_mark_fail.setEnabled(False)

            self.lbl_inspection_score.setText(f"Score: {result.best_score:.2f}\nVer: {dataset_version}")

            # Update VideoLabel Overlays (Pass/Fail Colors)
            # We need to map result.roi_results back to video label ROIs
            current_rois_ui = []

            # We need to scale rects again because VideoLabel expects IMAGE coords (which we have in roi_data)
            # and scales them itself. But we need to update the `status` field.

            # The `roi_data` loaded from disk matches `captured_img` resolution IF configured correctly.
            # But the UI might have transient scaling if the cached ROI was from different resolution?
            # Assumption: ROI resolution matches capture resolution.

            for r_def in roi_data.get("rois", []):
                 rid = r_def["id"]
                 # Find result
                 res = result.roi_results.get(rid)
                 status = 'none'
                 if res:
                     status = 'pass' if res.passed else 'fail'

                 rect = QRect(r_def["x"], r_def["y"], r_def["w"], r_def["h"])
                 current_rois_ui.append({
                     'rect': rect,
                     'id': rid,
                     'status': status
                 })

            self.image_label.set_rois(current_rois_ui)

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

        record = OverrideRecord(
            inspection_id=ctx["id"],
            original_result=ctx["passed"],
            overridden_result=new_result,
            score=ctx["score"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            image_path=ctx["image_path"],
            roi=ctx["roi"] # Full ROI dict
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
            self.refresh_roi_cache()
            self.reset_roi_visuals()
        else:
            QMessageBox.warning(self, "Learning Error", msg)

    def update_pending_count(self):
        count = self.dataset_manager.get_pending_count()
        self.lbl_pending_count.setText(f"Pending Overrides: {count}")

    def on_capture_error(self, error_msg):
        QMessageBox.critical(self, "Capture Error", f"Capture failed:\n{error_msg}")
        self.set_state(AppState.LIVE_VIEW)

    def handle_resume_click(self):
        if self.current_state != AppState.CAPTURED:
            return

        self.image_label.setText("Resuming...")
        self.set_state(AppState.LIVE_VIEW)

    @pyqtSlot(QImage)
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.set_frame(pixmap)

        # Apply stored ROI overlay
        self.apply_stored_roi_overlay(pixmap.width(), pixmap.height())

    def refresh_roi_cache(self):
        """Load ROI from disk into memory."""
        self.cached_roi_data = self.roi_manager.load_roi()

    def reset_roi_visuals(self):
        """Reset all ROI overlays to neutral color (e.g. on live view resume)."""
        if self.cached_roi_data:
             rois = []
             for r in self.cached_roi_data.get("rois", []):
                 rect = QRect(r["x"], r["y"], r["w"], r["h"])
                 rois.append({'rect': rect, 'id': r["id"], 'status': 'none'})
             self.image_label.set_rois(rois)
        else:
             self.image_label.clear_rois()

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

            scale_x = current_w / orig_w
            scale_y = current_h / orig_h

            rois_ui = []
            for r in roi_data.get("rois", []):
                x = int(r["x"] * scale_x)
                y = int(r["y"] * scale_y)
                w = int(r["w"] * scale_x)
                h = int(r["h"] * scale_y)

                # Check if we already have a status for this ROI in the label?
                # Usually we overwrite with 'none' unless inspection just ran.
                # If we are in Live View, status is none.
                # If we are in Captured View, we might be applying overlay before inspection finishes?
                # Let's assume neutral unless set otherwise explicitly.
                # Actually, `VideoLabel` holds state. If we call `set_rois`, we replace it.
                # We should try to preserve status if IDs match?
                # For now, simplistic: if this function is called (e.g. frame update), we might reset status?
                # Wait, in Live View, frame updates continuously. We want to show the boxes.
                # Inspection sets the status ONCE.
                # If we call `set_rois` every frame, we might flicker or reset colors.
                # `update_image` calls this.
                # So we SHOULD NOT reset status if we are in CAPTURED mode and inspection ran.
                # But `update_image` is mostly for Live View. In Captured mode, `update_image` is not called by worker.
                # Captured mode sets pixmap once.
                # So this is safe for Live View (status='none').

                rois_ui.append({
                    'rect': QRect(x, y, w, h),
                    'id': r["id"],
                    'status': 'none'
                })

            # Only update if we are in Live View to avoid overwriting inspection results in Captured view
            # But wait, `update_image` is only connected to `worker.frame_ready` which runs in Live View.
            if self.current_state == AppState.LIVE_VIEW:
                self.image_label.set_rois(rois_ui)
        else:
            if self.current_state == AppState.LIVE_VIEW:
                self.image_label.clear_rois()

    def handle_set_roi_click(self):
        """Commit the current defined ROIs to disk."""
        # Get ROIs from VideoLabel
        ui_rois = self.image_label.rois
        if not ui_rois:
            QMessageBox.information(self, "ROI Info", "Please draw at least one region.")
            return

        # Current image dimensions
        if not self.image_label.pixmap_frame:
            return

        current_w = self.image_label.pixmap_frame.width()
        current_h = self.image_label.pixmap_frame.height()

        # Convert UI ROIs (which are in IMAGE coords of the current frame) to Data format
        rois_to_save = []
        for ur in ui_rois:
            r = ur['rect']
            rois_to_save.append({
                "id": ur['id'],
                "x": r.x(),
                "y": r.y(),
                "w": r.width(),
                "h": r.height()
            })

        # Save ROI JSON
        self.roi_manager.save_rois(rois_to_save, current_w, current_h)

        # Save Reference Crops
        # We need the actual image data.
        # We are in CAPTURED state, so the image is at /tmp/capture.jpg
        try:
            full_img = cv2.imread("/tmp/capture.jpg")
            if full_img is None:
                raise Exception("Could not read capture file")

            # Clear existing references for the active version to avoid pollution/accumulation
            self.dataset_manager.clear_active_references()

            # For each ROI, crop and save
            for r_def in rois_to_save:
                rid = r_def['id']
                rx, ry, rw, rh = r_def['x'], r_def['y'], r_def['w'], r_def['h']

                # Check bounds
                if rx < 0 or ry < 0 or rx+rw > full_img.shape[1] or ry+rh > full_img.shape[0]:
                    print(f"ROI {rid} out of bounds, skipping reference save.")
                    continue

                crop = full_img[ry:ry+rh, rx:rx+rw]

                # We need a temporary path for the crop to pass to dataset_manager.save_roi_reference
                # Or we can update save_roi_reference to accept image data?
                # DatasetManager.save_roi_reference takes `source_path`.
                # So we write to temp.
                tmp_crop_path = f"/tmp/ref_{rid}.png"
                cv2.imwrite(tmp_crop_path, crop)

                self.dataset_manager.save_roi_reference(rid, tmp_crop_path)

            msg = "Regions and References saved successfully."
        except Exception as e:
            msg = f"Regions saved, but error saving references: {e}"

        QMessageBox.information(self, "ROI Saved", msg)

        # Refresh
        self.refresh_roi_cache()
        # In Captured state, we want to update the visual to show the ID and 'none' status (until re-inspected?)
        # Actually, if we just saved new ROIs, we probably want to run inspection immediately?
        # Or just show them.
        self.reset_roi_visuals()

    def handle_clear_roi_click(self):
        self.roi_manager.clear_roi()
        self.image_label.clear_rois()
        self.cached_roi_data = None
