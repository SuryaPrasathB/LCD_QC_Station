import threading
import time
import os
import cv2
import numpy as np
import uuid
from typing import Optional, Dict, List, Any
from datetime import datetime
import asyncio

from camera.real_camera import RealCamera
from camera.mock_camera import MockCamera
from core.dataset import DatasetManager, OverrideRecord
from core.migration import migrate_legacy_structure
from roi import ROIManager
from core.inspection import perform_inspection, InspectionResult
from core.roi_model import normalize_roi

class ServerState:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ServerState()
        return cls._instance

    def __init__(self):
        if ServerState._instance is not None:
            raise Exception("This class is a singleton!")

        self.lock = threading.Lock()
        self.inspection_lock = threading.Lock()

        # 0. Legacy Migration (Once on startup)
        try:
             migrate_legacy_structure(".")
        except Exception as e:
             print(f"[Server] Migration warning: {e}")

        # 1. Initialize Components
        self.datasets_root = os.path.join(".", DatasetManager.DATA_ROOT)
        self.active_dataset_name = "default"

        # Check if default exists, if not, maybe find first available?
        # For now, default is fine as migration ensures it exists if legacy did.

        self.dataset_manager = DatasetManager(".", self.active_dataset_name)
        self.dataset_manager.initialize()
        self.roi_manager = ROIManager(self.dataset_manager)

        # Detect Camera
        use_mock = os.environ.get("USE_MOCK_CAMERA", "0") == "1"
        if use_mock:
            print("[Server] Starting in MOCK mode")
            self.camera = MockCamera()
        elif RealCamera.check_camera_availability():
            print("[Server] Real Camera Detected")
            self.camera = RealCamera()
        else:
            print("[Server] No camera found, falling back to MOCK")
            self.camera = MockCamera()

        # State Variables
        self.latest_frame: Optional[np.ndarray] = None
        self.roi_data: Optional[Dict[str, Any]] = self.roi_manager.load_roi() # Load initially

        self.last_inspection_result: Optional[Dict[str, Any]] = None
        self.last_inspection_frame_path: Optional[str] = None

        self.running = False
        self.worker_thread = None

    def start(self):
        """Starts the camera and the frame grabber thread."""
        with self.lock:
            if self.running:
                return

            print("[Server] Starting Camera...")
            self.camera.start()
            self.running = True

            self.worker_thread = threading.Thread(target=self._frame_loop, daemon=True)
            self.worker_thread.start()

    def stop(self):
        """Stops the camera and thread."""
        print("[Server] Stopping...")
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

        self.camera.stop()

    def _frame_loop(self):
        """Continuously fetches frames for live view."""
        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[Server] Frame loop error: {e}")
                time.sleep(1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    # --- Dataset Management ---

    def list_datasets(self) -> List[str]:
        if not os.path.exists(self.datasets_root):
            return []

        datasets = []
        for name in os.listdir(self.datasets_root):
             if os.path.isdir(os.path.join(self.datasets_root, name)):
                 datasets.append(name)
        return sorted(datasets)

    def create_dataset(self, name: str) -> bool:
        if not name or ".." in name or "/" in name:
            return False

        target_path = os.path.join(self.datasets_root, name)
        if os.path.exists(target_path):
            return False # Already exists

        os.makedirs(target_path, exist_ok=True)

        # Initialize it properly
        dm = DatasetManager(".", name)
        dm.initialize()
        return True

    def set_active_dataset(self, name: str) -> bool:
        target_path = os.path.join(self.datasets_root, name)
        if not os.path.exists(target_path):
             return False

        with self.lock:
             self.active_dataset_name = name
             self.dataset_manager = DatasetManager(".", name)
             # Don't re-initialize fully (don't overwrite version file if exists), but ensure paths
             self.dataset_manager.initialize()

             # Re-bind ROI Manager
             self.roi_manager = ROIManager(self.dataset_manager)

             # Reload ROIs for new dataset
             self.roi_data = self.roi_manager.load_roi()

             # Reload References Cache
             self.dataset_manager.load_all_references()

             print(f"[Server] Switched to dataset: {name}")
             return True

    def get_active_dataset_name(self) -> str:
        return self.active_dataset_name

    # --------------------------

    def set_roi_override(self, roi_id: str, force_pass: bool):
        """
        Updates the override status for a specific ROI.
        """
        with self.lock:
            rois = self.get_roi_list()
            updated = False
            for r in rois:
                if r["id"] == roi_id:
                    r["force_pass"] = force_pass
                    updated = True
                    break

            if updated:
                # Save changes
                img_w = self.roi_data.get("image_width", 0)
                img_h = self.roi_data.get("image_height", 0)
                self.update_rois(rois, img_w, img_h)
                print(f"[Server] ROI {roi_id} force_pass set to {force_pass}")
                return True
            return False

    def get_roi_list(self):
        # Refresh from disk/memory if needed, or return cached
        if not self.roi_data:
            self.roi_data = self.roi_manager.load_roi()

        if not self.roi_data:
            return []

        return self.roi_data.get("rois", [])

    def update_rois(self, rois: List[Dict[str, Any]], img_w: int, img_h: int):
        with self.lock:
            self.roi_manager.save_rois(rois, img_w, img_h)
            self.roi_data = self.roi_manager.load_roi() # Reload

    def clear_rois(self):
        with self.lock:
            self.roi_manager.clear_roi()
            self.roi_data = None

    def commit_rois(self) -> Dict[str, Any]:
        """
        Captures current frame, crops references, saves, updates version.
        """
        # Lock inspection logic to prevent races during commit
        if not self.inspection_lock.acquire(blocking=False):
            raise Exception("System busy with inspection")

        try:
            capture_path = f"/tmp/setup_capture_{uuid.uuid4().hex}.jpg"

            # Capture synchronously (blocking but okay for setup)
            self.camera.capture_still(capture_path)

            img = cv2.imread(capture_path)
            if img is None:
                raise Exception("Failed to capture image for commit")

            rois = self.get_roi_list()
            if not rois:
                raise Exception("No ROIs defined")

            # Clear old refs
            self.dataset_manager.clear_active_references()

            for r_dict in rois:
                # NORMALIZE
                roi = normalize_roi(r_dict.get('id', 'unknown'), r_dict)

                x, y, w, h = roi.x, roi.y, roi.w, roi.h

                # Bounds check
                if x<0 or y<0 or x+w > img.shape[1] or y+h > img.shape[0]:
                    continue

                crop = img[y:y+h, x:x+w]
                tmp_crop = f"/tmp/ref_{roi.id}.png"
                cv2.imwrite(tmp_crop, crop)
                self.dataset_manager.save_roi_reference(roi.id, tmp_crop)

            return {
                "status": "committed",
                "dataset_version": self.dataset_manager.active_version,
                "roi_count": len(rois)
            }
        finally:
            self.inspection_lock.release()

    def override_inspection(self, inspection_id: str, action: str, roi_id: Optional[str] = None):
        """
        Records an override for a specific inspection.
        Action: "pass" or "fail"
        roi_id: Optional, if None applies to global (legacy), otherwise specific ROI.
        """
        print(f"[Server] Override requested for {inspection_id} -> {action} (ROI: {roi_id})")
        with self.lock:
            last_res = self.last_inspection_result
            if not last_res or last_res.get("inspection_id") != inspection_id:
                print(f"[Server] Override Failed: ID mismatch or missing.")
                raise Exception("Inspection not found or expired")

            frame_path = self.last_inspection_frame_path
            if not frame_path or not os.path.exists(frame_path):
                print(f"[Server] Override Failed: Frame path missing.")
                raise Exception("Inspection frame missing")

            current_rois = self.roi_data

            # Original Global Decision
            original_passed = last_res.get("passed", False)

            # Determine New Status
            new_passed = (action.lower() == "pass")

            # --- Per-ROI Logic ---
            roi_results = last_res.get("roi_results", {})

            # If roi_id is provided, we only override that ROI's status
            # And then re-evaluate Global Status
            if roi_id:
                 if roi_id not in roi_results:
                     raise Exception(f"ROI {roi_id} not found in inspection")

                 # Update the specific ROI result in memory
                 roi_results[roi_id]["passed"] = new_passed
                 roi_results[roi_id]["overridden"] = True
                 roi_results[roi_id]["override_status"] = "PASS" if new_passed else "FAIL"

                 # Re-calculate Global Passed
                 # Global is PASS only if ALL ROIs are PASS
                 all_pass = all(r.get("passed", False) for r in roi_results.values())

                 # Update Global Result
                 self.last_inspection_result["passed"] = all_pass

                 # For the Record, we store the *specific* ROI data if possible,
                 # or we store the whole set but maybe we should rely on the ROI-specific learning commit logic.
                 # The 'OverrideRecord' stores `roi` which is the definition.
                 # It doesn't store 'which roi was overridden'.
                 # We should probably filter the definition to ONLY the relevant ROI for the record?
                 # So learning only learns that one.

                 # Create a partial ROI definition for the record
                 target_roi_def = next((r for r in current_rois.get("rois", []) if r["id"] == roi_id), None)

                 record_roi_data = {"rois": [target_roi_def]} if target_roi_def else current_rois

            else:
                 # Legacy Global Override
                 self.last_inspection_result["passed"] = new_passed
                 self.last_inspection_result["overridden"] = True
                 self.last_inspection_result["override_status"] = "PASS" if new_passed else "FAIL"
                 record_roi_data = current_rois

            # Score for record (min score of all or specific?)
            # If specific, use that ROI's score
            if roi_id and roi_id in roi_results:
                score = roi_results[roi_id].get("score", 0.0)
            else:
                scores = [r["score"] for r in roi_results.values() if "score" in r]
                score = min(scores) if scores else 0.0

            record = OverrideRecord(
                inspection_id=inspection_id,
                original_result=original_passed, # This is technically global original...
                overridden_result=new_passed,
                score=score,
                timestamp=datetime.utcnow().isoformat() + "Z",
                image_path=frame_path,
                roi=record_roi_data # Save specific ROI if applicable
            )

            self.dataset_manager.save_override(record)
            print(f"[Server] Override saved. Pending count: {self.dataset_manager.get_pending_count()}")

    def get_pending_learning_count(self) -> int:
        return self.dataset_manager.get_pending_count()

    def commit_learning(self) -> Dict[str, Any]:
        """
        Commits pending overrides to a new dataset version.
        """
        success, msg = self.dataset_manager.commit_learning()
        if not success:
            raise Exception(msg)
        return {"status": "committed", "message": msg, "new_version": self.dataset_manager.active_version}

    async def run_inspection_async(self) -> str:
        """
        Async wrapper to run inspection in background.
        Returns inspection_id immediately.
        """
        if self.inspection_lock.locked():
             raise Exception("Inspection already in progress")

        insp_id = f"insp_{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}"

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._run_inspection_sync, insp_id)

        return insp_id

    def _run_inspection_sync(self, insp_id: str):
        # Acquire lock to ensure exclusive access to hardware
        if not self.inspection_lock.acquire(blocking=False):
             print(f"[Server] Skipped inspection {insp_id}: Busy")
             return

        t_start = time.time()
        try:
            print(f"[Server] Running inspection {insp_id}")

            t0 = time.time()
            capture_path = f"/tmp/insp_{insp_id}.jpg"
            self.camera.capture_still(capture_path)
            t1 = time.time()
            print(f"[Timer] Capture: {t1 - t0:.3f}s")

            img = cv2.imread(capture_path)
            if img is None:
                raise Exception("Capture failed")

            # Use Cached References
            t2 = time.time()
            refs_nested = self.dataset_manager.get_cached_references()
            t3 = time.time()
            if t3 - t2 > 0.01:
                print(f"[Timer] Reference Fetch (WARNING): {t3 - t2:.3f}s")

            roi_data = self.roi_data
            version = self.dataset_manager.active_version

            t4 = time.time()
            result = perform_inspection(
                img,
                refs_nested,
                roi_data if roi_data else {"rois": []},
                0.35,
                version
            )
            t5 = time.time()
            print(f"[Timer] Inspection Logic: {t5 - t4:.3f}s")

            # Save Record
            roi_results_dict = {}
            # Need to get ROI types to include in logs
            rois_list = roi_data.get("rois", []) if roi_data else []

            # Helper to map types safely
            roi_type_map = {}
            for r_dict in rois_list:
                # NORMALIZE just to be sure we get the type
                roi = normalize_roi(r_dict.get('id', 'unknown'), r_dict)
                roi_type_map[roi.id] = roi.type

            for rid, res in result.roi_results.items():
                roi_results_dict[rid] = {
                    "type": roi_type_map.get(rid, "DIGIT"), # Add Type to log
                    "passed": res.passed,
                    "score": res.best_score,
                    "failure_reason": res.failure_reason,
                    "failure_detail": res.failure_detail
                }

            log_data = {
                "passed": result.passed,
                "dataset_version": version,
                "model_version": "embedding_v2",
                "roi_results": roi_results_dict
            }

            final_path = self.dataset_manager.record_inspection(insp_id, capture_path, log_data)

            with self.lock:
                self.last_inspection_result = {
                    "inspection_id": insp_id,
                    **log_data
                }
                self.last_inspection_frame_path = final_path

            t_end = time.time()
            print(f"[Server] Inspection {insp_id} complete. Passed: {result.passed}. Total: {t_end - t_start:.3f}s")

        except Exception as e:
            print(f"[Server] Inspection failed: {e}")
            import traceback
            traceback.print_exc()
            with self.lock:
                self.last_inspection_result = {
                    "inspection_id": insp_id,
                    "passed": False,
                    "error": str(e)
                }
        finally:
            self.inspection_lock.release()
