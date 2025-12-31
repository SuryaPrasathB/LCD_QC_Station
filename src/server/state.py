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
from roi import ROIManager
from core.inspection import perform_inspection, InspectionResult

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

        # 1. Initialize Components
        self.dataset_manager = DatasetManager(".")
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

            for r in rois:
                rid = r['id']
                x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])

                # Bounds check
                if x<0 or y<0 or x+w > img.shape[1] or y+h > img.shape[0]:
                    continue

                crop = img[y:y+h, x:x+w]
                tmp_crop = f"/tmp/ref_{rid}.png"
                cv2.imwrite(tmp_crop, crop)
                self.dataset_manager.save_roi_reference(rid, tmp_crop)

            return {
                "status": "committed",
                "dataset_version": self.dataset_manager.active_version,
                "roi_count": len(rois)
            }
        finally:
            self.inspection_lock.release()

    def override_inspection(self, inspection_id: str, action: str):
        """
        Records an override for a specific inspection.
        Action: "pass" or "fail"
        """
        print(f"[Server] Override requested for {inspection_id} -> {action}")
        with self.lock:
            last_res = self.last_inspection_result
            if not last_res or last_res.get("inspection_id") != inspection_id:
                print(f"[Server] Override Failed: Inspection ID mismatch or missing. Last: {last_res.get('inspection_id') if last_res else 'None'}")
                raise Exception("Inspection not found or expired")

            frame_path = self.last_inspection_frame_path
            if not frame_path or not os.path.exists(frame_path):
                print(f"[Server] Override Failed: Frame path missing {frame_path}")
                raise Exception("Inspection frame missing")

            current_rois = self.roi_data

            # Original decision
            original_passed = last_res.get("passed", False)

            # New decision
            new_passed = (action.lower() == "pass")

            print(f"[Server] Original: {original_passed}, New: {new_passed}")

            # Determine score (just use original global score or 0.0)
            # We might not have global score easily available in last_inspection_result dict
            # We can aggregate from roi_results
            score = 0.0 # Placeholder
            roi_results = last_res.get("roi_results", {})
            if roi_results:
                scores = [r["score"] for r in roi_results.values() if "score" in r]
                if scores:
                    score = min(scores)

            record = OverrideRecord(
                inspection_id=inspection_id,
                original_result=original_passed,
                overridden_result=new_passed,
                score=score,
                timestamp=datetime.utcnow().isoformat() + "Z",
                image_path=frame_path,
                roi=current_rois # Save the ROI definition used
            )

            self.dataset_manager.save_override(record)
            print(f"[Server] Override saved. Pending count: {self.dataset_manager.get_pending_count()}")

            # Update local result to reflect override (optional, but good for UI if we polled again)
            self.last_inspection_result["overridden"] = True
            self.last_inspection_result["override_status"] = "PASS" if new_passed else "FAIL"

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
        # Check if already running?
        if self.inspection_lock.locked():
             raise Exception("Inspection already in progress")

        insp_id = f"insp_{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}"

        # We must offload the blocking part to a thread
        loop = asyncio.get_running_loop()
        # Note: We acquire lock inside the thread or check before?
        # If we check before (here), there's a tiny race condition if multiple requests come in exactly same time.
        # But single threaded asyncio, `await` yields.
        # Safer to acquire lock here? But lock is thread-blocking.
        # We can't acquire threading.Lock non-blockingly in asyncio easily without blocking loop if we wait.
        # But we want to FAIL if busy.
        # `inspection_lock.locked()` is not thread-safe in python? It is.
        # So we check.

        loop.run_in_executor(None, self._run_inspection_sync, insp_id)

        return insp_id

    def _run_inspection_sync(self, insp_id: str):
        # Acquire lock to ensure exclusive access to hardware
        if not self.inspection_lock.acquire(blocking=False):
             print(f"[Server] Skipped inspection {insp_id}: Busy")
             return

        try:
            print(f"[Server] Running inspection {insp_id}")
            capture_path = f"/tmp/insp_{insp_id}.jpg"
            self.camera.capture_still(capture_path)

            img = cv2.imread(capture_path)
            if img is None:
                raise Exception("Capture failed")

            # Load refs
            refs_map = self.dataset_manager.get_active_references()

            # Nested refs
            refs_nested = {}
            for rid, rdict in refs_map.items():
                refs_nested[rid] = {}
                for k, v in rdict.items():
                     r_img = cv2.imread(v)
                     if r_img is not None:
                         refs_nested[rid][k] = r_img

            roi_data = self.roi_data

            version = self.dataset_manager.active_version

            result = perform_inspection(
                img,
                refs_nested,
                roi_data if roi_data else {"rois": []},
                0.35,
                version
            )

            # Save Record
            roi_results_dict = {}
            for rid, res in result.roi_results.items():
                roi_results_dict[rid] = {
                    "passed": res.passed,
                    "score": res.best_score
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

            print(f"[Server] Inspection {insp_id} complete. Passed: {result.passed}")

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
