from typing import Dict, List, Optional
import requests
import json

class InspectionClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.timeout = 5  # seconds

    def connect(self, ip: str, port: int) -> bool:
        """Sets the connection details and tests connectivity."""
        self.base_url = f"http://{ip}:{port}"
        print(f"[Client] Connecting to {self.base_url}...")
        try:
            status = self.check_health().get("status")
            print(f"[Client] Connection status: {status}")
            return status == "ok"
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[Client] Connection failed: {e}")
            return False

    def check_health(self) -> Dict:
        """GET /health"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Client] Health check error: {e}")
            raise

    def get_live_frame(self, with_rois: bool = False) -> bytes:
        """GET /live/frame or /live/frame_with_rois"""
        endpoint = "/live/frame_with_rois" if with_rois else "/live/frame"
        # print(f"[Client] Fetching live frame from {endpoint}") # Too noisy for loop
        resp = requests.get(f"{self.base_url}{endpoint}", timeout=2.0)
        resp.raise_for_status()
        return resp.content

    def get_roi_list(self) -> List[Dict]:
        """GET /roi/list"""
        print("[Client] Fetching ROI list...")
        resp = requests.get(f"{self.base_url}/roi/list", timeout=self.timeout)
        resp.raise_for_status()
        rois = resp.json().get("rois", [])
        print(f"[Client] Received {len(rois)} ROIs")
        return rois

    def set_roi(self, roi_id: str, x: float, y: float, w: float, h: float, roi_type: str = "DIGIT") -> Dict:
        """POST /roi/set"""
        print(f"[Client] Setting ROI {roi_id} ({roi_type}): {x},{y},{w},{h}")
        payload = {
            "id": roi_id,
            "type": roi_type,
            "normalized_bbox": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        }
        resp = requests.post(f"{self.base_url}/roi/set", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def set_roi_config(self, roi_id: str, force_pass: bool) -> Dict:
        """POST /roi/{roi_id}/config"""
        print(f"[Client] Setting ROI {roi_id} force_pass={force_pass}")
        payload = {"force_pass": force_pass}
        resp = requests.post(f"{self.base_url}/roi/{roi_id}/config", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def clear_rois(self) -> Dict:
        """POST /roi/clear"""
        print("[Client] Clearing ROIs...")
        resp = requests.post(f"{self.base_url}/roi/clear", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def commit_rois(self) -> Dict:
        """POST /roi/commit"""
        print("[Client] Committing ROIs...")
        resp = requests.post(f"{self.base_url}/roi/commit", timeout=self.timeout)
        resp.raise_for_status()
        print(f"[Client] Commit result: {resp.json()}")
        return resp.json()

    def start_inspection(self) -> str:
        """POST /inspection/start - Returns inspection ID"""
        print("[Client] Starting inspection...")
        resp = requests.post(f"{self.base_url}/inspection/start", timeout=self.timeout)
        resp.raise_for_status()
        insp_id = resp.json().get("inspection_id")
        print(f"[Client] Inspection started. ID: {insp_id}")
        return insp_id

    def get_inspection_result(self) -> Dict:
        """GET /inspection/result"""
        # print("[Client] Polling result...") # Noisy
        resp = requests.get(f"{self.base_url}/inspection/result", timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_inspection_frame(self) -> bytes:
        """GET /inspection/frame"""
        print("[Client] Fetching inspection frame...")
        resp = requests.get(f"{self.base_url}/inspection/frame", timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def override_inspection(self, inspection_id: str, action: str, roi_id: Optional[str] = None) -> Dict:
        """POST /inspection/override"""
        print(f"[Client] Overriding inspection {inspection_id} -> {action} (ROI: {roi_id})")
        payload = {
            "inspection_id": inspection_id,
            "action": action,
            "roi_id": roi_id
        }
        resp = requests.post(f"{self.base_url}/inspection/override", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        print(f"[Client] Override success: {resp.json()}")
        return resp.json()

    def get_learning_status(self) -> Dict:
        """GET /learning/status"""
        resp = requests.get(f"{self.base_url}/learning/status", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def commit_learning(self) -> Dict:
        """POST /learning/commit"""
        print("[Client] Committing learning data...")
        resp = requests.post(f"{self.base_url}/learning/commit", timeout=self.timeout)
        resp.raise_for_status()
        print(f"[Client] Learning commit success: {resp.json()}")
        return resp.json()

    def list_datasets(self) -> Dict:
        """GET /datasets"""
        resp = requests.get(f"{self.base_url}/datasets", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_dataset(self, name: str) -> Dict:
        """POST /datasets"""
        payload = {"name": name}
        resp = requests.post(f"{self.base_url}/datasets", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def select_dataset(self, name: str) -> Dict:
        """POST /datasets/select"""
        payload = {"name": name}
        resp = requests.post(f"{self.base_url}/datasets/select", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # --- Reference Management ---

    def list_roi_references(self, roi_id: str) -> Dict:
        """GET /roi/{roi_id}/references"""
        resp = requests.get(f"{self.base_url}/roi/{roi_id}/references", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_roi_reference_image(self, roi_id: str, ref_id: str) -> bytes:
        """GET /roi/{roi_id}/references/{ref_id}"""
        resp = requests.get(f"{self.base_url}/roi/{roi_id}/references/{ref_id}", timeout=5.0)
        if resp.status_code == 200:
            return resp.content
        return None

    def delete_roi_reference(self, roi_id: str, ref_id: str) -> Dict:
        """DELETE /roi/{roi_id}/references/{ref_id}"""
        resp = requests.delete(f"{self.base_url}/roi/{roi_id}/references/{ref_id}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def delete_all_roi_references(self, roi_id: str) -> Dict:
        """DELETE /roi/{roi_id}/references"""
        resp = requests.delete(f"{self.base_url}/roi/{roi_id}/references", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
