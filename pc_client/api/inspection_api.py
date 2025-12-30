from typing import Dict, List
import requests

class InspectionClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.timeout = 5  # seconds

    def connect(self, ip: str, port: int) -> bool:
        """Sets the connection details and tests connectivity."""
        self.base_url = f"http://{ip}:{port}"
        try:
            return self.check_health().get("status") == "ok"
        except Exception: # pylint: disable=broad-exception-caught
            return False

    def check_health(self) -> Dict:
        """GET /health"""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_live_frame(self, with_rois: bool = False) -> bytes:
        """GET /live/frame or /live/frame_with_rois"""
        endpoint = "/live/frame_with_rois" if with_rois else "/live/frame"
        resp = requests.get(f"{self.base_url}{endpoint}", timeout=2.0)
        resp.raise_for_status()
        return resp.content

    def get_roi_list(self) -> List[Dict]:
        """GET /roi/list"""
        resp = requests.get(f"{self.base_url}/roi/list", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("rois", [])

    def set_roi(self, roi_id: str, x: float, y: float, w: float, h: float) -> Dict:
        """POST /roi/set"""
        payload = {
            "id": roi_id,
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

    def clear_rois(self) -> Dict:
        """POST /roi/clear"""
        resp = requests.post(f"{self.base_url}/roi/clear", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def commit_rois(self) -> Dict:
        """POST /roi/commit"""
        resp = requests.post(f"{self.base_url}/roi/commit", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def start_inspection(self) -> str:
        """POST /inspection/start - Returns inspection ID"""
        resp = requests.post(f"{self.base_url}/inspection/start", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("inspection_id")

    def get_inspection_result(self) -> Dict:
        """GET /inspection/result"""
        resp = requests.get(f"{self.base_url}/inspection/result", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_inspection_frame(self) -> bytes:
        """GET /inspection/frame"""
        resp = requests.get(f"{self.base_url}/inspection/frame", timeout=self.timeout)
        resp.raise_for_status()
        return resp.content
