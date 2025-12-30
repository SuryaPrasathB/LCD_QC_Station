from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict
import cv2
import numpy as np
import io

from .state import ServerState

app = FastAPI(title="Inspection Server")

# Models
class NormalizedBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

class ROIDefinition(BaseModel):
    id: str
    normalized_bbox: NormalizedBBox

class ROIList(BaseModel):
    rois: List[ROIDefinition]

class ROIStatus(BaseModel):
    status: str

# Helper to encode image
def encode_image(img: np.ndarray) -> bytes:
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return encoded_img.tobytes()

def draw_rois_on_frame(img: np.ndarray, rois: List[Dict], results: Dict = None, stored_w: int = 1, stored_h: int = 1) -> np.ndarray:
    """
    Draws ROIs on the image.
    Colors:
    - Yellow: Setup / No result
    - Green: PASS
    - Red: FAIL
    """
    out = img.copy()
    h, w = out.shape[:2]

    # Calculate scale factor
    scale_x = w / stored_w if stored_w > 0 else 1
    scale_y = h / stored_h if stored_h > 0 else 1

    for r in rois:
        rid = r['id']
        rx = int(r['x'] * scale_x)
        ry = int(r['y'] * scale_y)
        rw = int(r['w'] * scale_x)
        rh = int(r['h'] * scale_y)

        # Determine color
        color = (0, 255, 255) # Yellow (BGR)
        thickness = 2

        if results:
            res = results.get(rid)
            if res:
                if res.get("passed", False):
                    color = (0, 255, 0) # Green
                else:
                    color = (0, 0, 255) # Red

        cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), color, thickness)
        cv2.putText(out, rid, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return out

@app.get("/health")
def health_check():
    state = ServerState.get_instance()
    return {
        "status": "ok",
        "device": "raspberry_pi",
        "camera": "ready" if state.camera.is_running() else "stopped",
        "inspection_method": "hybrid",
        "active_dataset_version": state.dataset_manager.active_version,
        "model_version": "embedding_v2"
    }

@app.get("/live/frame")
def get_live_frame():
    state = ServerState.get_instance()
    frame = state.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    return Response(content=encode_image(frame), media_type="image/jpeg")

@app.get("/live/frame_with_rois")
def get_live_frame_rois():
    state = ServerState.get_instance()
    frame = state.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")

    # Get ROIs
    rois = state.get_roi_list()
    # Get stored resolution to handle scaling
    roi_data = state.roi_data or {}
    stored_w = roi_data.get("image_width", 1)
    stored_h = roi_data.get("image_height", 1)

    results = None
    if state.last_inspection_result:
        results = state.last_inspection_result.get("roi_results")

    annotated = draw_rois_on_frame(frame, rois, results, stored_w, stored_h)

    return Response(content=encode_image(annotated), media_type="image/jpeg")

@app.get("/roi/list", response_model=ROIList)
def list_rois():
    state = ServerState.get_instance()
    raw_rois = state.get_roi_list()

    roi_data = state.roi_data
    if not roi_data:
        return {"rois": []}

    w = roi_data.get("image_width", 1)
    h = roi_data.get("image_height", 1)

    out_rois = []
    for r in raw_rois:
        out_rois.append({
            "id": r["id"],
            "normalized_bbox": {
                "x": r["x"] / w,
                "y": r["y"] / h,
                "w": r["w"] / w,
                "h": r["h"] / h
            }
        })
    return {"rois": out_rois}

@app.post("/roi/set")
def set_roi(roi: ROIDefinition):
    state = ServerState.get_instance()

    # Use config from camera if available (for CAPTURE resolution)
    # If Mock or generic, try to get from current ROI data or default.
    w, h = 0, 0
    if hasattr(state.camera, "main_config"):
         # RealCamera has main_config
         cfg = state.camera.main_config
         if cfg and "size" in cfg:
             w, h = cfg["size"]

    # Fallback to defaults if not found (e.g. MockCamera might not expose it identically)
    if w == 0 or h == 0:
        # Check if Mock
        if state.camera.__class__.__name__ == "MockCamera":
             # Mock usually 1920x1080? Or whatever it mocks.
             # Let's check if we can get it from 'roi_data' if it exists.
             if state.roi_data and "image_width" in state.roi_data:
                  w = state.roi_data["image_width"]
                  h = state.roi_data["image_height"]
             else:
                  # Default High Res
                  w, h = 3280, 2464
        else:
             # Default High Res
             w, h = 3280, 2464

    # Spec: "Overwrites ROI if ID exists"
    current_rois = state.get_roi_list()
    new_rois = []
    updated = False

    abs_x = int(roi.normalized_bbox.x * w)
    abs_y = int(roi.normalized_bbox.y * h)
    abs_w = int(roi.normalized_bbox.w * w)
    abs_h = int(roi.normalized_bbox.h * h)

    new_entry = {
        "id": roi.id,
        "x": abs_x, "y": abs_y, "w": abs_w, "h": abs_h
    }

    for r in current_rois:
        if r["id"] == roi.id:
            new_rois.append(new_entry)
            updated = True
        else:
            new_rois.append(r)

    if not updated:
        new_rois.append(new_entry)

    state.update_rois(new_rois, w, h)
    return {"status": "updated", "count": len(new_rois)}

@app.post("/roi/clear")
def clear_rois():
    state = ServerState.get_instance()
    state.clear_rois()
    return {"status": "cleared"}

@app.post("/roi/commit")
def commit_rois():
    state = ServerState.get_instance()
    try:
        result = state.commit_rois()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inspection/start")
async def start_inspection():
    state = ServerState.get_instance()
    # Attempt to start inspection (non-blocking call, returns ID or raises if busy)
    try:
        insp_id = await state.run_inspection_async()
        return {"inspection_id": insp_id, "status": "started"}
    except Exception as e:
         # If busy
         raise HTTPException(status_code=409, detail=str(e))

@app.get("/inspection/result")
def get_inspection_result():
    state = ServerState.get_instance()
    res = state.last_inspection_result
    if not res:
        raise HTTPException(status_code=404, detail="No inspection found")
    return res

@app.get("/inspection/frame")
def get_inspection_frame():
    state = ServerState.get_instance()
    path = state.last_inspection_frame_path

    if not path or not state.last_inspection_result:
        raise HTTPException(status_code=404, detail="No inspection frame")

    # Read from disk
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=404, detail="Image file missing")

    # Draw ROIs with Result Colors
    rois = state.get_roi_list()

    roi_data = state.roi_data or {}
    stored_w = roi_data.get("image_width", 1)
    stored_h = roi_data.get("image_height", 1)

    results = state.last_inspection_result.get("roi_results")
    annotated = draw_rois_on_frame(img, rois, results, stored_w, stored_h)

    return Response(content=encode_image(annotated), media_type="image/jpeg")
