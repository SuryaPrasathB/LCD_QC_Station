from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
import cv2
import numpy as np
import io

from .state import ServerState

app = FastAPI(title="Inspection Server")

# Models
class ROIType(str, Enum):
    DIGIT = "DIGIT"
    ICON = "ICON"
    TEXT = "TEXT"

class NormalizedBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

class ROIDefinition(BaseModel):
    id: str
    type: ROIType = ROIType.DIGIT
    normalized_bbox: NormalizedBBox

class ROIList(BaseModel):
    rois: List[ROIDefinition]

class ROIStatus(BaseModel):
    status: str

class OverrideRequest(BaseModel):
    inspection_id: str
    action: str # "pass" or "fail"

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
    - Yellow: Setup / No result (Legacy default)

    Semantic Colors (Outline):
    - DIGIT: Blue
    - ICON: Green (Wait, spec says Green for ICON, but Green is also PASS... Spec check)
      Spec: "DIGIT -> Blue outline, ICON -> Green outline, TEXT -> Yellow outline"
      Spec: "Green for PASS and Red for FAIL" (Visual feedback uses colored outlines)

    Conflict Resolution:
    - If Result exists: Use PASS/FAIL colors (Green/Red).
    - If Setup/No Result: Use Type colors.
    """
    out = img.copy()
    h_img, w_img = out.shape[:2]

    # Calculate scale factor
    scale_x = w_img / stored_w if stored_w > 0 else 1
    scale_y = h_img / stored_h if stored_h > 0 else 1

    for r in rois:
        rid = r['id']
        rtype = r.get('type', 'DIGIT')

        # BBox
        bbox = r.get('bbox', r) # Handle flat if somehow passed (though State should have upgraded)

        rx = int(bbox.get('x', 0) * scale_x)
        ry = int(bbox.get('y', 0) * scale_y)
        rw = int(bbox.get('w', 0) * scale_x)
        rh = int(bbox.get('h', 0) * scale_y)

        # Determine color
        # Default based on Type
        if rtype == "DIGIT":
            color = (255, 0, 0) # Blue (BGR)
        elif rtype == "ICON":
            color = (0, 255, 0) # Green (BGR)
        elif rtype == "TEXT":
            color = (0, 255, 255) # Yellow (BGR)
        else:
            color = (255, 0, 0) # Default Blue

        thickness = 2

        # Override with Result Color if available
        if results:
            res = results.get(rid)
            if res:
                if res.get("passed", False):
                    color = (0, 255, 0) # Green (PASS)
                else:
                    color = (0, 0, 255) # Red (FAIL)

        cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), color, thickness)

        # Draw Label: ID (Type)
        label = f"{rid} ({rtype})"
        cv2.putText(out, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
        # State.get_roi_list() returns dicts with 'bbox' and 'type' now (after State upgrade)
        # But wait, State returns what ROIManager loads. ROIManager upgrades to new format.
        bbox = r.get("bbox", {})

        out_rois.append({
            "id": r["id"],
            "type": r.get("type", "DIGIT"),
            "normalized_bbox": {
                "x": bbox.get("x", 0) / w,
                "y": bbox.get("y", 0) / h,
                "w": bbox.get("w", 0) / w,
                "h": bbox.get("h", 0) / h
            }
        })
    return {"rois": out_rois}

@app.post("/roi/set")
def set_roi(roi: ROIDefinition):
    state = ServerState.get_instance()

    # Use config from camera if available (for CAPTURE resolution)
    w, h = 0, 0
    if hasattr(state.camera, "main_config"):
         cfg = state.camera.main_config
         if cfg and "size" in cfg:
             w, h = cfg["size"]

    if w == 0 or h == 0:
        if state.camera.__class__.__name__ == "MockCamera":
             if state.roi_data and "image_width" in state.roi_data:
                  w = state.roi_data["image_width"]
                  h = state.roi_data["image_height"]
             else:
                  w, h = 3280, 2464
        else:
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
        "type": roi.type.value,
        "bbox": {
            "x": abs_x, "y": abs_y, "w": abs_w, "h": abs_h
        }
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
    try:
        insp_id = await state.run_inspection_async()
        return {"inspection_id": insp_id, "status": "started"}
    except Exception as e:
         raise HTTPException(status_code=409, detail=str(e))

@app.get("/inspection/result")
def get_inspection_result():
    state = ServerState.get_instance()
    res = state.last_inspection_result
    if not res:
        raise HTTPException(status_code=404, detail="No inspection found")
    return res

@app.post("/inspection/override")
def override_inspection(req: OverrideRequest):
    state = ServerState.get_instance()
    try:
        print(f"[API] Override Request: {req.inspection_id}, {req.action}")
        state.override_inspection(req.inspection_id, req.action)
        return {"status": "ok", "action": req.action}
    except Exception as e:
        print(f"[API] Override Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/learning/status")
def get_learning_status():
    state = ServerState.get_instance()
    return {"pending_count": state.get_pending_learning_count()}

@app.post("/learning/commit")
def commit_learning():
    state = ServerState.get_instance()
    try:
        return state.commit_learning()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
