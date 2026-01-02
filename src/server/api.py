from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
import cv2
import numpy as np
import io
import os

from .state import ServerState
from src.core.roi_model import normalize_roi

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
    roi_id: Optional[str] = None # Optional for legacy compatibility, but preferred

class DatasetCreateRequest(BaseModel):
    name: str

class DatasetSelectRequest(BaseModel):
    name: str

class ROIOverrideConfig(BaseModel):
    force_pass: bool

# Helper to encode image
def encode_image(img: np.ndarray) -> bytes:
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return encoded_img.tobytes()

def draw_rois_on_frame(img: np.ndarray, rois: List[Dict], results: Dict = None, stored_w: int = 1, stored_h: int = 1) -> np.ndarray:
    """
    Draws ROIs on the image.
    Uses NormalizedROI for safe access.
    """
    out = img.copy()
    h_img, w_img = out.shape[:2]

    # Calculate scale factor
    scale_x = w_img / stored_w if stored_w > 0 else 1
    scale_y = h_img / stored_h if stored_h > 0 else 1

    for r_dict in rois:
        # NORMALIZE
        roi = normalize_roi(r_dict.get('id', 'unknown'), r_dict)

        rx = int(roi.x * scale_x)
        ry = int(roi.y * scale_y)
        rw = int(roi.w * scale_x)
        rh = int(roi.h * scale_y)

        # Determine color
        # Default based on Type
        if roi.type == "DIGIT":
            color = (255, 0, 0) # Blue (BGR)
        elif roi.type == "ICON":
            color = (0, 255, 0) # Green (BGR)
        elif roi.type == "TEXT":
            color = (0, 255, 255) # Yellow (BGR)
        else:
            color = (255, 0, 0) # Default Blue

        thickness = 2

        # Override with Result Color if available
        if results:
            res = results.get(roi.id)
            if res:
                if res.get("passed", False):
                    color = (0, 255, 0) # Green (PASS)
                else:
                    color = (0, 0, 255) # Red (FAIL)

        cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), color, thickness)

        # Draw Label: ID (Type)
        label = f"{roi.id} ({roi.type})"
        if roi.force_pass:
            label += " [FORCE]"

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
        "dataset_name": state.get_active_dataset_name(),
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

@app.get("/roi/list") # Removed response_model to handle new fields easily
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
        # NORMALIZE
        roi = normalize_roi(r.get("id"), r)

        out_rois.append({
            "id": roi.id,
            "type": roi.type,
            "force_pass": roi.force_pass,
            "normalized_bbox": {
                "x": roi.x / w,
                "y": roi.y / h,
                "w": roi.w / w,
                "h": roi.h / h
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

    # Check if existing to preserve flags like force_pass
    existing_force_pass = False
    for r in current_rois:
        if r["id"] == roi.id:
            existing_force_pass = r.get("force_pass", False)
            break

    new_entry = {
        "id": roi.id,
        "type": roi.type.value,
        "force_pass": existing_force_pass,
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

@app.post("/roi/{roi_id}/config")
def set_roi_config(roi_id: str, config: ROIOverrideConfig):
    state = ServerState.get_instance()
    success = state.set_roi_override(roi_id, config.force_pass)
    if not success:
         raise HTTPException(status_code=404, detail="ROI not found")
    return {"status": "updated", "roi_id": roi_id, "force_pass": config.force_pass}

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

# --- Reference Management API ---

@app.get("/roi/{roi_id}/references")
def list_roi_references(roi_id: str):
    """List all reference filenames for a specific ROI."""
    state = ServerState.get_instance()
    refs = state.dataset_manager.get_active_references()

    if roi_id not in refs:
        # If no refs, return empty list instead of 404 to allow for UI handling
        return {"roi_id": roi_id, "references": []}

    # refs[roi_id] is a dict {ref_id: path}
    # We return list of dicts with id and filename
    ref_list = []
    for ref_id, path in refs[roi_id].items():
        ref_list.append({
            "id": ref_id,
            "filename": os.path.basename(path)
        })

    return {"roi_id": roi_id, "references": ref_list}

@app.get("/roi/{roi_id}/references/{ref_id}")
def get_roi_reference_image(roi_id: str, ref_id: str):
    """Serve the actual image file for a reference."""
    state = ServerState.get_instance()
    refs = state.dataset_manager.get_active_references()

    if roi_id not in refs:
        raise HTTPException(status_code=404, detail="ROI not found")

    if ref_id not in refs[roi_id]:
        raise HTTPException(status_code=404, detail="Reference not found")

    path = refs[roi_id][ref_id]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path, media_type="image/png")

@app.delete("/roi/{roi_id}/references/{ref_id}")
def delete_roi_reference(roi_id: str, ref_id: str):
    """Delete a specific reference image."""
    state = ServerState.get_instance()

    # We need to expose a delete method in DatasetManager
    # For now, we'll implement a helper in State or access DatasetManager directly?
    # Better to keep logic in DatasetManager.

    # Let's add a quick helper here or assume we update DatasetManager in next step?
    # The plan said "Modify src/server/api.py and src/core/dataset.py".
    # I will update DatasetManager in the same step if possible or rely on simple os actions here (risky).
    # Let's do it properly via DatasetManager method.

    success = state.dataset_manager.delete_reference(roi_id, ref_id)
    if not success:
         raise HTTPException(status_code=404, detail="Reference not found or delete failed")

    # Reload cache!
    state.dataset_manager.load_all_references()

    return {"status": "deleted", "roi_id": roi_id, "ref_id": ref_id}

@app.delete("/roi/{roi_id}/references")
def delete_all_roi_references(roi_id: str):
    """Delete ALL references for an ROI."""
    state = ServerState.get_instance()

    success = state.dataset_manager.delete_all_references_for_roi(roi_id)
    if not success:
         raise HTTPException(status_code=404, detail="ROI not found or delete failed")

    # Reload cache!
    state.dataset_manager.load_all_references()

    return {"status": "cleared_all", "roi_id": roi_id}

# -------------------------------

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
        print(f"[API] Override Request: {req.inspection_id}, {req.action}, ROI: {req.roi_id}")
        state.override_inspection(req.inspection_id, req.action, req.roi_id)
        return {"status": "ok", "action": req.action, "roi_id": req.roi_id}
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

# --- Datasets API ---

@app.get("/datasets")
def list_datasets():
    state = ServerState.get_instance()
    return {"datasets": state.list_datasets(), "active": state.get_active_dataset_name()}

@app.post("/datasets")
def create_dataset(req: DatasetCreateRequest):
    state = ServerState.get_instance()
    success = state.create_dataset(req.name)
    if not success:
        raise HTTPException(status_code=400, detail="Dataset already exists or invalid name")
    return {"status": "created", "name": req.name}

@app.post("/datasets/select")
def select_dataset(req: DatasetSelectRequest):
    state = ServerState.get_instance()
    success = state.set_active_dataset(req.name)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "selected", "name": req.name}
