from dataclasses import dataclass
from typing import Dict, Any, Union

@dataclass
class NormalizedROI:
    id: str
    type: str  # DIGIT | ICON | TEXT
    x: int
    y: int
    w: int
    h: int

def normalize_roi(roi_id: str, roi_dict: Dict[str, Any]) -> NormalizedROI:
    """
    Converts a raw ROI dictionary (legacy flat or new nested bbox) into a NormalizedROI.
    Handles defaults and schema normalization.
    """
    # 1. Determine Type
    rtype = roi_dict.get("type", "DIGIT")

    # 2. Extract Coordinates (Handle Legacy vs New Schema)
    if "bbox" in roi_dict:
        # New Schema
        bbox = roi_dict["bbox"]
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
    else:
        # Legacy Flat Schema (Fallback)
        # Check for both "x" and "normalized_bbox" (if passed from API pydantic model as dict)
        # But usually this comes from roi.json or state which should be dicts.
        x = roi_dict.get("x", 0)
        y = roi_dict.get("y", 0)
        w = roi_dict.get("w", 0)
        h = roi_dict.get("h", 0)

        # Note: If it's a completely different format, we default to 0.
        # But we assume the system guarantees one of these.

    # 3. Create Object
    return NormalizedROI(
        id=roi_id,
        type=rtype,
        x=int(x),
        y=int(y),
        w=int(w),
        h=int(h)
    )
