import json
import os
import shutil
from typing import Optional, Dict, List, Any

class ROIManager:
    """
    Manages the persistence of Region of Interest (ROI) data and Reference Images.
    Delegates path resolution to the DatasetManager.
    """
    ROI_FILENAME = "roi.json"

    def __init__(self, dataset_manager):
        self.dm = dataset_manager

    def _get_roi_path(self):
        return os.path.join(self.dm.get_active_version_path(), self.ROI_FILENAME)

    def load_roi(self) -> Optional[Dict[str, Any]]:
        """
        Loads the ROI data from disk.
        Returns a dictionary with a "rois" list.
        Detects legacy flat format and converts it to a single-ROI list.
        Detects legacy Multi-ROI format (missing type/bbox) and upgrades it.

        Returns:
            {
                "rois": [
                    {
                        "id": "...",
                        "type": "DIGIT|ICON|TEXT",
                        "bbox": {"x": int, "y": int, "w": int, "h": int}
                    },
                    ...
                ],
                "image_width": int,
                "image_height": int
            }
            or None if not found/invalid.
        """
        filepath = self._get_roi_path()
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check for Newest Format (Multi-ROI with Semantics)
            if "rois" in data and isinstance(data["rois"], list):
                # Check first ROI for structure
                if data["rois"] and "type" in data["rois"][0] and "bbox" in data["rois"][0]:
                    return data

                # Check for Previous Multi-ROI Format (Step 10)
                # Structure: {"id": "...", "x": ..., "y": ..., "w": ..., "h": ...}
                print(f"Detecting legacy Multi-ROI format in {filepath}. Upgrading...")
                upgraded_rois = []
                for r in data["rois"]:
                    # Default to DIGIT if missing
                    # Move x,y,w,h into bbox
                    upgraded_rois.append({
                        "id": r["id"],
                        "type": r.get("type", "DIGIT"),
                        "bbox": {
                            "x": r["x"],
                            "y": r["y"],
                            "w": r["w"],
                            "h": r["h"]
                        }
                    })
                return {
                    "rois": upgraded_rois,
                    "image_width": data.get("image_width", 0),
                    "image_height": data.get("image_height", 0)
                }

            # Check for Legacy Flat Format (Single ROI)
            required_keys = {"x", "y", "width", "height", "image_width", "image_height"}
            if all(k in data for k in required_keys):
                print(f"Detecting legacy flat format in {filepath}. Upgrading...")
                # Convert to new format in-memory
                return {
                    "rois": [{
                        "id": "digits_main", # Default ID for legacy
                        "type": "DIGIT",
                        "bbox": {
                            "x": data["x"],
                            "y": data["y"],
                            "w": data["width"],
                            "h": data["height"]
                        }
                    }],
                    "image_width": data["image_width"],
                    "image_height": data["image_height"]
                }

            print(f"ROI file {filepath} has unrecognized format.")
            return None

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading ROI file: {e}")
            return None

    def save_rois(self, rois: List[Dict[str, Any]], image_width: int, image_height: int) -> None:
        """
        Saves the list of ROIs to disk.
        rois: List of dicts with keys 'id', 'type', 'bbox' (containing x,y,w,h).
        """
        self.dm.ensure_active_version_writable()
        filepath = self._get_roi_path()

        data = {
            "rois": rois,
            "image_width": int(image_width),
            "image_height": int(image_height)
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving ROI file: {e}")

    def clear_roi(self) -> None:
        """
        Deletes the ROI file and cleans up reference images from the active version.
        """
        # Delete ROI JSON
        roi_path = self._get_roi_path()
        if os.path.exists(roi_path):
            try:
                os.remove(roi_path)
            except OSError as e:
                print(f"Error removing ROI file: {e}")

        # We also need to clear references.
        self.dm.clear_active_references()
