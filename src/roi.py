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

        Returns:
            {
                "rois": [
                    {"id": "...", "x": int, "y": int, "w": int, "h": int},
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

            # Check for New Format (Multi-ROI)
            if "rois" in data and isinstance(data["rois"], list):
                return data

            # Check for Legacy Format (Single ROI)
            required_keys = {"x", "y", "width", "height", "image_width", "image_height"}
            if all(k in data for k in required_keys):
                # Convert to new format in-memory
                return {
                    "rois": [{
                        "id": "digits_main", # Default ID for legacy
                        "x": data["x"],
                        "y": data["y"],
                        "w": data["width"],
                        "h": data["height"]
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
        rois: List of dicts with keys 'id', 'x', 'y', 'w', 'h'.
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
        # DatasetManager should probably handle this deeper cleanup or we ask it to.
        # For now, let's rely on DM or remove known paths.
        self.dm.clear_active_references()
