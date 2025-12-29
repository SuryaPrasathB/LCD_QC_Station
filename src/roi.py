import json
import os
import shutil
from typing import Optional, Dict

# We import only for type hinting if needed, but to avoid circular imports
# (if DatasetManager imported ROIManager), we just assume the interface.
# However, this file is low level.

class ROIManager:
    """
    Manages the persistence of Region of Interest (ROI) data and Reference Image.
    Delegates path resolution to the DatasetManager.
    """
    ROI_FILENAME = "roi.json"
    REF_FILENAME_JPG = "reference.jpg"
    REF_FILENAME_PNG = "reference.png"

    def __init__(self, dataset_manager):
        self.dm = dataset_manager

    def _get_roi_path(self):
        return os.path.join(self.dm.get_active_version_path(), self.ROI_FILENAME)

    def _get_ref_path(self):
        # We need to find which one exists, or default to one for saving.
        # For loading, we ask the DM or check both.
        # DM has `get_active_paths` which resolves this.
        _, ref_path = self.dm.get_active_paths()
        return ref_path

    def load_roi(self) -> Optional[Dict[str, int]]:
        """
        Loads the ROI from disk if it exists.
        Returns:
            Dict containing x, y, width, height, image_width, image_height
            or None if file doesn't exist or is invalid.
        """
        filepath = self._get_roi_path()
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Basic validation of keys
                required_keys = {"x", "y", "width", "height", "image_width", "image_height"}
                if not all(k in data for k in required_keys):
                    print(f"ROI file {filepath} missing required keys. Ignoring.")
                    return None
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading ROI file: {e}")
            return None

    def save_roi(self, x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> None:
        """
        Saves the ROI to disk in the active version folder.
        """
        self.dm.ensure_active_version_writable()
        filepath = self._get_roi_path()

        data = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "image_width": int(image_width),
            "image_height": int(image_height)
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving ROI file: {e}")

    def save_reference(self, source_path: str) -> bool:
        """
        Saves (copies) the reference image to the active version folder.
        Returns True if successful, False otherwise.
        """
        self.dm.ensure_active_version_writable()

        # Determine target extension based on source or default to png?
        # Let's keep source extension if jpg/png, else png.
        ext = os.path.splitext(source_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            ext = '.png'

        filename = self.REF_FILENAME_JPG if ext in ['.jpg', '.jpeg'] else self.REF_FILENAME_PNG
        target_path = os.path.join(self.dm.get_active_version_path(), filename)

        try:
            shutil.copy(source_path, target_path)
            return True
        except IOError as e:
            print(f"Error saving reference image: {e}")
            return False

    def get_reference_path(self) -> str:
        """Returns the full path to the reference image."""
        return self._get_ref_path()

    def has_reference(self) -> bool:
        """Checks if reference image exists."""
        return os.path.exists(self._get_ref_path())

    def clear_roi(self) -> None:
        """
        Deletes the ROI file and Reference image from the active version.
        """
        # Delete ROI JSON
        roi_path = self._get_roi_path()
        if os.path.exists(roi_path):
            try:
                os.remove(roi_path)
            except OSError as e:
                print(f"Error removing ROI file: {e}")

        # Delete Reference Image
        ref_path = self._get_ref_path()
        if os.path.exists(ref_path):
            try:
                os.remove(ref_path)
            except OSError as e:
                print(f"Error removing reference image: {e}")
