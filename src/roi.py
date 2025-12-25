import json
import os
from typing import Optional, Dict

class ROIManager:
    """
    Manages the persistence of Region of Interest (ROI) data.
    Stores data in 'roi.json' in the project root.
    """
    FILENAME = "roi.json"

    def __init__(self, root_dir: str = "."):
        self.filepath = os.path.join(root_dir, self.FILENAME)

    def load_roi(self) -> Optional[Dict[str, int]]:
        """
        Loads the ROI from disk if it exists.
        Returns:
            Dict containing x, y, width, height, image_width, image_height
            or None if file doesn't exist or is invalid.
        """
        if not os.path.exists(self.filepath):
            return None

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                # Basic validation of keys
                required_keys = {"x", "y", "width", "height", "image_width", "image_height"}
                if not all(k in data for k in required_keys):
                    print(f"ROI file {self.filepath} missing required keys. Ignoring.")
                    return None
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading ROI file: {e}")
            return None

    def save_roi(self, x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> None:
        """
        Saves the ROI to disk.
        """
        data = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "image_width": int(image_width),
            "image_height": int(image_height)
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving ROI file: {e}")

    def clear_roi(self) -> None:
        """
        Deletes the ROI file from disk if it exists.
        """
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError as e:
                print(f"Error removing ROI file: {e}")
