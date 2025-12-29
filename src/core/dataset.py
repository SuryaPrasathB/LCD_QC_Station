import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class OverrideRecord:
    inspection_id: str
    original_result: bool
    overridden_result: bool
    score: float
    timestamp: str
    image_path: str
    roi: List[int]

class DatasetManager:
    """
    Manages the dataset versioning, inspections, and overrides.
    """
    DATA_DIR = "data"
    REF_DIR = "reference"
    INSPECTIONS_DIR = "inspections"
    OVERRIDES_DIR = "overrides"
    VERSION_FILE = "dataset_version.json"
    PENDING_FILE = "pending_learning.json"

    ROOT_FILES = ["roi.json", "reference.jpg", "reference.png"]

    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, self.DATA_DIR)
        self.ref_base_path = os.path.join(self.data_path, self.REF_DIR)
        self.inspections_path = os.path.join(self.data_path, self.INSPECTIONS_DIR)
        self.overrides_path = os.path.join(self.data_path, self.OVERRIDES_DIR)
        self.version_file_path = os.path.join(self.data_path, self.VERSION_FILE)
        self.pending_file_path = os.path.join(self.data_path, self.PENDING_FILE)

        self.active_version = "v1"

    def initialize(self):
        """
        Sets up the directory structure and migrates existing files if needed.
        """
        # Create directories
        os.makedirs(self.ref_base_path, exist_ok=True)
        os.makedirs(self.inspections_path, exist_ok=True)
        os.makedirs(self.overrides_path, exist_ok=True)

        # Check version file
        if os.path.exists(self.version_file_path):
            with open(self.version_file_path, 'r') as f:
                data = json.load(f)
                self.active_version = data.get("active_version", "v1")
        else:
            # Initialize v1
            self.active_version = "v1"
            self._write_version_file(self.active_version, "Initial version", None)

            # Migration Logic
            v1_path = os.path.join(self.ref_base_path, "v1")
            os.makedirs(v1_path, exist_ok=True)

            migrated = False
            for filename in self.ROOT_FILES:
                src = os.path.join(self.root_dir, filename)
                if os.path.exists(src):
                    dst = os.path.join(v1_path, filename)
                    try:
                        shutil.move(src, dst)
                        print(f"Migrated {src} to {dst}")
                        migrated = True
                    except Exception as e:
                        print(f"Error migrating {src}: {e}")

            if not migrated and not os.listdir(v1_path):
                print("No existing reference files found to migrate.")

    def _write_version_file(self, version: str, description: str, based_on: Optional[str]):
        data = {
            "active_version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "based_on": based_on,
            "description": description
        }
        with open(self.version_file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_active_version_path(self) -> str:
        return os.path.join(self.ref_base_path, self.active_version)

    def get_active_paths(self) -> Tuple[str, str]:
        """
        Returns (roi_path, ref_path).
        Tries reference.png first, then reference.jpg.
        """
        v_path = self.get_active_version_path()
        roi_path = os.path.join(v_path, "roi.json")

        # Check for png or jpg
        ref_path = os.path.join(v_path, "reference.png")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(v_path, "reference.jpg")

        return roi_path, ref_path

    def save_inspection(self, inspection_id: str, image: bytes, result_data: Dict) -> str:
        """
        Saves inspection image and result JSON.
        Returns the image path.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        filename_base = f"{timestamp}_{inspection_id}"

        # Save Image
        image_filename = f"{filename_base}.png"
        image_path = os.path.join(self.inspections_path, image_filename)

        # Write bytes or cv2 image?
        # The caller usually has a cv2 image or bytes.
        # Assuming caller handles image writing?
        # No, "Saves inspection image".
        # But passing bytes is generic.
        # Let's assume the caller saves the image to a temp path and we copy/move it?
        # Or better, caller passes the path where they saved it temporarily?
        # Requirement: "inspections/2025-01-01...png"
        pass
        # Actually, let's let the caller save the image to the final path returned by this function?
        # No, that breaks encapsulation of paths.
        # Let's assume the caller passes the image data (numpy array) but I don't want to depend on cv2 here if possible?
        # src/core/inspection.py uses cv2. It's fine.
        # But to keep this file clean, let's accept a `source_image_path` and we copy it.

        return image_path

    def record_inspection(self, inspection_id: str, source_image_path: str, result_data: Dict) -> str:
        """
        Moves/Copies the source image to the inspection folder and saves the result JSON.
        result_data should include 'passed', 'score', etc.
        Returns the final image path.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        # Final image path
        ext = os.path.splitext(source_image_path)[1]
        if not ext: ext = ".png"

        image_filename = f"{timestamp}_{inspection_id}{ext}"
        final_image_path = os.path.join(self.inspections_path, image_filename)

        try:
            shutil.copy(source_image_path, final_image_path)
        except Exception as e:
            print(f"Error saving inspection image: {e}")
            return ""

        # Save Result JSON
        result_filename = f"result_{timestamp}_{inspection_id}.json"
        result_path = os.path.join(self.inspections_path, result_filename)

        data = {
            "id": inspection_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset_version": self.active_version,
            "image_path": image_filename, # Relative to inspections folder
            **result_data
        }

        with open(result_path, 'w') as f:
            json.dump(data, f, indent=2)

        return final_image_path

    def save_override(self, record: OverrideRecord):
        """
        Saves override record and adds to pending pool if applicable.
        """
        filename = f"override_{record.timestamp.replace(':', '-')}_{record.inspection_id}.json"
        filepath = os.path.join(self.overrides_path, filename)

        data = {
            "inspection_id": record.inspection_id,
            "original_result": record.original_result,
            "overridden_result": record.overridden_result,
            "score": record.score,
            "timestamp": record.timestamp,
            "image_path": record.image_path,
            "roi": record.roi
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        # Logic: Only False Negatives (System: FAIL(False), Operator: PASS(True)) are candidates for learning.
        if not record.original_result and record.overridden_result:
            self._add_to_pending(filepath)

    def _add_to_pending(self, override_filepath: str):
        pending = []
        if os.path.exists(self.pending_file_path):
            try:
                with open(self.pending_file_path, 'r') as f:
                    pending = json.load(f)
            except:
                pending = []

        if override_filepath not in pending:
            pending.append(override_filepath)

        with open(self.pending_file_path, 'w') as f:
            json.dump(pending, f, indent=2)

    def get_pending_count(self) -> int:
        if os.path.exists(self.pending_file_path):
            try:
                with open(self.pending_file_path, 'r') as f:
                    pending = json.load(f)
                return len(pending)
            except:
                return 0
        return 0

    def commit_learning(self) -> Tuple[bool, str]:
        """
        Commits pending overrides to a new dataset version.
        Returns (success, message).
        """
        count = self.get_pending_count()
        if count == 0:
            return False, "No pending overrides to commit."

        # 1. Determine new version
        try:
            current_v_num = int(self.active_version.replace("v", ""))
            new_v_num = current_v_num + 1
        except ValueError:
            new_v_num = 2

        new_version = f"v{new_v_num}"
        old_version_path = self.get_active_version_path()
        new_version_path = os.path.join(self.ref_base_path, new_version)

        os.makedirs(new_version_path, exist_ok=True)

        # 2. Copy Base Reference Files (roi.json, reference.png)
        # We copy from the *previous* version to ensure continuity
        roi_src, ref_src = self.get_active_paths()

        if os.path.exists(roi_src):
            shutil.copy(roi_src, os.path.join(new_version_path, "roi.json"))
        if os.path.exists(ref_src):
            ref_name = os.path.basename(ref_src)
            shutil.copy(ref_src, os.path.join(new_version_path, ref_name))

        # 3. Process Pending Overrides
        try:
            with open(self.pending_file_path, 'r') as f:
                pending_files = json.load(f)
        except:
            return False, "Failed to read pending list."

        learned_count = 0
        for p_file in pending_files:
            if not os.path.exists(p_file):
                continue

            try:
                with open(p_file, 'r') as pf:
                    record = json.load(pf)

                # Copy image to new version folder
                # Image path in record is absolute or relative to inspections?
                # It is stored as full path or relative?
                # In record_inspection, we returned full path (or relative to cwd).
                # Let's assume absolute or valid relative path.
                src_img = record['image_path']
                if os.path.exists(src_img):
                    dst_img_name = f"learned_{uuid.uuid4().hex[:8]}.png"
                    dst_img = os.path.join(new_version_path, dst_img_name)
                    shutil.copy(src_img, dst_img)
                    learned_count += 1
            except Exception as e:
                print(f"Error processing pending file {p_file}: {e}")

        # 4. Clear Pending
        if os.path.exists(self.pending_file_path):
            os.remove(self.pending_file_path)

        # 5. Update Version File
        self.active_version = new_version
        self._write_version_file(new_version, f"Learned from {learned_count} overrides", f"v{current_v_num}")

        return True, f"Committed {learned_count} images to {new_version}."

    def ensure_active_version_writable(self):
        """
        Ensures the active version folder exists.
        """
        path = self.get_active_version_path()
        os.makedirs(path, exist_ok=True)
