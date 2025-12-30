import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass

@dataclass
class OverrideRecord:
    inspection_id: str
    original_result: bool
    overridden_result: bool
    score: float
    timestamp: str
    image_path: str
    roi: Any # Can be List[int] (legacy) or Dict (multi-roi)

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

    # Files that might exist at root of a version (Legacy)
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

            # Migration Logic (Legacy -> v1 folder)
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
                # Only log if empty, no error
                pass

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

    def get_active_references(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a dictionary mapping ROI IDs to a dict of {ref_id: path}.
        Structure:
        {
           "roi_id_1": { "ref_001": "/path/to/ref_001.png", ... },
           "roi_id_2": ...
        }
        Handles Legacy format (single reference at root) by mapping it to "digits_main".
        """
        v_path = self.get_active_version_path()
        if not os.path.exists(v_path):
            return {}

        refs = {}

        # 1. Check for Subdirectories (Multi-ROI)
        # We iterate through subdirectories in v_path.
        has_subdirs = False
        for item in os.listdir(v_path):
            item_path = os.path.join(v_path, item)
            if os.path.isdir(item_path):
                # Assume this is an ROI folder
                roi_id = item
                roi_refs = {}
                for f in os.listdir(item_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ref_id = os.path.splitext(f)[0]
                        roi_refs[ref_id] = os.path.join(item_path, f)

                # Even if empty, we might want to know it exists?
                # But inspection needs images.
                if roi_refs:
                    refs[roi_id] = roi_refs
                    has_subdirs = True

        # 2. Check for Root References (Legacy)
        if not has_subdirs:
            legacy_refs = {}
            for f in os.listdir(v_path):
                if f.lower() in ["roi.json", "meta.json", "dataset_version.json"]:
                    continue
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                     ref_id = os.path.splitext(f)[0]
                     legacy_refs[ref_id] = os.path.join(v_path, f)

            if legacy_refs:
                # Default legacy ID matches what ROIManager uses
                refs["digits_main"] = legacy_refs

        return refs

    def save_roi_reference(self, roi_id: str, source_path: str) -> bool:
        """
        Saves a reference image for a specific ROI.
        Creates the ROI subdirectory if needed.
        """
        self.ensure_active_version_writable()
        v_path = self.get_active_version_path()
        roi_dir = os.path.join(v_path, roi_id)
        os.makedirs(roi_dir, exist_ok=True)

        existing = [f for f in os.listdir(roi_dir) if f.startswith("ref_")]
        count = len(existing) + 1
        filename = f"ref_{count:03d}.png"
        target_path = os.path.join(roi_dir, filename)

        try:
            shutil.copy(source_path, target_path)
            return True
        except IOError as e:
            print(f"Error saving reference for {roi_id}: {e}")
            return False

    def clear_active_references(self):
        """
        Clears all references in the active version (files and subdirs).
        """
        v_path = self.get_active_version_path()
        if not os.path.exists(v_path):
            return

        for item in os.listdir(v_path):
            item_path = os.path.join(v_path, item)
            # Don't delete dataset metadata if any
            if item == "roi.json":
                continue

            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except OSError as e:
                print(f"Error clearing path {item_path}: {e}")

    def save_inspection(self, inspection_id: str, image: bytes, result_data: Dict) -> str:
        pass

    def record_inspection(self, inspection_id: str, source_image_path: str, result_data: Dict) -> str:
        """
        Moves/Copies the source image to the inspection folder and saves the result JSON.
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
            "image_path": image_filename,
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

        # 2. Copy Base Reference Files & Structure
        # Copy roi.json
        roi_src = os.path.join(old_version_path, "roi.json")
        if os.path.exists(roi_src):
            shutil.copy(roi_src, os.path.join(new_version_path, "roi.json"))

        # Copy Reference Folders / Files
        for item in os.listdir(old_version_path):
            if item == "roi.json" or item == "dataset_version.json":
                continue

            src_path = os.path.join(old_version_path, item)
            dst_path = os.path.join(new_version_path, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            elif os.path.isfile(src_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(src_path, dst_path)

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

                # Resolve image path
                src_img_path = record.get('image_path')
                if not os.path.isabs(src_img_path):
                    # Try resolving relative to CWD
                    if not os.path.exists(src_img_path):
                        # Try inside inspections
                        candidate = os.path.join(self.inspections_path, os.path.basename(src_img_path))
                        if os.path.exists(candidate):
                             src_img_path = candidate

                if not os.path.exists(src_img_path):
                    print(f"Skipping override {p_file}: Image not found at {src_img_path}")
                    continue

                import cv2
                img = cv2.imread(src_img_path)
                if img is None:
                    print(f"Skipping override {p_file}: Failed to read image")
                    continue

                roi_data = record.get('roi')
                # Support new Multi-ROI structure
                if isinstance(roi_data, dict) and "rois" in roi_data:
                    for r in roi_data["rois"]:
                        rid = r["id"]
                        rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]

                        # Validate bounds
                        if rx < 0 or ry < 0 or rx+rw > img.shape[1] or ry+rh > img.shape[0]:
                            print(f"Skipping ROI {rid}: Bounds invalid {rx},{ry},{rw},{rh} for img {img.shape}")
                            continue

                        # Crop
                        crop = img[ry:ry+rh, rx:rx+rw]

                        target_dir = os.path.join(new_version_path, rid)
                        os.makedirs(target_dir, exist_ok=True)

                        fname = f"learned_{uuid.uuid4().hex[:8]}.png"
                        cv2.imwrite(os.path.join(target_dir, fname), crop)

                    # We count this record as learned
                    learned_count += 1

                # Legacy handling (single roi list [x,y,w,h])
                elif isinstance(roi_data, list) and len(roi_data) == 4:
                     # This is ambiguous in multi-roi context, but likely "digits_main" if migrating
                     pass

            except Exception as e:
                print(f"Error processing pending file {p_file}: {e}")
                import traceback
                traceback.print_exc()

        # 4. Clear Pending
        if os.path.exists(self.pending_file_path):
            os.remove(self.pending_file_path)

        # 5. Update Version File
        self.active_version = new_version
        self._write_version_file(new_version, f"Learned from {learned_count} overrides", f"v{current_v_num}")

        return True, f"Committed {learned_count} overrides to {new_version}."

    def ensure_active_version_writable(self):
        path = self.get_active_version_path()
        os.makedirs(path, exist_ok=True)
