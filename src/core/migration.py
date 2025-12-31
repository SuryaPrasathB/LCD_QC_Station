import os
import shutil
import logging

logger = logging.getLogger(__name__)

def migrate_legacy_structure(root_dir: str):
    """
    Migrates legacy structure:
    data/reference/v1/... -> data/default/reference/v1/...
    data/inspections/... -> data/default/inspections/...
    data/overrides/... -> data/default/overrides/...
    data/*.json -> data/default/*.json
    """
    data_dir = os.path.join(root_dir, "data")
    default_dir = os.path.join(data_dir, "default")

    # Check if we need migration (i.e. if 'reference' exists in 'data' but 'default' does not)
    ref_path = os.path.join(data_dir, "reference")
    if os.path.exists(ref_path) and not os.path.exists(default_dir):
        logger.info("Migrating legacy dataset structure to 'default' dataset...")
        os.makedirs(default_dir, exist_ok=True)

        # Directories to move
        dirs_to_move = ["reference", "inspections", "overrides"]
        for d in dirs_to_move:
            src = os.path.join(data_dir, d)
            if os.path.exists(src):
                dst = os.path.join(default_dir, d)
                try:
                    shutil.move(src, dst)
                    logger.info(f"Moved {src} -> {dst}")
                except Exception as e:
                    logger.error(f"Failed to move {src}: {e}")

        # Files to move
        files_to_move = ["dataset_version.json", "pending_learning.json"]
        for f in files_to_move:
            src = os.path.join(data_dir, f)
            if os.path.exists(src):
                dst = os.path.join(default_dir, f)
                try:
                    shutil.move(src, dst)
                    logger.info(f"Moved {src} -> {dst}")
                except Exception as e:
                    logger.error(f"Failed to move {src}: {e}")

        logger.info("Migration complete.")
    else:
        # Create default if nothing exists
        if not os.path.exists(default_dir):
             os.makedirs(default_dir, exist_ok=True)
