from typing import List, Optional
from datetime import datetime
from uuid import UUID
import json
from sqlalchemy.orm import Session
from src.core.types import OverrideStatus, InspectionResultType
from src.infra.database import OverrideDB, DatasetDB, SessionLocal
from src.infra.logging import get_logger

logger = get_logger(__name__)

class DatasetManager:
    """
    Manages dataset versions and learning updates (overrides).
    """
    def __init__(self):
        self.active_version_id = None
        self._load_active_version()

    def _load_active_version(self):
        with SessionLocal() as db:
            active = db.query(DatasetDB).filter(DatasetDB.is_active == True).first()
            if active:
                self.active_version_id = active.id
                logger.info(f"Loaded active dataset version: {active.version_tag}")
            else:
                logger.info("No active dataset found. Using default/empty state.")

    def add_override(self, inspection_id: UUID, original_result: str, new_result: str, reason: str = None):
        """
        Registers a user override request (Pending State).
        """
        with SessionLocal() as db:
            override = OverrideDB(
                inspection_id=str(inspection_id),
                original_result=original_result,
                new_result=new_result,
                status=OverrideStatus.PENDING.value
            )
            db.add(override)
            db.commit()
            logger.info(f"Override pending for inspection {inspection_id}: {original_result} -> {new_result}")

    def commit_version(self, new_version_tag: str):
        """
        Processes all PENDING overrides and creates a new Dataset Version.
        This includes 'Learning' logic.
        """
        with SessionLocal() as db:
            # 1. Fetch pending overrides
            pending = db.query(OverrideDB).filter(OverrideDB.status == OverrideStatus.PENDING.value).all()

            if not pending:
                logger.info("No pending overrides to commit.")
                return

            logger.info(f"Committing {len(pending)} overrides to new version {new_version_tag}...")

            # 2. Analyze Overrides (Learning Logic placeholder)
            false_fails = [o for o in pending if o.original_result == "FAIL" and o.new_result == "PASS"]
            false_passes = [o for o in pending if o.original_result == "PASS" and o.new_result == "FAIL"]

            # logic to update "parameters_json" based on these lists would go here.
            # E.g.
            # for ff in false_fails:
            #    Add image to "Allowed References" list
            # for fp in false_passes:
            #    Tighten threshold?

            # For this architectural phase, we simply mark them as committed.
            for o in pending:
                o.status = OverrideStatus.COMMITTED.value

            # 3. Create New Dataset Record
            # In a real implementation, we would clone the old config and apply changes.
            new_dataset = DatasetDB(
                version_tag=new_version_tag,
                is_active=True,
                config_json={"base_threshold": 0.8} # Placeholder
            )

            # Deactivate old
            db.query(DatasetDB).update({DatasetDB.is_active: False})

            db.add(new_dataset)
            db.commit()

            self.active_version_id = new_dataset.id
            logger.info(f"Version {new_version_tag} committed and active.")

    def get_overrides(self, status: Optional[str] = None):
        with SessionLocal() as db:
            query = db.query(OverrideDB)
            if status:
                query = query.filter(OverrideDB.status == status)
            return query.all()
