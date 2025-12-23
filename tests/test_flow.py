import cv2
import numpy as np
import pytest
from pathlib import Path
from uuid import uuid4

from src.core.engine import InspectionEngine
from src.core.dataset import DatasetManager
from src.core.types import RawImage, InspectionResultType

# Fixture to create dummy images
@pytest.fixture
def dummy_data(tmp_path):
    # 1. Reference Image (White circle on black)
    ref = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(ref, (50, 50), 30, (255, 255, 255), -1)
    ref_path = tmp_path / "ref.jpg"
    cv2.imwrite(str(ref_path), ref)

    # 2. Good Test Image (Same, slight shift)
    good = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(good, (52, 52), 30, (255, 255, 255), -1)

    # 3. Bad Test Image (Missing circle / broken)
    bad = np.zeros((100, 100, 3), dtype=np.uint8) # Empty

    return str(ref_path), good, bad

def test_inspection_flow(dummy_data):
    ref_path, good_img, bad_img = dummy_data

    # Init Engine
    engine = InspectionEngine()
    engine.load_reference(ref_path)

    # Test PASS
    raw_good = RawImage(data=good_img)
    result_good = engine.process(raw_good)
    print(f"\nGood Image Score: {result_good.confidence_score}")
    assert result_good.result == InspectionResultType.PASS

    # Test FAIL
    raw_bad = RawImage(data=bad_img)
    result_bad = engine.process(raw_bad)
    print(f"Bad Image Score: {result_bad.confidence_score}")
    assert result_bad.result == InspectionResultType.FAIL

import shutil
import os
from sqlalchemy import create_engine
from src.infra.database import Base, SessionLocal, engine as prod_engine
from src.core.config import settings

# Override DB for testing
# We create a new engine pointing to test DB
test_engine = create_engine(settings.TEST_DB_PATH, connect_args={"check_same_thread": False})

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create test DB tables
    Base.metadata.create_all(bind=test_engine)
    yield
    # Cleanup (optional, keep for debug)
    # Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("data/db/test.sqlite"):
        os.remove("data/db/test.sqlite")

@pytest.fixture(autouse=True)
def clean_db_session(monkeypatch):
    """
    Ensure we use the test engine/session for all DB calls in tests.
    Also clean data between tests.
    """
    # Clean tables
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    # Mock SessionLocal in src.infra.database to use test engine
    # And mock the global 'engine' if used directly

    # We need to monkeypatch SessionLocal where it is USED, or the factory itself
    # Since SessionLocal is a class (sessionmaker), we can replace it
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr("src.infra.database.SessionLocal", TestSession)
    monkeypatch.setattr("src.infra.database.engine", test_engine)
    monkeypatch.setattr("src.core.dataset.SessionLocal", TestSession)
    monkeypatch.setattr("src.api.app.SessionLocal", TestSession)

    yield

def test_dataset_override():
    manager = DatasetManager()

    # Simulate an override
    insp_id = uuid4()
    manager.add_override(insp_id, "FAIL", "PASS", "False Alarm")

    # Check it's pending
    overrides = manager.get_overrides(status="PENDING")
    assert len(overrides) >= 1
    assert str(overrides[-1].inspection_id) == str(insp_id)

    # Commit
    # Ensure unique version tag for test to avoid UniqueConstraint error if DB persists
    tag = f"v1.0-test-{uuid4()}"
    manager.commit_version(tag)

    # Check it's committed
    overrides = manager.get_overrides(status="COMMITTED")
    assert str(overrides[-1].inspection_id) == str(insp_id)

if __name__ == "__main__":
    # Manual run for quick check
    # Need tmp_path simulation if running main
    pass
