from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
import uuid
from src.core.config import settings

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)

class InspectionDB(Base):
    __tablename__ = "inspections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=utcnow)
    image_path = Column(String, nullable=False)
    result = Column(String, nullable=False) # PASS/FAIL
    score = Column(Float, nullable=False)
    dataset_version = Column(String, nullable=False)
    meta_json = Column(JSON, nullable=True) # processing time, detailed metrics

class OverrideDB(Base):
    __tablename__ = "overrides"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    inspection_id = Column(String, nullable=False)
    original_result = Column(String, nullable=False)
    new_result = Column(String, nullable=False)
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=utcnow)

class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    is_active = Column(Boolean, default=False)
    config_json = Column(JSON, nullable=False) # Thresholds, ROI, etc.

# Database Engine
engine = create_engine(settings.DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
