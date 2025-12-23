from enum import Enum
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
import numpy as np

class RawImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    data: Any # Numpy array (marked as Any to avoid Pydantic validation issues with ndarray)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = {}

class InspectionResultType(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"

class OverrideStatus(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    DISCARDED = "DISCARDED"

class ImageMetadata(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_path: str
    width: int
    height: int

class InspectionResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    image_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: InspectionResultType
    confidence_score: float
    processing_time_ms: int
    dataset_version: str
    details: Dict[str, Any] = {} # e.g. {"diff_metric": 0.05, "alignment_status": "OK"}

class PendingOverride(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    inspection_id: UUID
    original_result: InspectionResultType
    new_result: InspectionResultType
    reason: Optional[str] = None
    status: OverrideStatus = OverrideStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DatasetVersion(BaseModel):
    id: int
    version_tag: str
    created_at: datetime
    is_active: bool
    parameters: Dict[str, Any]
