import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple
from uuid import UUID

from src.core.types import InspectionResult, InspectionResultType, RawImage
from src.core.vision.alignment import AlignmentEngine
from src.core.vision.comparison import ComparisonEngine
from src.core.config import settings
from src.infra.logging import get_logger

logger = get_logger(__name__)

class InspectionEngine:
    """
    Orchestrates the inspection process: Align -> Compare -> Decide.
    """
    def __init__(self):
        self.aligner = AlignmentEngine()
        self.comparator = ComparisonEngine()
        self.reference_image: Optional[np.ndarray] = None
        self.current_dataset_version = "v0"

    def load_reference(self, image_path: str):
        """Loads the golden reference image."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load reference image: {image_path}")
            return
        self.reference_image = img
        logger.info(f"Reference image loaded from {image_path}")

    def process(self, raw_image: RawImage) -> InspectionResult:
        start_time = datetime.now(timezone.utc)

        if self.reference_image is None:
            logger.error("No reference image loaded! Returning FAIL.")
            return self._build_result(raw_image, InspectionResultType.FAIL, 0.0, {"error": "No Reference"})

        input_img = raw_image.data

        # 1. Alignment
        aligned_img, align_success, align_method = self.aligner.align(input_img, self.reference_image)

        # 2. Comparison
        score, diff_map, metrics = self.comparator.compare(aligned_img, self.reference_image)

        # 3. Decision
        result = InspectionResultType.PASS if score >= settings.SCORE_PASS_THRESHOLD else InspectionResultType.FAIL

        # Add alignment info to metrics
        metrics["alignment_method"] = align_method
        metrics["alignment_success"] = align_success

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # TODO: Save diff map if needed for debug?

        return InspectionResult(
            image_id=raw_image.id,
            result=result,
            confidence_score=score,
            processing_time_ms=int(duration),
            dataset_version=self.current_dataset_version,
            details=metrics
        )

    def _build_result(self, raw_image, result, score, details):
        return InspectionResult(
            image_id=raw_image.id,
            result=result,
            confidence_score=score,
            processing_time_ms=0,
            dataset_version=self.current_dataset_version,
            details=details
        )
