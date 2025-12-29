from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

@dataclass
class InspectionResult:
    passed: bool
    score: float  # Deprecated or alias for best_score? We'll keep it for now.
    best_score: float
    best_reference_id: str
    all_scores: Dict[str, float]
    dataset_version: str
    heatmap: Optional[np.ndarray] = None

def perform_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> InspectionResult:
    """
    Performs deterministic inspection by comparing the ROI of captured image
    against multiple reference images.

    Args:
        captured_img: HxWx3 BGR image (captured)
        reference_images: Dictionary mapping reference_id (filename) to HxWx3 BGR image
        roi_coords: Dictionary containing x, y, width, height (relative to image dims)
        threshold: SSIM score threshold for passing
        dataset_version: The version string of the active dataset

    Returns:
        InspectionResult containing passed status, best score, and other metadata.
    """

    # 1. Validation of ROI (Captured Image)
    x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['width'], roi_coords['height']
    img_h, img_w = captured_img.shape[:2]

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        # ROI out of bounds for captured image
        return InspectionResult(
            passed=False,
            score=0.0,
            best_score=0.0,
            best_reference_id="none",
            all_scores={},
            dataset_version=dataset_version,
            heatmap=None
        )

    # Extract ROI from captured image
    roi_cap = captured_img[y:y+h, x:x+w]
    gray_cap = cv2.cvtColor(roi_cap, cv2.COLOR_BGR2GRAY)
    gray_cap = cv2.GaussianBlur(gray_cap, (5, 5), 0)

    scores: Dict[str, float] = {}
    best_score: float = -1.0
    best_ref_id: str = "none"
    best_heatmap: Optional[np.ndarray] = None

    # 2. Iterate over all references
    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]

        # Check ROI bounds for this reference
        if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
            # Skip invalid reference dimensions
            scores[ref_id] = 0.0
            continue

        # Extract ROI
        roi_ref = ref_img[y:y+h, x:x+w]

        # Preprocessing
        gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

        # Comparison Algorithm (SSIM)
        score, diff = ssim(
            gray_ref,
            gray_cap,
            data_range=255,
            full=True
        )

        # Cast to float
        score_val = float(score)
        scores[ref_id] = score_val

        if score_val > best_score:
            best_score = score_val
            best_ref_id = ref_id
            # Normalize heatmap to 0-255
            best_heatmap = ((diff + 1) / 2 * 255).astype(np.uint8)

    # 3. Decision Rule
    # If no valid references were processed (e.g. empty dict), fail.
    if best_score == -1.0:
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_score >= threshold)

    return InspectionResult(
        passed=passed,
        score=best_score, # Keeping score as alias for best_score
        best_score=best_score,
        best_reference_id=best_ref_id,
        all_scores=scores,
        dataset_version=dataset_version,
        heatmap=best_heatmap
    )
