from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
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

def _perform_ssim_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> InspectionResult:
    """
    Legacy SSIM-based inspection.
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

def _perform_orb_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> InspectionResult:
    """
    Deterministic feature-based inspection using ORB.
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

    # ORB Parameters
    # Fixed constants as per spec
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp_cap, des_cap = orb.detectAndCompute(gray_cap, None)

    if des_cap is None:
        # No features in capture -> Fail
        return InspectionResult(
            passed=False,
            score=0.0,
            best_score=0.0,
            best_reference_id="none",
            all_scores={},
            dataset_version=dataset_version,
            heatmap=None
        )

    scores: Dict[str, float] = {}
    best_score: float = -1.0
    best_ref_id: str = "none"
    best_heatmap: Optional[np.ndarray] = None

    # 2. Iterate over all references
    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]

        # Check ROI bounds for this reference
        if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
            scores[ref_id] = 0.0
            continue

        # Extract ROI
        roi_ref = ref_img[y:y+h, x:x+w]

        # Preprocessing
        gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

        # Feature Extraction (Reference)
        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

        if des_ref is None or len(des_ref) == 0:
            scores[ref_id] = 0.0
            continue

        # Matching
        try:
            matches = bf.knnMatch(des_ref, des_cap, k=2)
        except Exception:
            # Should not happen if des_ref and des_cap are valid
            scores[ref_id] = 0.0
            continue

        # Ratio Test
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Scoring
        denom = min(len(des_ref), len(des_cap))
        if denom == 0:
            score = 0.0
        else:
            score = len(good) / denom

        scores[ref_id] = score

        if score > best_score:
            best_score = score
            best_ref_id = ref_id

            # Generate visualization for explainability
            # Draw matches
            best_heatmap = cv2.drawMatches(
                roi_ref, kp_ref,
                roi_cap, kp_cap,
                good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        # Short-circuit
        if score >= threshold:
            break

    # 3. Decision Rule
    if best_score == -1.0:
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_score >= threshold)

    return InspectionResult(
        passed=passed,
        score=best_score,
        best_score=best_score,
        best_reference_id=best_ref_id,
        all_scores=scores,
        dataset_version=dataset_version,
        heatmap=best_heatmap
    )

def perform_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str,
    method: str = "orb"
) -> InspectionResult:
    """
    Performs deterministic inspection by comparing the ROI of captured image
    against multiple reference images.

    Args:
        captured_img: HxWx3 BGR image (captured)
        reference_images: Dictionary mapping reference_id (filename) to HxWx3 BGR image
        roi_coords: Dictionary containing x, y, width, height (relative to image dims)
        threshold: Score threshold for passing
        dataset_version: The version string of the active dataset
        method: "orb" (default) or "ssim"

    Returns:
        InspectionResult containing passed status, best score, and other metadata.
    """
    if method.lower() == "ssim":
        return _perform_ssim_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
    else:
        # Default to ORB
        return _perform_orb_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
