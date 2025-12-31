from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import logging
import os
from src.core.embedding import EmbeddingModel
from src.core.roi_model import NormalizedROI, normalize_roi

logger = logging.getLogger(__name__)

ORB_GATE_THRESHOLD = 0.50
MISSING_STRUCTURE_RATIO_THRESHOLD = 0.03

@dataclass
class ROIResult:
    roi_id: str
    passed: bool
    best_score: float
    best_reference_id: str
    decision_path: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    heatmap: Optional[np.ndarray] = None
    orb_passed: Optional[bool] = None
    embedding_passed: Optional[bool] = None
    # Semantic fields
    type: str = "DIGIT"
    similarity_passed: Optional[bool] = None
    semantic_passed: Optional[bool] = None
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None

@dataclass
class InspectionResult:
    passed: bool
    roi_results: Dict[str, ROIResult]
    dataset_version: str
    model_version: str = "v2" # Frozen embedding model version

    # Legacy / Compatibility fields
    score: float = 0.0          # Minimum score across ROIs
    best_score: float = 0.0     # Same as score
    best_reference_id: str = "" # Reference of the worst score ROI
    all_scores: Dict[str, float] = field(default_factory=dict) # Flat dict?
    heatmap: Optional[np.ndarray] = None # Heatmap of the worst ROI
    orb_passed: Optional[bool] = None
    embedding_passed: Optional[bool] = None
    decision_path: Optional[str] = None # Aggregate path

def _get_cropped_reference(ref_img: np.ndarray, roi: NormalizedROI) -> np.ndarray:
    """
    Helper to crop or resize reference image to match ROI dimensions.
    Handles legacy full-frame references.
    """
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    ref_h, ref_w = ref_img.shape[:2]

    # Detect legacy full-frame reference
    if ref_h > h * 1.5 or ref_w > w * 1.5:
        # Legacy Mode: Crop the reference using the ROI coordinates
        if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
            # Fallback: simple resize if crop is out of bounds
            return cv2.resize(ref_img, (w, h))
        else:
            return ref_img[y:y+h, x:x+w]
    else:
        # Modern Mode: Reference is already cropped
        return ref_img

def _validate_semantic_structure(
    captured_roi: np.ndarray,
    reference_roi: np.ndarray,
    roi_type: str
) -> Tuple[bool, float, Optional[str]]:
    """
    Deterministic semantic validation using structural difference.
    Returns (passed, score, detail).
    Score is the mismatch ratio (lower is better).
    """
    # 1. Ensure sizes match
    if captured_roi.shape != reference_roi.shape:
        reference_roi = cv2.resize(reference_roi, (captured_roi.shape[1], captured_roi.shape[0]))

    # 2. Convert to Grayscale
    gray_cap = cv2.cvtColor(captured_roi, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)

    # 3. Apply Otsu Thresholding to get binary structure masks
    # We use Otsu to automatically find the split between foreground/background
    _, mask_cap = cv2.threshold(gray_cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_ref = cv2.threshold(gray_ref, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Compute Structural Difference (Absolute Difference)
    diff = cv2.absdiff(mask_ref, mask_cap)

    # 5. Calculate Mismatch Ratio
    mismatch_pixels = np.count_nonzero(diff)
    total_pixels = diff.size

    if total_pixels == 0:
        return False, 1.0, "zero_pixels"

    ratio = mismatch_pixels / total_pixels

    passed = ratio <= MISSING_STRUCTURE_RATIO_THRESHOLD

    detail = None
    if not passed:
        detail = f"structure_mismatch_ratio_{ratio:.4f}"

    return passed, ratio, detail

def _perform_ssim_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi: NormalizedROI,
    threshold: float,
) -> ROIResult:
    """
    Legacy SSIM-based inspection for a SINGLE ROI.
    """
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    roi_id = roi.id
    img_h, img_w = captured_img.shape[:2]

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return ROIResult(roi_id, False, 0.0, "none", "BOUNDS_ERROR")

    roi_cap = captured_img[y:y+h, x:x+w]
    gray_cap = cv2.cvtColor(roi_cap, cv2.COLOR_BGR2GRAY)
    gray_cap = cv2.GaussianBlur(gray_cap, (5, 5), 0)

    scores = {}
    best_score = -1.0
    best_ref_id = "none"
    best_heatmap = None

    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]
        # Skip if ref is smaller than ROI (should match since we cropped from it, but safe check)
        if ref_h < h or ref_w < w:
            continue

        ref_roi = _get_cropped_reference(ref_img, roi)

        # Ensure exact size match for SSIM
        if ref_roi.shape != roi_cap.shape:
             ref_roi = cv2.resize(ref_roi, (w, h))

        gray_ref = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

        score, diff = ssim(gray_ref, gray_cap, data_range=255, full=True)
        score_val = float(score)
        scores[ref_id] = score_val

        if score_val > best_score:
            best_score = score_val
            best_ref_id = ref_id
            best_heatmap = ((diff + 1) / 2 * 255).astype(np.uint8)

    if best_score == -1.0:
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_score >= threshold)

    return ROIResult(
        roi_id=roi_id,
        passed=passed,
        best_score=best_score,
        best_reference_id=best_ref_id,
        decision_path="SSIM_PASS" if passed else "SSIM_FAIL",
        all_scores=scores,
        heatmap=best_heatmap
    )

def _perform_orb_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi: NormalizedROI,
    threshold: float,
) -> ROIResult:
    """
    ORB inspection for a SINGLE ROI.
    """
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    roi_id = roi.id

    # Extract ROI from Capture
    img_h, img_w = captured_img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
         return ROIResult(roi_id, False, 0.0, "none", "BOUNDS_ERROR")

    roi_cap = captured_img[y:y+h, x:x+w]
    gray_cap = cv2.cvtColor(roi_cap, cv2.COLOR_BGR2GRAY)
    gray_cap = cv2.GaussianBlur(gray_cap, (5, 5), 0)

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp_cap, des_cap = orb.detectAndCompute(gray_cap, None)

    # Debug Logging
    logger.debug(f"ROI {roi_id}: Capture Features: {len(kp_cap) if kp_cap else 0}")

    if des_cap is None:
        return ROIResult(roi_id, False, 0.0, "none", "ORB_NO_FEATURES")

    scores = {}
    best_score = -1.0
    best_ref_id = "none"
    best_heatmap = None

    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]

        ref_roi = _get_cropped_reference(ref_img, roi)

        gray_ref = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

        logger.debug(f"ROI {roi_id} Ref {ref_id}: Features: {len(kp_ref) if kp_ref else 0}")

        if des_ref is None or len(des_ref) == 0:
            scores[ref_id] = 0.0
            continue

        try:
            matches = bf.knnMatch(des_ref, des_cap, k=2)
        except Exception as e:
            logger.debug(f"ROI {roi_id}: ORB match error: {e}")
            scores[ref_id] = 0.0
            continue

        good = []
        for m_n in matches:
            if len(m_n) != 2: continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        denom = min(len(des_ref), len(des_cap))
        score = len(good) / denom if denom > 0 else 0.0
        scores[ref_id] = score

        logger.debug(f"ROI {roi_id} Ref {ref_id}: Score {score:.4f} ({len(good)} matches)")

        if score > best_score:
            best_score = score
            best_ref_id = ref_id
            best_heatmap = cv2.drawMatches(
                ref_img, kp_ref,
                roi_cap, kp_cap,
                good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        if score >= threshold:
            break

    if best_score == -1.0:
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_score >= threshold)

    return ROIResult(
        roi_id=roi_id,
        passed=passed,
        best_score=best_score,
        best_reference_id=best_ref_id,
        decision_path="ORB_PASS" if passed else "ORB_FAIL",
        all_scores=scores,
        heatmap=best_heatmap,
        orb_passed=passed,
        embedding_passed=None
    )

def _perform_embedding_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi: NormalizedROI,
    threshold: float,
    dataset_version: str
) -> ROIResult:
    """
    Embedding inspection for a SINGLE ROI.
    """
    model = EmbeddingModel.get_instance()
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    roi_id = roi.id

    if model.session is None:
        # Fallback to ORB if model missing (unlikely in prod but safe)
        return _perform_orb_inspection(captured_img, reference_images, roi, 0.75)

    img_h, img_w = captured_img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
         return ROIResult(roi_id, False, 0.0, "none", "BOUNDS_ERROR")

    roi_cap = captured_img[y:y+h, x:x+w]
    emb_cap = model.compute_embedding(roi_cap)
    
    if emb_cap is None:
         return ROIResult(roi_id, False, 0.0, "none", "EMBEDDING_FAIL")

    scores = {}
    best_distance = float('inf')
    best_ref_id = "none"

    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]

        # Detect legacy full-frame reference
        if ref_h > h * 1.5 or ref_w > w * 1.5:
             # Legacy Mode: Crop
             if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
                  ref_roi = cv2.resize(ref_img, (w, h))
             else:
                  ref_roi = ref_img[y:y+h, x:x+w]
        else:
             # Modern Mode
             ref_roi = ref_img

        unique_ref_key = f"{roi_id}/{ref_id}"
        emb_ref = model.get_reference_embedding(dataset_version, unique_ref_key, ref_roi)
        if emb_ref is None:
            continue
            
        distance = model.compute_cosine_distance(emb_cap, emb_ref)
        scores[ref_id] = 1.0 - distance
        
        if distance < best_distance:
            best_distance = distance
            best_ref_id = ref_id

    if best_distance == float('inf'):
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_distance <= threshold)
        best_score = 1.0 - best_distance

    return ROIResult(
        roi_id=roi_id,
        passed=passed,
        best_score=best_score,
        best_reference_id=best_ref_id,
        decision_path="EMB_PASS" if passed else "EMB_FAIL",
        all_scores=scores,
        heatmap=None,
        orb_passed=None,
        embedding_passed=passed
    )

def _perform_hybrid_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi: NormalizedROI,
    threshold: float,
    dataset_version: str
) -> ROIResult:
    """
    Hybrid inspection for a SINGLE ROI.
    """
    # 1. ORB Gate
    orb_res = _perform_orb_inspection(captured_img, reference_images, roi, ORB_GATE_THRESHOLD)

    if not orb_res.passed:
        orb_res.decision_path = "ORB_REJECT"
        orb_res.orb_passed = False
        orb_res.embedding_passed = None
        return orb_res

    # 2. Embedding
    emb_res = _perform_embedding_inspection(captured_img, reference_images, roi, threshold, dataset_version)

    emb_res.orb_passed = True
    emb_res.embedding_passed = emb_res.passed
    emb_res.heatmap = orb_res.heatmap

    if emb_res.passed:
        emb_res.decision_path = "EMBEDDING_ACCEPT"
    else:
        emb_res.decision_path = "EMBEDDING_REJECT"

    return emb_res

def inspect_roi(
    captured_img: np.ndarray,
    refs: Dict[str, np.ndarray],
    roi: NormalizedROI,
    threshold: float,
    dataset_version: str,
    method: str
) -> ROIResult:
    """
    Type-aware inspection router with semantic validation layer.
    """
    # 1. Similarity Inspection
    res = _dispatch_method(captured_img, refs, roi, threshold, dataset_version, method)

    # Populate Type
    res.type = roi.type
    res.similarity_passed = res.passed

    # 2. Semantic Validation (only if Similarity passed)
    if res.passed and res.best_reference_id != "none":
        best_ref_img = refs.get(res.best_reference_id)

        if best_ref_img is None:
            # Should not happen if logic is correct, but safe fallback
            logger.error(f"ROI {roi.id}: Best reference {res.best_reference_id} not found in refs.")
            # Fallback to similarity result as per requirements
            res.semantic_passed = True
            res.failure_reason = "semantic_ref_missing"
        else:
            # Extract Capture ROI
            x, y, w, h = roi.x, roi.y, roi.w, roi.h
            img_h, img_w = captured_img.shape[:2]

            # Bounds check again (redundant but safe)
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                 res.passed = False
                 res.semantic_passed = False
                 res.failure_reason = "bounds_error_in_semantic"
            else:
                 roi_cap = captured_img[y:y+h, x:x+w]

                 # Prepare Reference
                 roi_ref = _get_cropped_reference(best_ref_img, roi)

                 # Perform Check
                 try:
                     sem_passed, sem_score, sem_detail = _validate_semantic_structure(roi_cap, roi_ref, roi.type)

                     res.semantic_passed = sem_passed

                     if not sem_passed:
                         res.passed = False # Override result to FAIL
                         res.failure_reason = "semantic_failure"
                         res.failure_detail = sem_detail
                         logger.debug(f"ROI {roi.id} Semantic FAIL: {sem_detail} (Score {sem_score:.4f})")
                     else:
                         logger.debug(f"ROI {roi.id} Semantic PASS (Score {sem_score:.4f})")
                 except Exception as e:
                     logger.error(f"ROI {roi.id}: Semantic validation crashed: {e}")
                     # Fallback safely: "Do NOT crash inspection" -> Fallback to similarity result
                     # User said: "Log error, Fallback to similarity result"
                     res.semantic_passed = True # Assume Pass since we can't verify and similarity passed
                     res.failure_reason = "semantic_error_fallback"

    else:
        # Similarity failed or no ref found
        res.semantic_passed = None # Skipped

    return res

def _dispatch_method(captured_img, refs, roi, threshold, dataset_version, method):
    if method == "orb":
        return _perform_orb_inspection(captured_img, refs, roi, threshold)
    elif method == "embedding":
        return _perform_embedding_inspection(captured_img, refs, roi, threshold, dataset_version)
    elif method == "ssim":
         return _perform_ssim_inspection(captured_img, refs, roi, threshold)
    else:
         return _perform_hybrid_inspection(captured_img, refs, roi, threshold, dataset_version)

def perform_inspection(
    captured_img: np.ndarray,
    reference_images_nested: Dict[str, Dict[str, np.ndarray]],
    roi_data_full: Dict[str, Any],
    threshold: float,
    dataset_version: str
) -> InspectionResult:
    """
    Main entry point for Multi-ROI inspection.
    """
    raw_rois = roi_data_full.get("rois", [])
    
    roi_results: Dict[str, ROIResult] = {}

    # Strategy Determination (Global env var applies to all ROIs)
    env_method = os.environ.get("INSPECTION_METHOD", "hybrid").lower()

    for r_dict in raw_rois:
        # NORMALIZE ROI HERE
        roi_id = r_dict["id"]
        roi = normalize_roi(roi_id, r_dict)

        refs = reference_images_nested.get(roi.id, {})

        if not refs:
            print(f"DEBUG: No references found for ROI {roi.id}")
            roi_results[roi.id] = ROIResult(
                roi.id, False, 0.0, "none", "NO_REFERENCES"
            )
            continue

        # Use Router
        res = inspect_roi(captured_img, refs, roi, threshold, dataset_version, env_method)
        roi_results[roi.id] = res

    # Final Decision
    passed = all(r.passed for r in roi_results.values()) if roi_results else False

    if not roi_results:
        min_score = 0.0
        worst_roi = None
    else:
        min_score = min(r.best_score for r in roi_results.values())
        worst_roi = min(roi_results.values(), key=lambda r: r.best_score)

    return InspectionResult(
        passed=passed,
        roi_results=roi_results,
        dataset_version=dataset_version,
        score=min_score,
        best_score=min_score,
        best_reference_id=worst_roi.best_reference_id if worst_roi else "none",
        all_scores={},
        heatmap=worst_roi.heatmap if worst_roi else None,
        decision_path=f"ALL_PASS" if passed else "ROI_FAIL",
        orb_passed=all(r.orb_passed for r in roi_results.values() if r.orb_passed is not None),
        embedding_passed=all(r.embedding_passed for r in roi_results.values() if r.embedding_passed is not None)
    )
