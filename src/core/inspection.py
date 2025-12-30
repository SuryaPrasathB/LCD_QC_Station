from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import logging
import os
from src.core.embedding import EmbeddingModel

logger = logging.getLogger(__name__)

ORB_GATE_THRESHOLD = 0.50

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

def _perform_ssim_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_rect: Dict[str, int],
    threshold: float,
) -> ROIResult:
    """
    Legacy SSIM-based inspection for a SINGLE ROI.
    """
    x, y, w, h = roi_rect['x'], roi_rect['y'], roi_rect['w'], roi_rect['h']
    roi_id = roi_rect.get('id', 'unknown')
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

        # ROI from ref (assuming ref is the crop itself? Step 10 says "Reference images are stored inside ROI-named folders".
        # Yes, so ref_img IS the cropped reference.)
        # Wait, if ref_img IS the crop, we don't need to slice it using x,y.
        # BUT, legacy code sliced it.
        # "Reference images are stored inside ROI-named folders".
        # "Capture once on Save ... Crop the frame ... Save as initial reference".
        # So the stored image IS the ROI.
        # So we should compare `roi_cap` to `ref_img` directly (maybe resize if minor off by 1?)
        # Step 10 spec: "For each ROI: Extract ROI from captured image ... Run ORB".

        # Assumption: Reference images in ROI folders are PRE-CROPPED.
        # So we do NOT crop `ref_img` using x,y.

        # Check sizes. If exact match is required for SSIM:
        if ref_img.shape != roi_cap.shape:
             ref_img = cv2.resize(ref_img, (w, h))

        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
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
    roi_rect: Dict[str, int],
    threshold: float,
) -> ROIResult:
    """
    ORB inspection for a SINGLE ROI.
    """
    x, y, w, h = roi_rect['x'], roi_rect['y'], roi_rect['w'], roi_rect['h']
    roi_id = roi_rect.get('id', 'unknown')

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
    if des_cap is None:
        return ROIResult(roi_id, False, 0.0, "none", "ORB_NO_FEATURES")

    scores = {}
    best_score = -1.0
    best_ref_id = "none"
    best_heatmap = None

    for ref_id, ref_img in reference_images.items():
        # ref_img is pre-cropped.
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
        if des_ref is None or len(des_ref) == 0:
            scores[ref_id] = 0.0
            continue

        try:
            matches = bf.knnMatch(des_ref, des_cap, k=2)
        except Exception:
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
    roi_rect: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> ROIResult:
    """
    Embedding inspection for a SINGLE ROI.
    """
    model = EmbeddingModel.get_instance()
    x, y, w, h = roi_rect['x'], roi_rect['y'], roi_rect['w'], roi_rect['h']
    roi_id = roi_rect.get('id', 'unknown')

    if model.session is None:
        # Fallback to ORB if model missing (unlikely in prod but safe)
        return _perform_orb_inspection(captured_img, reference_images, roi_rect, 0.75)

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
        # ref_img is pre-cropped.
        # Get/Cache Reference Embedding
        # The cache key needs to include roi_id now?
        # `get_reference_embedding` uses dataset_version + filename.
        # Filename is usually unique enough if it includes path?
        # `get_reference_embedding(dataset_version, ref_id, ref_img)`
        # `ref_id` comes from dict keys. In `DatasetManager`, ref_id was just filename base (e.g. "ref_001").
        # This collides across ROIs! ("digits_main/ref_001" vs "icon/ref_001").
        # We MUST ensure the key is unique.
        # We should probably pass `f"{roi_id}/{ref_id}"` as the key to `get_reference_embedding`.

        unique_ref_key = f"{roi_id}/{ref_id}"
        emb_ref = model.get_reference_embedding(dataset_version, unique_ref_key, ref_img)
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
    roi_rect: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> ROIResult:
    """
    Hybrid inspection for a SINGLE ROI.
    """
    # 1. ORB Gate
    orb_res = _perform_orb_inspection(captured_img, reference_images, roi_rect, ORB_GATE_THRESHOLD)

    if not orb_res.passed:
        orb_res.decision_path = "ORB_REJECT"
        orb_res.orb_passed = False
        orb_res.embedding_passed = None
        return orb_res

    # 2. Embedding
    emb_res = _perform_embedding_inspection(captured_img, reference_images, roi_rect, threshold, dataset_version)

    emb_res.orb_passed = True
    emb_res.embedding_passed = emb_res.passed

    # Preserve ORB visualization if embedding fails? Or just use Embedding result?
    # Embedding doesn't have heatmap. We might want to keep ORB heatmap for context?
    # Spec says "heatmap... contains RGB visualization... when using ORB".
    # If Embedding passes, we don't have heatmap.
    # If we want to show why it passed/failed, maybe keep ORB heatmap if available.
    emb_res.heatmap = orb_res.heatmap

    if emb_res.passed:
        emb_res.decision_path = "EMBEDDING_ACCEPT"
    else:
        emb_res.decision_path = "EMBEDDING_REJECT"

    return emb_res


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
    rois = roi_data_full.get("rois", [])
    
    roi_results: Dict[str, ROIResult] = {}

    # Strategy Determination (Global env var applies to all ROIs)
    env_method = os.environ.get("INSPECTION_METHOD", "hybrid").lower()

    for roi in rois:
        roi_id = roi["id"]

        # Get references for this ROI
        # If legacy, `reference_images_nested` might just be {ref_id: img} if caller didn't adapt?
        # We rely on caller (MainWindow) to pass correct structure.
        # But wait, MainWindow calls `dataset.get_active_references` which we updated.
        # It returns `{roi_id: {ref_id: img}}`.

        refs = reference_images_nested.get(roi_id, {})

        if not refs:
            # No references for this ROI -> FAIL or SKIP?
            # "Final PASS requires all ROIs to PASS".
            # If no reference, we can't pass.
            logger.warning(f"No references found for ROI {roi_id}")
            roi_results[roi_id] = ROIResult(
                roi_id, False, 0.0, "none", "NO_REFERENCES"
            )
            continue

        # Dispatch
        if env_method == "orb":
            res = _perform_orb_inspection(captured_img, refs, roi, threshold)
        elif env_method == "embedding":
            res = _perform_embedding_inspection(captured_img, refs, roi, threshold, dataset_version)
        elif env_method == "ssim":
             res = _perform_ssim_inspection(captured_img, refs, roi, threshold)
        else:
             res = _perform_hybrid_inspection(captured_img, refs, roi, threshold, dataset_version)

        roi_results[roi_id] = res

    # Final Decision
    # "If any ROI fails -> Final FAIL"
    passed = all(r.passed for r in roi_results.values()) if roi_results else False

    # Aggregates for legacy fields
    # Score = min score (bottleneck)
    if not roi_results:
        min_score = 0.0
        worst_roi = None
    else:
        min_score = min(r.best_score for r in roi_results.values())
        # Find the worst ROI for metadata
        worst_roi = min(roi_results.values(), key=lambda r: r.best_score)

    return InspectionResult(
        passed=passed,
        roi_results=roi_results,
        dataset_version=dataset_version,
        score=min_score,
        best_score=min_score,
        best_reference_id=worst_roi.best_reference_id if worst_roi else "none",
        all_scores={}, # Flattening is ambiguous
        heatmap=worst_roi.heatmap if worst_roi else None,
        decision_path=f"ALL_PASS" if passed else "ROI_FAIL",
        orb_passed=all(r.orb_passed for r in roi_results.values() if r.orb_passed is not None),
        embedding_passed=all(r.embedding_passed for r in roi_results.values() if r.embedding_passed is not None)
    )
