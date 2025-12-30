from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import logging
import os
from src.core.embedding import EmbeddingModel

logger = logging.getLogger(__name__)

ORB_GATE_THRESHOLD = 0.50

@dataclass
class InspectionResult:
    passed: bool
    score: float  # Deprecated or alias for best_score? We'll keep it for now.
    best_score: float
    best_reference_id: str
    all_scores: Dict[str, float]
    dataset_version: str
    heatmap: Optional[np.ndarray] = None
    # STEP 9 additions
    orb_passed: Optional[bool] = None
    embedding_passed: Optional[bool] = None
    decision_path: Optional[str] = None

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

def _perform_embedding_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float, # default is likely 0.35 distance
    dataset_version: str
) -> InspectionResult:
    """
    Embedding-based inspection using cosine distance.
    Decision Rule: best_distance = min(distances) <= threshold -> PASS
    """
    model = EmbeddingModel.get_instance()
    if model.session is None:
        logger.warning("Embedding model not loaded. Falling back to ORB.")
        return _perform_orb_inspection(captured_img, reference_images, roi_coords, 0.75, dataset_version)

    # 1. Validation of ROI (Captured Image)
    x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['width'], roi_coords['height']
    img_h, img_w = captured_img.shape[:2]

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
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
    
    # Compute Embedding
    emb_cap = model.compute_embedding(roi_cap)
    
    if emb_cap is None:
        # Failed to compute embedding
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
    best_distance: float = float('inf')
    best_ref_id: str = "none"

    # 2. Iterate over all references
    for ref_id, ref_img in reference_images.items():
        ref_h, ref_w = ref_img.shape[:2]

        # Check ROI bounds for this reference
        if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
            # Skip invalid reference
            continue
            
        # Extract ROI
        roi_ref = ref_img[y:y+h, x:x+w]
        
        # Get/Cache Reference Embedding
        emb_ref = model.get_reference_embedding(dataset_version, ref_id, roi_ref)
        if emb_ref is None:
            continue
            
        # Compute Distance
        distance = model.compute_cosine_distance(emb_cap, emb_ref)
        
        # For compatibility with UI which expects "score" (higher is better),
        # we will report score = 1.0 - distance.
        score_val = 1.0 - distance
        scores[ref_id] = score_val
        
        if distance < best_distance:
            best_distance = distance
            best_ref_id = ref_id

    # 3. Decision Rule
    if best_distance == float('inf'):
        passed = False
        best_score = 0.0
    else:
        passed = bool(best_distance <= threshold)
        best_score = 1.0 - best_distance

    return InspectionResult(
        passed=passed,
        score=best_score,
        best_score=best_score,
        best_reference_id=best_ref_id,
        all_scores=scores,
        dataset_version=dataset_version,
        heatmap=None # No heatmap for embedding
    )

def _perform_hybrid_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str
) -> InspectionResult:
    """
    Hybrid inspection: ORB Gate -> Embedding.
    """
    # STAGE 1: ORB Gate
    orb_result = _perform_orb_inspection(
        captured_img,
        reference_images,
        roi_coords,
        ORB_GATE_THRESHOLD, # Locked constant
        dataset_version
    )

    if not orb_result.passed:
        # ORB FAIL -> Immediate Final FAIL
        orb_result.orb_passed = False
        orb_result.embedding_passed = None # Not run
        orb_result.decision_path = "ORB_REJECT"
        return orb_result

    # ORB PASS -> Continue to Embedding

    # STAGE 2: Embedding Inspection
    emb_result = _perform_embedding_inspection(
        captured_img,
        reference_images,
        roi_coords,
        threshold, # Passed threshold applies to embedding
        dataset_version
    )

    # Augment result with Hybrid info
    emb_result.orb_passed = True
    emb_result.embedding_passed = emb_result.passed

    if emb_result.passed:
        emb_result.decision_path = "EMBEDDING_ACCEPT"
    else:
        emb_result.decision_path = "EMBEDDING_REJECT"

    return emb_result

def perform_inspection(
    captured_img: np.ndarray,
    reference_images: Dict[str, np.ndarray],
    roi_coords: Dict[str, int],
    threshold: float,
    dataset_version: str,
    method: str = "hybrid" # Changed default to reflect intent, but handled via Env var mostly
) -> InspectionResult:
    """
    Performs deterministic inspection.
    Dispatches based on INSPECTION_METHOD env var or method argument.
    """
    env_method = os.environ.get("INSPECTION_METHOD", "").lower()

    # Priority: Env Var > Argument (if arg matches intent? No, usually Arg overrides Env,
    # but the requirement says INSPECTION_METHOD dictates logic).
    # "INSPECTION_METHOD=orb -> Step 7 ORB only"
    # "INSPECTION_METHOD unset -> Hybrid (default)"

    # If the caller passes "orb" or "embedding" explicitly (like UI might), should we honor it?
    # The requirement says "INSPECTION_METHOD unset -> Step 9 Hybrid".
    # And "Do not pass threshold through perform_inspection for ORB" (Wait, that was for hybrid construction).

    # Let's assume if env var is set, it wins. If not, check method arg.
    # If method arg is also default/unset, use Hybrid.

    # However, existing calls might pass "embedding".
    # If env var is unset, and method="embedding" (from window.py), should we force Hybrid?
    # "Default -> Step 9 hybrid logic"
    # "INSPECTION_METHOD unset -> Step 9 Hybrid (default)"
    
    # But window.py is calling it with method="embedding".
    # If I don't change window.py, it will keep asking for embedding.
    # The user instructions said: "INSPECTION_METHOD unset -> Step 9 Hybrid".
    # This implies that even if `method` arg says "embedding", if env is unset, we might want Hybrid?
    # Or should I change `window.py` to NOT pass "embedding"?
    # Modifying `window.py` was part of my plan (to update log data).
    # I should also update `window.py` to NOT hardcode "embedding" or pass "hybrid" or rely on default.

    # Logic implementation:
    active_method = "hybrid" # Default

    if env_method == "orb":
        active_method = "orb"
    elif env_method == "embedding":
        active_method = "embedding"
    else:
        # Env unset or unknown -> Hybrid
        # Check explicit method arg if provided and not default?
        # If the caller passed "ssim", maybe we should respect it?
        if method == "ssim":
            active_method = "ssim"
        elif method == "orb" and not env_method:
            active_method = "orb"
        elif method == "embedding" and not env_method:
            # Here is the catch. Window.py currently passes "embedding".
            # If I don't change window.py, and env is unset, do we run Hybrid?
            # User said "INSPECTION_METHOD unset -> Hybrid".
            # So I should probably treat method="embedding" as eligible for Hybrid override if env is unset?
            # OR better: I will change window.py to pass `method="hybrid"` or remove the arg.
            # I will assume for `perform_inspection`, if env is unset, "hybrid" is the target.
            active_method = "hybrid"

    if active_method == "orb":
        return _perform_orb_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
    elif active_method == "embedding":
        return _perform_embedding_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
    elif active_method == "ssim":
        return _perform_ssim_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
    else:
        return _perform_hybrid_inspection(
            captured_img, reference_images, roi_coords, threshold, dataset_version
        )
