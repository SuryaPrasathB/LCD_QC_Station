from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

@dataclass
class InspectionResult:
    passed: bool
    score: float
    heatmap: Optional[np.ndarray] = None

def perform_inspection(
    captured_img: np.ndarray,
    reference_img: np.ndarray,
    roi_coords: Dict[str, int],
    threshold: float = 0.90
) -> InspectionResult:
    """
    Performs deterministic inspection by comparing the ROI of captured and reference images.

    Args:
        captured_img: HxWx3 BGR image (captured)
        reference_img: HxWx3 BGR image (reference)
        roi_coords: Dictionary containing x, y, width, height (relative to image dims)
        threshold: SSIM score threshold for passing (default 0.90)

    Returns:
        InspectionResult containing passed status, score, and optional heatmap.
    """

    # 1. Validation of ROI
    x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['width'], roi_coords['height']
    img_h, img_w = captured_img.shape[:2]
    ref_h, ref_w = reference_img.shape[:2]

    # Check if images have compatible dimensions (or at least valid ROI)
    # The requirement is "Use exact pixel coordinates".
    # If dimensions mismatch, we should probably fail or handle it.
    # For now, we assume the camera output is consistent (locked mode).
    # But let's check ROI bounds.

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        # ROI out of bounds for captured image
        # This is a critical failure.
        # Let's fail gracefully with score 0.
        return InspectionResult(passed=False, score=0.0)

    if x < 0 or y < 0 or x + w > ref_w or y + h > ref_h:
        # ROI out of bounds for reference image
        return InspectionResult(passed=False, score=0.0)

    # 2. Extract ROI
    # NumPy slicing: [y:y+h, x:x+w]
    roi_cap = captured_img[y:y+h, x:x+w]
    roi_ref = reference_img[y:y+h, x:x+w]

    # 3. Preprocessing
    # Allowed: Grayscale, Gaussian blur (small kernel)

    # Convert to Grayscale
    gray_cap = cv2.cvtColor(roi_cap, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur (small kernel, e.g., 3x3 or 5x5)
    # Standard deviation 0 means calculated from kernel size
    gray_cap = cv2.GaussianBlur(gray_cap, (5, 5), 0)
    gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

    # 4. Comparison Algorithm (SSIM)
    # scikit-image ssim expects arrays.
    # returns score, and optionally diff image (heatmap)
    # data_range should be specified (255 for uint8)

    score, diff = ssim(
        gray_ref,
        gray_cap,
        data_range=255,
        full=True
    )

    # diff is the structural similarity image.
    # It is in range [-1, 1] theoretically, but practically [0, 1] for identical?
    # Actually SSIM map can be negative?
    # The diff map returned by full=True has the same shape as inputs.

    # Convert diff to heatmap (0-255 uint8) for potential visualization
    # The diff from ssim is the local SSIM value map.
    # We might want the absolute difference or (1-ssim) for "heatmap".
    # But for now, let's just store the raw diff map or a visualized version.
    # Requirements: "Difference heatmap... Optional... toggleable"
    # Let's normalize it to 0-255 for easy display later if needed.
    # ssim map is typically -1 to 1.
    heatmap = ((diff + 1) / 2 * 255).astype(np.uint8)

    # 5. Output
    passed = bool(score >= threshold)

    return InspectionResult(passed=passed, score=float(score), heatmap=heatmap)
