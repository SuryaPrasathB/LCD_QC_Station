import cv2
import numpy as np
from src.infra.logging import get_logger

logger = get_logger(__name__)

class ComparisonEngine:
    def __init__(self):
        pass

    def compare(self, aligned_img: np.ndarray, ref_img: np.ndarray, mask: np.ndarray = None):
        """
        Compares aligned image with reference.
        Returns: (score, diff_map, details)
        """
        # Ensure grayscale
        g_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY) if len(aligned_img.shape) == 3 else aligned_img
        g_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        # 1. Absolute Difference
        diff = cv2.absdiff(g_aligned, g_ref)

        # 2. Apply Mask (if any regions should be ignored)
        if mask is not None:
            diff = cv2.bitwise_and(diff, diff, mask=mask)

        # 3. Clean Noise
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Morphological opening to remove specks
        kernel = np.ones((3,3), np.uint8)
        clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Blob Analysis (Robust Metric)
        contours, _ = cv2.findContours(clean_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_error_area = 0.0
        max_blob_area = 0.0
        blob_count = 0

        image_area = g_ref.shape[0] * g_ref.shape[1]

        # Cap for single blob contribution (e.g., 5% of screen)
        BLOB_CAP_PERCENT = 0.05
        blob_cap_pixels = image_area * BLOB_CAP_PERCENT

        for c in contours:
            area = cv2.contourArea(c)
            if area < 10: # Ignore tiny noise
                continue

            blob_count += 1
            max_blob_area = max(max_blob_area, area)

            # Robustness: Cap contribution
            effective_area = min(area, blob_cap_pixels)
            total_error_area += effective_area

        # 5. Score Calculation
        # Normalize error: 0.0 (perfect) to 1.0 (bad)
        # Let's say 10% total error area is a FAIL (score=0)
        MAX_TOLERABLE_ERROR_PCT = 0.10
        normalized_error = total_error_area / (image_area * MAX_TOLERABLE_ERROR_PCT)

        score = max(0.0, 1.0 - normalized_error)

        logger.info(f"Comparison: Score={score:.3f}, Blobs={blob_count}, MaxBlob={max_blob_area}")

        return score, clean_diff, {
            "blob_count": blob_count,
            "max_blob_area": max_blob_area,
            "total_error_area": total_error_area
        }
