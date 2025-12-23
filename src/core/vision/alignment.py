import cv2
import numpy as np
from src.infra.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)

class AlignmentEngine:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, input_img: np.ndarray, ref_img: np.ndarray):
        """
        Aligns input_img to ref_img.
        Returns: (aligned_image, success_flag, method_used)
        """
        # 1. Convert to Grayscale
        gray_in = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if len(input_img.shape) == 3 else input_img
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        # 2. Try ORB + Homography
        try:
            kp1, des1 = self.orb.detectAndCompute(gray_in, None)
            kp2, des2 = self.orb.detectAndCompute(gray_ref, None)

            if des1 is not None and des2 is not None and len(kp1) > settings.ALIGNMENT_MIN_KEYPOINTS:
                matches = self.matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Take top matches
                good_matches = matches[:int(len(matches) * 0.15)]

                if len(good_matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = gray_ref.shape
                        aligned = cv2.warpPerspective(input_img, M, (w, h))
                        logger.info("Alignment successful using ORB+Homography")
                        return aligned, True, "ORB"
        except Exception as e:
            logger.warning(f"ORB Alignment failed: {e}")

        # 3. Fallback: Template Matching / ECC (or just simple Resize/Center if assumption is loose)
        # Using Template Matching for translation correction
        logger.info("Falling back to Translation Alignment")
        return self._align_translation(input_img, ref_img), True, "Translation_Fallback"

    def _align_translation(self, input_img, ref_img):
        """
        Simple translation alignment using Phase Correlation or Template Matching.
        Here we use findTransformECC for better accuracy if motion is small,
        or just return resized input if completely failed.
        For robust fallback, we'll try to just match centroids or simple resize.
        Let's assume ROI is roughly correct and just Resize for now as 'dumb' fallback,
        or better, Phase Correlation for shift.
        """
        # Resize input to match reference dims
        h, w = ref_img.shape[:2]
        resized = cv2.resize(input_img, (w, h))

        # Optional: Calculate phase correlation for shift
        gray_in = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        try:
            (x, y), response = cv2.phaseCorrelate(np.float32(gray_in), np.float32(gray_ref))
            M = np.float32([[1, 0, x], [0, 1, y]])
            aligned = cv2.warpAffine(resized, M, (w, h))
            return aligned
        except:
            return resized
