import unittest
import numpy as np
import cv2
import os
from src.core.inspection import perform_inspection

class TestInspection(unittest.TestCase):
    def setUp(self):
        # Create synthetic images
        self.width = 640
        self.height = 480

        # 1. Reference Image (Random noise or solid color with pattern)
        self.ref_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw a white rectangle in the middle
        cv2.rectangle(self.ref_img, (100, 100), (300, 300), (255, 255, 255), -1)

        # 2. Identical Image
        self.identical_img = self.ref_img.copy()

        # 3. Different Image (Missing the rectangle)
        self.diff_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 4. Slightly Shifted Image (or noise added)
        self.noisy_img = self.ref_img.copy()
        noise = np.random.normal(0, 10, self.noisy_img.shape).astype(np.uint8)
        self.noisy_img = cv2.add(self.noisy_img, noise)

        self.roi = {"x": 100, "y": 100, "width": 200, "height": 200}

    def test_identical_images_pass(self):
        result = perform_inspection(self.identical_img, self.ref_img, self.roi)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 0.99)

    def test_different_images_fail(self):
        result = perform_inspection(self.diff_img, self.ref_img, self.roi)
        self.assertFalse(result.passed)
        self.assertLess(result.score, 0.90)

    def test_roi_cropping(self):
        # ROI focuses on the white rectangle.
        # If we use an ROI that looks at black space in both, they should match.
        empty_roi = {"x": 0, "y": 0, "width": 50, "height": 50}
        result = perform_inspection(self.diff_img, self.ref_img, empty_roi)
        # Both are black in (0,0,50,50)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 0.99)

    def test_invalid_roi_bounds(self):
        # ROI out of bounds
        bad_roi = {"x": 600, "y": 400, "width": 100, "height": 100} # Extends beyond 640x480
        result = perform_inspection(self.identical_img, self.ref_img, bad_roi)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_grayscale_conversion(self):
        # Ensure it works with colorful images too
        color_ref = np.zeros((100, 100, 3), dtype=np.uint8)
        color_ref[:] = (0, 0, 255) # Red

        color_cap = np.zeros((100, 100, 3), dtype=np.uint8)
        color_cap[:] = (0, 255, 0) # Green

        roi = {"x": 0, "y": 0, "width": 100, "height": 100}

        result = perform_inspection(color_cap, color_ref, roi)
        # Red vs Green in grayscale:
        # Red (0,0,255) -> 0.114*255 = 29
        # Green (0,255,0) -> 0.587*255 = 150
        # Should be different
        self.assertFalse(result.passed)

if __name__ == '__main__':
    unittest.main()
