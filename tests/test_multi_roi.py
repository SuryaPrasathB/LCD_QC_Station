import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
from src.core.inspection import (
    perform_inspection, InspectionResult, ROIResult,
    _perform_orb_inspection, _perform_embedding_inspection, _perform_hybrid_inspection
)

class TestMultiROIInspection(unittest.TestCase):
    def setUp(self):
        # Create dummy images
        self.img_h, self.img_w = 100, 100
        self.captured_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.ref_img = np.zeros((50, 50, 3), dtype=np.uint8) # pre-cropped reference

        # Legacy/Single ROI definition
        self.roi_data_legacy = {
            "rois": [{"id": "main", "x": 10, "y": 10, "w": 50, "h": 50}],
            "image_width": 100,
            "image_height": 100
        }

        # Multi ROI definition
        self.roi_data_multi = {
            "rois": [
                {"id": "roi_1", "x": 10, "y": 10, "w": 20, "h": 20},
                {"id": "roi_2", "x": 50, "y": 50, "w": 20, "h": 20}
            ],
            "image_width": 100,
            "image_height": 100
        }

        self.refs_nested_multi = {
            "roi_1": {"ref_a": self.ref_img},
            "roi_2": {"ref_b": self.ref_img}
        }

        self.refs_nested_legacy = {
            "main": {"ref_main": self.ref_img}
        }

    @patch("src.core.inspection._perform_embedding_inspection")
    def test_perform_inspection_multi_all_pass(self, mock_embedding):
        # Setup mock return values for each ROI
        mock_embedding.side_effect = [
            ROIResult("roi_1", True, 0.9, "ref_a", "PASS"),
            ROIResult("roi_2", True, 0.8, "ref_b", "PASS")
        ]

        result = perform_inspection(
            self.captured_img,
            self.refs_nested_multi,
            self.roi_data_multi,
            threshold=0.35,
            dataset_version="v2"
        )

        self.assertTrue(result.passed)
        self.assertEqual(len(result.roi_results), 2)
        self.assertEqual(result.score, 0.8) # Min score
        self.assertEqual(result.decision_path, "ALL_PASS")

    @patch("src.core.inspection._perform_embedding_inspection")
    def test_perform_inspection_multi_one_fail(self, mock_embedding):
        mock_embedding.side_effect = [
            ROIResult("roi_1", True, 0.9, "ref_a", "PASS"),
            ROIResult("roi_2", False, 0.2, "ref_b", "FAIL")
        ]

        result = perform_inspection(
            self.captured_img,
            self.refs_nested_multi,
            self.roi_data_multi,
            threshold=0.35,
            dataset_version="v2"
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.2)
        self.assertEqual(result.roi_results["roi_2"].decision_path, "FAIL")
        self.assertEqual(result.decision_path, "ROI_FAIL")

    def test_perform_inspection_no_refs(self):
        # Empty refs
        result = perform_inspection(
            self.captured_img,
            {},
            self.roi_data_multi,
            threshold=0.35,
            dataset_version="v2"
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.roi_results["roi_1"].decision_path, "NO_REFERENCES")

    @patch("src.core.inspection._perform_orb_inspection")
    def test_perform_inspection_orb_mode(self, mock_orb):
        # Set ENV variable
        with patch.dict(os.environ, {"INSPECTION_METHOD": "orb"}):
            mock_orb.return_value = ROIResult("roi_1", True, 0.9, "ref_a", "PASS", orb_passed=True)

            result = perform_inspection(
                self.captured_img,
                self.refs_nested_legacy,
                self.roi_data_legacy,
                threshold=0.75,
                dataset_version="v1"
            )

            self.assertTrue(result.passed)
            mock_orb.assert_called_once()
            self.assertTrue(result.orb_passed)

if __name__ == "__main__":
    unittest.main()
