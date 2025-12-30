import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Optional
import cv2

# Import the code to be tested.
# Note: We need to import InspectionResult to mock it or check it.
from src.core.inspection import InspectionResult, perform_inspection, ORB_GATE_THRESHOLD

# Mock EmbeddingModel since we don't want to load ONNX in unit tests
class MockEmbeddingModel:
    _instance = None
    def __init__(self):
        self.session = MagicMock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_embedding(self, image):
        # Return a dummy vector
        return np.ones(128, dtype=np.float32)

    def get_reference_embedding(self, dataset_version, ref_id, ref_image):
        return np.ones(128, dtype=np.float32)

    def compute_cosine_distance(self, vec1, vec2):
        # Return a controllable distance
        # We can patch this method in tests to simulate pass/fail
        return 0.1

class TestHybridInspection(unittest.TestCase):

    def setUp(self):
        self.captured_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.reference_images = {"ref1": np.zeros((100, 100, 3), dtype=np.uint8)}
        self.roi_coords = {'x': 0, 'y': 0, 'width': 50, 'height': 50}
        self.threshold = 0.35 # Embedding threshold
        self.dataset_version = "v1"

    @patch('src.core.inspection.EmbeddingModel', new=MockEmbeddingModel)
    @patch('src.core.inspection._perform_orb_inspection')
    @patch('src.core.inspection._perform_embedding_inspection')
    def test_hybrid_orb_reject(self, mock_embedding, mock_orb):
        """Test that if ORB fails, the result is FAIL (ORB_REJECT) and Embedding is NOT called."""

        # Setup ORB to FAIL
        # ORB returns an InspectionResult
        mock_orb.return_value = InspectionResult(
            passed=False,
            score=0.1,
            best_score=0.1,
            best_reference_id="ref1",
            all_scores={},
            dataset_version="v1",
            orb_passed=False,
            embedding_passed=None,
            decision_path="ORB_REJECT" # Assuming _perform_orb sets this or we set it in hybrid
        )

        # We force environment to be empty (default -> Hybrid)
        with patch.dict(os.environ, {}, clear=True):
            result = perform_inspection(
                self.captured_img, self.reference_images, self.roi_coords,
                self.threshold, self.dataset_version
            )

        # Checks
        self.assertFalse(result.passed)
        self.assertEqual(result.decision_path, "ORB_REJECT")
        self.assertFalse(result.orb_passed)
        self.assertIsNone(result.embedding_passed) # Should be None as it wasn't run

        # Verify calls
        # perform_inspection should call _perform_orb_inspection with gate threshold
        mock_orb.assert_called_once()
        args, _ = mock_orb.call_args
        # Check threshold passed to ORB is ORB_GATE_THRESHOLD (0.50)
        self.assertEqual(args[3], 0.50)

        # Verify Embedding was NOT called
        mock_embedding.assert_not_called()

    @patch('src.core.inspection.EmbeddingModel', new=MockEmbeddingModel)
    @patch('src.core.inspection._perform_orb_inspection')
    @patch('src.core.inspection._perform_embedding_inspection')
    def test_hybrid_orb_pass_embedding_pass(self, mock_embedding, mock_orb):
        """Test ORB Pass -> Embedding Pass -> Final Pass."""

        # Setup ORB to PASS
        mock_orb.return_value = InspectionResult(
            passed=True,
            score=0.8,
            best_score=0.8,
            best_reference_id="ref1",
            all_scores={"ref1": 0.8},
            dataset_version="v1",
            orb_passed=True,
            embedding_passed=None,
            decision_path="ORB_PASS"
        )

        # Setup Embedding to PASS
        mock_embedding.return_value = InspectionResult(
            passed=True,
            score=0.9,
            best_score=0.9,
            best_reference_id="ref1",
            all_scores={"ref1": 0.9},
            dataset_version="v1",
            orb_passed=True,
            embedding_passed=True,
            decision_path="EMBEDDING_ACCEPT"
        )

        with patch.dict(os.environ, {}, clear=True):
            result = perform_inspection(
                self.captured_img, self.reference_images, self.roi_coords,
                self.threshold, self.dataset_version
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.decision_path, "EMBEDDING_ACCEPT")
        self.assertTrue(result.orb_passed)
        self.assertTrue(result.embedding_passed)

        mock_orb.assert_called_once()
        mock_embedding.assert_called_once()

    @patch('src.core.inspection.EmbeddingModel', new=MockEmbeddingModel)
    @patch('src.core.inspection._perform_orb_inspection')
    @patch('src.core.inspection._perform_embedding_inspection')
    def test_hybrid_orb_pass_embedding_fail(self, mock_embedding, mock_orb):
        """Test ORB Pass -> Embedding Fail -> Final Fail."""

        # Setup ORB to PASS
        mock_orb.return_value = InspectionResult(
            passed=True,
            score=0.8,
            best_score=0.8,
            best_reference_id="ref1",
            all_scores={"ref1": 0.8},
            dataset_version="v1",
            orb_passed=True,
            embedding_passed=None,
            decision_path="ORB_PASS"
        )

        # Setup Embedding to FAIL
        mock_embedding.return_value = InspectionResult(
            passed=False,
            score=0.2, # Low score / high distance
            best_score=0.2,
            best_reference_id="ref1",
            all_scores={"ref1": 0.2},
            dataset_version="v1",
            orb_passed=True, # Context passed down?
            embedding_passed=False,
            decision_path="EMBEDDING_REJECT"
        )

        with patch.dict(os.environ, {}, clear=True):
            result = perform_inspection(
                self.captured_img, self.reference_images, self.roi_coords,
                self.threshold, self.dataset_version
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.decision_path, "EMBEDDING_REJECT")
        self.assertTrue(result.orb_passed)
        self.assertFalse(result.embedding_passed)

    @patch('src.core.inspection.EmbeddingModel', new=MockEmbeddingModel)
    @patch('src.core.inspection._perform_orb_inspection')
    @patch('src.core.inspection._perform_embedding_inspection')
    def test_env_method_orb(self, mock_embedding, mock_orb):
        """Test INSPECTION_METHOD=orb runs only ORB logic."""

        mock_orb.return_value = InspectionResult(
            passed=True, score=0.8, best_score=0.8, best_reference_id="ref1",
            all_scores={}, dataset_version="v1",
            orb_passed=True, embedding_passed=None, decision_path="LEGACY_ORB"
        )

        with patch.dict(os.environ, {"INSPECTION_METHOD": "orb"}, clear=True):
            perform_inspection(
                self.captured_img, self.reference_images, self.roi_coords,
                self.threshold, self.dataset_version
            )

        mock_orb.assert_called_once()
        mock_embedding.assert_not_called()

    @patch('src.core.inspection.EmbeddingModel', new=MockEmbeddingModel)
    @patch('src.core.inspection._perform_orb_inspection')
    @patch('src.core.inspection._perform_embedding_inspection')
    def test_env_method_embedding(self, mock_embedding, mock_orb):
        """Test INSPECTION_METHOD=embedding runs only Embedding logic."""

        mock_embedding.return_value = InspectionResult(
            passed=True, score=0.9, best_score=0.9, best_reference_id="ref1",
            all_scores={}, dataset_version="v1",
            orb_passed=None, embedding_passed=True, decision_path="LEGACY_EMBEDDING"
        )

        with patch.dict(os.environ, {"INSPECTION_METHOD": "embedding"}, clear=True):
            perform_inspection(
                self.captured_img, self.reference_images, self.roi_coords,
                self.threshold, self.dataset_version
            )

        mock_embedding.assert_called_once()
        mock_orb.assert_not_called()

if __name__ == '__main__':
    unittest.main()
