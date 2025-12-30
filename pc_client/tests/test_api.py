import unittest
from unittest.mock import MagicMock, patch
from pc_client.api.inspection_api import InspectionClient
import requests

class TestInspectionClient(unittest.TestCase):
    def setUp(self):
        self.client = InspectionClient()
        self.client.base_url = "http://test-server:8000"

    @patch('requests.get')
    def test_check_health(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.client.check_health()
        self.assertEqual(result, {"status": "ok"})
        mock_get.assert_called_with("http://test-server:8000/health", timeout=5)

    @patch('requests.post')
    def test_set_roi(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "updated"}
        mock_post.return_value = mock_resp

        self.client.set_roi("roi_1", 0.1, 0.1, 0.2, 0.2)

        expected_payload = {
            "id": "roi_1",
            "normalized_bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        }
        mock_post.assert_called_with(
            "http://test-server:8000/roi/set",
            json=expected_payload,
            timeout=5
        )

    @patch('requests.get')
    def test_get_roi_list(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rois": [{"id": "roi_1"}]}
        mock_get.return_value = mock_resp

        rois = self.client.get_roi_list()
        self.assertEqual(rois, [{"id": "roi_1"}])

    @patch('requests.post')
    def test_start_inspection(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"inspection_id": "123"}
        mock_post.return_value = mock_resp

        insp_id = self.client.start_inspection()
        self.assertEqual(insp_id, "123")

if __name__ == '__main__':
    unittest.main()
