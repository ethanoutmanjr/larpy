"""Unit tests for workers."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from larpy.workers.tasks import estimate_price_job, batch_estimate_price_job


class TestEstimatePriceJob:
    """Tests for estimate_price_job."""

    @patch("larpy.workers.tasks.get_model")
    def test_estimate_price_success(self, mock_get_model):
        """Test successful price estimation."""
        # Mock model
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.item.return_value = 7.5
        mock_get_model.return_value = mock_model

        # Create dummy image bytes
        dummy_image = b"fake_image_bytes"
        result = estimate_price_job(dummy_image)

        assert result["status"] == "success"
        assert result["price_tier"] == 8  # round(7.5) = 8
        assert "processing_time_ms" in result

    @patch("larpy.workers.tasks.get_model")
    def test_estimate_price_edge_cases(self, mock_get_model):
        """Test edge cases for price tiers."""
        mock_model = Mock()
        mock_model.return_value.item.side_effect = [0.1, 9.9, 5.0]
        mock_get_model.return_value = mock_model

        # Low score
        result_low = estimate_price_job(b"test")
        assert result_low["price_tier"] == 1  # max(1, round(0.1*10)) = 1

        # High score
        mock_model.return_value.item.return_value = 9.9
        result_high = estimate_price_job(b"test")
        assert result_high["price_tier"] == 10  # min(10, round(9.9)) = 10


class TestBatchEstimate:
    """Tests for batch estimation."""

    @patch("larpy.workers.tasks.estimate_price_job")
    def test_batch_estimation(self, mock_estimate):
        """Test batch processing."""
        mock_estimate.return_value = {"status": "success", "price_tier": 5}
        image_batch = [{"data": b"img1"}, {"data": b"img2"}, {"data": b"img3"}]
        result = batch_estimate_price_job(image_batch)
        assert len(result) == 3
        assert all(r["status"] == "success" for r in result)
