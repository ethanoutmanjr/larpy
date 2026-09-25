"""Unit tests for PriceEstimator model."""

import pytest
import torch
from larpy.models.price_estimator import priceEstimator, clipPriceEstimator


class TestPriceEstimator:
    """Tests for priceEstimator."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return priceEstimator(model_name="efficientnet_b0", num_classes=10)

    def test_model_creation(self, model):
        """Test model is created correctly."""
        assert model.model_name == "efficientnet_b0"
        assert model.num_classes == 10
        assert model.training

    def test_output_shape(self, model):
        """Test output shape is correct."""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (batch_size, 10)
        assert torch.all(output >= 0) and torch.all(output <= 10)

    def test_single_prediction(self, model):
        """Test single image prediction."""
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model.predict(dummy_input)
        assert output.shape == (1, 10)
        assert output.min() >= 0 and output.max() <= 10

    def test_save_load(self, model, tmp_path):
        """Test model save and load."""
        path = tmp_path / "test_model.pth"
        model.save(str(path))

        loaded = priceEstimator.load(model_name="efficientnet_b0", path=str(path))
        assert loaded.model_name == model.model_name

    def test_model_requires_grad(self, model):
        """Test model parameters require grad."""
        for param in model.parameters():
            assert param.requires_grad

    def test_backbone_freezable(self, model):
        """Test that backbone can be frozen."""
        for param in model.backbone.parameters():
            param.requires_grad = False
        assert not any(p.requires_grad for p in model.backbone.parameters())


class TestClipPriceEstimator:
    """Tests for clipPriceEstimator."""

    def test_model_creation(self):
        """Test CLIP model creation."""
        model = clipPriceEstimator(clip_model_name="openai/clip-vit-base-patch32", num_classes=10)
        assert model is not None

    def test_output_shape(self):
        """Test output shape."""
        model = clipPriceEstimator(clip_model_name="openai/clip-vit-base-patch32", num_classes=10)
        dummy_images = torch.randn(1, 3, 224, 224)
        output = model(dummy_images)
        assert output.shape == (1, 10)


class TestModelConfig:
    """Test available model configurations."""

    def test_available_models(self):
        """Test that available_models dict is populated."""
        from larpy.config.settings import modelConfig
        assert len(modelConfig.available_models) > 0
        assert "efficientnet_b0" in modelConfig.available_models
        assert "vit_base" in modelConfig.available_models
        assert "clip_vit_l14" in modelConfig.available_models
