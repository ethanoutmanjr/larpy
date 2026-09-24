"""Unit tests for data loading utilities."""

import pytest
import tempfile
import os
import csv
from larpy.data.loader import create_price_tier_label, split_dataset


class TestCreatePriceTierLabel:
    """Tests for create_price_tier_label."""

    def test_basic_tier(self):
        """Test basic tier assignment."""
        assert create_price_tier_label(500, scale=10, max_price=1000) == 6

    def test_minimum_tier(self):
        """Test minimum tier."""
        assert create_price_tier_label(0, scale=10, max_price=1000) == 1

    def test_maximum_tier(self):
        """Test maximum tier."""
        assert create_price_tier_label(1000, scale=10, max_price=1000) == 10

    def test_clamping(self):
        """Test that tiers are clamped to [1, scale]."""
        assert create_price_tier_label(-100, scale=10) == 1
        assert create_price_tier_label(2000, scale=10) == 10


class TestSplitDataset:
    """Tests for dataset splitting."""

    def test_split_ratios(self):
        """Test train/val/test split ratios."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        prices = [float(i) for i in range(100)]

        result = split_dataset(paths, prices, seed=42)

        assert len(result["train"]["image_paths"]) == 80
        assert len(result["val"]["image_paths"]) == 10
        assert len(result["test"]["image_paths"]) == 10

    def test_no_overlap(self):
        """Test that splits don't overlap."""
        paths = [f"img_{i}.jpg" for i in range(100)]
        prices = [float(i) for i in range(100)]

        result = split_dataset(paths, prices, seed=42)

        train_set = set(result["train"]["image_paths"])
        val_set = set(result["val"]["image_paths"])
        test_set = set(result["test"]["image_paths"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_custom_ratios(self):
        """Test custom split ratios."""
        paths = [f"img_{i}.jpg" for i in range(50)]
        prices = [float(i) for i in range(50)]

        result = split_dataset(paths, prices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        assert len(result["train"]["image_paths"]) == 35
        assert len(result["val"]["image_paths"]) == 7
        assert len(result["test"]["image_paths"]) == 8
