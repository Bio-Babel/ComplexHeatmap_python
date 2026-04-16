"""Tests for density_heatmap and frequency_heatmap."""

from __future__ import annotations

import numpy as np
import pytest

from complexheatmap.density_heatmap import (
    density_heatmap,
    frequency_heatmap,
    _compute_density_list,
)
from complexheatmap.heatmap import Heatmap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_matrix():
    """A small matrix for testing density/frequency heatmaps."""
    np.random.seed(123)
    return np.random.randn(50, 4)


@pytest.fixture
def sample_list():
    """A list of arrays with different lengths."""
    np.random.seed(456)
    return [np.random.randn(30), np.random.randn(50), np.random.randn(40)]


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestComputeDensityList:
    """Tests for _compute_density_list (the actual internal API)."""

    def test_returns_list_of_tuples(self, sample_matrix):
        data_list = [sample_matrix[:, i] for i in range(sample_matrix.shape[1])]
        result = _compute_density_list(data_list)
        assert isinstance(result, list)
        assert len(result) == 4
        for x, y in result:
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)

    def test_non_negative_density(self, sample_matrix):
        data_list = [sample_matrix[:, i] for i in range(sample_matrix.shape[1])]
        result = _compute_density_list(data_list)
        for x, y in result:
            assert np.all(y >= 0)

    def test_handles_nan(self):
        data_list = [
            np.array([1.0, np.nan, 4.0, 5.0]),
            np.array([2.0, 3.0, np.nan, 6.0]),
        ]
        result = _compute_density_list(data_list)
        assert len(result) == 2
        for x, y in result:
            assert np.any(y > 0)

    def test_single_finite_value(self):
        """Column with fewer than 2 finite values => fallback."""
        data_list = [np.array([1.0, np.nan])]
        result = _compute_density_list(data_list)
        assert len(result) == 1
        x, y = result[0]
        assert np.allclose(y, 0.0)


# ---------------------------------------------------------------------------
# density_heatmap tests
# ---------------------------------------------------------------------------

class TestDensityHeatmap:
    def test_returns_heatmap(self, sample_matrix):
        hm = density_heatmap(sample_matrix, title="test_density")
        assert isinstance(hm, Heatmap)

    def test_name(self, sample_matrix):
        # R: ht@name = paste0("density", "_", random_str) — unique VP names
        hm = density_heatmap(sample_matrix, title="my_density")
        assert hm.name.startswith("density_")

    def test_default_name(self, sample_matrix):
        hm = density_heatmap(sample_matrix)
        assert hm.name.startswith("density_")

    def test_shape(self, sample_matrix):
        hm = density_heatmap(sample_matrix)
        assert hm.ncol == 4
        assert hm.nrow > 0  # grid size depends on R's density() default

    def test_column_names_auto(self, sample_matrix):
        hm = density_heatmap(sample_matrix)
        assert hm.column_labels is not None
        labels = list(hm.column_labels)
        assert labels == ["V1", "V2", "V3", "V4"]

    def test_column_names_custom(self, sample_matrix):
        hm = density_heatmap(sample_matrix, column_names=["a", "b", "c", "d"])
        assert list(hm.column_labels) == ["a", "b", "c", "d"]

    def test_list_input(self, sample_list):
        hm = density_heatmap(sample_list)
        assert isinstance(hm, Heatmap)
        assert hm.ncol == 3

    def test_cluster_columns_false(self, sample_matrix):
        hm = density_heatmap(sample_matrix, cluster_columns=False)
        assert isinstance(hm, Heatmap)


# ---------------------------------------------------------------------------
# frequency_heatmap tests
# ---------------------------------------------------------------------------

class TestFrequencyHeatmap:
    def test_returns_heatmap(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix)
        assert isinstance(hm, Heatmap)

    def test_shape_int_breaks(self, sample_matrix):
        # breaks=N is a suggestion (like R's hist()), not exact
        hm = frequency_heatmap(sample_matrix, breaks=10)
        assert hm.nrow > 0
        assert hm.ncol == 4

    def test_shape_array_breaks(self, sample_matrix):
        edges = np.linspace(-3, 3, 16)
        hm = frequency_heatmap(sample_matrix, breaks=edges)
        assert hm.ncol == 4

    def test_name(self, sample_matrix):
        # R: ht@name = paste0(stat, "_", random_str)
        hm = frequency_heatmap(sample_matrix)
        assert hm.name.startswith("count_")

    def test_list_input(self, sample_list):
        hm = frequency_heatmap(sample_list)
        assert isinstance(hm, Heatmap)
        assert hm.ncol == 3

    def test_column_names(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix, column_names=["x", "y", "z", "w"])
        assert list(hm.column_labels) == ["x", "y", "z", "w"]

    def test_counts_sum(self, sample_matrix):
        """Total counts across bins should equal number of finite values per column."""
        hm = frequency_heatmap(sample_matrix, breaks=20)
        mat_data = hm.matrix
        for j in range(4):
            total = mat_data[:, j].sum()
            expected = np.sum(np.isfinite(sample_matrix[:, j]))
            assert total == pytest.approx(expected, abs=1)  # histogram edge effects
