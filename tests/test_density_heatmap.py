"""Tests for density_heatmap and frequency_heatmap."""

from __future__ import annotations

import numpy as np
import pytest

from complexheatmap.density_heatmap import (
    density_heatmap,
    frequency_heatmap,
    _compute_density_matrix,
    _quantile_indices,
    _resolve_palette,
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

class TestComputeDensityMatrix:
    def test_shape(self, sample_matrix):
        grid, dm = _compute_density_matrix(sample_matrix, n_grid=100)
        assert grid.shape == (100,)
        assert dm.shape == (100, 4)

    def test_non_negative_density(self, sample_matrix):
        _, dm = _compute_density_matrix(sample_matrix, n_grid=64)
        assert np.all(dm >= 0)

    def test_ylim_respected(self, sample_matrix):
        grid, _ = _compute_density_matrix(sample_matrix, n_grid=50, ylim=(-5.0, 5.0))
        assert grid[0] == pytest.approx(-5.0)
        assert grid[-1] == pytest.approx(5.0)

    def test_handles_nan(self):
        mat = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0]])
        grid, dm = _compute_density_matrix(mat, n_grid=32)
        assert dm.shape == (32, 2)
        # Both columns should have non-zero density
        assert np.any(dm[:, 0] > 0)
        assert np.any(dm[:, 1] > 0)

    def test_single_value_column(self):
        """Column with fewer than 2 finite values => density stays zero."""
        mat = np.array([[1.0, 2.0], [np.nan, 3.0]])
        grid, dm = _compute_density_matrix(mat, n_grid=16)
        # Column 0 has only 1 finite value => all zeros
        assert np.allclose(dm[:, 0], 0.0)


class TestQuantileIndices:
    def test_basic(self):
        grid = np.linspace(0, 10, 101)
        data = np.array([2.0, 5.0, 8.0])
        indices = _quantile_indices(grid, data, [0.0, 0.5, 1.0])
        assert len(indices) == 3
        # Median should be close to index 50 (value 5.0)
        assert abs(grid[indices[1]] - 5.0) < 0.2


class TestResolvePalette:
    def test_none_returns_callable(self):
        result = _resolve_palette(None, n=256)
        assert callable(result)

    def test_explicit_list_returned(self):
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        result = _resolve_palette(colors)
        assert result == colors

    def test_callable_passed_through(self):
        fn = lambda x: "#000000"
        result = _resolve_palette(fn)
        assert result is fn


# ---------------------------------------------------------------------------
# density_heatmap tests
# ---------------------------------------------------------------------------

class TestDensityHeatmap:
    def test_returns_heatmap(self, sample_matrix):
        hm = density_heatmap(sample_matrix, title="test_density")
        assert isinstance(hm, Heatmap)

    def test_name(self, sample_matrix):
        hm = density_heatmap(sample_matrix, title="my_density")
        assert hm.name == "my_density"

    def test_default_name(self, sample_matrix):
        hm = density_heatmap(sample_matrix)
        assert hm.name == "density"

    def test_shape(self, sample_matrix):
        hm = density_heatmap(sample_matrix, n_grid=128)
        assert hm.nrow == 128
        assert hm.ncol == 4

    def test_column_names_auto(self, sample_matrix):
        hm = density_heatmap(sample_matrix)
        assert hm.column_labels is not None
        labels = list(hm.column_labels)
        assert labels == ["V1", "V2", "V3", "V4"]

    def test_column_names_custom(self, sample_matrix):
        hm = density_heatmap(sample_matrix, column_names=["a", "b", "c", "d"])
        assert list(hm.column_labels) == ["a", "b", "c", "d"]

    def test_list_input(self, sample_list):
        hm = density_heatmap(sample_list, n_grid=64)
        assert isinstance(hm, Heatmap)
        assert hm.ncol == 3
        assert hm.nrow == 64

    def test_quantile_metadata_attached(self, sample_matrix):
        hm = density_heatmap(sample_matrix, show_quantiles=True)
        assert hasattr(hm, "_density_quantile_indices")
        assert hasattr(hm, "_density_quantile_values")
        assert hasattr(hm, "_density_grid")
        assert hm._density_quantile_values == [0.25, 0.5, 0.75]
        assert len(hm._density_quantile_indices) == 4

    def test_no_quantile_metadata(self, sample_matrix):
        hm = density_heatmap(sample_matrix, show_quantiles=False)
        assert not hasattr(hm, "_density_quantile_indices")

    def test_custom_quantiles(self, sample_matrix):
        hm = density_heatmap(sample_matrix, quantile_values=[0.1, 0.9])
        assert hm._density_quantile_values == [0.1, 0.9]
        for j in range(4):
            assert len(hm._density_quantile_indices[j]) == 2

    def test_ylim(self, sample_matrix):
        hm = density_heatmap(sample_matrix, ylim=(-10, 10), n_grid=100)
        grid = hm._density_grid
        assert grid[0] == pytest.approx(-10.0)
        assert grid[-1] == pytest.approx(10.0)

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
        hm = frequency_heatmap(sample_matrix, breaks=10)
        assert hm.nrow == 10
        assert hm.ncol == 4

    def test_shape_array_breaks(self, sample_matrix):
        edges = np.linspace(-3, 3, 16)
        hm = frequency_heatmap(sample_matrix, breaks=edges)
        assert hm.nrow == 15  # 16 edges => 15 bins
        assert hm.ncol == 4

    def test_bin_edges_metadata(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix, breaks=10)
        assert hasattr(hm, "_frequency_bin_edges")
        assert len(hm._frequency_bin_edges) == 11

    def test_name(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix, title="freq_test")
        assert hm.name == "freq_test"

    def test_default_name(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix)
        assert hm.name == "frequency"

    def test_list_input(self, sample_list):
        hm = frequency_heatmap(sample_list, breaks=8)
        assert isinstance(hm, Heatmap)
        assert hm.ncol == 3
        assert hm.nrow == 8

    def test_column_names(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix, column_names=["x", "y", "z", "w"])
        assert list(hm.column_labels) == ["x", "y", "z", "w"]

    def test_ylim(self, sample_matrix):
        hm = frequency_heatmap(sample_matrix, ylim=(-5.0, 5.0), breaks=10)
        edges = hm._frequency_bin_edges
        assert edges[0] == pytest.approx(-5.0)
        assert edges[-1] == pytest.approx(5.0)

    def test_counts_sum(self, sample_matrix):
        """Total counts across bins should equal number of finite values per column."""
        hm = frequency_heatmap(sample_matrix, breaks=20)
        mat_data = hm.matrix
        for j in range(4):
            total = mat_data[:, j].sum()
            expected = np.sum(np.isfinite(sample_matrix[:, j]))
            assert total == pytest.approx(expected, abs=1)  # histogram edge effects
