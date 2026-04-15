"""Tests for Heatmap3D and bar3D."""

from __future__ import annotations

import numpy as np
import pytest

import grid_py
from complexheatmap.heatmap_3d import (
    Heatmap3D,
    bar3D,
    _darken,
    _lighten,
    _value_to_hex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    return np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])


@pytest.fixture
def square_matrix():
    np.random.seed(42)
    return np.random.rand(4, 4)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestDarken:
    def test_basic(self):
        result = _darken("#FF8040", factor=0.5)
        assert result.startswith("#")
        assert len(result) == 7

    def test_black(self):
        result = _darken("#000000", factor=0.5)
        assert result == "#000000"

    def test_full_factor(self):
        result = _darken("#FF8040", factor=1.0)
        assert result == "#FF8040"


class TestLighten:
    def test_basic(self):
        result = _lighten("#FF0000", factor=0.5)
        assert result.startswith("#")
        r = int(result[1:3], 16)
        assert r > 0xFF * 0.9  # should be close to max

    def test_zero_factor(self):
        result = _lighten("#804020", factor=0.0)
        assert result == "#804020"

    def test_white(self):
        result = _lighten("#FFFFFF", factor=0.5)
        assert result == "#FFFFFF"


class TestValueToHex:
    def test_returns_hex_list(self):
        values = np.array([0.0, 0.5, 1.0])
        result = _value_to_hex(values)
        assert len(result) == 3
        for c in result:
            assert isinstance(c, str)
            assert c.startswith("#")

    def test_custom_col(self):
        fn = lambda v: "#FF0000"
        values = np.array([1.0, 2.0, 3.0])
        result = _value_to_hex(values, col=fn)
        assert all(c == "#FF0000" for c in result)

    def test_uniform_values(self):
        """All same values should not crash (span = 0)."""
        values = np.array([5.0, 5.0, 5.0])
        result = _value_to_hex(values)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Heatmap3D tests
# ---------------------------------------------------------------------------

class TestHeatmap3D:
    def test_construction(self, small_matrix):
        h3d = Heatmap3D(small_matrix, title="test")
        assert h3d.n_rows == 2
        assert h3d.n_cols == 3
        assert h3d.title == "test"

    def test_default_names(self, small_matrix):
        h3d = Heatmap3D(small_matrix)
        assert h3d.row_names == ["R1", "R2"]
        assert h3d.column_names == ["C1", "C2", "C3"]

    def test_custom_names(self, small_matrix):
        h3d = Heatmap3D(small_matrix, row_names=["a", "b"], column_names=["x", "y", "z"])
        assert h3d.row_names == ["a", "b"]
        assert h3d.column_names == ["x", "y", "z"]

    def test_build_grob(self, small_matrix):
        h3d = Heatmap3D(small_matrix)
        grob = h3d._build_grob()
        assert isinstance(grob, grid_py.GTree)
        assert h3d.grob is grob

    def test_build_grob_with_title(self, small_matrix):
        h3d = Heatmap3D(small_matrix, title="Hello")
        grob = h3d._build_grob()
        # Should have a title text grob among children
        names = [c.name for c in grob.get_children() if hasattr(c, "name")]
        assert "title" in names

    def test_build_grob_no_title(self, small_matrix):
        h3d = Heatmap3D(small_matrix, title="")
        grob = h3d._build_grob()
        names = [c.name for c in grob.get_children() if hasattr(c, "name")]
        assert "title" not in names

    def test_theta_stored(self, small_matrix):
        h3d = Heatmap3D(small_matrix, theta=45.0)
        assert h3d.theta == 45.0

    def test_bar_height_scale(self, small_matrix):
        h3d = Heatmap3D(small_matrix, bar_height_scale=2.0)
        assert h3d.bar_height_scale == 2.0
        grob = h3d._build_grob()
        assert isinstance(grob, grid_py.GTree)

    def test_children_count(self, small_matrix):
        """Each cell generates 1-3 polygons (front + top + right for non-zero)."""
        h3d = Heatmap3D(small_matrix)
        grob = h3d._build_grob()
        # 2*3 = 6 cells, all have positive values => 3 faces each = 18
        n_children = len(grob.get_children())
        assert n_children >= 6  # at least front faces

    def test_zero_height_cells(self):
        mat = np.array([[0.0, 0.0], [0.0, 0.0]])
        h3d = Heatmap3D(mat)
        grob = h3d._build_grob()
        # Zero height means only front faces (4 cells)
        # since all values are equal, heights will be 0
        assert isinstance(grob, grid_py.GTree)

    def test_nan_handling(self):
        mat = np.array([[1.0, np.nan], [3.0, 4.0]])
        h3d = Heatmap3D(mat)
        # Should not raise
        grob = h3d._build_grob()
        assert isinstance(grob, grid_py.GTree)

    def test_name_default(self, small_matrix):
        h3d = Heatmap3D(small_matrix)
        assert h3d.name == "heatmap_3d"

    def test_name_custom(self, small_matrix):
        h3d = Heatmap3D(small_matrix, name="custom")
        grob = h3d._build_grob()
        assert grob.name == "custom"


# ---------------------------------------------------------------------------
# bar3D tests
# ---------------------------------------------------------------------------

class TestBar3D:
    def test_returns_gtree(self, small_matrix):
        grob = bar3D(small_matrix)
        assert isinstance(grob, grid_py.GTree)

    def test_with_title(self, small_matrix):
        grob = bar3D(small_matrix, title="bar_test")
        assert isinstance(grob, grid_py.GTree)

    def test_custom_theta(self, small_matrix):
        grob = bar3D(small_matrix, theta=30.0)
        assert isinstance(grob, grid_py.GTree)

    def test_custom_scale(self, small_matrix):
        grob = bar3D(small_matrix, bar_height_scale=0.5)
        assert isinstance(grob, grid_py.GTree)
