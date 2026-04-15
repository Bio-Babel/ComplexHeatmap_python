"""Tests for complexheatmap._utils utility functions."""

import numpy as np
import pytest

from complexheatmap._utils import (
    pindex,
    subset_gp,
    max_text_width,
    max_text_height,
    is_abs_unit,
    list_to_matrix,
    restore_matrix,
    default_axis_param,
    cluster_within_group,
    dist2,
)


# ---------------------------------------------------------------------------
# pindex
# ---------------------------------------------------------------------------

class TestPindex:
    def test_basic(self):
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, [0, 1, 2], [0, 1, 2])
        np.testing.assert_array_equal(result, [0, 5, 10])

    def test_recycle_i(self):
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, [1], [0, 1, 2])
        np.testing.assert_array_equal(result, [4, 5, 6])

    def test_recycle_j(self):
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, [0, 1, 2], [0])
        np.testing.assert_array_equal(result, [0, 4, 8])


# ---------------------------------------------------------------------------
# subset_gp
# ---------------------------------------------------------------------------

class TestSubsetGp:
    def test_vector_entries_subsetted(self):
        gp = {"col": ["red", "blue", "green"], "lwd": 2}
        result = subset_gp(gp, [0, 2])
        assert result["col"] == ["red", "green"]
        assert result["lwd"] == 2

    def test_single_index(self):
        gp = {"col": ["red", "blue", "green"]}
        result = subset_gp(gp, 1)
        assert result["col"] == ["blue"]

    def test_numpy_array_entry(self):
        gp = {"sizes": np.array([1.0, 2.0, 3.0])}
        result = subset_gp(gp, [0, 2])
        np.testing.assert_array_equal(result["sizes"], [1.0, 3.0])


# ---------------------------------------------------------------------------
# max_text_width / max_text_height  (using grid_py)
# ---------------------------------------------------------------------------

class TestMaxTextWidth:
    def test_single_string(self):
        import grid_py
        result = max_text_width("hello")
        assert grid_py.is_unit(result)

    def test_list_of_strings(self):
        import grid_py
        result = max_text_width(["hi", "hello world"])
        assert grid_py.is_unit(result)

    def test_empty_list(self):
        import grid_py
        result = max_text_width([])
        assert grid_py.is_unit(result)


class TestMaxTextHeight:
    def test_single_string(self):
        import grid_py
        result = max_text_height("hello")
        assert grid_py.is_unit(result)

    def test_list_of_strings(self):
        import grid_py
        result = max_text_height(["hi", "hello world"])
        assert grid_py.is_unit(result)


# ---------------------------------------------------------------------------
# is_abs_unit
# ---------------------------------------------------------------------------

class TestIsAbsUnit:
    def test_int(self):
        assert is_abs_unit(5) is True

    def test_float(self):
        assert is_abs_unit(3.14) is True

    def test_tuple(self):
        assert is_abs_unit((5, "mm")) is True

    def test_string(self):
        assert is_abs_unit("5mm") is False

    def test_grid_unit_absolute(self):
        import grid_py
        u = grid_py.Unit(1, "cm")
        assert is_abs_unit(u) is True

    def test_grid_unit_relative(self):
        import grid_py
        u = grid_py.Unit(0.5, "npc")
        assert is_abs_unit(u) is False


# ---------------------------------------------------------------------------
# list_to_matrix
# ---------------------------------------------------------------------------

class TestListToMatrix:
    def test_basic(self):
        lt = {"A": {"x", "y"}, "B": {"y", "z"}}
        mat, rows, cols = list_to_matrix(lt)
        assert mat.shape == (3, 2)
        assert set(rows) == {"x", "y", "z"}
        assert cols == ["A", "B"]

    def test_with_universal_set(self):
        lt = {"A": {"x"}, "B": {"y"}}
        mat, rows, cols = list_to_matrix(lt, universal_set=["x", "y", "z"])
        assert mat.shape == (3, 2)
        assert rows == ["x", "y", "z"]
        # z should be 0 in both columns
        z_idx = rows.index("z")
        assert mat[z_idx, 0] == 0
        assert mat[z_idx, 1] == 0

    def test_membership_correct(self):
        lt = {"A": {"x", "y"}, "B": {"y", "z"}}
        mat, rows, cols = list_to_matrix(lt)
        y_idx = rows.index("y")
        a_idx = cols.index("A")
        b_idx = cols.index("B")
        assert mat[y_idx, a_idx] == 1
        assert mat[y_idx, b_idx] == 1


# ---------------------------------------------------------------------------
# restore_matrix
# ---------------------------------------------------------------------------

class TestRestoreMatrix:
    def test_basic(self):
        i = np.array([0, 0, 1, 1])
        j = np.array([0, 1, 0, 1])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0])  # dummy
        mat = restore_matrix(j, i, x, y)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(mat, expected)


# ---------------------------------------------------------------------------
# default_axis_param
# ---------------------------------------------------------------------------

class TestDefaultAxisParam:
    def test_column(self):
        params = default_axis_param("column")
        assert params["side"] == "bottom"
        assert params["labels_rot"] == 0

    def test_row(self):
        params = default_axis_param("row")
        assert params["side"] == "left"


# ---------------------------------------------------------------------------
# cluster_within_group
# ---------------------------------------------------------------------------

class TestClusterWithinGroup:
    def test_basic(self):
        np.random.seed(42)
        mat = np.random.randn(10, 3)
        factor = ["A"] * 5 + ["B"] * 5
        order = cluster_within_group(mat, factor)
        assert len(order) == 10
        # First 5 should be indices from group A (0-4)
        assert set(order[:5]) == {0, 1, 2, 3, 4}
        # Last 5 should be indices from group B (5-9)
        assert set(order[5:]) == {5, 6, 7, 8, 9}

    def test_small_group(self):
        """Groups with <= 2 members should not fail."""
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        factor = ["A", "B", "B"]
        order = cluster_within_group(mat, factor)
        assert len(order) == 3
        assert order[0] == 0


# ---------------------------------------------------------------------------
# dist2
# ---------------------------------------------------------------------------

class TestDist2:
    def test_euclidean(self):
        x = np.array([[0, 0], [1, 0], [0, 1]])
        d = dist2(x)
        assert d.shape == (3, 3)
        np.testing.assert_almost_equal(d[0, 0], 0.0)
        np.testing.assert_almost_equal(d[0, 1], 1.0)
        np.testing.assert_almost_equal(d[0, 2], 1.0)
        np.testing.assert_almost_equal(d[1, 2], np.sqrt(2))

    def test_symmetric(self):
        x = np.random.randn(5, 3)
        d = dist2(x)
        np.testing.assert_array_almost_equal(d, d.T)

    def test_custom_distance(self):
        x = np.array([[0, 0], [1, 1]])
        d = dist2(x, pairwise_fun=lambda u, v: np.sum(np.abs(u - v)))
        np.testing.assert_almost_equal(d[0, 1], 2.0)  # Manhattan distance
