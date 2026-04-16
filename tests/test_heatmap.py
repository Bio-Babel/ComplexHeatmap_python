"""Tests for the Heatmap class."""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest

from complexheatmap.heatmap import (
    Heatmap,
    AdditiveUnit,
    _compute_dist,
    _compute_linkage,
    _leaves_from_linkage,
    _kmeans_split,
    _factor_to_slices,
)
from complexheatmap._color import color_ramp2
import grid_py


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_matrix():
    """A small numeric matrix for testing."""
    np.random.seed(42)
    return np.random.rand(10, 8)


@pytest.fixture
def small_matrix():
    """A tiny matrix for fast tests."""
    return np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0]])


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestHeatmapConstruction:
    def test_basic_construction(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="test1")
        assert h.name == "test1"
        assert h.nrow == 10
        assert h.ncol == 8
        assert h._is_numeric_matrix is True
        assert h._layout_computed is False

    def test_auto_name(self, small_matrix):
        h = Heatmap(small_matrix)
        assert h.name.startswith("matrix_")

    def test_empty_name_raises(self, small_matrix):
        with pytest.raises(ValueError, match="cannot be empty"):
            Heatmap(small_matrix, name="")

    def test_1d_input(self):
        vec = np.array([1.0, 2.0, 3.0])
        h = Heatmap(vec, name="vec_test")
        assert h.nrow == 3
        assert h.ncol == 1

    def test_discrete_matrix(self):
        mat = np.array([["A", "B"], ["C", "A"]])
        h = Heatmap(mat, name="discrete")
        assert h._is_numeric_matrix is False
        assert h._color_mapping is not None

    def test_col_dict(self, small_matrix):
        # Not really meaningful for numeric, but tests the code path
        col_fun = color_ramp2([1, 6, 12], ["blue", "white", "red"])
        h = Heatmap(small_matrix, col=col_fun, name="col_fun_test")
        assert h._color_mapping is not None

    def test_col_list(self, small_matrix):
        h = Heatmap(small_matrix, col=["blue", "white", "red"], name="col_list_test")
        assert h._color_mapping is not None

    def test_border_bool(self, small_matrix):
        h = Heatmap(small_matrix, border=True, name="border_bool")
        assert h.border == "black"

    def test_border_str(self, small_matrix):
        h = Heatmap(small_matrix, border="red", name="border_str")
        assert h.border == "red"

    def test_default_gp_types(self, small_matrix):
        h = Heatmap(small_matrix, name="gp_test")
        assert isinstance(h.rect_gp, grid_py.Gpar)
        assert isinstance(h.row_names_gp, grid_py.Gpar)
        assert isinstance(h.column_names_gp, grid_py.Gpar)

    def test_size_defaults(self, small_matrix):
        h = Heatmap(small_matrix, name="size_test")
        assert isinstance(h.heatmap_width, grid_py.Unit)
        assert isinstance(h.heatmap_height, grid_py.Unit)
        assert isinstance(h.row_dend_width, grid_py.Unit)
        assert isinstance(h.column_dend_height, grid_py.Unit)

    def test_repr(self, small_matrix):
        h = Heatmap(small_matrix, name="repr_test")
        s = repr(h)
        assert "repr_test" in s
        assert "nrow=4" in s
        assert "ncol=3" in s


# ---------------------------------------------------------------------------
# Distance and linkage helpers
# ---------------------------------------------------------------------------

class TestDistanceLinkage:
    def test_euclidean_dist(self, small_matrix):
        d = _compute_dist(small_matrix, "euclidean")
        assert d.ndim == 1
        # 4 rows -> 6 pairwise distances
        assert len(d) == 6

    def test_correlation_dist(self, numeric_matrix):
        d = _compute_dist(numeric_matrix, "pearson")
        assert d.ndim == 1

    def test_callable_dist_one_param(self, small_matrix):
        from scipy.spatial.distance import pdist
        def my_dist(mat):
            return pdist(mat, metric="cityblock")
        d = _compute_dist(small_matrix, my_dist)
        assert d.ndim == 1

    def test_callable_dist_two_params(self, small_matrix):
        def my_metric(u, v):
            return np.sum(np.abs(u - v))
        d = _compute_dist(small_matrix, my_metric)
        assert d.ndim == 1

    def test_linkage(self, small_matrix):
        d = _compute_dist(small_matrix, "euclidean")
        Z = _compute_linkage(d, method="complete")
        assert Z.shape == (3, 4)  # 4 samples -> 3 merges

    def test_leaves(self, small_matrix):
        d = _compute_dist(small_matrix, "euclidean")
        Z = _compute_linkage(d, method="complete")
        leaves = _leaves_from_linkage(Z)
        assert len(leaves) == 4
        assert set(leaves) == {0, 1, 2, 3}

    def test_nan_handling(self):
        mat = np.array([[1.0, np.nan, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, np.nan]])
        d = _compute_dist(mat, "euclidean")
        assert d.ndim == 1
        assert not np.any(np.isnan(d))


# ---------------------------------------------------------------------------
# K-means splitting
# ---------------------------------------------------------------------------

class TestKMeans:
    def test_basic_kmeans(self, numeric_matrix):
        labels = _kmeans_split(numeric_matrix, k=3)
        assert len(labels) == 10
        assert len(set(labels)) <= 3

    def test_factor_to_slices(self):
        factor = np.array(["A", "B", "A", "C", "B"])
        levels, groups = _factor_to_slices(factor)
        assert levels == ["A", "B", "C"]
        assert len(groups) == 3
        np.testing.assert_array_equal(groups[0], [0, 2])
        np.testing.assert_array_equal(groups[1], [1, 4])
        np.testing.assert_array_equal(groups[2], [3])


# ---------------------------------------------------------------------------
# Layout computation tests
# ---------------------------------------------------------------------------

class TestMakeLayout:
    def test_basic_clustering(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="layout_basic")
        h.make_layout()
        assert h._layout_computed is True
        ro = h.get_row_order()
        co = h.get_column_order()
        assert isinstance(ro, np.ndarray)
        assert len(ro) == 10
        assert isinstance(co, np.ndarray)
        assert len(co) == 8
        # All indices present
        assert set(ro) == set(range(10))
        assert set(co) == set(range(8))

    def test_no_clustering(self, small_matrix):
        h = Heatmap(small_matrix, cluster_rows=False, cluster_columns=False,
                     name="no_clust")
        h.make_layout()
        ro = h.get_row_order()
        co = h.get_column_order()
        np.testing.assert_array_equal(ro, [0, 1, 2, 3])
        np.testing.assert_array_equal(co, [0, 1, 2])

    def test_manual_row_order(self, small_matrix):
        h = Heatmap(small_matrix, row_order=[3, 1, 0, 2], name="manual_ro")
        h.make_layout()
        ro = h.get_row_order()
        np.testing.assert_array_equal(ro, [3, 1, 0, 2])

    def test_manual_column_order(self, small_matrix):
        h = Heatmap(small_matrix, column_order=[2, 0, 1], name="manual_co")
        h.make_layout()
        co = h.get_column_order()
        np.testing.assert_array_equal(co, [2, 0, 1])

    def test_row_split_factor(self, numeric_matrix):
        split = np.array(["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"])
        h = Heatmap(numeric_matrix, row_split=split, name="row_split_factor")
        h.make_layout()
        ro = h.get_row_order()
        assert isinstance(ro, list)
        assert len(ro) == 3  # 3 groups
        all_indices = np.concatenate(ro)
        assert set(all_indices) == set(range(10))

    def test_row_km_split(self, numeric_matrix):
        h = Heatmap(numeric_matrix, row_km=2, name="row_km")
        h.make_layout()
        ro = h.get_row_order()
        assert isinstance(ro, list)
        assert len(ro) == 2

    def test_column_km_split(self, numeric_matrix):
        h = Heatmap(numeric_matrix, column_km=2, name="col_km")
        h.make_layout()
        co = h.get_column_order()
        assert isinstance(co, list)
        assert len(co) == 2

    def test_row_split_integer_cutree(self, numeric_matrix):
        h = Heatmap(numeric_matrix, row_split=3, name="cutree_test")
        h.make_layout()
        ro = h.get_row_order()
        assert isinstance(ro, list)
        assert len(ro) == 3

    def test_dendrogram_access(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="dend_access")
        h.make_layout()
        rd = h.get_row_dend()
        cd = h.get_column_dend()
        assert rd is not None
        assert cd is not None
        assert rd.shape[1] == 4  # linkage matrix
        assert cd.shape[1] == 4

    def test_idempotent_layout(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="idempotent")
        h.make_layout()
        ro1 = h.get_row_order()
        h.make_layout()  # should not recompute
        ro2 = h.get_row_order()
        np.testing.assert_array_equal(ro1, ro2)

    def test_lazy_layout(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="lazy")
        # Accessing order triggers layout
        ro = h.get_row_order()
        assert h._layout_computed is True
        assert len(ro) == 10


# ---------------------------------------------------------------------------
# Component sizes
# ---------------------------------------------------------------------------

class TestComponentSizes:
    def test_component_height(self, small_matrix):
        h = Heatmap(small_matrix, name="comp_h",
                     column_title="Title", show_column_names=True)
        body_h = h.component_height("heatmap_body")
        assert isinstance(body_h, grid_py.Unit)
        title_h = h.component_height("column_title_top")
        assert isinstance(title_h, grid_py.Unit)

    def test_component_width(self, small_matrix):
        h = Heatmap(small_matrix, name="comp_w",
                     row_title="Title", show_row_names=True)
        body_w = h.component_width("heatmap_body")
        assert isinstance(body_w, grid_py.Unit)


# ---------------------------------------------------------------------------
# Drawing tests
# ---------------------------------------------------------------------------

class TestDraw:
    def test_basic_draw(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="draw_basic")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.exists(fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_no_clustering(self, small_matrix):
        h = Heatmap(small_matrix, cluster_rows=False, cluster_columns=False,
                     name="draw_no_clust")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_with_titles(self, small_matrix):
        h = Heatmap(small_matrix, name="draw_titles",
                     column_title="Column Title",
                     row_title="Row Title")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_with_split(self, numeric_matrix):
        split = np.array(["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"])
        h = Heatmap(numeric_matrix, row_split=split, name="draw_split")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_with_border(self, small_matrix):
        h = Heatmap(small_matrix, border=True, name="draw_border")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_with_cell_fun(self, small_matrix):
        calls = []
        def my_cell_fun(j, i, x, y, w, h, fill):
            calls.append((j, i))

        h = Heatmap(small_matrix, cell_fun=my_cell_fun, name="draw_cellfun",
                     cluster_rows=False, cluster_columns=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            # 4 rows * 3 cols = 12 cells
            assert len(calls) == 12
        finally:
            os.unlink(fname)

    def test_draw_with_custom_col(self, small_matrix):
        col_fun = color_ramp2([1, 6, 12], ["green", "white", "purple"])
        h = Heatmap(small_matrix, col=col_fun, name="draw_custom_col")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_show_dend_sides(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="dend_sides",
                     row_dend_side="right", column_dend_side="bottom")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_hide_names(self, small_matrix):
        h = Heatmap(small_matrix, show_row_names=False, show_column_names=False,
                     name="no_names")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)

    def test_draw_custom_labels(self, small_matrix):
        h = Heatmap(small_matrix, name="custom_labels",
                     row_labels=["r1", "r2", "r3", "r4"],
                     column_labels=["c1", "c2", "c3"],
                     cluster_rows=False, cluster_columns=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            h.draw(show=False, filename=fname)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# AdditiveUnit tests
# ---------------------------------------------------------------------------

class TestAdditiveUnit:
    def test_is_subclass(self):
        assert issubclass(Heatmap, AdditiveUnit)

    def test_radd_none(self, small_matrix):
        h = Heatmap(small_matrix, name="radd_test")
        result = 0 + h
        assert result is h

    def test_add_returns_not_implemented_for_other(self, small_matrix):
        h = Heatmap(small_matrix, name="add_other")
        result = h.__add__("not_a_heatmap")
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# Copy and re_size tests
# ---------------------------------------------------------------------------

class TestCopyResize:
    def test_copy_all(self, small_matrix):
        h = Heatmap(small_matrix, name="copy_test")
        h.make_layout()
        h2 = h.copy_all()
        assert h2.name == h.name
        assert h2._layout_computed == h._layout_computed
        # Modifying copy doesn't affect original
        h2.name = "modified"
        assert h.name == "copy_test"

    def test_re_size(self, small_matrix):
        h = Heatmap(small_matrix, name="resize_test")
        new_w = grid_py.Unit(5, "cm")
        h.re_size(width=new_w)
        assert h.width is new_w


# ---------------------------------------------------------------------------
# make_row_cluster / make_column_cluster
# ---------------------------------------------------------------------------

class TestClusterMethods:
    def test_make_row_cluster(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="row_cluster")
        h.make_row_cluster()
        assert h._row_order_list is not None

    def test_make_column_cluster(self, numeric_matrix):
        h = Heatmap(numeric_matrix, name="col_cluster")
        h.make_column_cluster()
        assert h._column_order_list is not None
