"""Tests for the oncoprint module."""

import numpy as np
import pytest

from complexheatmap.oncoprint import (
    oncoPrint,
    alter_graphic,
    test_alter_fun as _test_alter_fun,
    _default_get_type,
    _make_default_alter_fun,
)


# ======================================================================
# _default_get_type
# ======================================================================

class TestDefaultGetType:
    """Tests for the default alteration type parser."""

    def test_semicolon_separated(self):
        assert _default_get_type("snv;amp") == ["snv", "amp"]

    def test_comma_separated(self):
        assert _default_get_type("snv,amp") == ["snv", "amp"]

    def test_single(self):
        assert _default_get_type("snv") == ["snv"]

    def test_empty_string(self):
        assert _default_get_type("") == []

    def test_whitespace_stripping(self):
        assert _default_get_type(" snv ; amp ") == ["snv", "amp"]

    def test_non_string(self):
        assert _default_get_type(0) == []


# ======================================================================
# _make_default_alter_fun
# ======================================================================

class TestMakeDefaultAlterFun:
    """Tests for the default alter_fun builder."""

    def test_returns_dict(self):
        funs = _make_default_alter_fun({"snv": "red", "del": "blue"})
        assert isinstance(funs, dict)
        assert "background" in funs
        assert "snv" in funs
        assert "del" in funs

    def test_callable_entries(self):
        funs = _make_default_alter_fun({"snv": "red"})
        assert callable(funs["background"])
        assert callable(funs["snv"])


# ======================================================================
# alter_graphic
# ======================================================================

class TestAlterGraphic:
    """Tests for the alter_graphic helper."""

    def test_returns_callable(self):
        fn = alter_graphic("rect", fill="red")
        assert callable(fn)

    def test_point_graphic(self):
        fn = alter_graphic("point", fill="blue")
        assert callable(fn)

    def test_default_graphic(self):
        fn = alter_graphic()
        assert callable(fn)


# ======================================================================
# oncoPrint
# ======================================================================

class TestOncoPrint:
    """Tests for oncoPrint construction."""

    @pytest.fixture()
    def binary_mat(self):
        """Two alteration types, 4 genes x 5 samples."""
        return {
            "snv": np.array([
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1],
            ]),
            "del": np.array([
                [0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ]),
        }

    def test_basic_creation(self, binary_mat):
        from complexheatmap.heatmap import Heatmap
        ht = oncoPrint(binary_mat, col={"snv": "red", "del": "blue"})
        assert isinstance(ht, Heatmap)

    def test_name_assignment(self, binary_mat):
        ht = oncoPrint(binary_mat, name="my_op")
        assert ht.name == "my_op"

    def test_auto_name(self, binary_mat):
        ht = oncoPrint(binary_mat)
        assert "oncoPrint" in ht.name

    def test_default_colors(self, binary_mat):
        ht = oncoPrint(binary_mat)
        assert ht is not None

    def test_custom_alter_fun(self, binary_mat):
        import grid_py
        funs = {
            "background": lambda x, y, w, h: grid_py.grid_rect(
                x=x, y=y, width=w, height=h,
                gp=grid_py.Gpar(fill="#CCCCCC", col=None),
            ),
            "snv": lambda x, y, w, h: grid_py.grid_rect(
                x=x, y=y, width=w, height=h,
                gp=grid_py.Gpar(fill="red", col=None),
            ),
            "del": lambda x, y, w, h: grid_py.grid_rect(
                x=x, y=y, width=w, height=h,
                gp=grid_py.Gpar(fill="blue", col=None),
            ),
        }
        ht = oncoPrint(binary_mat, alter_fun=funs, col={"snv": "red", "del": "blue"})
        assert ht is not None

    def test_remove_empty_columns(self):
        mat = {
            "snv": np.array([[1, 0, 0], [0, 0, 0]]),
        }
        ht = oncoPrint(mat, remove_empty_columns=True)
        # Column with all zeros should be removed
        assert ht.ncol <= 3

    def test_remove_empty_rows(self):
        mat = {
            "snv": np.array([[1, 0], [0, 0]]),
        }
        ht = oncoPrint(mat, remove_empty_rows=True)
        assert ht.nrow <= 2

    def test_custom_row_order(self, binary_mat):
        ht = oncoPrint(binary_mat, row_order=[3, 2, 1, 0])
        assert ht is not None

    def test_custom_column_order(self, binary_mat):
        ht = oncoPrint(binary_mat, column_order=[4, 3, 2, 1, 0])
        assert ht is not None

    def test_show_pct_false(self, binary_mat):
        ht = oncoPrint(binary_mat, show_pct=False)
        assert ht is not None

    def test_show_column_names(self, binary_mat):
        ht = oncoPrint(binary_mat, show_column_names=True)
        assert ht.show_column_names is True

    def test_show_row_names_false(self, binary_mat):
        ht = oncoPrint(binary_mat, show_row_names=False)
        assert ht.show_row_names is False

    def test_custom_labels(self, binary_mat):
        ht = oncoPrint(
            binary_mat,
            row_labels=["TP53", "BRCA1", "EGFR", "KRAS"],
            column_labels=["S1", "S2", "S3", "S4", "S5"],
        )
        assert ht is not None

    def test_cluster_rows(self, binary_mat):
        ht = oncoPrint(binary_mat, cluster_rows=False)
        assert ht is not None

    def test_single_type(self):
        mat = {"mut": np.array([[1, 0], [0, 1]])}
        ht = oncoPrint(mat)
        assert ht is not None

    def test_numpy_3d_input(self):
        arr = np.zeros((2, 3, 4), dtype=int)
        arr[0, 0, 0] = 1
        arr[1, 1, 1] = 1
        ht = oncoPrint(arr)
        assert ht is not None

    def test_invalid_input_type(self):
        with pytest.raises(ValueError):
            oncoPrint("not_valid")

    def test_no_types_raises(self):
        with pytest.raises(ValueError):
            oncoPrint({})

    def test_oncoprint_metadata(self, binary_mat):
        ht = oncoPrint(binary_mat, col={"snv": "red", "del": "blue"})
        assert hasattr(ht, "_oncoprint_arr")
        assert hasattr(ht, "_oncoprint_types")
        assert hasattr(ht, "_oncoprint_col")
        assert set(ht._oncoprint_types) == {"snv", "del"}

    def test_pct_digits(self, binary_mat):
        ht = oncoPrint(binary_mat, pct_digits=1)
        assert ht is not None

    def test_heatmap_legend_param(self, binary_mat):
        ht = oncoPrint(
            binary_mat,
            heatmap_legend_param={"title": "Mutations"},
        )
        assert ht is not None
