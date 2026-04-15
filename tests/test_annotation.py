"""Tests for the annotation system (Slice 3).

Tests construction, subsetting, and properties of AnnotationFunction,
SingleAnnotation, HeatmapAnnotation, and anno_* factory functions.
Does NOT test actual drawing (no grid_py rendering).
"""

import numpy as np
import pytest

from complexheatmap.annotation_function import AnnotationFunction
from complexheatmap.annotation_functions import (
    anno_simple,
    anno_barplot,
    anno_boxplot,
    anno_points,
    anno_lines,
    anno_text,
    anno_histogram,
    anno_density,
    anno_joyplot,
    anno_horizon,
    anno_image,
    anno_link,
    anno_mark,
    anno_block,
    anno_summary,
    anno_empty,
    anno_textbox,
    anno_customize,
    anno_numeric,
    anno_oncoprint_barplot,
)
from complexheatmap.single_annotation import SingleAnnotation
from complexheatmap.heatmap_annotation import (
    HeatmapAnnotation,
    rowAnnotation,
    columnAnnotation,
)


# ---------------------------------------------------------------------------
# AnnotationFunction
# ---------------------------------------------------------------------------

class TestAnnotationFunction:
    """Test the AnnotationFunction base class."""

    def test_basic_construction(self):
        def my_draw(index, k, n):
            pass

        af = AnnotationFunction(
            fun=my_draw,
            fun_name="test_fun",
            which="column",
            n=10,
        )
        assert af.fun_name == "test_fun"
        assert af.which == "column"
        assert af.nobs == 10
        assert af.data_scale == (0.0, 1.0)
        assert af.subsettable is False
        assert af.show_name is True

    def test_invalid_which_raises(self):
        with pytest.raises(ValueError, match="'column' or 'row'"):
            AnnotationFunction(fun=lambda i, k, n: None, which="diagonal")

    def test_row_construction(self):
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            fun_name="row_fun",
            which="row",
            n=5,
        )
        assert af.which == "row"
        assert af.nobs == 5

    def test_properties_width_height(self):
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            width=15.0,
            height=20.0,
        )
        assert af.width is not None
        assert af.height is not None

    def test_subset_basic(self):
        data = np.arange(10)
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            fun_name="sub_fun",
            which="column",
            var_env={"x": data},
            n=10,
            subsettable=True,
            subset_rule={"x": "array"},
        )
        af2 = af.subset(np.array([0, 2, 4]))
        assert af2.nobs == 3
        np.testing.assert_array_equal(af2.var_env["x"], np.array([0, 2, 4]))

    def test_subset_matrix_row(self):
        data = np.arange(20).reshape(10, 2)
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            var_env={"x": data},
            n=10,
            subsettable=True,
            subset_rule={"x": "matrix_row"},
        )
        af2 = af.subset(np.array([1, 3]))
        assert af2.nobs == 2
        assert af2.var_env["x"].shape == (2, 2)
        np.testing.assert_array_equal(af2.var_env["x"][0], data[1])

    def test_subset_not_subsettable_raises(self):
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            subsettable=False,
        )
        with pytest.raises(RuntimeError, match="not subsettable"):
            af.subset(np.array([0]))

    def test_copy(self):
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            fun_name="copy_test",
            var_env={"x": np.array([1, 2, 3])},
            n=3,
        )
        af2 = af.copy()
        assert af2.fun_name == "copy_test"
        assert af2 is not af
        assert af2.var_env["x"] is not af.var_env["x"]

    def test_repr(self):
        af = AnnotationFunction(
            fun=lambda i, k, n: None,
            fun_name="repr_test",
            which="row",
            n=5,
        )
        r = repr(af)
        assert "repr_test" in r
        assert "row" in r


# ---------------------------------------------------------------------------
# anno_* factory functions
# ---------------------------------------------------------------------------

class TestAnnoFactories:
    """Test that each anno_* function returns a proper AnnotationFunction."""

    def test_anno_simple_vector(self):
        af = anno_simple(np.array(["A", "B", "C", "A"]))
        assert af.fun_name == "anno_simple"
        assert af.nobs == 4
        assert af.subsettable is True

    def test_anno_simple_matrix(self):
        af = anno_simple(np.arange(20).reshape(10, 2))
        assert af.nobs == 10
        assert af.subsettable is True

    def test_anno_simple_with_col(self):
        af = anno_simple(
            np.array(["A", "B", "A"]),
            col={"A": "red", "B": "blue"},
        )
        assert af.nobs == 3

    def test_anno_simple_row(self):
        af = anno_simple(np.arange(5), which="row")
        assert af.which == "row"

    def test_anno_barplot(self):
        af = anno_barplot(np.random.rand(10))
        assert af.fun_name == "anno_barplot"
        assert af.nobs == 10
        assert af.subsettable is True

    def test_anno_barplot_matrix(self):
        af = anno_barplot(np.random.rand(10, 3))
        assert af.nobs == 10

    def test_anno_boxplot(self):
        af = anno_boxplot(np.random.rand(10, 20))
        assert af.fun_name == "anno_boxplot"
        assert af.nobs == 10

    def test_anno_boxplot_list(self):
        data = [np.random.rand(5) for _ in range(8)]
        af = anno_boxplot(data)
        assert af.nobs == 8

    def test_anno_points(self):
        af = anno_points(np.random.rand(15))
        assert af.fun_name == "anno_points"
        assert af.nobs == 15
        assert af.subsettable is True

    def test_anno_points_row(self):
        af = anno_points(np.random.rand(10), which="row")
        assert af.which == "row"

    def test_anno_lines(self):
        af = anno_lines(np.random.rand(12))
        assert af.fun_name == "anno_lines"
        assert af.nobs == 12

    def test_anno_text(self):
        af = anno_text(["gene1", "gene2", "gene3"])
        assert af.fun_name == "anno_text"
        assert af.nobs == 3
        assert af.show_name is False

    def test_anno_histogram(self):
        af = anno_histogram(np.random.rand(8, 100))
        assert af.fun_name == "anno_histogram"
        assert af.nobs == 8

    def test_anno_density(self):
        af = anno_density(np.random.rand(6, 50))
        assert af.fun_name == "anno_density"
        assert af.nobs == 6

    def test_anno_joyplot(self):
        af = anno_joyplot(np.random.rand(5, 30))
        assert af.fun_name == "anno_joyplot"
        assert af.nobs == 5

    def test_anno_horizon(self):
        af = anno_horizon(np.random.rand(4, 20))
        assert af.fun_name == "anno_horizon"
        assert af.nobs == 4

    def test_anno_image(self):
        af = anno_image(["a.png", "b.png", "c.png"])
        assert af.fun_name == "anno_image"
        assert af.nobs == 3

    def test_anno_link(self):
        af = anno_link(align_to={"grp1": [0, 1, 2], "grp2": [3, 4]})
        assert af.fun_name == "anno_link"
        assert af.subsettable is False

    def test_anno_mark(self):
        af = anno_mark(at=[1, 5, 8], labels=["a", "b", "c"])
        assert af.fun_name == "anno_mark"
        assert af.subsettable is False

    def test_anno_empty(self):
        af = anno_empty()
        assert af.fun_name == "anno_empty"
        assert af.show_name is False

    def test_anno_empty_with_border(self):
        af = anno_empty(border=True, which="row")
        assert af.which == "row"

    def test_anno_block(self):
        af = anno_block(labels=["A", "B"])
        assert af.fun_name == "anno_block"
        assert af.show_name is False

    def test_anno_summary(self):
        af = anno_summary()
        assert af.fun_name == "anno_summary"
        assert af.subsettable is False

    def test_anno_textbox(self):
        af = anno_textbox(
            align_to={"g1": [0, 1], "g2": [2, 3]},
            text={"g1": "hello", "g2": "world"},
        )
        assert af.fun_name == "anno_textbox"

    def test_anno_customize(self):
        af = anno_customize(np.array([1, 2, 3]))
        assert af.fun_name == "anno_customize"
        assert af.nobs == 3

    def test_anno_numeric(self):
        af = anno_numeric(np.array([1.0, 2.0, 3.0, 4.0]))
        assert af.fun_name == "anno_numeric"
        assert af.nobs == 4
        assert af.which == "row"

    def test_anno_oncoprint_barplot(self):
        af = anno_oncoprint_barplot(type=["snv", "indel"])
        assert af.fun_name == "anno_oncoprint_barplot"
        assert af.subsettable is False

    def test_subsetting_anno_barplot(self):
        af = anno_barplot(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        af2 = af.subset(np.array([0, 2, 4]))
        assert af2.nobs == 3
        np.testing.assert_array_equal(af2.var_env["x"], np.array([1.0, 3.0, 5.0]))

    def test_subsetting_anno_points(self):
        af = anno_points(np.array([10.0, 20.0, 30.0, 40.0]))
        af2 = af.subset(np.array([1, 3]))
        assert af2.nobs == 2

    def test_subsetting_anno_simple(self):
        af = anno_simple(np.array(["A", "B", "C", "D", "E"]))
        af2 = af.subset(np.array([0, 4]))
        assert af2.nobs == 2


# ---------------------------------------------------------------------------
# SingleAnnotation
# ---------------------------------------------------------------------------

class TestSingleAnnotation:
    """Test SingleAnnotation class."""

    def test_construction_with_value(self):
        sa = SingleAnnotation(
            name="test",
            value=np.array(["A", "B", "C"]),
            col={"A": "red", "B": "blue", "C": "green"},
        )
        assert sa.name == "test"
        assert sa.nobs == 3
        assert sa.which == "column"
        assert sa.color_mapping is not None

    def test_construction_with_fun(self):
        af = anno_barplot(np.random.rand(5))
        sa = SingleAnnotation(name="bar", fun=af, which="column")
        assert sa.name == "bar"
        assert sa.nobs == 5

    def test_construction_requires_value_or_fun(self):
        with pytest.raises(ValueError, match="Either `value` or `fun`"):
            SingleAnnotation(name="empty")

    def test_invalid_which(self):
        with pytest.raises(ValueError, match="'column' or 'row'"):
            SingleAnnotation(name="bad", value=[1, 2], which="top")

    def test_label_default(self):
        sa = SingleAnnotation(name="my_anno", value=[1, 2, 3])
        assert sa.label == "my_anno"

    def test_label_custom(self):
        sa = SingleAnnotation(name="my_anno", value=[1, 2, 3], label="Custom")
        assert sa.label == "Custom"

    def test_name_side_defaults(self):
        sa_col = SingleAnnotation(name="a", value=[1, 2], which="column")
        assert sa_col.name_side == "right"

        sa_row = SingleAnnotation(name="a", value=[1, 2], which="row")
        assert sa_row.name_side == "bottom"

    def test_width_height_defaults(self):
        sa_col = SingleAnnotation(name="a", value=[1, 2], which="column")
        assert sa_col.height is not None

        sa_row = SingleAnnotation(name="a", value=[1, 2], which="row")
        assert sa_row.width is not None

    def test_is_anno_matrix(self):
        sa = SingleAnnotation(name="m", value=np.arange(20).reshape(10, 2))
        assert sa.is_anno_matrix is True

        sa2 = SingleAnnotation(name="v", value=np.arange(10))
        assert sa2.is_anno_matrix is False

    def test_subset(self):
        sa = SingleAnnotation(
            name="sub",
            value=np.array(["A", "B", "C", "D"]),
            col={"A": "red", "B": "blue", "C": "green", "D": "yellow"},
        )
        sa2 = sa.subset(np.array([0, 3]))
        assert sa2.nobs == 2

    def test_repr(self):
        sa = SingleAnnotation(name="repr_test", value=[1, 2, 3])
        r = repr(sa)
        assert "repr_test" in r
        assert "column" in r


# ---------------------------------------------------------------------------
# HeatmapAnnotation
# ---------------------------------------------------------------------------

class TestHeatmapAnnotation:
    """Test HeatmapAnnotation class."""

    def test_basic_construction(self):
        ha = HeatmapAnnotation(
            x=np.array(["A", "B", "C"]),
            y=np.array([1.0, 2.0, 3.0]),
        )
        assert len(ha) == 2
        assert "x" in ha
        assert "y" in ha
        assert ha.which == "column"

    def test_row_annotation(self):
        ha = HeatmapAnnotation(
            which="row",
            x=np.array([1, 2, 3]),
        )
        assert ha.which == "row"

    def test_names(self):
        ha = HeatmapAnnotation(
            a=np.array([1, 2]),
            b=np.array([3, 4]),
        )
        assert ha.names == ["a", "b"]

    def test_nobs(self):
        ha = HeatmapAnnotation(
            x=np.array([1, 2, 3, 4, 5]),
        )
        assert ha.nobs == 5

    def test_getitem(self):
        ha = HeatmapAnnotation(
            x=np.array([1, 2, 3]),
        )
        sa = ha["x"]
        assert isinstance(sa, SingleAnnotation)
        assert sa.name == "x"

    def test_iter(self):
        ha = HeatmapAnnotation(
            a=np.array([1]),
            b=np.array([2]),
        )
        names = list(ha)
        assert names == ["a", "b"]

    def test_contains(self):
        ha = HeatmapAnnotation(foo=np.array([1, 2]))
        assert "foo" in ha
        assert "bar" not in ha

    def test_invalid_which(self):
        with pytest.raises(ValueError, match="'column' or 'row'"):
            HeatmapAnnotation(which="diagonal", x=[1])

    def test_with_annotation_function(self):
        af = anno_barplot(np.random.rand(5))
        ha = HeatmapAnnotation(bars=af)
        assert "bars" in ha
        assert ha["bars"].nobs == 5

    def test_with_col(self):
        ha = HeatmapAnnotation(
            col={"x": {"A": "red", "B": "blue"}},
            x=np.array(["A", "B", "A"]),
        )
        assert ha["x"].color_mapping is not None

    def test_subset(self):
        ha = HeatmapAnnotation(
            x=np.array([10, 20, 30, 40, 50]),
            y=np.array(["a", "b", "c", "d", "e"]),
        )
        ha2 = ha.subset(np.array([0, 2, 4]))
        assert ha2.nobs == 3

    def test_height_auto_computed(self):
        ha = HeatmapAnnotation(
            x=np.array([1, 2]),
            y=np.array([3, 4]),
        )
        # Should have a non-None height for column annotations
        assert ha.height is not None

    def test_repr(self):
        ha = HeatmapAnnotation(x=[1, 2], y=[3, 4])
        r = repr(ha)
        assert "HeatmapAnnotation" in r
        assert "column" in r

    def test_radd_zero(self):
        ha = HeatmapAnnotation(x=[1, 2])
        assert ha.__radd__(0) is ha

    def test_add_not_implemented(self):
        ha = HeatmapAnnotation(x=[1, 2])
        result = ha.__add__("something")
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

class TestConvenienceConstructors:
    """Test rowAnnotation and columnAnnotation."""

    def test_rowAnnotation(self):
        ha = rowAnnotation(x=np.array([1, 2, 3]))
        assert ha.which == "row"
        assert "x" in ha

    def test_columnAnnotation(self):
        ha = columnAnnotation(x=np.array([1, 2, 3]))
        assert ha.which == "column"
        assert "x" in ha

    def test_rowAnnotation_with_options(self):
        ha = rowAnnotation(
            x=np.array([1, 2, 3]),
            gap=2.0,
            border=True,
        )
        assert ha.gap == 2.0
        assert ha.border is True

    def test_columnAnnotation_with_col(self):
        ha = columnAnnotation(
            col={"grp": {"A": "red", "B": "blue"}},
            grp=np.array(["A", "B", "A"]),
        )
        assert ha["grp"].color_mapping is not None


# ---------------------------------------------------------------------------
# DataFrame integration
# ---------------------------------------------------------------------------

class TestDataFrameAnnotation:
    """Test HeatmapAnnotation with a pandas DataFrame."""

    def test_df_construction(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ha = HeatmapAnnotation(df=df)
        assert len(ha) == 2
        assert "a" in ha
        assert "b" in ha

    def test_df_with_kwargs_override(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3]})
        ha = HeatmapAnnotation(df=df, a=np.array([10, 20, 30]))
        # kwargs override df columns
        assert ha["a"].nobs == 3
