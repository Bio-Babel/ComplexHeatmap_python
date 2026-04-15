"""Tests for complexheatmap.heatmap_list module."""

import numpy as np
import pytest

from complexheatmap.heatmap import Heatmap, AdditiveUnit
from complexheatmap.heatmap_list import (
    HeatmapList,
    _COMPONENT_REGISTRY,
    _clear_registry,
    _register_component,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHeatmapListConstruction:
    """Tests for HeatmapList construction and add_heatmap."""

    def test_empty_init(self):
        hl = HeatmapList()
        assert len(hl) == 0
        assert hl.direction == "horizontal"

    def test_init_with_direction(self):
        hl = HeatmapList(direction="vertical")
        assert hl.direction == "vertical"

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            HeatmapList(direction="diagonal")

    def test_init_with_list(self):
        mat = np.random.randn(5, 4)
        ht1 = Heatmap(mat, name="m1")
        ht2 = Heatmap(mat, name="m2")
        hl = HeatmapList([ht1, ht2])
        assert len(hl) == 2

    def test_add_heatmap(self):
        mat = np.random.randn(5, 4)
        hl = HeatmapList()
        ht = Heatmap(mat, name="m1")
        result = hl.add_heatmap(ht)
        assert result is hl
        assert len(hl) == 1

    def test_add_another_heatmap_list(self):
        mat = np.random.randn(5, 4)
        hl1 = HeatmapList([Heatmap(mat, name="a")])
        hl2 = HeatmapList([Heatmap(mat, name="b")])
        hl1.add_heatmap(hl2)
        assert len(hl1) == 2

    def test_duplicate_names_warning(self):
        mat = np.random.randn(5, 4)
        hl = HeatmapList([Heatmap(mat, name="dup")])
        with pytest.warns(UserWarning, match="Duplicate"):
            hl.add_heatmap(Heatmap(mat, name="dup"))


# ---------------------------------------------------------------------------
# Operator overloading
# ---------------------------------------------------------------------------


class TestOperators:
    """Tests for +, +=, and AdditiveUnit integration."""

    def test_add_creates_heatmap_list(self):
        mat = np.random.randn(5, 4)
        ht1 = Heatmap(mat, name="x1")
        ht2 = Heatmap(mat, name="x2")
        hl = ht1 + ht2
        assert isinstance(hl, HeatmapList)
        assert len(hl) == 2

    def test_add_three_heatmaps(self):
        mat = np.random.randn(5, 4)
        ht1 = Heatmap(mat, name="a")
        ht2 = Heatmap(mat, name="b")
        ht3 = Heatmap(mat, name="c")
        hl = ht1 + ht2 + ht3
        assert isinstance(hl, HeatmapList)
        assert len(hl) == 3

    def test_iadd(self):
        mat = np.random.randn(5, 4)
        hl = HeatmapList([Heatmap(mat, name="p")])
        hl += Heatmap(mat, name="q")
        assert len(hl) == 2

    def test_heatmap_list_add_heatmap(self):
        mat = np.random.randn(5, 4)
        hl = HeatmapList([Heatmap(mat, name="p")])
        hl2 = hl + Heatmap(mat, name="q")
        assert len(hl2) == 2
        # Original unchanged
        assert len(hl) == 1

    def test_getitem(self):
        mat = np.random.randn(5, 4)
        ht1 = Heatmap(mat, name="g1")
        ht2 = Heatmap(mat, name="g2")
        hl = ht1 + ht2
        assert hl[0].name == "g1"
        assert hl[1].name == "g2"

    def test_repr(self):
        mat = np.random.randn(5, 4)
        hl = Heatmap(mat, name="r1") + Heatmap(mat, name="r2")
        r = repr(hl)
        assert "r1" in r
        assert "r2" in r


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    """Tests for make_layout and associated methods."""

    def test_make_layout_basic(self):
        mat1 = np.random.randn(10, 5)
        mat2 = np.random.randn(10, 8)
        hl = Heatmap(mat1, name="h1") + Heatmap(mat2, name="h2")
        hl.make_layout()
        assert hl._layout is not None
        assert hl._main_heatmap_index == 0

    def test_make_layout_main_by_name(self):
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="first") + Heatmap(mat, name="second")
        hl.make_layout(main_heatmap="second")
        assert hl._main_heatmap_index == 1

    def test_make_layout_main_by_index(self):
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="aa") + Heatmap(mat, name="bb")
        hl.make_layout(main_heatmap=1)
        assert hl._main_heatmap_index == 1

    def test_width_ratios(self):
        hl = HeatmapList([
            Heatmap(np.random.randn(5, 3), name="w1"),
            Heatmap(np.random.randn(5, 7), name="w2"),
        ])
        ratios = hl._compute_width_ratios()
        assert ratios == [3.0, 7.0]

    def test_height_ratios(self):
        hl = HeatmapList([
            Heatmap(np.random.randn(4, 5), name="h1"),
            Heatmap(np.random.randn(6, 5), name="h2"),
        ], direction="vertical")
        ratios = hl._compute_height_ratios()
        assert ratios == [4.0, 6.0]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    """Tests for get_row_order, get_column_order, etc."""

    def test_get_row_order(self):
        np.random.seed(42)
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="ro1") + Heatmap(mat, name="ro2")
        hl.make_layout()
        ro = hl.get_row_order()
        assert "ro1" in ro
        assert "ro2" in ro

    def test_get_column_order(self):
        np.random.seed(42)
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="co1") + Heatmap(mat, name="co2")
        hl.make_layout()
        co = hl.get_column_order()
        assert "co1" in co
        assert "co2" in co

    def test_get_row_dend(self):
        np.random.seed(42)
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="d1") + Heatmap(mat, name="d2")
        hl.make_layout()
        rd = hl.get_row_dend()
        assert "d1" in rd

    def test_get_column_dend(self):
        np.random.seed(42)
        mat = np.random.randn(10, 5)
        hl = Heatmap(mat, name="cd1") + Heatmap(mat, name="cd2")
        hl.make_layout()
        cd = hl.get_column_dend()
        assert "cd1" in cd


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDraw:
    """Tests for the draw() method."""

    def test_draw_single_heatmap(self):
        np.random.seed(42)
        mat = np.random.randn(8, 6)
        hl = HeatmapList([Heatmap(mat, name="single")])
        result = hl.draw(show=False)
        assert result is hl
        assert hl._drawn is True

    def test_draw_two_heatmaps(self):
        np.random.seed(42)
        mat1 = np.random.randn(8, 5)
        mat2 = np.random.randn(8, 4)
        hl = Heatmap(mat1, name="d1") + Heatmap(mat2, name="d2")
        result = hl.draw(show=False)
        assert result is hl
        # Check that body viewports were registered
        reg = _COMPONENT_REGISTRY
        assert any("heatmap_body_d1" in k for k in reg)
        assert any("heatmap_body_d2" in k for k in reg)

    def test_draw_with_titles(self):
        np.random.seed(42)
        mat = np.random.randn(6, 4)
        hl = HeatmapList([Heatmap(mat, name="titled")])
        hl.draw(
            show=False,
            row_title="My Rows",
            column_title="My Columns",
        )
        assert "global_row_title" in _COMPONENT_REGISTRY
        assert "global_column_title" in _COMPONENT_REGISTRY

    def test_draw_empty_list(self):
        hl = HeatmapList()
        result = hl.draw(show=False)
        assert result is hl

    def test_draw_vertical(self):
        np.random.seed(42)
        mat1 = np.random.randn(5, 8)
        mat2 = np.random.randn(6, 8)
        hl = HeatmapList(
            [Heatmap(mat1, name="v1"), Heatmap(mat2, name="v2")],
            direction="vertical",
        )
        hl.draw(show=False)
        assert hl._drawn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the component registry."""

    def test_register_and_clear(self):
        _clear_registry()
        assert len(_COMPONENT_REGISTRY) == 0
        _register_component("test_comp", "test_vp")
        assert "test_comp" in _COMPONENT_REGISTRY
        _clear_registry()
        assert len(_COMPONENT_REGISTRY) == 0
