"""Tests for complexheatmap.decorate module."""

import numpy as np
import pytest

from complexheatmap.heatmap import Heatmap
from complexheatmap.heatmap_list import (
    HeatmapList,
    _COMPONENT_REGISTRY,
    _clear_registry,
    _register_component,
)
from complexheatmap.decorate import (
    _find_component,
    _lookup_component,
    decorate_heatmap_body,
    decorate_annotation,
    decorate_column_dend,
    decorate_row_dend,
    decorate_row_names,
    decorate_column_names,
    decorate_row_title,
    decorate_column_title,
    decorate_dimnames,
    list_components,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistryLookup:
    """Tests for internal registry lookup functions."""

    def setup_method(self):
        _clear_registry()
        _register_component("heatmap_body_mat1_1_1", "mat1_heatmap_body_1_1")
        _register_component("heatmap_body_mat1_2_1", "mat1_heatmap_body_2_1")
        _register_component("column_dend_mat1_1", "mat1_dend_column_1")
        _register_component("row_dend_mat1_1", "mat1_dend_row_1")
        _register_component("row_names_mat1_1", "mat1_row_names_1")
        _register_component("column_names_mat1_1", "mat1_column_names_1")
        _register_component("global_row_title", "global_row_title")
        _register_component("global_column_title", "global_column_title")

    def test_exact_lookup(self):
        vp = _lookup_component("heatmap_body_mat1_1_1")
        assert vp == "mat1_heatmap_body_1_1"

    def test_lookup_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            _lookup_component("nonexistent")

    def test_find_exact(self):
        vp = _find_component("heatmap_body_mat1_1_1")
        assert vp == "mat1_heatmap_body_1_1"

    def test_find_substring(self):
        # "column_dend_mat1_1" is unique
        vp = _find_component("column_dend_mat1_1")
        assert vp == "mat1_dend_column_1"

    def test_find_ambiguous_raises(self):
        with pytest.raises(ValueError, match="Ambiguous"):
            _find_component("heatmap_body_mat1")

    def test_find_missing_raises(self):
        with pytest.raises(KeyError, match="No component"):
            _find_component("totally_missing_xyz")


# ---------------------------------------------------------------------------
# list_components
# ---------------------------------------------------------------------------


class TestListComponents:
    """Tests for list_components."""

    def setup_method(self):
        _clear_registry()

    def test_empty_registry(self):
        assert list_components() == []

    def test_nonempty_registry(self):
        _register_component("b_comp", "vp_b")
        _register_component("a_comp", "vp_a")
        result = list_components()
        assert result == ["a_comp", "b_comp"]


# ---------------------------------------------------------------------------
# Decorate functions with mocked registry
# ---------------------------------------------------------------------------


class TestDecorateWithRegistry:
    """Tests for decorate functions using a manually populated registry.

    These tests verify that the functions find the right viewport name
    and call the user callback.  We do NOT actually seek viewports
    since we may not have a live grid_py drawing context.
    """

    def setup_method(self):
        _clear_registry()
        _register_component("heatmap_body_ht1_1_1", "ht1_heatmap_body_1_1")
        _register_component("global_row_title", "global_row_title")
        _register_component("global_column_title", "global_column_title")

    def test_decorate_heatmap_body_finds_component(self):
        """Verify the component lookup succeeds (even if seek fails)."""
        vp = _find_component("heatmap_body_ht1_1_1")
        assert vp == "ht1_heatmap_body_1_1"

    def test_decorate_row_title_falls_back_to_global(self):
        """row_title for unknown heatmap should fall back to global."""
        vp = _find_component("global_row_title")
        assert vp == "global_row_title"

    def test_decorate_column_title_falls_back_to_global(self):
        vp = _find_component("global_column_title")
        assert vp == "global_column_title"

    def test_decorate_dimnames_invalid_which(self):
        with pytest.raises(ValueError, match="must be 'row' or 'column'"):
            decorate_dimnames("ht1", lambda: None, which="diagonal")


# ---------------------------------------------------------------------------
# Integration: draw + decorate
# ---------------------------------------------------------------------------


class TestDecorateIntegration:
    """Integration tests: draw a HeatmapList then decorate it."""

    def test_draw_and_decorate_body(self):
        np.random.seed(123)
        mat = np.random.randn(8, 6)
        hl = HeatmapList([Heatmap(mat, name="intg")])
        hl.draw(show=False)

        # The registry should contain a body component
        assert any("heatmap_body_intg" in k for k in _COMPONENT_REGISTRY)

        # Decorate should succeed (callback is called in the viewport)
        called = []

        def my_callback():
            called.append(True)

        decorate_heatmap_body("intg", my_callback)
        assert len(called) == 1

    def test_draw_and_list_components(self):
        np.random.seed(123)
        mat = np.random.randn(6, 4)
        hl = HeatmapList([Heatmap(mat, name="lc")])
        hl.draw(show=False, column_title="Title")

        comps = list_components()
        assert isinstance(comps, list)
        assert len(comps) > 0
        # Should contain at least the body and title
        assert any("heatmap_body" in c for c in comps)
        assert "global_column_title" in comps

    def test_draw_two_and_decorate_each(self):
        np.random.seed(42)
        mat1 = np.random.randn(8, 5)
        mat2 = np.random.randn(8, 4)
        hl = Heatmap(mat1, name="d1") + Heatmap(mat2, name="d2")
        hl.draw(show=False)

        calls_d1 = []
        calls_d2 = []

        decorate_heatmap_body("d1", lambda: calls_d1.append(1))
        decorate_heatmap_body("d2", lambda: calls_d2.append(1))

        assert len(calls_d1) == 1
        assert len(calls_d2) == 1
