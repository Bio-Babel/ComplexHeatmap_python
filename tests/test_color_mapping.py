"""Tests for complexheatmap.color_mapping."""

import numpy as np
import pytest

from complexheatmap.color_mapping import ColorMapping
from complexheatmap._color import color_ramp2


# ---------------------------------------------------------------------------
# Discrete ColorMapping
# ---------------------------------------------------------------------------

class TestDiscreteColorMapping:
    def test_from_dict(self):
        cm = ColorMapping(colors={"A": "red", "B": "blue"})
        assert cm.is_discrete
        assert not cm.is_continuous
        assert cm.levels == ["A", "B"]
        # R: colors always stored as hex
        assert cm.color_map == {"A": "#FF0000", "B": "#0000FF"}

    def test_from_list_and_levels(self):
        cm = ColorMapping(
            colors=["red", "blue", "green"],
            levels=["x", "y", "z"],
        )
        assert cm.levels == ["x", "y", "z"]
        assert cm.map_to_colors("x") == "#FF0000"
        assert cm.map_to_colors("z") == "#00FF00"  # R's green = #00FF00

    def test_missing_levels_raises(self):
        with pytest.raises(ValueError, match="levels"):
            ColorMapping(colors=["red", "blue"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            ColorMapping(colors=["red"], levels=["a", "b"])

    def test_map_scalar(self):
        cm = ColorMapping(colors={"A": "#FF0000", "B": "#0000FF"})
        assert cm.map_to_colors("A") == "#FF0000"
        assert cm.map_to_colors("B") == "#0000FF"

    def test_map_na_returns_na_col(self):
        cm = ColorMapping(colors={"A": "red"}, na_col="#CCCCCC")
        assert cm.map_to_colors(None) == "#CCCCCC"
        assert cm.map_to_colors(float("nan")) == "#CCCCCC"

    def test_map_unknown_returns_na_col(self):
        cm = ColorMapping(colors={"A": "red"}, na_col="#FFFFFF")
        assert cm.map_to_colors("Z") == "#FFFFFF"

    def test_map_array(self):
        cm = ColorMapping(colors={"A": "red", "B": "blue"})
        result = cm.map_to_colors(np.array(["A", "B", "A"]))
        assert list(result) == ["#FF0000", "#0000FF", "#FF0000"]

    def test_map_list(self):
        cm = ColorMapping(colors={"A": "red", "B": "blue"})
        result = cm.map_to_colors(["B", "A"])
        assert list(result) == ["#0000FF", "#FF0000"]

    def test_float_key_matching(self):
        cm = ColorMapping(colors={"1": "red", "2": "blue"})
        assert cm.map_to_colors(1.0) == "#FF0000"

    def test_auto_name(self):
        cm = ColorMapping(colors={"A": "red"})
        assert cm.name.startswith("color_mapping_")

    def test_explicit_name(self):
        cm = ColorMapping(name="mymap", colors={"A": "red"})
        assert cm.name == "mymap"

    def test_repr_discrete(self):
        cm = ColorMapping(colors={"A": "red", "B": "blue"})
        r = repr(cm)
        assert "discrete" in r
        assert "n_levels=2" in r


# ---------------------------------------------------------------------------
# Continuous ColorMapping
# ---------------------------------------------------------------------------

class TestContinuousColorMapping:
    def test_from_col_fun(self):
        col_fun = color_ramp2([0, 1], ["white", "red"])
        cm = ColorMapping(col_fun=col_fun)
        assert cm.is_continuous
        assert not cm.is_discrete
        assert cm.breaks is not None
        assert len(cm.breaks) == 2

    def test_explicit_breaks(self):
        col_fun = color_ramp2([0, 1], ["white", "red"])
        cm = ColorMapping(col_fun=col_fun, breaks=[0, 0.5, 1])
        assert len(cm.breaks) == 3

    def test_map_scalar(self):
        col_fun = color_ramp2([0, 1], ["#000000", "#FFFFFF"], space="RGB")
        cm = ColorMapping(col_fun=col_fun)
        c = cm.map_to_colors(0.0)
        assert isinstance(c, str)
        assert c.startswith("#")

    def test_map_array(self):
        col_fun = color_ramp2([0, 1], ["black", "white"], space="RGB")
        cm = ColorMapping(col_fun=col_fun)
        result = cm.map_to_colors(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3

    def test_map_na(self):
        col_fun = color_ramp2([0, 1], ["black", "white"])
        cm = ColorMapping(col_fun=col_fun, na_col="#AAAAAA")
        assert cm.map_to_colors(float("nan")) == "#AAAAAA"

    def test_repr_continuous(self):
        col_fun = color_ramp2([0, 10, 20], ["blue", "white", "red"])
        cm = ColorMapping(col_fun=col_fun)
        r = repr(cm)
        assert "continuous" in r
        assert "n_breaks=3" in r


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_two(self):
        cm1 = ColorMapping(name="a", colors={"X": "red"})
        cm2 = ColorMapping(name="b", colors={"Y": "blue"})
        merged = ColorMapping.merge(cm1, cm2)
        assert merged.levels == ["X", "Y"]
        assert merged.color_map == {"X": "#FF0000", "Y": "#0000FF"}
        assert merged.name == "a+b"

    def test_merge_overlap(self):
        cm1 = ColorMapping(colors={"A": "red", "B": "green"})
        cm2 = ColorMapping(colors={"B": "blue", "C": "yellow"})
        merged = ColorMapping.merge(cm1, cm2)
        # First occurrence wins
        assert merged.color_map["B"] == "#00FF00"  # R's green = #00FF00
        assert "C" in merged.levels

    def test_merge_continuous_raises(self):
        col_fun = color_ramp2([0, 1], ["white", "red"])
        cm = ColorMapping(col_fun=col_fun)
        with pytest.raises(ValueError, match="continuous"):
            ColorMapping.merge(cm)

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError):
            ColorMapping.merge()


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------

class TestConstructionErrors:
    def test_no_colors_no_col_fun(self):
        with pytest.raises(ValueError, match="Either"):
            ColorMapping()
