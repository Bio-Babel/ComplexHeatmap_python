"""Tests for complexheatmap._color colour utilities."""

import re

import numpy as np
import pytest

from complexheatmap._color import color_ramp2, add_transparency, rand_color


# ---------------------------------------------------------------------------
# color_ramp2
# ---------------------------------------------------------------------------

class TestColorRamp2:
    def test_basic_rgb(self):
        f = color_ramp2([0, 1], ["#000000", "#FFFFFF"], space="RGB")
        assert f(0) == "#000000"
        assert f(1) == "#FFFFFF"

    def test_midpoint_rgb(self):
        f = color_ramp2([0, 1], ["#000000", "#FFFFFF"], space="RGB")
        mid = f(0.5)
        # Should be close to gray
        assert mid.startswith("#")
        # Parse and check roughly gray
        r = int(mid[1:3], 16)
        g = int(mid[3:5], 16)
        b = int(mid[5:7], 16)
        assert abs(r - 128) <= 2
        assert abs(g - 128) <= 2
        assert abs(b - 128) <= 2

    def test_lab_space(self):
        f = color_ramp2([0, 10], ["blue", "red"], space="LAB")
        result = f(5)
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_vector_input(self):
        f = color_ramp2([0, 1], ["black", "white"], space="RGB")
        result = f([0, 0.5, 1])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_ndarray_input(self):
        f = color_ramp2([0, 1], ["black", "white"], space="RGB")
        result = f(np.array([0.0, 1.0]))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_scalar_returns_string(self):
        f = color_ramp2([0, 1], ["black", "white"])
        result = f(0.5)
        assert isinstance(result, str)

    def test_breaks_colors_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            color_ramp2([0, 1, 2], ["black", "white"])

    def test_unsorted_breaks_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            color_ramp2([1, 0], ["black", "white"])

    def test_metadata(self):
        f = color_ramp2([0, 5, 10], ["blue", "white", "red"])
        assert hasattr(f, "breaks")
        assert hasattr(f, "colors")
        assert hasattr(f, "space")

    def test_three_colors(self):
        f = color_ramp2([-1, 0, 1], ["blue", "white", "red"])
        assert f(-1).startswith("#")
        assert f(0).startswith("#")
        assert f(1).startswith("#")

    def test_unsupported_space_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            color_ramp2([0, 1], ["black", "white"], space="HCL")


# ---------------------------------------------------------------------------
# add_transparency
# ---------------------------------------------------------------------------

class TestAddTransparency:
    def test_single_color(self):
        result = add_transparency("red", 0.5)
        assert isinstance(result, str)
        assert len(result) == 9  # #RRGGBBAA
        # Alpha should be ~128 for 0.5 transparency
        alpha = int(result[7:9], 16)
        assert abs(alpha - 128) <= 1

    def test_fully_opaque(self):
        result = add_transparency("red", 0.0)
        alpha = int(result[7:9], 16)
        assert alpha == 255

    def test_fully_transparent(self):
        result = add_transparency("red", 1.0)
        alpha = int(result[7:9], 16)
        assert alpha == 0

    def test_list_input(self):
        result = add_transparency(["red", "blue"], 0.5)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_hex_format(self):
        result = add_transparency("#FF0000", 0.3)
        assert re.match(r"^#[0-9A-F]{8}$", result)


# ---------------------------------------------------------------------------
# rand_color
# ---------------------------------------------------------------------------

class TestRandColor:
    def test_returns_correct_count(self):
        result = rand_color(5)
        assert len(result) == 5

    def test_hex_format(self):
        result = rand_color(3)
        for c in result:
            assert re.match(r"^#[0-9A-F]{6}$", c)

    def test_zero_colors(self):
        result = rand_color(0)
        assert result == []

    def test_luminosity_bright(self):
        result = rand_color(3, luminosity="bright")
        assert len(result) == 3

    def test_luminosity_dark(self):
        result = rand_color(3, luminosity="dark")
        assert len(result) == 3

    def test_luminosity_light(self):
        result = rand_color(3, luminosity="light")
        assert len(result) == 3
