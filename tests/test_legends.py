"""Tests for complexheatmap.legends."""

import numpy as np
import pytest

import grid_py

from complexheatmap.legends import Legend, Legends, pack_legend
from complexheatmap._color import color_ramp2


# ---------------------------------------------------------------------------
# Legend factory - Discrete
# ---------------------------------------------------------------------------

class TestDiscreteLegend:
    def test_basic_creation(self):
        lgd = Legend(
            at=[1, 2, 3],
            labels=["a", "b", "c"],
            legend_gp={"fill": ["red", "green", "blue"]},
            title="test",
        )
        assert isinstance(lgd, Legends)
        assert lgd.type == "single_legend"
        assert lgd.n == 1
        assert lgd.grob is not None

    def test_no_title(self):
        lgd = Legend(
            at=[1, 2],
            labels=["x", "y"],
            legend_gp={"fill": ["red", "blue"]},
        )
        assert lgd.type == "single_legend_no_title"

    def test_empty_title(self):
        lgd = Legend(
            at=[1],
            labels=["a"],
            legend_gp={"fill": ["red"]},
            title="",
        )
        assert lgd.type == "single_legend_no_title"

    def test_labels_only(self):
        lgd = Legend(labels=["A", "B", "C"], legend_gp={"fill": ["red", "green", "blue"]})
        assert lgd.n == 1

    def test_grob_is_gtree(self):
        lgd = Legend(
            at=[1, 2],
            labels=["a", "b"],
            legend_gp={"fill": ["red", "blue"]},
            title="T",
        )
        assert isinstance(lgd.grob, grid_py.GTree)

    def test_points_type(self):
        lgd = Legend(
            at=[1, 2],
            labels=["a", "b"],
            legend_gp={"fill": ["red", "blue"]},
            type="points",
            title="Points",
        )
        assert isinstance(lgd, Legends)

    def test_lines_type(self):
        lgd = Legend(
            at=[1, 2],
            labels=["a", "b"],
            legend_gp={"fill": ["red", "blue"]},
            type="lines",
            title="Lines",
        )
        assert isinstance(lgd, Legends)

    def test_multirow_layout(self):
        lgd = Legend(
            at=list(range(6)),
            labels=[f"L{i}" for i in range(6)],
            legend_gp={"fill": ["red"] * 6},
            nrow=2,
            ncol=3,
            title="Multi",
        )
        assert isinstance(lgd, Legends)

    def test_repr_single(self):
        lgd = Legend(at=[1], labels=["a"], legend_gp={"fill": ["red"]}, title="T")
        assert "single legend" in repr(lgd)


# ---------------------------------------------------------------------------
# Legend factory - Continuous
# ---------------------------------------------------------------------------

class TestContinuousLegend:
    def test_vertical_colorbar(self):
        col_fun = color_ramp2([0, 1], ["white", "red"])
        lgd = Legend(
            col_fun=col_fun,
            at=[0, 0.5, 1],
            title="Colorbar",
        )
        assert isinstance(lgd, Legends)
        assert lgd.type == "single_legend"

    def test_horizontal_colorbar(self):
        col_fun = color_ramp2([0, 10], ["blue", "yellow"])
        lgd = Legend(
            col_fun=col_fun,
            at=[0, 5, 10],
            direction="horizontal",
            title="H-bar",
        )
        assert isinstance(lgd, Legends)

    def test_auto_breaks(self):
        col_fun = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        lgd = Legend(col_fun=col_fun, title="Auto")
        assert isinstance(lgd, Legends)

    def test_border(self):
        col_fun = color_ramp2([0, 1], ["white", "red"])
        lgd = Legend(
            col_fun=col_fun,
            at=[0, 1],
            title="Border",
            border="black",
        )
        assert isinstance(lgd, Legends)


# ---------------------------------------------------------------------------
# Legends container
# ---------------------------------------------------------------------------

class TestLegends:
    def test_empty(self):
        obj = Legends()
        assert obj.grob is None
        assert obj.n == 1

    def test_repr(self):
        obj = Legends(type="pack_legend", n=3)
        assert "3" in repr(obj)


# ---------------------------------------------------------------------------
# pack_legend
# ---------------------------------------------------------------------------

class TestPackLegend:
    def test_pack_two(self):
        lgd1 = Legend(
            at=[1, 2],
            labels=["a", "b"],
            legend_gp={"fill": ["red", "blue"]},
            title="L1",
        )
        lgd2 = Legend(
            at=[3, 4],
            labels=["c", "d"],
            legend_gp={"fill": ["green", "yellow"]},
            title="L2",
        )
        packed = pack_legend(lgd1, lgd2)
        assert isinstance(packed, Legends)
        assert packed.type == "pack_legend"
        assert packed.n == 2

    def test_pack_empty(self):
        packed = pack_legend()
        assert packed.n == 0

    def test_pack_direction(self):
        lgd = Legend(
            at=[1],
            labels=["a"],
            legend_gp={"fill": ["red"]},
            title="D",
        )
        packed = pack_legend(lgd, direction="horizontal")
        assert packed.direction == "horizontal"

    def test_pack_invalid_type_raises(self):
        with pytest.raises(TypeError):
            pack_legend("not_a_legend")  # type: ignore[arg-type]
