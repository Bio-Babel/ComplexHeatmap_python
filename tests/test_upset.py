"""Tests for the upset module."""

import numpy as np
import pytest

from complexheatmap.upset import (
    CombMat,
    make_comb_mat,
    comb_degree,
    comb_name,
    comb_size,
    set_name,
    set_size,
    extract_comb,
    normalize_comb_mat,
    UpSet,
    upset_top_annotation,
    upset_right_annotation,
    upset_left_annotation,
)


# ======================================================================
# CombMat container
# ======================================================================

class TestCombMat:
    """Tests for the CombMat dataclass."""

    def test_basic_attributes(self):
        cm = CombMat(
            set_names=["A", "B"],
            comb_mat=np.array([[1, 0, 1], [0, 1, 1]]),
            comb_sizes=np.array([2, 3, 1]),
            set_sizes=np.array([3, 4]),
            mode="distinct",
        )
        assert cm.n_sets == 2
        assert cm.n_comb == 3
        assert cm.mode == "distinct"

    def test_repr(self):
        cm = CombMat(
            set_names=["X"],
            comb_mat=np.array([[1]]),
            comb_sizes=np.array([5]),
            set_sizes=np.array([5]),
            mode="intersect",
        )
        assert "CombMat" in repr(cm)
        assert "intersect" in repr(cm)


# ======================================================================
# make_comb_mat
# ======================================================================

class TestMakeCombMat:
    """Tests for make_comb_mat."""

    def test_from_dict_distinct(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}}
        m = make_comb_mat(sets, mode="distinct")
        assert isinstance(m, CombMat)
        assert m.n_sets == 3
        # Non-empty combinations (R default: filter out size-0 entries)
        assert m.n_comb == 5

    def test_from_dict_intersect(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        m = make_comb_mat(sets, mode="intersect")
        assert m.n_sets == 2
        assert m.n_comb == 3

    def test_from_dict_union(self):
        sets = {"A": {1, 2}, "B": {3, 4}}
        m = make_comb_mat(sets, mode="union")
        assert m.n_sets == 2
        assert m.n_comb == 3

    def test_from_array(self):
        arr = np.array([[1, 0], [1, 1], [0, 1]])
        m = make_comb_mat(arr, mode="distinct")
        assert m.n_sets == 2
        assert m.n_comb == 3

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            make_comb_mat({"A": {1}}, mode="bad")

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            make_comb_mat("not a dict or array")

    def test_invalid_array_dim(self):
        with pytest.raises(ValueError, match="2-D"):
            make_comb_mat(np.array([1, 2, 3]))

    def test_min_set_size(self):
        sets = {"A": {1, 2, 3}, "B": {4}}
        m = make_comb_mat(sets, min_set_size=2)
        assert m.n_sets == 1
        assert m.set_names == ["A"]

    def test_top_n_sets(self):
        sets = {"A": {1, 2, 3}, "B": {4, 5}, "C": {6}}
        m = make_comb_mat(sets, top_n_sets=2)
        assert m.n_sets == 2

    def test_complement_size(self):
        sets = {"A": {1, 2}, "B": {2, 3}}
        m = make_comb_mat(sets, complement_size=10)
        # 3 normal + 1 complement
        assert m.n_comb == 4

    def test_custom_value_fun(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        m = make_comb_mat(sets, value_fun=lambda s: len(s) * 2)
        # Sizes should be doubled
        for i in range(m.n_comb):
            # All sizes should be even
            assert m.comb_sizes[i] % 2 == 0

    def test_universal_set(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        m = make_comb_mat(sets, universal_set={2, 3})
        # Only elements 2 and 3 are in the universe
        total = m.comb_sizes.sum()
        assert total <= 2

    def test_distinct_sizes_sum_to_universe(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}}
        m = make_comb_mat(sets, mode="distinct")
        # In distinct mode, sizes should sum to universe size
        assert m.comb_sizes.sum() == 5  # {1,2,3,4,5}


# ======================================================================
# Accessors
# ======================================================================

class TestAccessors:
    """Tests for accessor functions."""

    @pytest.fixture()
    def sample_comb_mat(self) -> CombMat:
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}}
        return make_comb_mat(sets, mode="distinct")

    def test_comb_degree(self, sample_comb_mat):
        deg = comb_degree(sample_comb_mat)
        assert len(deg) == sample_comb_mat.n_comb
        assert all(d >= 1 for d in deg)

    def test_comb_name(self, sample_comb_mat):
        names = comb_name(sample_comb_mat)
        assert len(names) == sample_comb_mat.n_comb
        # Each name should be a binary string of length n_sets
        for n in names:
            assert len(n) == sample_comb_mat.n_sets
            assert all(c in ("0", "1") for c in n)

    def test_comb_size(self, sample_comb_mat):
        sizes = comb_size(sample_comb_mat)
        assert len(sizes) == sample_comb_mat.n_comb
        assert sizes.dtype == int

    def test_set_name(self, sample_comb_mat):
        names = set_name(sample_comb_mat)
        assert names == ["A", "B", "C"]

    def test_set_size(self, sample_comb_mat):
        sizes = set_size(sample_comb_mat)
        assert len(sizes) == 3
        assert list(sizes) == [3, 3, 3]

    def test_extract_comb(self, sample_comb_mat):
        names = comb_name(sample_comb_mat)
        # Find the "A only" combination (100)
        for cn in names:
            elems = extract_comb(sample_comb_mat, cn)
            assert isinstance(elems, set)

    def test_extract_comb_not_found(self, sample_comb_mat):
        with pytest.raises(KeyError):
            extract_comb(sample_comb_mat, "999")


# ======================================================================
# normalize_comb_mat
# ======================================================================

class TestNormalizeCombMat:
    """Tests for normalize_comb_mat."""

    def test_full_normalization(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        m = make_comb_mat(sets)
        normed = normalize_comb_mat(m, full=True)
        assert abs(normed.comb_sizes.sum() - 1.0) < 1e-10

    def test_per_set_normalization(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        m = make_comb_mat(sets)
        normed = normalize_comb_mat(m, full=False)
        assert normed.n_comb == m.n_comb
        assert normed.mode == m.mode

    def test_returns_new_object(self):
        sets = {"A": {1, 2}, "B": {2, 3}}
        m = make_comb_mat(sets)
        normed = normalize_comb_mat(m)
        assert normed is not m


# ======================================================================
# Annotation helpers
# ======================================================================

class TestAnnotationHelpers:
    """Tests for upset annotation factory functions."""

    @pytest.fixture()
    def sample_comb_mat(self) -> CombMat:
        return make_comb_mat({"A": {1, 2, 3}, "B": {2, 3, 4}})

    def test_upset_top_annotation(self, sample_comb_mat):
        anno = upset_top_annotation(sample_comb_mat)
        assert anno is not None

    def test_upset_right_annotation(self, sample_comb_mat):
        anno = upset_right_annotation(sample_comb_mat)
        assert anno is not None

    def test_upset_left_annotation(self, sample_comb_mat):
        anno = upset_left_annotation(sample_comb_mat)
        assert anno is not None


# ======================================================================
# UpSet plot construction
# ======================================================================

class TestUpSet:
    """Tests for UpSet plot creation."""

    @pytest.fixture()
    def sample_comb_mat(self) -> CombMat:
        return make_comb_mat(
            {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}},
            mode="distinct",
        )

    def test_creates_heatmap(self, sample_comb_mat):
        from complexheatmap.heatmap import Heatmap
        ht = UpSet(sample_comb_mat)
        assert isinstance(ht, Heatmap)

    def test_custom_comb_order(self, sample_comb_mat):
        order = list(range(sample_comb_mat.n_comb))[::-1]
        ht = UpSet(sample_comb_mat, comb_order=order)
        assert ht is not None

    def test_custom_set_order(self, sample_comb_mat):
        ht = UpSet(sample_comb_mat, set_order=[2, 1, 0])
        assert ht is not None

    def test_custom_comb_col(self, sample_comb_mat):
        cols = ["red"] * sample_comb_mat.n_comb
        ht = UpSet(sample_comb_mat, comb_col=cols)
        assert ht is not None

    def test_no_annotations(self, sample_comb_mat):
        from complexheatmap.heatmap_annotation import HeatmapAnnotation
        ht = UpSet(
            sample_comb_mat,
            top_annotation=None,
            right_annotation=None,
        )
        # Should use defaults
        assert ht is not None

    def test_show_comb_name(self, sample_comb_mat):
        ht = UpSet(sample_comb_mat, show_comb_name=True)
        assert ht.show_column_names is True

    def test_show_row_names_false(self, sample_comb_mat):
        ht = UpSet(sample_comb_mat, show_row_names=False)
        assert ht.show_row_names is False

    def test_name_kwarg(self, sample_comb_mat):
        ht = UpSet(sample_comb_mat, name="my_upset")
        assert ht.name == "my_upset"
