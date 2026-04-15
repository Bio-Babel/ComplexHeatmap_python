"""Tests for complexheatmap._globals option system."""

import pytest

from complexheatmap._globals import ht_opt, reset_ht_opt


class TestHtOptGet:
    def setup_method(self):
        reset_ht_opt()

    def test_get_single_option(self):
        assert ht_opt("verbose") is False

    def test_get_all_options(self):
        opts = ht_opt()
        assert isinstance(opts, dict)
        assert "verbose" in opts

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown option"):
            ht_opt("nonexistent_option")


class TestHtOptSet:
    def setup_method(self):
        reset_ht_opt()

    def test_set_option(self):
        ht_opt(verbose=True)
        assert ht_opt("verbose") is True

    def test_set_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown option"):
            ht_opt(nonexistent_option=42)

    def test_mixed_args_raises(self):
        with pytest.raises(TypeError):
            ht_opt("verbose", verbose=True)


class TestHtOptContextManager:
    def setup_method(self):
        reset_ht_opt()

    def test_context_restores(self):
        assert ht_opt("verbose") is False
        with ht_opt(verbose=True):
            assert ht_opt("verbose") is True
        assert ht_opt("verbose") is False

    def test_bare_set_persists(self):
        ht_opt(verbose=True)
        assert ht_opt("verbose") is True


class TestResetHtOpt:
    def test_reset(self):
        ht_opt(verbose=True)
        reset_ht_opt()
        assert ht_opt("verbose") is False

    def test_reset_restores_defaults(self):
        ht_opt(legend_grid_width=99)
        reset_ht_opt()
        assert ht_opt("legend_grid_width") == 4


class TestHtOptDeepCopy:
    def setup_method(self):
        reset_ht_opt()

    def test_get_returns_copy(self):
        """Mutating the returned value should not affect the stored option."""
        gp = ht_opt("legend_title_gp")
        gp["fontsize"] = 999
        assert ht_opt("legend_title_gp")["fontsize"] == 10
