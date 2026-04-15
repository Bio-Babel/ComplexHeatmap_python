"""Tests for complexheatmap._data data loaders."""

import numpy as np
import pandas as pd
import pytest

from complexheatmap._data import (
    load_gene_expression,
    load_measles,
    load_tcga_oncoprint,
    load_sample_order,
    load_dmr_summary,
    load_color_space_comparison,
    load_genome_level_data,
    load_meth_data,
    load_mouse_scrnaseq,
    load_mouse_cell_cycle_genes,
    load_mouse_ribonucleoprotein_genes,
    load_random_meth_expr_data,
)


class TestLoadGeneExpression:
    def test_returns_dataframe(self):
        df = load_gene_expression()
        assert isinstance(df, pd.DataFrame)

    def test_has_annotation_columns(self):
        df = load_gene_expression()
        for col in ("length", "type", "chr"):
            assert col in df.columns

    def test_has_rows(self):
        df = load_gene_expression()
        assert df.shape[0] > 0


class TestLoadMeasles:
    def test_returns_tuple(self):
        result = load_measles()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_matrix_shape(self):
        mat, rows, cols = load_measles()
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (len(rows), len(cols))

    def test_values_numeric(self):
        mat, _, _ = load_measles()
        assert mat.dtype == float


class TestLoadTcgaOncoprint:
    def test_returns_dataframe(self):
        df = load_tcga_oncoprint()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0


class TestLoadSampleOrder:
    def test_returns_list(self):
        result = load_sample_order()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, str) for s in result)


class TestLoadDmrSummary:
    def test_returns_dict(self):
        result = load_dmr_summary()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = load_dmr_summary()
        for key in ("label", "mean_meth", "n_gr", "n_corr", "dist_tss",
                     "gene_anno", "cgi_anno", "mat_enrich_gf"):
            assert key in result, f"Missing key: {key}"

    def test_label_is_list(self):
        result = load_dmr_summary()
        assert isinstance(result["label"], list)
        assert len(result["label"]) > 0


class TestLoadColorSpaceComparison:
    def test_returns_dict(self):
        result = load_color_space_comparison()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_values_are_lists_of_strings(self):
        result = load_color_space_comparison()
        for key, val in result.items():
            assert isinstance(val, list)


class TestLoadGenomeLevelData:
    def test_returns_dict(self):
        result = load_genome_level_data()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = load_genome_level_data()
        for key in ("num_mat", "char_mat", "chr", "chr_level",
                     "subgroup", "at", "labels", "v"):
            assert key in result, f"Missing key: {key}"

    def test_chr_is_list(self):
        result = load_genome_level_data()
        assert isinstance(result["chr"], list)
        assert len(result["chr"]) > 0


class TestLoadMethData:
    def test_returns_dict(self):
        result = load_meth_data()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = load_meth_data()
        for key in ("type", "mat_meth", "mat_expr", "direction",
                     "cor_pvalue", "gene_type"):
            assert key in result, f"Missing key: {key}"


class TestLoadMouseScrnaseq:
    def test_returns_dataframe(self):
        df = load_mouse_scrnaseq()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0


class TestLoadMouseCellCycleGenes:
    def test_returns_list(self):
        result = load_mouse_cell_cycle_genes()
        assert isinstance(result, list)
        assert len(result) > 0


class TestLoadMouseRibonucleoproteinGenes:
    def test_returns_list(self):
        result = load_mouse_ribonucleoprotein_genes()
        assert isinstance(result, list)
        assert len(result) > 0


class TestLoadRandomMethExprData:
    def test_returns_dict(self):
        result = load_random_meth_expr_data()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = load_random_meth_expr_data()
        for key in ("mat_meth", "mat_expr", "direction", "cor_pvalue"):
            assert key in result, f"Missing key: {key}"
