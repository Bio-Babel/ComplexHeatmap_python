"""Data loaders for ComplexHeatmap example datasets.

Each function reads pre-bundled CSV / TSV / TXT files from the
``resources/`` directory shipped alongside this package.

R source correspondence
-----------------------
These loaders provide access to datasets that ship with the R
``ComplexHeatmap`` package (and companion vignettes).  The original R
objects were exported to CSV/TXT via the data-export pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "load_gene_expression",
    "load_measles",
    "load_tcga_oncoprint",
    "load_sample_order",
    "load_dmr_summary",
    "load_color_space_comparison",
    "load_genome_level_data",
    "load_meth_data",
    "load_mouse_scrnaseq",
    "load_mouse_cell_cycle_genes",
    "load_mouse_ribonucleoprotein_genes",
    "load_random_meth_expr_data",
]

_RESOURCES = Path(__file__).parent / "resources"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(name: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV from the resources directory.

    Parameters
    ----------
    name : str
        File name relative to the ``resources/`` directory.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(_RESOURCES / name, index_col=0, **kwargs)


def _read_csv_no_index(name: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV without treating first column as index.

    Parameters
    ----------
    name : str
        File name relative to the ``resources/`` directory.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(_RESOURCES / name, **kwargs)


def _read_lines(name: str) -> List[str]:
    """Read non-empty lines from a text file.

    Parameters
    ----------
    name : str
        File name relative to the ``resources/`` directory.

    Returns
    -------
    list of str
    """
    return [
        line.strip()
        for line in (_RESOURCES / name).read_text().splitlines()
        if line.strip()
    ]


def _read_single_column_csv(name: str, col: str = "") -> List[str]:
    """Read a single-column CSV and return its values as a list of strings.

    Parameters
    ----------
    name : str
        File name relative to the ``resources/`` directory.
    col : str, optional
        Column name to extract.  If empty, uses the first column.

    Returns
    -------
    list of str
    """
    df = pd.read_csv(_RESOURCES / name)
    if col:
        return df[col].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_gene_expression() -> pd.DataFrame:
    """Load the example gene-expression matrix.

    Returns
    -------
    pandas.DataFrame
        Genes in rows, samples in columns, with additional ``length``,
        ``type``, and ``chr`` annotation columns.
    """
    return _read_csv("gene_expression.csv")


def load_measles() -> Tuple[np.ndarray, List[str], List[str]]:
    """Load the US measles incidence data.

    Returns
    -------
    matrix : numpy.ndarray
        Incidence counts, shape ``(n_states, n_years)``.
    row_names : list of str
        State names.
    col_names : list of str
        Year labels.
    """
    df = _read_csv("measles.csv")
    return df.values.astype(float), list(df.index), list(df.columns)


def load_tcga_oncoprint() -> pd.DataFrame:
    """Load the TCGA lung oncoprint alteration matrix.

    Returns
    -------
    pandas.DataFrame
        Genes in rows, samples in columns; entries encode mutation types.
    """
    return _read_csv("tcga_lung.csv")


def load_sample_order() -> List[str]:
    """Load the predefined TCGA sample ordering.

    Returns
    -------
    list of str
        Sample identifiers.
    """
    return _read_lines("sample_order.txt")


def load_dmr_summary() -> Dict[str, Any]:
    """Load the differentially methylated region summary data.

    Returns
    -------
    dict
        Keys and types:

        - ``label`` : list of str
        - ``mean_meth`` : pandas.DataFrame
        - ``n_gr`` : dict mapping name to int
        - ``n_corr`` : pandas.DataFrame
        - ``dist_tss`` : pandas.DataFrame
        - ``gene_anno`` : pandas.DataFrame
        - ``cgi_anno`` : pandas.DataFrame
        - ``mat_enrich_gf`` : pandas.DataFrame
        - ``mat_pct_st`` : pandas.DataFrame
        - ``mat_enrich_st`` : pandas.DataFrame
    """
    n_gr_df = pd.read_csv(_RESOURCES / "dmr_n_gr.csv")
    n_gr = dict(zip(n_gr_df["name"].tolist(), n_gr_df["value"].astype(int).tolist()))

    return {
        "label": _read_lines("dmr_label.txt"),
        "mean_meth": _read_csv("dmr_mean_meth.csv"),
        "n_gr": n_gr,
        "n_corr": _read_csv("dmr_n_corr.csv"),
        "dist_tss": _read_csv("dmr_dist_tss.csv"),
        "gene_anno": _read_csv("dmr_gene_anno.csv"),
        "cgi_anno": _read_csv("dmr_cgi_anno.csv"),
        "mat_enrich_gf": _read_csv("dmr_mat_enrich_gf.csv"),
        "mat_pct_st": _read_csv("dmr_mat_pct_st.csv"),
        "mat_enrich_st": _read_csv("dmr_mat_enrich_st.csv"),
    }


def load_color_space_comparison() -> Dict[str, List[str]]:
    """Load precomputed colour-ramp comparisons across colour spaces.

    Returns
    -------
    dict of list of str
        Keys are colour-space names (e.g. ``"RGB"``, ``"LAB"``), values
        are lists of hex colour strings.
    """
    df = pd.read_csv(_RESOURCES / "color_space_comparison.csv")
    return {col: df[col].tolist() for col in df.columns}


def load_genome_level_data() -> Dict[str, Any]:
    """Load genome-level visualisation data.

    Returns
    -------
    dict
        Keys and types:

        - ``num_mat`` : pandas.DataFrame  -- numeric matrix
        - ``char_mat`` : pandas.DataFrame -- character matrix
        - ``chr`` : list of str           -- chromosome per row
        - ``chr_level`` : list of str     -- ordered chromosome levels
        - ``subgroup`` : list of str      -- sample subgroup labels
        - ``at`` : list of int            -- label positions
        - ``labels`` : list of str        -- gene labels
        - ``v`` : pandas.DataFrame        -- additional numeric data
    """
    markers = pd.read_csv(_RESOURCES / "genome_level_markers.csv")
    chr_df = pd.read_csv(_RESOURCES / "genome_level_chr.csv")
    subgroup_df = pd.read_csv(_RESOURCES / "genome_level_subgroup.csv")

    return {
        "num_mat": _read_csv("genome_level_num_mat.csv"),
        "char_mat": _read_csv("genome_level_char_mat.csv"),
        "chr": chr_df["chr"].tolist(),
        "chr_level": _read_lines("genome_level_chr_level.txt"),
        "subgroup": subgroup_df["subgroup"].tolist(),
        "at": markers["at"].tolist(),
        "labels": markers["labels"].tolist(),
        "v": _read_csv("genome_level_v.csv"),
    }


def load_meth_data() -> Dict[str, Any]:
    """Load methylation vs expression comparison data.

    Returns
    -------
    dict
        Keys and types:

        - ``type`` : list of str
        - ``mat_meth`` : pandas.DataFrame
        - ``mat_expr`` : pandas.DataFrame
        - ``direction`` : list of str
        - ``cor_pvalue`` : list of float
        - ``gene_type`` : list of str
        - ``anno_gene`` : list of str
        - ``dist`` : list of float
        - ``anno_enhancer`` : pandas.DataFrame
    """
    anno = pd.read_csv(_RESOURCES / "meth_anno.csv")
    type_df = pd.read_csv(_RESOURCES / "meth_type.csv")
    return {
        "type": type_df["type"].tolist(),
        "mat_meth": _read_csv("meth_mat_meth.csv"),
        "mat_expr": _read_csv("meth_mat_expr.csv"),
        "direction": anno["direction"].tolist(),
        "cor_pvalue": anno["cor_pvalue"].tolist(),
        "gene_type": anno["gene_type"].tolist(),
        "anno_gene": anno["anno_gene"].tolist(),
        "dist": anno["dist"].tolist(),
        "anno_enhancer": _read_csv("meth_anno_enhancer.csv"),
    }


def load_mouse_scrnaseq() -> pd.DataFrame:
    """Load the mouse single-cell RNA-seq expression matrix.

    Returns
    -------
    pandas.DataFrame
        Genes in rows, cells in columns.
    """
    return pd.read_csv(
        _RESOURCES / "mouse_scRNAseq_corrected.txt",
        sep="\t",
        index_col=0,
    )


def load_mouse_cell_cycle_genes() -> List[str]:
    """Load mouse cell-cycle gene names.

    Returns
    -------
    list of str
    """
    df = pd.read_csv(_RESOURCES / "mouse_cell_cycle_gene.csv")
    return df.iloc[:, 0].tolist()


def load_mouse_ribonucleoprotein_genes() -> List[str]:
    """Load mouse ribonucleoprotein complex gene names.

    Returns
    -------
    list of str
    """
    df = pd.read_csv(_RESOURCES / "mouse_ribonucleoprotein.csv")
    return df.iloc[:, 0].tolist()


def load_random_meth_expr_data() -> Dict[str, Any]:
    """Load random methylation / expression example data.

    Returns
    -------
    dict
        Keys and types:

        - ``anno`` : pandas.DataFrame
        - ``mat_meth`` : pandas.DataFrame
        - ``mat_expr`` : pandas.DataFrame
        - ``tss_dist`` : list of float
        - ``anno_col`` : pandas.DataFrame  -- colour mapping per annotation column
        - ``direction`` : list of str
        - ``cor_pvalue`` : list of float
        - ``gene_type`` : list of str
        - ``anno_gene`` : list of str
        - ``anno_states`` : pandas.DataFrame (if available)
    """
    anno = pd.read_csv(_RESOURCES / "random_anno.csv")
    result: Dict[str, Any] = {
        "anno": _read_csv_no_index("random_anno_col.csv"),
        "mat_meth": _read_csv("random_mat_meth.csv"),
        "mat_expr": _read_csv("random_mat_expr.csv"),
        "tss_dist": anno["tss_dist"].tolist(),
        "anno_col": _read_csv_no_index("random_anno_col.csv"),
        "direction": anno["direction"].tolist(),
        "cor_pvalue": anno["cor_pvalue"].tolist(),
        "gene_type": anno["gene_type"].tolist(),
    }
    # anno_gene may not be present in all exports
    if "anno_gene" in anno.columns:
        result["anno_gene"] = anno["anno_gene"].tolist()
    return result
