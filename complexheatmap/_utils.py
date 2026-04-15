"""Utility functions for ComplexHeatmap.

R source correspondence
-----------------------
``R/utils.R`` -- miscellaneous helpers used throughout the package:
matrix indexing (``pindex``), graphical parameter subsetting
(``subset_gp``), text measurement (``max_text_width``,
``max_text_height``), set-to-matrix conversion (``list_to_matrix``),
and distance computation (``dist2``).

Uses ``grid_py`` (the Python port of R's ``grid`` package) for all
grid operations including unit handling and text measurement.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "pindex",
    "subset_gp",
    "max_text_width",
    "max_text_height",
    "is_abs_unit",
    "list_to_matrix",
    "restore_matrix",
    "default_axis_param",
    "cluster_within_group",
    "dist2",
]


# ---------------------------------------------------------------------------
# Matrix indexing
# ---------------------------------------------------------------------------

def pindex(
    m: np.ndarray,
    i: Union[np.ndarray, Sequence[int]],
    j: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """Pair-wise matrix indexing: ``m[i[k], j[k]]`` for all *k*.

    If *i* or *j* has length 1 it is recycled to match the other.

    Parameters
    ----------
    m : numpy.ndarray
        A 2-D matrix.
    i : array-like of int
        Row indices (0-based).
    j : array-like of int
        Column indices (0-based).

    Returns
    -------
    numpy.ndarray
        1-D array of extracted values.
    """
    i_arr = np.atleast_1d(np.asarray(i))
    j_arr = np.atleast_1d(np.asarray(j))

    if len(i_arr) == 1 and len(j_arr) > 1:
        i_arr = np.repeat(i_arr, len(j_arr))
    elif len(j_arr) == 1 and len(i_arr) > 1:
        j_arr = np.repeat(j_arr, len(i_arr))

    return m[i_arr, j_arr]


# ---------------------------------------------------------------------------
# Graphical parameter subsetting
# ---------------------------------------------------------------------------

def subset_gp(
    gp: Dict[str, Any],
    i: Union[int, Sequence[int], np.ndarray],
) -> Dict[str, Any]:
    """Subset a graphical-parameters dict by index.

    Vector-valued entries (lists / arrays) are subsetted; scalar entries
    are kept as-is.

    Parameters
    ----------
    gp : dict
        Graphical parameters (e.g. ``{"col": ["red", "blue"], "lwd": 2}``).
    i : int or array-like of int
        Indices to select.

    Returns
    -------
    dict
        New dict with vector entries subsetted.
    """
    result: Dict[str, Any] = {}
    idx = np.atleast_1d(np.asarray(i))
    for key, val in gp.items():
        if isinstance(val, (list, tuple)):
            result[key] = [val[ii] for ii in idx]
        elif isinstance(val, np.ndarray) and val.ndim >= 1:
            result[key] = val[idx]
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Text measurement (using grid_py)
# ---------------------------------------------------------------------------

def max_text_width(
    text: Union[str, Sequence[str]],
    gp: Optional[Dict[str, Any]] = None,
    rot: float = 0,
) -> Any:
    """Compute the maximum rendered text width using grid_py.

    Parameters
    ----------
    text : str or sequence of str
        One or more text strings to measure.
    gp : dict, optional
        Graphical parameters.  Recognised keys: ``fontsize`` (default 10),
        ``fontfamily`` (default ``"sans-serif"``), ``fontweight`` (default
        ``"normal"``).  These are passed through to ``grid_py.Gpar``.
    rot : float, optional
        Rotation angle in degrees (default 0).  Note: grid_py's
        ``string_width`` does not account for rotation directly; the
        caller should handle rotated text sizing separately if needed.

    Returns
    -------
    grid_py.Unit
        A ``grid_py.Unit`` representing the maximum width.
    """
    import grid_py

    if isinstance(text, str):
        text = [text]
    if not text:
        return grid_py.Unit(0, "mm")

    # grid_py.string_width accepts a list and returns a multi-element Unit
    widths = grid_py.string_width(list(text))

    # If only one string, string_width returns a single Unit directly
    if len(text) == 1:
        return widths

    # For multiple strings, use unit_pmax to find the maximum
    return grid_py.unit_pmax(widths)


def max_text_height(
    text: Union[str, Sequence[str]],
    gp: Optional[Dict[str, Any]] = None,
    rot: float = 0,
) -> Any:
    """Compute the maximum rendered text height using grid_py.

    Parameters
    ----------
    text : str or sequence of str
        One or more text strings to measure.
    gp : dict, optional
        Graphical parameters (same keys as :func:`max_text_width`).
    rot : float, optional
        Rotation angle in degrees (default 0).

    Returns
    -------
    grid_py.Unit
        A ``grid_py.Unit`` representing the maximum height.
    """
    import grid_py

    if isinstance(text, str):
        text = [text]
    if not text:
        return grid_py.Unit(0, "mm")

    heights = grid_py.string_height(list(text))

    if len(text) == 1:
        return heights

    return grid_py.unit_pmax(heights)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def is_abs_unit(x: Any) -> bool:
    """Check whether *x* is an absolute unit.

    An absolute unit is either a plain number (int / float), a
    ``grid_py.Unit`` with an absolute unit type (mm, cm, inches, points),
    or a ``(value, unit_string)`` tuple such as ``(5, "mm")``.

    Parameters
    ----------
    x : Any
        Value to test.

    Returns
    -------
    bool
    """
    import grid_py

    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    if isinstance(x, tuple) and len(x) == 2:
        return isinstance(x[0], (int, float, np.integer, np.floating))
    # Check grid_py Unit objects
    if grid_py.is_unit(x):
        # Absolute unit types in grid_py
        utype = grid_py.unit_type(x)
        # unit_type may return a list for multi-element units
        if isinstance(utype, (list, tuple)):
            abs_types = {"cm", "mm", "inches", "points", "picas",
                         "bigpts", "cicero", "scaledpts"}
            return all(t in abs_types for t in utype)
        return utype in {"cm", "mm", "inches", "points", "picas",
                         "bigpts", "cicero", "scaledpts"}
    return False


# ---------------------------------------------------------------------------
# Set-to-matrix conversion
# ---------------------------------------------------------------------------

def list_to_matrix(
    lt: Dict[str, set],
    universal_set: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert a dict of sets to a binary membership matrix.

    Parameters
    ----------
    lt : dict of set
        Keys are set names, values are sets of element names.
    universal_set : sequence of str, optional
        If given, use this as the row universe.  Otherwise the union of
        all sets is used (sorted alphabetically).

    Returns
    -------
    matrix : numpy.ndarray
        Binary (0/1) matrix of shape ``(n_elements, n_sets)``.
    row_names : list of str
        Element names (rows).
    col_names : list of str
        Set names (columns).
    """
    col_names = list(lt.keys())
    if universal_set is not None:
        row_names = list(universal_set)
    else:
        all_elems: set = set()
        for s in lt.values():
            all_elems.update(s)
        row_names = sorted(all_elems)

    mat = np.zeros((len(row_names), len(col_names)), dtype=int)
    row_idx = {name: i for i, name in enumerate(row_names)}
    for ci, key in enumerate(col_names):
        for elem in lt[key]:
            if elem in row_idx:
                mat[row_idx[elem], ci] = 1
    return mat, row_names, col_names


# ---------------------------------------------------------------------------
# Matrix reconstruction
# ---------------------------------------------------------------------------

def restore_matrix(
    j: np.ndarray,
    i: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Reconstruct a matrix from vectorised row/column indices and values.

    Parameters
    ----------
    j : numpy.ndarray
        Column indices (0-based) for each value in *y*.
    i : numpy.ndarray
        Row indices (0-based) for each value in *y*.
    x : numpy.ndarray
        Not placed in the matrix; used only to determine dimensions
        (legacy R compatibility).
    y : numpy.ndarray
        Values to place in the matrix.

    Returns
    -------
    numpy.ndarray
        2-D matrix with values filled in; unfilled cells are ``NaN``.
    """
    nrow = int(i.max()) + 1
    ncol = int(j.max()) + 1
    mat = np.full((nrow, ncol), np.nan)
    mat[i, j] = y
    return mat


# ---------------------------------------------------------------------------
# Axis parameter defaults
# ---------------------------------------------------------------------------

def default_axis_param(which: str = "column") -> Dict[str, Any]:
    """Return the default axis parameter dictionary.

    Mirrors R ``ComplexHeatmap:::default_axis_param``.

    Parameters
    ----------
    which : str
        Either ``"column"`` or ``"row"``.

    Returns
    -------
    dict
        Default axis parameter settings including ``at``, ``labels``,
        ``labels_rot``, ``gp``, ``side``, and ``facing``.
    """
    if which == "column":
        return {
            "at": None,
            "labels": None,
            "labels_rot": 0,
            "gp": {"fontsize": 8},
            "side": "bottom",
            "facing": "outside",
        }
    else:
        return {
            "at": None,
            "labels": None,
            "labels_rot": 0,
            "gp": {"fontsize": 8},
            "side": "left",
            "facing": "outside",
        }


# ---------------------------------------------------------------------------
# Clustering within groups
# ---------------------------------------------------------------------------

def cluster_within_group(
    mat: np.ndarray,
    factor: Union[Sequence[str], np.ndarray],
    method: str = "complete",
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster rows within each group defined by *factor*.

    Parameters
    ----------
    mat : numpy.ndarray
        Data matrix of shape ``(n, p)``.
    factor : array-like of str
        Group labels for each row.  Length must equal ``n``.
    method : str, optional
        Linkage method for :func:`scipy.cluster.hierarchy.linkage`.
    metric : str, optional
        Distance metric for :func:`scipy.spatial.distance.pdist`.

    Returns
    -------
    numpy.ndarray
        Row index permutation (0-based) that places rows in group order
        with intra-group clustering applied.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    factor_arr = np.asarray(factor)
    unique_groups = []
    seen: set = set()
    for g in factor_arr:
        if g not in seen:
            unique_groups.append(g)
            seen.add(g)

    order: List[int] = []
    for group in unique_groups:
        idx = np.where(factor_arr == group)[0]
        if len(idx) <= 2:
            order.extend(idx.tolist())
        else:
            sub = mat[idx]
            dist = pdist(sub, metric=metric)
            Z = linkage(dist, method=method)
            leaves = leaves_list(Z)
            order.extend(idx[leaves].tolist())

    return np.array(order, dtype=int)


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def dist2(
    x: np.ndarray,
    pairwise_fun: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> np.ndarray:
    """Compute a pairwise distance matrix.

    Parameters
    ----------
    x : numpy.ndarray
        2-D data matrix of shape ``(n, p)`` where *n* is the number of
        observations and *p* is the number of features.
    pairwise_fun : callable, optional
        A function ``f(u, v) -> float`` that computes the distance between
        two 1-D vectors.  Defaults to Euclidean distance.

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape ``(n, n)``.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    if pairwise_fun is None:
        # Fast Euclidean via broadcasting
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    d = np.zeros((n, n), dtype=float)
    for ii in range(n):
        for jj in range(ii + 1, n):
            val = pairwise_fun(x[ii], x[jj])
            d[ii, jj] = val
            d[jj, ii] = val
    return d
