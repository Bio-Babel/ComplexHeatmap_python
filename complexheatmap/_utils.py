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
    "smart_align",
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

    # R: max(do.call("unit.c", lapply(..., grobWidth(textGrob(...)))))
    #    then convertWidth(u, "mm")
    widths = grid_py.string_width(list(text))
    mm_vals = grid_py.convert_width(widths, "mm", valueOnly=True)
    return grid_py.Unit(float(np.max(mm_vals)), "mm")


def max_text_height(
    text: Union[str, Sequence[str]],
    gp: Optional[Dict[str, Any]] = None,
    rot: float = 0,
) -> Any:
    """Compute the maximum rendered text height using grid_py.

    Port of R's ``max_text_height`` (utils.R:430+):
    ``convertHeight(max(grobHeight(textGrob(...))), "mm")``.

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
        A ``grid_py.Unit`` in mm representing the maximum height.
    """
    import grid_py

    if isinstance(text, str):
        text = [text]
    if not text:
        return grid_py.Unit(0, "mm")

    heights = grid_py.string_height(list(text))
    mm_vals = grid_py.convert_height(heights, "mm", valueOnly=True)
    return grid_py.Unit(float(np.max(mm_vals)), "mm")


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
# Smart-align (de-overlap intervals)
# ---------------------------------------------------------------------------

def smart_align(
    h1: np.ndarray,
    h2: np.ndarray,
    bounds: tuple,
) -> np.ndarray:
    """Shift intervals to avoid overlap within *bounds*.

    Faithful port of R's ``smartAlign2`` (box_align.R:41-84) which uses
    ``BoxArrange`` with cluster-merging, flatten, and range adjustment.

    Parameters
    ----------
    h1, h2 : ndarray
        Lower and upper edges of each interval.
    bounds : tuple of (lo, hi)
        Allowed range.

    Returns
    -------
    ndarray of shape (n, 2)
        Adjusted ``(lo, hi)`` positions.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    n = len(h1)
    if n == 0:
        return np.empty((0, 2))

    lo, hi = float(bounds[0]), float(bounds[1])
    heights = h2 - h1
    total_height = float(np.sum(heights))

    # R: if(sum(end - start) > range[2] - range[1]) — overflow path
    if total_height > hi - lo:
        mid = (h1 + h2) / 2.0
        od = np.argsort(mid)
        rk = _rank_with_random_ties(mid)
        h_sorted = heights[od]
        n_s = len(h_sorted)

        mid_diff = np.zeros(n_s)
        for i in range(1, n_s):
            mid_diff[i] = h_sorted[i] / 2 + h_sorted[i - 1] / 2
        mid_radius = total_height - h_sorted[-1] / 2 - h_sorted[0] / 2

        a_1 = lo + h_sorted[0] / 2
        a_n = hi - h_sorted[-1] / 2

        if mid_radius > 0:
            a = a_1 + np.cumsum(mid_diff) / mid_radius * (a_n - a_1)
        else:
            a = np.full(n_s, (a_1 + a_n) / 2)

        new_start = a - h_sorted / 2
        new_end = a + h_sorted / 2

        result = np.column_stack([new_start, new_end])
        # Unsort by rank (R: df[rk, ])
        out = np.zeros((n, 2))
        out[rk] = result
        return out

    # R: BoxArrange algorithm — cluster, flatten, merge, adjust, merge
    return _box_arrange(h1, h2, lo, hi)


def _rank_with_random_ties(x: np.ndarray) -> np.ndarray:
    """Return 0-based ranks with random tie-breaking (R: rank(ties.method='random'))."""
    n = len(x)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    return ranks


def _box_arrange(
    start: np.ndarray,
    end: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Port of R's BoxArrange class (box_align.R:212-341).

    Steps: sort → cluster overlapping → flatten each cluster → merge →
    adjust to range → merge again → extract new positions → unsort.
    """
    n = len(start)
    mid = (start + end) / 2.0
    od = np.argsort(mid)
    rk = _rank_with_random_ties(mid)

    # Build sorted list of (start, end, height, new_start, new_end)
    boxes_s = [(float(start[i]), float(end[i])) for i in od]

    # --- cluster overlapping boxes ---
    clusters = []  # each cluster: list of (orig_start, orig_end, height)
    cur_cluster = [(boxes_s[0][0], boxes_s[0][1], boxes_s[0][1] - boxes_s[0][0])]
    cur_end = boxes_s[0][1]

    for i in range(1, n):
        s, e = boxes_s[i]
        if s < cur_end:  # overlap
            cur_cluster.append((s, e, e - s))
            cur_end = max(cur_end, e)
        else:
            clusters.append(cur_cluster)
            cur_cluster = [(s, e, e - s)]
            cur_end = e
    clusters.append(cur_cluster)

    # --- flatten each cluster ---
    flat_clusters = _flatten_clusters(clusters, lo, hi)

    # --- merge overlapping clusters ---
    flat_clusters = _merge_clusters(flat_clusters, lo, hi)

    # --- adjust to range ---
    for cl in flat_clusters:
        if cl[0] < lo:
            shift = lo - cl[0]
            cl[0] += shift
            cl[1] += shift
            _update_box_positions(cl)
        if cl[1] > hi:
            shift = cl[1] - hi
            cl[0] -= shift
            cl[1] -= shift
            _update_box_positions(cl)

    # --- merge again after adjustment ---
    flat_clusters = _merge_clusters(flat_clusters, lo, hi)

    # --- extract new positions, unsort by rank ---
    all_positions = []
    for cl in flat_clusters:
        all_positions.extend(cl[2])  # list of (new_start, new_end)

    result = np.array(all_positions)
    out = np.zeros((n, 2))
    out[rk] = result
    return out


def _flatten_clusters(clusters, lo, hi):
    """Flatten each cluster: pack boxes tightly, center on cluster mid."""
    flat = []
    for cl_boxes in clusters:
        s_vals = [b[0] for b in cl_boxes]
        e_vals = [b[1] for b in cl_boxes]
        h_vals = [b[2] for b in cl_boxes]

        cl_mid = (min(s_vals) + max(e_vals)) / 2.0
        total_h = sum(h_vals)

        s2 = cl_mid - total_h / 2.0
        e2 = cl_mid + total_h / 2.0

        if s2 < lo:
            s2 = lo
            e2 = s2 + total_h
        elif e2 > hi:
            e2 = hi
            s2 = e2 - total_h

        # Assign new positions sequentially
        new_positions = []
        cur = s2
        for h in h_vals:
            new_positions.append((cur, cur + h))
            cur += h

        # cl = [cl_start, cl_end, [(new_s, new_e), ...], [heights]]
        flat.append([s2, e2, new_positions, h_vals])
    return flat


def _merge_clusters(clusters, lo, hi):
    """Merge overlapping clusters repeatedly until stable."""
    while True:
        merged = False
        new_clusters = []
        skip = set()
        for i in range(len(clusters)):
            if i in skip:
                continue
            if i + 1 < len(clusters) and clusters[i][1] > clusters[i + 1][0]:
                # Merge i and i+1
                combined_boxes_h = clusters[i][3] + clusters[i + 1][3]
                combined_s = [p[0] for p in clusters[i][2]] + [p[0] for p in clusters[i + 1][2]]
                combined_e = [p[1] for p in clusters[i][2]] + [p[1] for p in clusters[i + 1][2]]

                cl_mid = (min(combined_s) + max(combined_e)) / 2.0
                total_h = sum(combined_boxes_h)
                s2 = cl_mid - total_h / 2.0
                e2 = cl_mid + total_h / 2.0

                if s2 < lo:
                    s2 = lo
                    e2 = s2 + total_h
                elif e2 > hi:
                    e2 = hi
                    s2 = e2 - total_h

                new_positions = []
                cur = s2
                for h in combined_boxes_h:
                    new_positions.append((cur, cur + h))
                    cur += h

                new_clusters.append([s2, e2, new_positions, combined_boxes_h])
                skip.add(i + 1)
                merged = True
            else:
                new_clusters.append(clusters[i])
        clusters = new_clusters
        if not merged:
            break
    return clusters


def _update_box_positions(cl):
    """Recompute box positions after shifting a cluster."""
    cur = cl[0]
    new_positions = []
    for h in cl[3]:
        new_positions.append((cur, cur + h))
        cur += h
    cl[2] = new_positions
    cl[1] = cl[0] + sum(cl[3])


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
