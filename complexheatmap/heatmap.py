"""Core :class:`Heatmap` class for ComplexHeatmap.

This module implements the primary ``Heatmap`` class, the Python equivalent
of the ``Heatmap`` S4 class in the R *ComplexHeatmap* package.  A single
``Heatmap`` manages data, colour mapping, hierarchical clustering,
row/column splitting, annotations, and grid_py-based rendering.
"""

from __future__ import annotations

import copy
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

import grid_py

from .color_mapping import ColorMapping
from ._color import color_ramp2
from ._globals import ht_opt

__all__ = ["AdditiveUnit", "Heatmap"]

# Module-level counter for auto-generated heatmap names.
_heatmap_id: int = 0


def _next_heatmap_name() -> str:
    """Return an auto-generated heatmap name."""
    global _heatmap_id
    _heatmap_id += 1
    return f"matrix_{_heatmap_id}"


# ---------------------------------------------------------------------------
# Distance / linkage helpers
# ---------------------------------------------------------------------------

_SCIPY_METRICS = {
    "euclidean", "cityblock", "cosine", "correlation",
    "chebyshev", "canberra", "braycurtis", "mahalanobis",
    "minkowski", "sqeuclidean", "seuclidean", "hamming",
    "jaccard", "matching",
}

_CORRELATION_METRICS = {"pearson", "spearman", "kendall"}


def _compute_dist(
    mat: np.ndarray,
    metric: Union[str, Callable[..., Any]],
) -> np.ndarray:
    """Compute a condensed distance vector.

    Mirrors R's ``ComplexHeatmap:::get_dist()``.
    """
    from scipy.spatial.distance import pdist, squareform

    if isinstance(metric, str) and metric in _CORRELATION_METRICS:
        if np.any(np.isnan(mat)):
            warnings.warn(
                "NA exists in the matrix, calculating distance by "
                "removing NA values."
            )
        import pandas as pd
        corr = pd.DataFrame(mat.T).corr(method=metric).values
        corr = np.clip(corr, -1.0, 1.0)
        dist_mat = (1.0 - corr) / 2.0
        np.fill_diagonal(dist_mat, 0.0)
        return squareform(dist_mat, checks=False)

    if callable(metric) and not isinstance(metric, str):
        import inspect
        try:
            sig = inspect.signature(metric)
            n_params = len(sig.parameters)
        except (ValueError, TypeError):
            n_params = 1
        if n_params == 1:
            result = metric(mat)
            arr = np.asarray(result, dtype=float)
            if arr.ndim == 2:
                return squareform(arr, checks=False)
            return arr
        return pdist(mat, metric=metric)

    has_nan = np.any(np.isnan(mat))
    if has_nan:
        n = mat.shape[0]
        n_pairs = n * (n - 1) // 2
        condensed = np.empty(n_pairs, dtype=float)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                valid = ~(np.isnan(mat[i]) | np.isnan(mat[j]))
                if np.any(valid):
                    condensed[idx] = pdist(
                        mat[np.ix_([i, j], np.where(valid)[0])],
                        metric=metric,
                    )[0]
                else:
                    condensed[idx] = 0.0
                idx += 1
        return condensed

    return pdist(mat, metric=metric)


def _compute_linkage(
    dist_condensed: np.ndarray,
    method: str = "complete",
) -> np.ndarray:
    """Compute hierarchical clustering linkage."""
    from scipy.cluster.hierarchy import linkage

    Z = linkage(dist_condensed, method=method)
    return Z


def _leaves_from_linkage(Z: np.ndarray) -> np.ndarray:
    """Return leaf ordering from a linkage matrix."""
    from scipy.cluster.hierarchy import leaves_list
    return leaves_list(Z)


def _reorder_linkage_by_weights(
    Z: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """Reorder a linkage tree to better match R's ``reorder.dendrogram``.

    R's ComplexHeatmap defaults to reordering dendrogram branches by
    ``-rowMeans`` / ``-colMeans`` rather than SciPy's optimal-leaf-ordering.
    This function approximates that behavior by recursively swapping the
    left/right children at each merge based on the mean weight of the
    descendant leaves.
    """
    if weights is None:
        return Z

    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        return Z

    Z2 = np.asarray(Z, dtype=float).copy()
    n_leaves = Z2.shape[0] + 1
    if len(arr) != n_leaves:
        return Z

    def _visit(node_id: int) -> float:
        if node_id < n_leaves:
            val = arr[node_id]
            return float(val) if not np.isnan(val) else 0.0

        row = node_id - n_leaves
        left = int(Z2[row, 0])
        right = int(Z2[row, 1])

        left_mean = _visit(left)
        right_mean = _visit(right)

        if left_mean > right_mean:
            Z2[row, 0], Z2[row, 1] = Z2[row, 1], Z2[row, 0]
            left_mean, right_mean = right_mean, left_mean

        return float(np.nanmean([left_mean, right_mean]))

    _visit(2 * n_leaves - 2)
    return Z2


def _sum_units(units: Sequence[grid_py.Unit]) -> grid_py.Unit:
    """Return the sum of a sequence of grid units."""
    if not units:
        return grid_py.Unit(0, "mm")
    total = units[0]
    for unit in units[1:]:
        total = total + unit
    return total


def _kmeans_split(mat: np.ndarray, k: int, repeats: int = 1) -> np.ndarray:
    """Assign rows to *k* clusters using k-means."""
    from scipy.cluster.vq import kmeans2, whiten

    mat_clean = mat.copy()
    col_means = np.nanmean(mat_clean, axis=0)
    for j in range(mat_clean.shape[1]):
        mask = np.isnan(mat_clean[:, j])
        mat_clean[mask, j] = col_means[j]

    whitened = whiten(mat_clean)
    if np.all(whitened == 0):
        whitened = mat_clean

    if repeats <= 1:
        _, labels = kmeans2(whitened, k, minit="points")
        return labels

    # Consensus k-means
    from collections import Counter
    label_runs = []
    for _ in range(repeats):
        _, labs = kmeans2(whitened, k, minit="points")
        label_runs.append(labs)
    # Simple consensus: use the most common assignment across runs
    n = mat.shape[0]
    consensus = np.zeros(n, dtype=int)
    for i in range(n):
        votes = [lr[i] for lr in label_runs]
        consensus[i] = Counter(votes).most_common(1)[0][0]
    return consensus


def _factor_to_slices(
    factor: np.ndarray,
) -> Tuple[List[Any], List[np.ndarray]]:
    """Convert a factor vector to ordered groups.

    If *factor* was created from a ``pd.Categorical`` with explicit
    categories, those categories define the level order (matching R's
    ``factor(x, levels=...)``) and only categories actually present in the
    data are included.  Otherwise, first-occurrence order is used.
    """
    import pandas as pd

    # Detect pd.Categorical stored inside an object-dtype ndarray
    # (np.asarray(pd.Categorical(...)) produces object array but the
    # original Categorical may be accessible via the source).
    cat_order: Optional[list] = None
    if hasattr(factor, "dtype"):
        if isinstance(factor.dtype, pd.CategoricalDtype):
            # pd.Categorical or pd.arrays.CategoricalArray
            cat_order = list(factor.categories)
        elif factor.dtype == object:
            pass  # fall through to first-occurrence

    if cat_order is not None:
        # Use category order, only keeping levels present in data
        data_set = set(factor)
        levels = [c for c in cat_order if c in data_set]
        val_to_idx = {v: i for i, v in enumerate(levels)}
        groups: List[List[int]] = [[] for _ in levels]
        for i, val in enumerate(factor):
            if val in val_to_idx:
                groups[val_to_idx[val]].append(i)
        return levels, [np.array(g, dtype=int) for g in groups]

    # Default: first-occurrence order
    seen: Dict[Any, int] = {}
    levels_fo: List[Any] = []
    groups_fo: List[List[int]] = []
    for i, val in enumerate(factor):
        key = val if not isinstance(val, float) or not np.isnan(val) else "__NA__"
        if key not in seen:
            seen[key] = len(levels_fo)
            levels_fo.append(val)
            groups_fo.append([])
        groups_fo[seen[key]].append(i)
    return levels_fo, [np.array(g, dtype=int) for g in groups_fo]


# ---------------------------------------------------------------------------
# AdditiveUnit mixin
# ---------------------------------------------------------------------------


class AdditiveUnit:
    """Mixin enabling the ``+`` operator for building HeatmapList objects.

    When ``A + B`` is evaluated where both are ``AdditiveUnit`` subclasses
    (e.g. ``Heatmap``), the result is a ``HeatmapList`` containing both.
    """

    def __add__(self, other: Any) -> Any:
        # Accept AdditiveUnit subclasses (Heatmap, HeatmapList) and
        # HeatmapAnnotation, mirroring R's "+.AdditiveUnit".
        from .heatmap_annotation import HeatmapAnnotation
        if not isinstance(other, (AdditiveUnit, HeatmapAnnotation)):
            return NotImplemented
        from .heatmap_list import HeatmapList
        hl = HeatmapList()
        hl.add(self)
        hl.add(other)
        return hl

    def __mod__(self, other: Any) -> Any:
        """Vertical concatenation operator, mirroring R's ``%v%``.

        ``ht1 % ht2`` creates a :class:`HeatmapList` with
        ``direction="vertical"``, matching R's
        ``ht1 %v% ht2`` (AdditiveUnit-class.R:112-134).

        For vertical lists, all heatmaps must have the same number of
        columns.  ``HeatmapAnnotation`` objects must have
        ``which="column"`` (they become column-annotation rows in the
        vertical stack).
        """
        from .heatmap_annotation import HeatmapAnnotation
        if not isinstance(other, (AdditiveUnit, HeatmapAnnotation)):
            return NotImplemented
        # Validate: HeatmapAnnotation must be column-type for %v%
        if isinstance(other, HeatmapAnnotation) and other.which != "column":
            raise ValueError(
                "Use `which='column'` or `columnAnnotation()` when adding "
                "annotations vertically with `%`."
            )
        if isinstance(self, HeatmapAnnotation) and self.which != "column":
            raise ValueError(
                "Use `which='column'` or `columnAnnotation()` when adding "
                "annotations vertically with `%`."
            )
        from .heatmap_list import HeatmapList
        hl = HeatmapList(direction="vertical")
        hl.add(self)
        hl.add(other)
        return hl

    def __radd__(self, other: Any) -> Any:
        if other == 0 or other is None:
            return self
        from .heatmap_annotation import HeatmapAnnotation
        if isinstance(other, HeatmapAnnotation):
            from .heatmap_list import HeatmapList
            hl = HeatmapList()
            hl.add(other)
            hl.add(self)
            return hl
        return NotImplemented


# ---------------------------------------------------------------------------
# Heatmap class
# ---------------------------------------------------------------------------


class Heatmap(AdditiveUnit):
    """A single heatmap with optional clustering, splitting, and annotations.

    This is the primary class in ComplexHeatmap.  It manages:

    - Data matrix and colour mapping
    - Row/column clustering and dendrogram computation
    - Row/column splitting (k-means or manual)
    - Annotations (top, bottom, left, right)
    - Rendering via grid_py
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        matrix: ArrayLike,
        col: Optional[Union[Dict[str, str], Callable[..., Any], List[str]]] = None,
        name: Optional[str] = None,
        na_col: str = "grey",
        color_space: str = "LAB",
        rect_gp: Optional[grid_py.Gpar] = None,
        border: Union[bool, str] = False,
        border_gp: Optional[grid_py.Gpar] = None,
        cell_fun: Optional[Callable[..., Any]] = None,
        layer_fun: Optional[Callable[..., Any]] = None,
        jitter: Union[bool, float] = False,
        # Row titles
        row_title: Optional[Union[str, List[str]]] = None,
        row_title_side: str = "left",
        row_title_gp: Optional[grid_py.Gpar] = None,
        row_title_rot: Optional[float] = None,
        # Column titles
        column_title: Optional[Union[str, List[str]]] = None,
        column_title_side: str = "top",
        column_title_gp: Optional[grid_py.Gpar] = None,
        column_title_rot: float = 0.0,
        # Row clustering
        cluster_rows: Union[bool, np.ndarray, Callable[..., Any]] = True,
        cluster_row_slices: bool = True,
        clustering_distance_rows: Union[str, Callable[..., Any]] = "euclidean",
        clustering_method_rows: str = "complete",
        row_dend_side: str = "left",
        row_dend_width: Optional[grid_py.Unit] = None,
        show_row_dend: bool = True,
        row_dend_reorder: Optional[bool] = None,
        row_dend_gp: Optional[grid_py.Gpar] = None,
        # Column clustering
        cluster_columns: Union[bool, np.ndarray, Callable[..., Any]] = True,
        cluster_column_slices: bool = True,
        clustering_distance_columns: Union[str, Callable[..., Any]] = "euclidean",
        clustering_method_columns: str = "complete",
        column_dend_side: str = "top",
        column_dend_height: Optional[grid_py.Unit] = None,
        show_column_dend: bool = True,
        column_dend_reorder: Optional[bool] = None,
        column_dend_gp: Optional[grid_py.Gpar] = None,
        # Manual ordering
        row_order: Optional[ArrayLike] = None,
        column_order: Optional[ArrayLike] = None,
        # Names
        row_labels: Optional[Sequence[str]] = None,
        row_names_side: str = "right",
        show_row_names: bool = True,
        row_names_max_width: Optional[grid_py.Unit] = None,
        row_names_gp: Optional[grid_py.Gpar] = None,
        row_names_rot: float = 0.0,
        row_names_centered: bool = False,
        column_labels: Optional[Sequence[str]] = None,
        column_names_side: str = "bottom",
        show_column_names: bool = True,
        column_names_max_height: Optional[grid_py.Unit] = None,
        column_names_gp: Optional[grid_py.Gpar] = None,
        column_names_rot: float = 90.0,
        column_names_centered: bool = False,
        # Annotations
        top_annotation: Optional[Any] = None,
        bottom_annotation: Optional[Any] = None,
        left_annotation: Optional[Any] = None,
        right_annotation: Optional[Any] = None,
        # Splitting
        km: int = 1,
        split: Optional[ArrayLike] = None,
        row_km: Optional[int] = None,
        row_km_repeats: int = 1,
        row_split: Optional[ArrayLike] = None,
        column_km: Optional[int] = None,
        column_km_repeats: int = 1,
        column_split: Optional[ArrayLike] = None,
        gap: Optional[grid_py.Unit] = None,
        row_gap: Optional[grid_py.Unit] = None,
        column_gap: Optional[grid_py.Unit] = None,
        show_parent_dend_line: bool = False,
        # Size
        width: Optional[grid_py.Unit] = None,
        height: Optional[grid_py.Unit] = None,
        heatmap_width: Optional[grid_py.Unit] = None,
        heatmap_height: Optional[grid_py.Unit] = None,
        # Legend
        show_heatmap_legend: bool = True,
        heatmap_legend_param: Optional[Dict[str, Any]] = None,
        # Raster
        use_raster: Optional[bool] = None,
        raster_quality: float = 1.0,
        raster_device_param: Optional[Dict[str, Any]] = None,
        raster_resize_mat: Union[bool, Callable] = False,
        # Post function
        post_fun: Optional[Callable] = None,
    ) -> None:
        # --- Matrix ---
        # R Heatmap-class.R:254,261: row_labels = rownames(matrix),
        # column_labels = colnames(matrix)
        # Extract labels from pandas DataFrame before converting to numpy
        import pandas as _pd
        if isinstance(matrix, _pd.DataFrame):
            if row_labels is None and matrix.index is not None:
                row_labels = [str(x) for x in matrix.index]
            if column_labels is None and matrix.columns is not None:
                column_labels = [str(x) for x in matrix.columns]

        raw = np.asarray(matrix)
        self._is_numeric_matrix: bool = np.issubdtype(raw.dtype, np.number)
        if self._is_numeric_matrix:
            mat = np.asarray(matrix, dtype=float)
        else:
            mat = np.asarray(matrix, dtype=object)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        if mat.ndim != 2:
            raise ValueError(
                f"'matrix' must be 1-D or 2-D, got {mat.ndim}-D array."
            )
        self.matrix: np.ndarray = mat
        self.nrow: int = mat.shape[0]
        self.ncol: int = mat.shape[1]

        # --- Name ---
        self.name: str = name if name is not None else _next_heatmap_name()
        if self.name == "":
            raise ValueError("Heatmap name cannot be empty string.")

        # --- Colour mapping ---
        self.na_col: str = na_col
        self.color_space: str = color_space
        self._color_mapping: Optional[ColorMapping] = None
        self._raw_col = col
        self._setup_color_mapping(col)

        # --- Graphical parameters ---
        self.rect_gp: grid_py.Gpar = rect_gp if rect_gp is not None else grid_py.Gpar(col="transparent")
        if isinstance(border, bool):
            self.border: Optional[str] = "black" if border else None
        else:
            self.border = str(border) if border else None
        self.border_gp: grid_py.Gpar = border_gp if border_gp is not None else grid_py.Gpar(col="black")

        # --- Clustering ---
        self._cluster_rows_input = cluster_rows
        self._cluster_columns_input = cluster_columns
        self.cluster_row_slices = cluster_row_slices
        self.cluster_column_slices = cluster_column_slices
        self.clustering_distance_rows = clustering_distance_rows
        self.clustering_distance_columns = clustering_distance_columns
        self.clustering_method_rows = clustering_method_rows
        self.clustering_method_columns = clustering_method_columns
        # Default reorder: True when cluster input is bool or callable
        if row_dend_reorder is None:
            self.row_dend_reorder = isinstance(cluster_rows, (bool,)) or callable(cluster_rows)
        else:
            self.row_dend_reorder = row_dend_reorder
        if column_dend_reorder is None:
            self.column_dend_reorder = isinstance(cluster_columns, (bool,)) or callable(cluster_columns)
        else:
            self.column_dend_reorder = column_dend_reorder

        # --- Splitting ---
        # Normalize km / split aliases
        if row_km is None:
            row_km = km
        if row_split is None and split is not None:
            row_split = split

        self._row_split_k: Optional[int] = None
        self._column_split_k: Optional[int] = None
        if isinstance(row_split, (int, np.integer)) and not isinstance(row_split, bool):
            self._row_split_k = int(row_split)
            row_split = None
        if isinstance(column_split, (int, np.integer)) and not isinstance(column_split, bool):
            self._column_split_k = int(column_split)
            column_split = None

        self.row_km: Optional[int] = row_km if row_km is not None and row_km > 1 else None
        self.row_km_repeats: int = row_km_repeats
        self.column_km: Optional[int] = column_km if column_km is not None and column_km > 1 else None
        self.column_km_repeats: int = column_km_repeats
        # Preserve pd.Categorical dtype for ordered factor support
        # (np.asarray strips CategoricalDtype → use pd.array or keep as-is)
        import pandas as _pd
        self._row_split_input = (
            _pd.array(row_split) if (row_split is not None and isinstance(
                getattr(row_split, 'dtype', None), _pd.CategoricalDtype))
            else (np.asarray(row_split) if row_split is not None else None)
        )
        self._column_split_input = (
            _pd.array(column_split) if (column_split is not None and isinstance(
                getattr(column_split, 'dtype', None), _pd.CategoricalDtype))
            else (np.asarray(column_split) if column_split is not None else None)
        )

        # Gaps
        _default_gap = gap if gap is not None else grid_py.Unit(1, "mm")
        self.row_gap: grid_py.Unit = row_gap if row_gap is not None else _default_gap
        self.column_gap: grid_py.Unit = column_gap if column_gap is not None else _default_gap
        self.show_parent_dend_line = show_parent_dend_line

        # --- Manual ordering ---
        self._row_order_input = (
            np.asarray(row_order, dtype=int) if row_order is not None else None
        )
        self._column_order_input = (
            np.asarray(column_order, dtype=int) if column_order is not None else None
        )

        # --- Dendrogram display ---
        self.show_row_dend: bool = show_row_dend
        self.show_column_dend: bool = show_column_dend
        self.row_dend_side: str = row_dend_side
        self.column_dend_side: str = column_dend_side
        self.row_dend_width: grid_py.Unit = (
            row_dend_width if row_dend_width is not None
            else grid_py.Unit(10, "mm")
        )
        self.column_dend_height: grid_py.Unit = (
            column_dend_height if column_dend_height is not None
            else grid_py.Unit(10, "mm")
        )
        self.row_dend_gp: grid_py.Gpar = row_dend_gp if row_dend_gp is not None else grid_py.Gpar()
        self.column_dend_gp: grid_py.Gpar = column_dend_gp if column_dend_gp is not None else grid_py.Gpar()

        # --- Titles ---
        self.row_title = row_title
        self.column_title = column_title
        self.row_title_side: str = row_title_side
        self.column_title_side: str = column_title_side
        self.row_title_gp: grid_py.Gpar = (
            row_title_gp if row_title_gp is not None
            else grid_py.Gpar(fontsize=13.2)
        )
        self.column_title_gp: grid_py.Gpar = (
            column_title_gp if column_title_gp is not None
            else grid_py.Gpar(fontsize=13.2)
        )
        if row_title_rot is not None:
            self.row_title_rot: float = row_title_rot
        else:
            self.row_title_rot = 90.0 if row_title_side == "left" else 270.0
        self.column_title_rot: float = column_title_rot

        # --- Names ---
        # R Heatmap-class.R:671-674: if matrix has no rownames and
        # row_labels is not provided, auto-hide row names.
        # Same for columns (line 688-691).
        if show_row_names and row_labels is None:
            # Check if the input matrix had row names (pandas index, etc.)
            has_rownames = False
            if hasattr(matrix, 'index') and matrix.index is not None:
                has_rownames = True  # pandas DataFrame
            elif hasattr(matrix, 'dtype') and matrix.dtype.names:
                has_rownames = True  # structured array
            if not has_rownames:
                show_row_names = False

        if show_column_names and column_labels is None:
            has_colnames = False
            if hasattr(matrix, 'columns') and matrix.columns is not None:
                has_colnames = True
            if not has_colnames:
                show_column_names = False

        self.show_row_names: bool = show_row_names
        self.show_column_names: bool = show_column_names
        self.row_labels: Optional[List[str]] = (
            list(row_labels) if row_labels is not None else None
        )
        self.column_labels: Optional[List[str]] = (
            list(column_labels) if column_labels is not None else None
        )
        self.row_names_side: str = row_names_side
        self.column_names_side: str = column_names_side
        self.row_names_gp: grid_py.Gpar = (
            row_names_gp if row_names_gp is not None
            else grid_py.Gpar(fontsize=12)
        )
        self.column_names_gp: grid_py.Gpar = (
            column_names_gp if column_names_gp is not None
            else grid_py.Gpar(fontsize=12)
        )
        self.row_names_rot: float = row_names_rot
        self.column_names_rot: float = column_names_rot
        self.row_names_centered: bool = row_names_centered
        self.column_names_centered: bool = column_names_centered
        self.row_names_max_width: grid_py.Unit = (
            row_names_max_width if row_names_max_width is not None
            else grid_py.Unit(6, "cm")
        )
        self.column_names_max_height: grid_py.Unit = (
            column_names_max_height if column_names_max_height is not None
            else grid_py.Unit(6, "cm")
        )

        # --- Annotations ---
        self.top_annotation = top_annotation
        self.bottom_annotation = bottom_annotation
        self.left_annotation = left_annotation
        self.right_annotation = right_annotation

        # --- Cell rendering ---
        self.cell_fun = cell_fun
        self.layer_fun = layer_fun
        if use_raster is None:
            self.use_raster = (self.nrow > 2000 or self.ncol > 2000) and cell_fun is None
        else:
            self.use_raster = use_raster
        if cell_fun is not None:
            self.use_raster = False
        self.raster_quality: float = raster_quality
        self.raster_device_param = raster_device_param or {}
        self.raster_resize_mat = raster_resize_mat

        # --- Legend ---
        self.show_heatmap_legend: bool = show_heatmap_legend
        self.heatmap_legend_param: Dict[str, Any] = (
            heatmap_legend_param if heatmap_legend_param is not None else {}
        )
        if "title" not in self.heatmap_legend_param:
            self.heatmap_legend_param["title"] = self.name

        # --- Size ---
        self.width = width
        self.height = height
        self.heatmap_width = heatmap_width if heatmap_width is not None else grid_py.Unit(1, "npc")
        self.heatmap_height = heatmap_height if heatmap_height is not None else grid_py.Unit(1, "npc")

        # --- Post function ---
        self.post_fun = post_fun

        # --- Layout state (populated by make_layout) ---
        self._row_order_list: Optional[List[np.ndarray]] = None
        self._column_order_list: Optional[List[np.ndarray]] = None
        self._row_dend_list: Optional[List[Optional[np.ndarray]]] = None
        self._column_dend_list: Optional[List[Optional[np.ndarray]]] = None
        self._row_dend_slice: Optional[np.ndarray] = None
        self._column_dend_slice: Optional[np.ndarray] = None
        self._row_split_labels: Optional[List[Any]] = None
        self._column_split_labels: Optional[List[Any]] = None
        self._layout_computed: bool = False
        self._layout: Dict[str, Any] = {"initialized": False}

    # ------------------------------------------------------------------
    # Colour mapping setup
    # ------------------------------------------------------------------

    def _setup_color_mapping(
        self, col: Optional[Union[Dict[str, str], Callable[..., Any], List[str]]]
    ) -> None:
        """Build a :class:`ColorMapping` from the *col* parameter."""
        if col is None and not self._is_numeric_matrix:
            unique_vals = sorted(set(self.matrix.ravel()))
            default_palette = [
                "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                "#FF7F00", "#FFFF33", "#A65628", "#F781BF",
            ]
            auto_colors = {
                str(v): default_palette[i % len(default_palette)]
                for i, v in enumerate(unique_vals)
            }
            self._color_mapping = ColorMapping(
                name=self.name, colors=auto_colors, na_col=self.na_col
            )
            return
        if col is None:
            data = self.matrix[~np.isnan(self.matrix)]
            if len(data) == 0:
                self._color_mapping = None
                return
            default_colors = ht_opt("COLOR")
            if default_colors is None:
                default_colors = ["blue", "white", "red"]
            vmin, vmax = float(np.min(data)), float(np.max(data))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            vmid = (vmin + vmax) / 2.0
            col_fun = color_ramp2([vmin, vmid, vmax], default_colors)
            self._color_mapping = ColorMapping(
                name=self.name, col_fun=col_fun, na_col=self.na_col
            )
        elif isinstance(col, dict):
            self._color_mapping = ColorMapping(
                name=self.name, colors=col, na_col=self.na_col
            )
        elif isinstance(col, list):
            data = self.matrix[~np.isnan(self.matrix)]
            if len(data) == 0:
                self._color_mapping = None
                return
            n_colors = len(col)
            vmin, vmax = float(np.min(data)), float(np.max(data))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            breaks = np.linspace(vmin, vmax, n_colors).tolist()
            col_fun = color_ramp2(breaks, col)
            self._color_mapping = ColorMapping(
                name=self.name, col_fun=col_fun, na_col=self.na_col
            )
        elif callable(col):
            self._color_mapping = ColorMapping(
                name=self.name, col_fun=col, na_col=self.na_col
            )
        else:
            raise ValueError(
                f"'col' must be a dict, callable, list of colours, or None; "
                f"got {type(col)!r}."
            )

    # ------------------------------------------------------------------
    # Matrix colour mapping
    # ------------------------------------------------------------------

    def _map_to_colors(self, mat: np.ndarray) -> np.ndarray:
        """Map a matrix to a 2-D array of colour strings.

        Returns
        -------
        numpy.ndarray
            String array of shape ``(nrow, ncol)`` with hex colour values.
        """
        if self._color_mapping is None:
            return np.full(mat.shape, self.na_col, dtype=object)
        result = self._color_mapping.map_to_colors(mat)
        return np.asarray(result, dtype=object).reshape(mat.shape)

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def make_layout(self) -> None:
        """Compute clustering, ordering, and splitting.

        After calling this method, the row/column order and dendrogram
        data structures are available via :meth:`row_order`,
        :meth:`column_order`, :meth:`row_dend`, and :meth:`column_dend`.
        """
        if self._layout_computed:
            return
        self._compute_row_layout()
        self._compute_column_layout()
        self._compute_slice_layout()
        self._layout_computed = True
        self._layout["initialized"] = True

    def _cluster_slice_order(
        self,
        order_list: List[np.ndarray],
        dend_list: List[Optional[np.ndarray]],
        levels: Optional[List[Any]],
        axis: str,
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], Optional[List[Any]], Optional[np.ndarray]]:
        """Cluster slice means and reorder slice groups like R ComplexHeatmap."""
        if len(order_list) <= 1:
            return order_list, dend_list, levels, None

        if axis == "row":
            if not self.cluster_row_slices:
                return order_list, dend_list, levels, None
            profiles = np.column_stack([
                np.nanmean(self.matrix[ind, :], axis=0) for ind in order_list
            ])
        else:
            if not self.cluster_column_slices:
                return order_list, dend_list, levels, None
            profiles = np.column_stack([
                np.nanmean(self.matrix[:, ind], axis=1) for ind in order_list
            ])

        if profiles.ndim == 1:
            profiles = profiles.reshape(1, -1)
        if profiles.shape[1] <= 1:
            return order_list, dend_list, levels, None

        Z_slice = _compute_linkage(
            _compute_dist(profiles.T, "euclidean"),
            method="complete",
        )
        Z_slice = _reorder_linkage_by_weights(
            Z_slice,
            -np.nanmean(profiles, axis=0),
        )
        slice_order = _leaves_from_linkage(Z_slice)

        order_list = [order_list[i] for i in slice_order]
        dend_list = [dend_list[i] for i in slice_order]
        if levels is not None:
            levels = [levels[i] for i in slice_order]

        return order_list, dend_list, levels, Z_slice

    def _compute_slice_layout(self) -> None:
        """Compute slice positions as unit expressions, matching R layout."""
        assert self._row_order_list is not None
        assert self._column_order_list is not None

        n_row_slices = len(self._row_order_list)
        n_col_slices = len(self._column_order_list)

        row_sizes = np.array([len(idx) for idx in self._row_order_list], dtype=float)
        col_sizes = np.array([len(idx) for idx in self._column_order_list], dtype=float)

        if row_sizes.sum() == 0:
            row_sizes = np.ones(max(n_row_slices, 1), dtype=float)
        if col_sizes.sum() == 0:
            col_sizes = np.ones(max(n_col_slices, 1), dtype=float)

        if n_row_slices == 1:
            slice_heights = [grid_py.Unit(1, "npc")]
        else:
            available_h = grid_py.Unit(1, "npc") - self.row_gap * (n_row_slices - 1)
            slice_heights = [
                available_h * (size / row_sizes.sum()) for size in row_sizes
            ]

        slice_y = [grid_py.Unit(1, "npc")]
        for i in range(1, n_row_slices):
            prior_h = _sum_units(slice_heights[:i])
            slice_y.append(grid_py.Unit(1, "npc") - prior_h - self.row_gap * i)

        if n_col_slices == 1:
            slice_widths = [grid_py.Unit(1, "npc")]
        else:
            available_w = grid_py.Unit(1, "npc") - self.column_gap * (n_col_slices - 1)
            slice_widths = [
                available_w * (size / col_sizes.sum()) for size in col_sizes
            ]

        slice_x = [grid_py.Unit(0, "npc")]
        for i in range(1, n_col_slices):
            prior_w = _sum_units(slice_widths[:i])
            slice_x.append(prior_w + self.column_gap * i)

        self._layout["slice"] = {
            "x": slice_x,
            "y": slice_y,
            "width": slice_widths,
            "height": slice_heights,
            "just": ["left", "top"],
        }

    # --- Row layout ---------------------------------------------------

    def _compute_row_layout(self) -> None:
        """Compute row ordering, clustering, and splitting."""
        mat = self.matrix
        n = self.nrow

        # integer row_split: cluster entire matrix, then cutree
        if (
            self._row_split_k is not None
            and self._row_split_k >= 2
            and self._should_cluster_rows()
            and n > 1
        ):
            from scipy.cluster.hierarchy import fcluster

            Z_full = self._get_row_linkage(mat)
            labels = fcluster(Z_full, t=self._row_split_k, criterion="maxclust")
            full_leaves = _leaves_from_linkage(Z_full)
            seen: Dict[int, int] = {}
            counter = 0
            for leaf in full_leaves:
                lab = labels[leaf]
                if lab not in seen:
                    seen[lab] = counter
                    counter += 1
            row_factor = np.array([seen[labels[i]] for i in range(n)])
            levels, groups = _factor_to_slices(row_factor)
            self._row_split_labels = levels

            row_order_list: List[np.ndarray] = []
            row_dend_list: List[Optional[np.ndarray]] = []
            for idx_group in groups:
                sub_mat = mat[idx_group, :]
                if len(idx_group) > 1:
                    Z = self._get_row_linkage(sub_mat)
                    leaves = _leaves_from_linkage(Z)
                    row_order_list.append(idx_group[leaves])
                    row_dend_list.append(Z)
                else:
                    row_order_list.append(idx_group)
                    row_dend_list.append(None)
            row_order_list, row_dend_list, levels, row_dend_slice = self._cluster_slice_order(
                row_order_list,
                row_dend_list,
                list(levels),
                axis="row",
            )
            self._row_order_list = row_order_list
            self._row_dend_list = row_dend_list
            self._row_split_labels = levels
            self._row_dend_slice = row_dend_slice
            return

        # Standard path: factor-based split or k-means
        row_factor = None
        _row_has_ordered_categories = False
        if self._row_split_input is not None:
            row_factor = self._row_split_input
            # Detect ordered pd.Categorical (like R's ordered factor)
            import pandas as _pd
            if isinstance(getattr(row_factor, 'dtype', None), _pd.CategoricalDtype):
                _row_has_ordered_categories = True
        elif self.row_km is not None and self.row_km > 1:
            row_factor = _kmeans_split(mat, self.row_km, self.row_km_repeats)

        if row_factor is not None:
            levels, groups = _factor_to_slices(row_factor)
            self._row_split_labels = levels
        else:
            levels = [None]
            groups = [np.arange(n, dtype=int)]
            self._row_split_labels = None

        row_order_list_std: List[np.ndarray] = []
        row_dend_list_std: List[Optional[np.ndarray]] = []

        for idx_group in groups:
            sub_mat = mat[idx_group, :]

            if self._row_order_input is not None:
                order_in_group = np.array(
                    [i for i in self._row_order_input if i in set(idx_group)],
                    dtype=int,
                )
                row_order_list_std.append(order_in_group)
                row_dend_list_std.append(None)
            elif self._should_cluster_rows() and len(idx_group) > 1:
                Z = self._get_row_linkage(sub_mat)
                leaves = _leaves_from_linkage(Z)
                row_order_list_std.append(idx_group[leaves])
                row_dend_list_std.append(Z)
            else:
                row_order_list_std.append(idx_group)
                row_dend_list_std.append(None)

        row_levels = list(levels) if self._row_split_labels is not None else None

        # R preserves factor level order when split is an ordered factor.
        # Only cluster-reorder slices when the split has no explicit order.
        if _row_has_ordered_categories:
            row_dend_slice = None
        else:
            row_order_list_std, row_dend_list_std, row_levels, row_dend_slice = self._cluster_slice_order(
                row_order_list_std,
                row_dend_list_std,
                row_levels,
                axis="row",
            )

        self._row_order_list = row_order_list_std
        self._row_dend_list = row_dend_list_std
        self._row_dend_slice = row_dend_slice
        self._row_split_labels = row_levels

    def _should_cluster_rows(self) -> bool:
        if not self._is_numeric_matrix:
            return False
        if self._row_order_input is not None:
            return False
        cr = self._cluster_rows_input
        if isinstance(cr, bool):
            return cr
        return cr is not False

    def _get_row_linkage(self, sub_mat: np.ndarray) -> np.ndarray:
        cr = self._cluster_rows_input
        if isinstance(cr, np.ndarray) and cr.ndim == 2 and cr.shape[1] == 4:
            return cr
        if callable(cr) and not isinstance(cr, bool):
            return np.asarray(cr(sub_mat))

        dist = _compute_dist(sub_mat, self.clustering_distance_rows)
        Z = _compute_linkage(dist, method=self.clustering_method_rows)

        reorder = self.row_dend_reorder
        weights: Optional[np.ndarray] = None
        if isinstance(reorder, np.ndarray):
            if reorder.ndim == 1 and len(reorder) == sub_mat.shape[0]:
                weights = np.asarray(reorder, dtype=float)
        elif isinstance(reorder, (list, tuple)):
            arr = np.asarray(reorder, dtype=float)
            if arr.ndim == 1 and len(arr) == sub_mat.shape[0]:
                weights = arr
        elif reorder:
            try:
                from scipy.cluster.hierarchy import optimal_leaf_ordering

                return optimal_leaf_ordering(Z, dist)
            except (np.linalg.LinAlgError, ValueError):
                return Z

        return _reorder_linkage_by_weights(Z, weights)

    # --- Column layout ------------------------------------------------

    def _compute_column_layout(self) -> None:
        """Compute column ordering, clustering, and splitting."""
        mat = self.matrix
        n = self.ncol

        if (
            self._column_split_k is not None
            and self._column_split_k >= 2
            and self._should_cluster_columns()
            and n > 1
        ):
            from scipy.cluster.hierarchy import fcluster

            Z_full = self._get_column_linkage(mat)
            labels = fcluster(Z_full, t=self._column_split_k, criterion="maxclust")
            full_leaves = _leaves_from_linkage(Z_full)
            seen: Dict[int, int] = {}
            counter = 0
            for leaf in full_leaves:
                lab = labels[leaf]
                if lab not in seen:
                    seen[lab] = counter
                    counter += 1
            col_factor = np.array([seen[labels[i]] for i in range(n)])
            levels, groups = _factor_to_slices(col_factor)
            self._column_split_labels = levels

            column_order_list: List[np.ndarray] = []
            column_dend_list: List[Optional[np.ndarray]] = []
            for idx_group in groups:
                sub_mat = mat[:, idx_group]
                if len(idx_group) > 1:
                    Z = self._get_column_linkage(sub_mat)
                    leaves = _leaves_from_linkage(Z)
                    column_order_list.append(idx_group[leaves])
                    column_dend_list.append(Z)
                else:
                    column_order_list.append(idx_group)
                    column_dend_list.append(None)
            column_order_list, column_dend_list, levels, column_dend_slice = self._cluster_slice_order(
                column_order_list,
                column_dend_list,
                list(levels),
                axis="column",
            )
            self._column_order_list = column_order_list
            self._column_dend_list = column_dend_list
            self._column_split_labels = levels
            self._column_dend_slice = column_dend_slice
            return

        col_factor_std: Optional[np.ndarray] = None
        if self._column_split_input is not None:
            col_factor_std = self._column_split_input
        elif self.column_km is not None and self.column_km > 1:
            col_factor_std = _kmeans_split(mat.T, self.column_km, self.column_km_repeats)

        if col_factor_std is not None:
            levels, groups = _factor_to_slices(col_factor_std)
            self._column_split_labels = levels
        else:
            levels = [None]
            groups = [np.arange(n, dtype=int)]
            self._column_split_labels = None

        column_order_list_std: List[np.ndarray] = []
        column_dend_list_std: List[Optional[np.ndarray]] = []

        for idx_group in groups:
            sub_mat = mat[:, idx_group]

            if self._column_order_input is not None:
                order_in_group = np.array(
                    [i for i in self._column_order_input if i in set(idx_group)],
                    dtype=int,
                )
                column_order_list_std.append(order_in_group)
                column_dend_list_std.append(None)
            elif self._should_cluster_columns() and len(idx_group) > 1:
                Z = self._get_column_linkage(sub_mat)
                leaves = _leaves_from_linkage(Z)
                column_order_list_std.append(idx_group[leaves])
                column_dend_list_std.append(Z)
            else:
                column_order_list_std.append(idx_group)
                column_dend_list_std.append(None)

        col_levels = list(levels) if self._column_split_labels is not None else None
        column_order_list_std, column_dend_list_std, col_levels, column_dend_slice = self._cluster_slice_order(
            column_order_list_std,
            column_dend_list_std,
            col_levels,
            axis="column",
        )

        self._column_order_list = column_order_list_std
        self._column_dend_list = column_dend_list_std
        self._column_dend_slice = column_dend_slice
        self._column_split_labels = col_levels

    def _should_cluster_columns(self) -> bool:
        if not self._is_numeric_matrix:
            return False
        if self._column_order_input is not None:
            return False
        cc = self._cluster_columns_input
        if isinstance(cc, bool):
            return cc
        return cc is not False

    def _get_column_linkage(self, sub_mat: np.ndarray) -> np.ndarray:
        cc = self._cluster_columns_input
        if isinstance(cc, np.ndarray) and cc.ndim == 2 and cc.shape[1] == 4:
            return cc
        if callable(cc) and not isinstance(cc, bool):
            return np.asarray(cc(sub_mat.T))

        dist = _compute_dist(sub_mat.T, self.clustering_distance_columns)
        Z = _compute_linkage(dist, method=self.clustering_method_columns)

        reorder = self.column_dend_reorder
        weights: Optional[np.ndarray] = None
        if isinstance(reorder, np.ndarray):
            if reorder.ndim == 1 and len(reorder) == sub_mat.shape[1]:
                weights = np.asarray(reorder, dtype=float)
        elif isinstance(reorder, (list, tuple)):
            arr = np.asarray(reorder, dtype=float)
            if arr.ndim == 1 and len(arr) == sub_mat.shape[1]:
                weights = arr
        elif reorder:
            try:
                from scipy.cluster.hierarchy import optimal_leaf_ordering

                return optimal_leaf_ordering(Z, dist)
            except (np.linalg.LinAlgError, ValueError):
                return Z

        return _reorder_linkage_by_weights(Z, weights)

    # ------------------------------------------------------------------
    # Accessors for layout results
    # ------------------------------------------------------------------

    def get_row_order(self) -> Union[List[np.ndarray], np.ndarray]:
        """Return the row ordering."""
        if not self._layout_computed:
            self.make_layout()
        assert self._row_order_list is not None
        if len(self._row_order_list) == 1:
            return self._row_order_list[0]
        return self._row_order_list

    def get_column_order(self) -> Union[List[np.ndarray], np.ndarray]:
        """Return the column ordering."""
        if not self._layout_computed:
            self.make_layout()
        assert self._column_order_list is not None
        if len(self._column_order_list) == 1:
            return self._column_order_list[0]
        return self._column_order_list

    def get_row_dend(self) -> Union[List[Optional[np.ndarray]], Optional[np.ndarray]]:
        """Return row dendrogram linkage data."""
        if not self._layout_computed:
            self.make_layout()
        assert self._row_dend_list is not None
        if len(self._row_dend_list) == 1:
            return self._row_dend_list[0]
        return self._row_dend_list

    def get_column_dend(self) -> Union[List[Optional[np.ndarray]], Optional[np.ndarray]]:
        """Return column dendrogram linkage data."""
        if not self._layout_computed:
            self.make_layout()
        assert self._column_dend_list is not None
        if len(self._column_dend_list) == 1:
            return self._column_dend_list[0]
        return self._column_dend_list

    # Also provide R-style property accessors
    def column_order_list(self) -> List[np.ndarray]:
        if not self._layout_computed:
            self.make_layout()
        return self._column_order_list

    def row_order_list(self) -> List[np.ndarray]:
        if not self._layout_computed:
            self.make_layout()
        return self._row_order_list

    # ------------------------------------------------------------------
    # Component sizes
    # ------------------------------------------------------------------

    def _max_text_width_mm(self, labels: list, rot: float = 0,
                            gp: Any = None, fallback: float = 10.0) -> float:
        """Measure the maximum text width in mm across *labels*.

        Port of R's ``max_text_width(text, gp, rot)`` (utils.R:393-402):
        ``max(grobWidth(textGrob(text[i], gp=gp, rot=rot)))`` converted to mm.

        Uses grid_py's ``widthDetails(textGrob(rot=rot))`` which computes
        the rotated bounding box — matching R's ``C_textBounds``.
        """
        if not labels:
            return fallback
        try:
            from grid_py._primitives import text_grob
            from grid_py._size import width_details
            max_w = 0.0
            for lbl in labels:
                g = text_grob(label=str(lbl), x=0.5, y=0.5, rot=rot, gp=gp)
                w_unit = width_details(g)
                w_mm = grid_py.convert_width(w_unit, "mm", valueOnly=True)
                val = float(w_mm[0]) if hasattr(w_mm, '__getitem__') else float(w_mm)
                max_w = max(max_w, val)
            return max_w
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return fallback

    def _max_text_height_mm(self, labels: list, rot: float = 0,
                             gp: Any = None, fallback: float = 5.0) -> float:
        """Measure the maximum text height in mm across *labels*.

        Port of R's ``max_text_height(text, gp, rot)`` (utils.R:430+):
        ``max(grobHeight(textGrob(text[i], gp=gp, rot=rot)))`` converted to mm.
        """
        if not labels:
            return fallback
        try:
            from grid_py._primitives import text_grob
            from grid_py._size import height_details
            max_h = 0.0
            for lbl in labels:
                g = text_grob(label=str(lbl), x=0.5, y=0.5, rot=rot, gp=gp)
                h_unit = height_details(g)
                h_mm = grid_py.convert_height(h_unit, "mm", valueOnly=True)
                val = float(h_mm[0]) if hasattr(h_mm, '__getitem__') else float(h_mm)
                max_h = max(max_h, val)
            return max_h
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return fallback

    def _get_row_titles(self) -> list:
        """Return the list of row title strings."""
        n = len(self._row_order_list) if self._row_order_list else 1
        t = self.row_title
        if t is None and n > 1:
            if self._row_split_labels is not None:
                return [str(l) for l in self._row_split_labels]
            return [str(i + 1) for i in range(n)]
        if t is None:
            return []
        return t if isinstance(t, list) else [t]

    def _get_column_titles(self) -> list:
        """Return the list of column title strings."""
        n = len(self._column_order_list) if self._column_order_list else 1
        t = self.column_title
        if t is None and n > 1:
            if self._column_split_labels is not None:
                return [str(l) for l in self._column_split_labels]
            return [str(i + 1) for i in range(n)]
        if t is None:
            return []
        return t if isinstance(t, list) else [t]

    def _title_size(self, titles: list, gp: dict, rot: float,
                    dimension: str) -> grid_py.Unit:
        """Compute title component size matching R's Heatmap-layout.R:103,135.

        R: max_text_height(title, gp, rot) + sum(title_padding) for height
        R: max_text_width(title, gp, rot) + sum(title_padding) for width

        title_padding = 5.5 points + grobDescent("jA", gp)
        """
        if not titles:
            return grid_py.Unit(0, "mm")
        # R: title_padding[1] = 5.5 points + grobDescent("jA", gp)
        # Approximate grobDescent ≈ 1.5 points for typical fonts
        padding_mm = (5.5 + 1.5) * 0.3528  # points → mm (1 pt ≈ 0.3528 mm)
        if dimension == "height":
            text_mm = self._max_text_height_mm(titles, rot=rot, gp=gp)
        else:
            text_mm = self._max_text_width_mm(titles, rot=rot, gp=gp)
        return grid_py.Unit(text_mm + padding_mm, "mm")

    def component_height(self, component: str) -> grid_py.Unit:
        """Return the height of a given component.

        Parameters
        ----------
        component : str
            One of ``"column_title_top"``, ``"column_dend_top"``,
            ``"column_names_top"``, ``"top_annotation"``,
            ``"heatmap_body"``, ``"bottom_annotation"``,
            ``"column_names_bottom"``, ``"column_dend_bottom"``,
            ``"column_title_bottom"``.
        """
        zero = grid_py.Unit(0, "mm")
        # R: set_component_height overrides from HeatmapList alignment
        _overrides = getattr(self, '_override_heights', {})
        if component in _overrides:
            return grid_py.Unit(_overrides[component], "mm")
        if component == "column_title_top":
            has_title = self.column_title is not None
            if not has_title and self._column_order_list and len(self._column_order_list) > 1:
                has_title = True
            if has_title and self.column_title_side == "top":
                titles = self._get_column_titles()
                gp = self.column_title_gp if isinstance(self.column_title_gp, dict) else {}
                return self._title_size(titles, gp, self.column_title_rot, "height")
            return zero
        if component == "column_title_bottom":
            has_title = self.column_title is not None
            if not has_title and self._column_order_list and len(self._column_order_list) > 1:
                has_title = True
            h = zero
            if has_title and self.column_title_side == "bottom":
                titles = self._get_column_titles()
                gp = self.column_title_gp if isinstance(self.column_title_gp, dict) else {}
                h = self._title_size(titles, gp, self.column_title_rot, "height")
            # Reserve space for row annotation extended (axis + name)
            # that overflows below the heatmap body (R extended mechanism).
            bottom_ext = 0.0
            for anno_attr in ("left_annotation", "right_annotation"):
                ha = getattr(self, anno_attr, None)
                if ha is not None:
                    ext = getattr(ha, "extended", (0, 0, 0, 0))
                    bottom_ext = max(bottom_ext, ext[2])  # bottom
            if bottom_ext > 0:
                h = h + grid_py.Unit(bottom_ext, "mm")
            return h
        if component == "column_dend_top":
            if self.show_column_dend and self.column_dend_side == "top" and self._should_cluster_columns():
                return self.column_dend_height
            return zero
        if component == "column_dend_bottom":
            if self.show_column_dend and self.column_dend_side == "bottom" and self._should_cluster_columns():
                return self.column_dend_height
            return zero
        if component in ("column_names_top", "column_names_bottom"):
            side = "top" if component == "column_names_top" else "bottom"
            if self.show_column_names and self.column_names_side == side:
                labels = self.column_labels if self.column_labels is not None else [str(i) for i in range(self.matrix.shape[1])]
                gp = getattr(self, 'column_names_gp', None)
                rot = self.column_names_rot
                # R anno_text (AnnotationFunction-function.R:2432-2434):
                # height = max_text_width(x, gp) * |sin(rot)| + grobHeight("A", gp) * |cos(rot)|
                import math
                sin_r = abs(math.sin(math.radians(rot)))
                cos_r = abs(math.cos(math.radians(rot)))
                text_w = self._max_text_width_mm(labels, rot=0, gp=gp)
                text_h = self._max_text_height_mm(["A"], rot=0, gp=gp)
                h = text_w * sin_r + text_h * cos_r
                padding = 1.0  # DIMNAME_PADDING (R default = 1mm)
                return grid_py.Unit(h + padding * 2, "mm")
            return zero
        if component in ("top_annotation", "bottom_annotation"):
            ha = self.top_annotation if component == "top_annotation" else self.bottom_annotation
            if ha is not None:
                h = ha.height
                if h is not None and isinstance(h, (int, float)):
                    h_unit = grid_py.Unit(float(h), "mm")
                elif h is not None and hasattr(h, '_values'):
                    h_unit = h
                else:
                    n = len(ha.anno_list) if hasattr(ha, 'anno_list') else 1
                    h_unit = grid_py.Unit(max(n * 7, 15), "mm")
                # R Heatmap-class.R:843,869: += COLUMN_ANNO_PADDING
                h_unit = h_unit + grid_py.Unit(float(ht_opt("COLUMN_ANNO_PADDING")), "mm")
                return h_unit
            return zero
        if component == "heatmap_body":
            return grid_py.Unit(1, "null")
        return zero

    def component_width(self, component: str) -> grid_py.Unit:
        """Return the width of a given component."""
        zero = grid_py.Unit(0, "mm")
        # R: set_component_width overrides from HeatmapList alignment
        _overrides_w = getattr(self, '_override_widths', {})
        if component in _overrides_w:
            return grid_py.Unit(_overrides_w[component], "mm")
        if component == "row_title_left":
            has_title = self.row_title is not None
            if not has_title and self._row_order_list and len(self._row_order_list) > 1:
                has_title = True
            w = zero
            if has_title and self.row_title_side == "left":
                titles = self._get_row_titles()
                gp = self.row_title_gp if isinstance(self.row_title_gp, dict) else {}
                w = self._title_size(titles, gp, self.row_title_rot, "width")
            # Reserve space for column annotation extended (name label)
            # that overflows to the left (R extended mechanism).
            left_ext = 0.0
            for anno_attr in ("top_annotation", "bottom_annotation"):
                ha = getattr(self, anno_attr, None)
                if ha is not None:
                    ext = getattr(ha, "extended", (0, 0, 0, 0))
                    left_ext = max(left_ext, ext[3])  # left
            if left_ext > 0:
                w = w + grid_py.Unit(left_ext, "mm")
            return w
        if component == "row_title_right":
            has_title = self.row_title is not None
            if not has_title and self._row_order_list and len(self._row_order_list) > 1:
                has_title = True
            if has_title and self.row_title_side == "right":
                titles = self._get_row_titles()
                gp = self.row_title_gp if isinstance(self.row_title_gp, dict) else {}
                return self._title_size(titles, gp, self.row_title_rot, "width")
            return zero
        if component == "row_dend_left":
            if self.show_row_dend and self.row_dend_side == "left" and self._should_cluster_rows():
                return self.row_dend_width
            return zero
        if component == "row_dend_right":
            if self.show_row_dend and self.row_dend_side == "right" and self._should_cluster_rows():
                return self.row_dend_width
            return zero
        if component in ("row_names_left", "row_names_right"):
            side = "left" if component == "row_names_left" else "right"
            if self.show_row_names and self.row_names_side == side:
                labels = self.row_labels if self.row_labels is not None else [str(i) for i in range(self.matrix.shape[0])]
                gp = getattr(self, 'row_names_gp', None)
                rot = self.row_names_rot
                # R anno_text (AnnotationFunction-function.R:2441-2443):
                # width = max_text_width(x, gp) * |cos(rot)| + grobHeight("A", gp) * |sin(rot)|
                import math
                sin_r = abs(math.sin(math.radians(rot)))
                cos_r = abs(math.cos(math.radians(rot)))
                text_w = self._max_text_width_mm(labels, rot=0, gp=gp)
                text_h = self._max_text_height_mm(["A"], rot=0, gp=gp)
                w = text_w * cos_r + text_h * sin_r
                padding = 1.0  # DIMNAME_PADDING
                return grid_py.Unit(w + padding * 2, "mm")
            return zero
        if component in ("left_annotation", "right_annotation"):
            ha = self.left_annotation if component == "left_annotation" else self.right_annotation
            if ha is not None:
                w = ha.width
                if w is not None and isinstance(w, (int, float)):
                    w_unit = grid_py.Unit(float(w), "mm")
                elif w is not None and hasattr(w, '_values'):
                    w_unit = w
                else:
                    n = len(ha.anno_list) if hasattr(ha, 'anno_list') else 1
                    w_unit = grid_py.Unit(max(n * 7, 15), "mm")
                # R Heatmap-class.R:895,921: += ROW_ANNO_PADDING
                w_unit = w_unit + grid_py.Unit(float(ht_opt("ROW_ANNO_PADDING")), "mm")
                return w_unit
            return zero
        if component == "heatmap_body":
            return grid_py.Unit(1, "null")
        return zero

    # ------------------------------------------------------------------
    # re_size
    # ------------------------------------------------------------------

    def re_size(
        self,
        width: Optional[grid_py.Unit] = None,
        height: Optional[grid_py.Unit] = None,
    ) -> None:
        """Adjust the body size of the heatmap."""
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

    # ------------------------------------------------------------------
    # Deep copy
    # ------------------------------------------------------------------

    def copy_all(self) -> "Heatmap":
        """Return a deep copy of this Heatmap."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        show: bool = True,
        filename: Optional[str] = None,
        width: float = 7.0,
        height: float = 7.0,
        dpi: float = 150.0,
        heatmap_legend_side: str = "right",
        show_heatmap_legend: bool = True,
        show_annotation_legend: bool = True,
        annotation_legend_list: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Render the heatmap using grid_py.

        Parameters
        ----------
        show : bool
            If ``True``, display the result (for interactive use).
        filename : str, optional
            If provided, save output to this file (PNG).
        width : float
            Device width in inches.
        height : float
            Device height in inches.
        dpi : float
            Dots per inch.
        heatmap_legend_side : str
            Side for legends: ``"right"``, ``"left"``, ``"top"``, or ``"bottom"``.
        show_heatmap_legend : bool
            Whether to show the heatmap color legend.
        show_annotation_legend : bool
            Whether to show annotation legends.
        annotation_legend_list : list, optional
            Additional custom legend objects to display.
        """
        # Delegate to HeatmapList for complete rendering including legends.
        # This mirrors R's behavior where Heatmap.draw() creates a
        # HeatmapList internally and calls its draw() method.
        from .heatmap_list import HeatmapList

        ht_list = HeatmapList()
        ht_list.add_heatmap(self)
        ht_list.draw(
            show=show,
            width=width,
            height=height,
            dpi=dpi,
            filename=filename,
            heatmap_legend_side=heatmap_legend_side,
            show_heatmap_legend=show_heatmap_legend,
            show_annotation_legend=show_annotation_legend,
            annotation_legend_list=annotation_legend_list,
        )

    def _draw_into_viewport(self) -> None:
        """Draw the complete heatmap into the current viewport.

        Called by ``HeatmapList._draw_single_heatmap`` so that body,
        dendrograms, names, titles, and annotations are all rendered
        and their viewports registered for ``decorate_*`` functions.
        """
        from .heatmap_list import _register_component

        if not self._layout_computed:
            self.make_layout()
        assert self._row_order_list is not None
        assert self._column_order_list is not None

        n_row_slices = len(self._row_order_list)
        n_col_slices = len(self._column_order_list)

        row_components = [
            "column_title_top", "column_dend_top", "column_names_top",
            "top_annotation", "heatmap_body", "bottom_annotation",
            "column_names_bottom", "column_dend_bottom", "column_title_bottom",
        ]
        col_components = [
            "row_title_left", "left_annotation", "row_names_left",
            "row_dend_left", "heatmap_body", "row_dend_right",
            "row_names_right", "right_annotation", "row_title_right",
        ]

        row_heights = [self.component_height(c) for c in row_components]
        col_widths = [self.component_width(c) for c in col_components]

        layout = grid_py.GridLayout(
            nrow=len(row_components), ncol=len(col_components),
            heights=grid_py.unit_c(*row_heights),
            widths=grid_py.unit_c(*col_widths),
        )
        main_vp = grid_py.Viewport(name=f"{self.name}_main", layout=layout)
        grid_py.push_viewport(main_vp)

        body_row = row_components.index("heatmap_body") + 1
        body_col = col_components.index("heatmap_body") + 1

        # Draw all components
        self._draw_heatmap_body(body_row, body_col, n_row_slices, n_col_slices)
        self._draw_column_dendrograms_grid(row_components, col_components)
        self._draw_row_dendrograms_grid(row_components, col_components)
        self._draw_column_names_grid(row_components, col_components)
        self._draw_row_names_grid(row_components, col_components)
        self._draw_column_title_grid(row_components, col_components)
        self._draw_row_title_grid(row_components, col_components)
        self._draw_annotations_grid(row_components, col_components, body_row, body_col)

        grid_py.up_viewport()  # main_vp

        if self.post_fun is not None:
            self.post_fun(self)

    # ------------------------------------------------------------------
    # Heatmap body drawing
    # ------------------------------------------------------------------

    def _prepare_raster_image(self, col_matrix: np.ndarray) -> Any:
        """Convert a color matrix into a raster image for ``grid_raster``."""
        image: Any = np.asarray(col_matrix, dtype=object)
        renderer = grid_py.get_state().get_renderer()
        if renderer is None:
            return image

        quality = max(float(self.raster_quality), 1.0)
        target_w = max(
            1,
            int(round(
                float(grid_py.convert_width(
                    grid_py.Unit(1, "npc"),
                    "inches",
                    valueOnly=True,
                )[0]) * renderer.dpi * quality
            )),
        )
        target_h = max(
            1,
            int(round(
                float(grid_py.convert_height(
                    grid_py.Unit(1, "npc"),
                    "inches",
                    valueOnly=True,
                )[0]) * renderer.dpi * quality
            )),
        )

        if image.shape[1] == target_w and image.shape[0] == target_h:
            return image

        try:
            from PIL import Image, ImageColor

            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    rgba[r, c] = ImageColor.getcolor(str(image[r, c]), "RGBA")

            pil_image = Image.fromarray(rgba, mode="RGBA")
            resized = pil_image.resize(
                (target_w, target_h),
                resample=Image.Resampling.NEAREST,
            )
            return np.asarray(resized)
        except Exception:
            return image

    def _draw_heatmap_body(
        self, body_row: int, body_col: int,
        n_row_slices: int, n_col_slices: int,
    ) -> None:
        """Draw heatmap body cells into the body viewport."""
        from .heatmap_list import _register_component
        # Push into the body cell of the main layout
        body_vp = grid_py.Viewport(
            layout_pos_row=body_row,
            layout_pos_col=body_col,
            name=f"{self.name}_heatmap_body_wrap",
        )
        grid_py.push_viewport(body_vp)

        slice_layout = self._layout["slice"]
        slice_x = slice_layout["x"]
        slice_y = slice_layout["y"]
        slice_width = slice_layout["width"]
        slice_height = slice_layout["height"]

        for ri in range(n_row_slices):
            for ci in range(n_col_slices):
                row_ord = self._row_order_list[ri]
                col_ord = self._column_order_list[ci]
                sub_mat = self.matrix[np.ix_(row_ord, col_ord)]
                col_matrix = self._map_to_colors(sub_mat)

                nr, nc = sub_mat.shape
                if nr == 0 or nc == 0:
                    continue

                # R: viewport(...) defaults to clip="inherit" (no clip).
                # Heatmap3D sets heatmap_param["clip_body"]=False so bar
                # projections can extend beyond the body boundary.
                _clip = getattr(self, "heatmap_param", {}).get(
                    "clip_body", True
                )
                slice_vp = grid_py.Viewport(
                    x=slice_x[ci],
                    y=slice_y[ri],
                    width=slice_width[ci],
                    height=slice_height[ri],
                    just=["left", "top"],
                    clip=_clip,
                    name=f"{self.name}_heatmap_body_{ri + 1}_{ci + 1}",
                )
                grid_py.push_viewport(slice_vp)

                # Draw cells
                cell_width = 1.0 / nc
                cell_height = 1.0 / nr

                x_positions = [(j + 0.5) / nc for j in range(nc)]
                y_positions = [1.0 - (i + 0.5) / nr for i in range(nr)]

                gp_kwargs: Dict[str, Any] = {}
                if hasattr(self.rect_gp, "params"):
                    gp_kwargs = dict(self.rect_gp.params)

                use_raster = bool(self.use_raster and self.cell_fun is None)
                if gp_kwargs.get("type") == "none":
                    use_raster = False

                if use_raster:
                    image = self._prepare_raster_image(col_matrix)
                    grid_py.grid_raster(
                        image=image,
                        x=0.5,
                        y=0.5,
                        width=1.0,
                        height=1.0,
                        default_units="npc",
                        interpolate=False,
                        name=f"{self.name}_body_raster_{ri}_{ci}",
                    )
                elif gp_kwargs.get("type") != "none":
                    all_x = []
                    all_y = []
                    all_fill = []
                    for i_r in range(nr):
                        for j_c in range(nc):
                            all_x.append(x_positions[j_c])
                            all_y.append(y_positions[i_r])
                            all_fill.append(col_matrix[i_r, j_c])

                    gp_kwargs["fill"] = all_fill

                    grid_py.grid_rect(
                        x=all_x,
                        y=all_y,
                        width=cell_width,
                        height=cell_height,
                        default_units="npc",
                        just="centre",
                        gp=grid_py.Gpar(**gp_kwargs),
                        name=f"{self.name}_body_rect_{ri}_{ci}",
                    )

                # Border around the whole slice
                if self.border is not None:
                    border_gp = (
                        grid_py.Gpar(**dict(self.border_gp.params))
                        if hasattr(self.border_gp, "params")
                        else grid_py.Gpar(col="black")
                    )
                    border_gp._params["fill"] = "transparent"
                    if self.border is not True:
                        border_gp._params["col"] = self.border
                    grid_py.grid_rect(
                        gp=border_gp,
                        name=f"{self.name}_body_border_{ri}_{ci}",
                    )

                # cell_fun callback
                if self.cell_fun is not None:
                    for i_r in range(nr):
                        for j_c in range(nc):
                            self.cell_fun(
                                col_ord[j_c], row_ord[i_r],
                                grid_py.Unit(x_positions[j_c], "npc"),
                                grid_py.Unit(y_positions[i_r], "npc"),
                                grid_py.Unit(cell_width, "npc"),
                                grid_py.Unit(cell_height, "npc"),
                                col_matrix[i_r, j_c],
                            )

                # layer_fun callback
                if self.layer_fun is not None:
                    n_cells = nr * nc
                    j_arr = np.tile(col_ord, nr)
                    i_arr = np.repeat(row_ord, nc)
                    x_arr = [grid_py.Unit(x_positions[j % nc], "npc") for j in range(n_cells)]
                    y_arr = [grid_py.Unit(y_positions[j // nc], "npc") for j in range(n_cells)]
                    w_arr = [grid_py.Unit(cell_width, "npc")] * n_cells
                    h_arr = [grid_py.Unit(cell_height, "npc")] * n_cells
                    fill_arr = [col_matrix[j // nc, j % nc] for j in range(n_cells)]
                    self.layer_fun(
                        j_arr, i_arr, x_arr, y_arr, w_arr, h_arr, fill_arr,
                    )

                # Register viewport for decorate_heatmap_body()
                _register_component(
                    f"heatmap_body_{self.name}_{ri + 1}_{ci + 1}",
                    f"{self.name}_heatmap_body_{ri + 1}_{ci + 1}",
                )

                grid_py.up_viewport()  # pop slice_vp

        grid_py.up_viewport()  # pop body_vp

    # ------------------------------------------------------------------
    # Dendrogram drawing
    # ------------------------------------------------------------------

    def _dendrogram_segments(self, Z: np.ndarray, orientation: str = "top"):
        """Convert a scipy linkage matrix to dendrogram segment coordinates.

        Returns lists of (x0, y0, x1, y1) in npc-like coordinates.
        """
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        dend_data = scipy_dendrogram(Z, no_plot=True)
        icoord = np.array(dend_data["icoord"])
        dcoord = np.array(dend_data["dcoord"])

        if len(icoord) == 0:
            return [], [], [], []

        # Normalize x to [0, 1] and y to [0, 1]
        all_x = icoord.ravel()
        all_y = dcoord.ravel()
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = 0.0, all_y.max()

        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        x0_list, y0_list, x1_list, y1_list = [], [], [], []
        for ic, dc in zip(icoord, dcoord):
            # Each U-shape has 3 segments
            for seg in range(3):
                nx0 = (ic[seg] - x_min) / (x_max - x_min)
                nx1 = (ic[seg + 1] - x_min) / (x_max - x_min)
                ny0 = (dc[seg] - y_min) / (y_max - y_min)
                ny1 = (dc[seg + 1] - y_min) / (y_max - y_min)

                if orientation == "top":
                    x0_list.append(nx0)
                    y0_list.append(ny0)
                    x1_list.append(nx1)
                    y1_list.append(ny1)
                elif orientation == "bottom":
                    x0_list.append(nx0)
                    y0_list.append(1.0 - ny0)
                    x1_list.append(nx1)
                    y1_list.append(1.0 - ny1)
                elif orientation == "left":
                    x0_list.append(1.0 - ny0)
                    y0_list.append(nx0)
                    x1_list.append(1.0 - ny1)
                    y1_list.append(nx1)
                elif orientation == "right":
                    x0_list.append(ny0)
                    y0_list.append(nx0)
                    x1_list.append(ny1)
                    y1_list.append(nx1)

        return x0_list, y0_list, x1_list, y1_list

    def _draw_column_dendrograms_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        """Draw column dendrograms using grid_py."""
        from .heatmap_list import _register_component
        if not self._should_cluster_columns():
            return
        if not self.show_column_dend:
            return
        assert self._column_dend_list is not None

        dend_comp = f"column_dend_{self.column_dend_side}"
        if dend_comp not in row_components:
            return

        dend_row = row_components.index(dend_comp) + 1
        body_col = col_components.index("heatmap_body") + 1

        orientation = self.column_dend_side  # "top" or "bottom"

        vp = grid_py.Viewport(
            layout_pos_row=dend_row,
            layout_pos_col=body_col,
            name=f"{self.name}_column_dend",
        )
        grid_py.push_viewport(vp)

        slice_layout = self._layout["slice"]
        slice_x = slice_layout["x"]
        slice_width = slice_layout["width"]

        for ci in range(len(self._column_order_list)):
            Z = self._column_dend_list[ci]
            if Z is None:
                continue

            vp_name = f"{self.name}_column_dend_{ci + 1}"
            slice_vp = grid_py.Viewport(
                x=slice_x[ci],
                y=grid_py.Unit(0, "npc"),
                width=slice_width[ci],
                height=grid_py.Unit(1, "npc"),
                just=["left", "bottom"],
                name=vp_name,
            )
            grid_py.push_viewport(slice_vp)
            # R: ht1_dend_column_1 → lookup: column_dend_ht1_1
            _register_component(
                f"column_dend_{self.name}_{ci + 1}", vp_name)

            x0, y0, x1, y1 = self._dendrogram_segments(Z, orientation)
            if x0:
                grid_py.grid_segments(
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    default_units="npc",
                    gp=self.column_dend_gp,
                    name=f"{self.name}_col_dend_seg_{ci}",
                )

            grid_py.up_viewport()

        grid_py.up_viewport()

    def _draw_row_dendrograms_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        """Draw row dendrograms using grid_py."""
        from .heatmap_list import _register_component
        if not self._should_cluster_rows():
            return
        if not self.show_row_dend:
            return
        assert self._row_dend_list is not None

        dend_comp = f"row_dend_{self.row_dend_side}"
        if dend_comp not in col_components:
            return

        dend_col = col_components.index(dend_comp) + 1
        body_row = row_components.index("heatmap_body") + 1
        orientation = self.row_dend_side  # "left" or "right"

        vp = grid_py.Viewport(
            layout_pos_row=body_row,
            layout_pos_col=dend_col,
            name=f"{self.name}_row_dend",
        )
        grid_py.push_viewport(vp)

        slice_layout = self._layout["slice"]
        slice_y = slice_layout["y"]
        slice_height = slice_layout["height"]

        for ri in range(len(self._row_order_list)):
            Z = self._row_dend_list[ri]
            if Z is None:
                continue

            vp_name = f"{self.name}_row_dend_{ri + 1}"
            slice_vp = grid_py.Viewport(
                x=grid_py.Unit(0, "npc"),
                y=slice_y[ri],
                width=grid_py.Unit(1, "npc"),
                height=slice_height[ri],
                just=["left", "top"],
                name=vp_name,
            )
            grid_py.push_viewport(slice_vp)
            _register_component(f"row_dend_{self.name}_{ri + 1}", vp_name)

            x0, y0, x1, y1 = self._dendrogram_segments(Z, orientation)
            if x0:
                grid_py.grid_segments(
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    default_units="npc",
                    gp=self.row_dend_gp,
                    name=f"{self.name}_row_dend_seg_{ri}",
                )

            grid_py.up_viewport()

        grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Column names drawing
    # ------------------------------------------------------------------

    def _draw_column_names_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        """Draw column names.

        Port of R ``draw_dimnames(which="column")``
        (Heatmap-draw_component.R:463-472) + ``anno_text.column_fun``
        (AnnotationFunction-function.R:2473-2486).

        R positioning logic:
          bottom: y = unit(1, "npc"), just = "right" (text hangs from top)
          top:    y = unit(0, "npc"), just = "left"  (text rises from bottom)
        """
        if not self.show_column_names:
            return
        from .heatmap_list import _register_component

        comp = f"column_names_{self.column_names_side}"
        if comp not in row_components:
            return

        names_row = row_components.index(comp) + 1
        body_col = col_components.index("heatmap_body") + 1

        vp = grid_py.Viewport(
            layout_pos_row=names_row,
            layout_pos_col=body_col,
            name=f"{self.name}_column_names",
        )
        grid_py.push_viewport(vp)

        slice_layout = self._layout["slice"]
        slice_x = slice_layout["x"]
        slice_width = slice_layout["width"]

        rot = self.column_names_rot

        # R anno_text location/just logic (AnnotationFunction-function.R:2394-2428)
        if self.column_names_side == "bottom":
            # R: location = unit(1, "npc"), just inferred from rot
            y_pos = grid_py.Unit(1, "npc") - grid_py.Unit(1, "mm")  # DIMNAME_PADDING
            if self.column_names_centered:
                just = "centre"
            elif rot >= 0 and rot < 180:
                just = "right"
            else:
                just = "left"
        else:  # top
            y_pos = grid_py.Unit(0, "npc") + grid_py.Unit(1, "mm")
            if self.column_names_centered:
                just = "centre"
            elif rot >= 0 and rot < 180:
                just = "left"
            else:
                just = "right"

        if rot == 0:
            just = "centre"

        for ci in range(len(self._column_order_list)):
            col_ord = self._column_order_list[ci]
            nc = len(col_ord)
            if nc == 0:
                continue

            slice_vp = grid_py.Viewport(
                x=slice_x[ci],
                y=grid_py.Unit(0, "npc"),
                width=slice_width[ci],
                height=grid_py.Unit(1, "npc"),
                just=["left", "bottom"],
                name=f"{self.name}_column_names_{ci + 1}",
            )
            grid_py.push_viewport(slice_vp)
            _register_component(
                f"column_names_{self.name}_{ci + 1}", f"{self.name}_column_names_{ci + 1}")

            labels = (
                [self.column_labels[c] for c in col_ord]
                if self.column_labels is not None
                else [str(c) for c in col_ord]
            )

            # R: grid.text(labels, x=(i-0.5)/n, y=location, just=just, rot=rot)
            x_positions = [(j + 0.5) / nc for j in range(nc)]

            grid_py.grid_text(
                label=labels,
                x=x_positions,
                y=y_pos,
                default_units="npc",
                rot=rot,
                just=just,
                gp=self.column_names_gp,
                name=f"{self.name}_col_names_text_{ci}",
            )

            grid_py.up_viewport()

        grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Row names drawing
    # ------------------------------------------------------------------

    def _draw_row_names_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        """Draw row names.

        Port of R ``draw_dimnames(which="row")``
        (Heatmap-draw_component.R:453-462) + ``anno_text.row_fun``
        (AnnotationFunction-function.R:2454-2466).

        R positioning logic:
          right: x = unit(0, "npc") + padding, just = "left"
          left:  x = unit(1, "npc") - padding, just = "right"
        """
        if not self.show_row_names:
            return

        from .heatmap_list import _register_component
        comp = f"row_names_{self.row_names_side}"
        if comp not in col_components:
            return

        names_col = col_components.index(comp) + 1
        body_row = row_components.index("heatmap_body") + 1

        vp = grid_py.Viewport(
            layout_pos_row=body_row,
            layout_pos_col=names_col,
            name=f"{self.name}_row_names",
        )
        grid_py.push_viewport(vp)

        slice_layout = self._layout["slice"]
        slice_y = slice_layout["y"]
        slice_height = slice_layout["height"]

        rot = self.row_names_rot

        # R draw_dimnames (Heatmap-draw_component.R:453-462)
        if self.row_names_side == "right":
            x_pos = grid_py.Unit(0, "npc") + grid_py.Unit(1, "mm")  # DIMNAME_PADDING
            just = "left"
        else:  # left
            x_pos = grid_py.Unit(1, "npc") - grid_py.Unit(1, "mm")
            just = "right"

        if self.row_names_centered:
            just = "centre"
            x_pos = grid_py.Unit(0.5, "npc")

        for ri in range(len(self._row_order_list)):
            row_ord = self._row_order_list[ri]
            nr = len(row_ord)
            if nr == 0:
                continue

            slice_vp = grid_py.Viewport(
                x=grid_py.Unit(0, "npc"),
                y=slice_y[ri],
                width=grid_py.Unit(1, "npc"),
                height=slice_height[ri],
                just=["left", "top"],
                name=f"{self.name}_row_names_{ri + 1}",
            )
            grid_py.push_viewport(slice_vp)
            _register_component(
                f"row_names_{self.name}_{ri + 1}", f"{self.name}_row_names_{ri + 1}")

            labels = (
                [self.row_labels[r] for r in row_ord]
                if self.row_labels is not None
                else [str(r) for r in row_ord]
            )

            # R: grid.text(labels, x=location, y=(n-i+0.5)/n, just=just, rot=rot)
            y_positions = [1.0 - (i + 0.5) / nr for i in range(nr)]

            grid_py.grid_text(
                label=labels,
                x=x_pos,
                y=y_positions,
                default_units="npc",
                rot=rot,
                just=just,
                gp=self.row_names_gp,
                name=f"{self.name}_row_names_text_{ri}",
            )

            grid_py.up_viewport()

        grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Title drawing
    # ------------------------------------------------------------------

    def _draw_column_title_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        # Auto-generate split titles — use split labels when available (R behavior)
        n_col_slices = len(self._column_order_list) if self._column_order_list else 1
        column_title = self.column_title
        if column_title is None and n_col_slices > 1:
            if self._column_split_labels is not None:
                column_title = [str(lbl) for lbl in self._column_split_labels]
            else:
                column_title = [str(i + 1) for i in range(n_col_slices)]
        if column_title is None:
            return

        comp = f"column_title_{self.column_title_side}"
        if comp not in row_components:
            return

        title_row = row_components.index(comp) + 1
        body_col = col_components.index("heatmap_body") + 1

        from .heatmap_list import _register_component
        vp = grid_py.Viewport(
            layout_pos_row=title_row,
            layout_pos_col=body_col,
            name=f"{self.name}_column_title",
        )
        grid_py.push_viewport(vp)

        gp = self.column_title_gp
        if isinstance(gp, dict):
            gp = grid_py.Gpar(**gp)

        titles = column_title if isinstance(column_title, list) else [column_title]

        if len(titles) == 1 or n_col_slices <= 1:
            grid_py.grid_text(
                label=titles[0], x=0.5, y=0.5, default_units="npc",
                rot=self.column_title_rot, gp=gp,
                name=f"{self.name}_col_title_text",
            )
        else:
            slice_layout = self._layout["slice"]
            slice_x = slice_layout["x"]
            slice_width = slice_layout["width"]
            for i in range(min(len(titles), n_col_slices)):
                slice_vp = grid_py.Viewport(
                    x=slice_x[i], y=grid_py.Unit(0, "npc"),
                    width=slice_width[i], height=grid_py.Unit(1, "npc"),
                    just=["left", "bottom"],
                    name=f"{self.name}_col_title_{i+1}",
                )
                grid_py.push_viewport(slice_vp)
                _register_component(
                    f"column_title_{self.name}_{i + 1}",
                    f"{self.name}_col_title_{i+1}")
                grid_py.grid_text(
                    label=titles[i], x=0.5, y=0.5,
                    default_units="npc", rot=0, gp=gp,
                )
                grid_py.up_viewport()

        grid_py.up_viewport()

    def _draw_row_title_grid(
        self,
        row_components: List[str],
        col_components: List[str],
    ) -> None:
        """Draw row title(s).

        Port of R ``Heatmap-layout.R:141-149``:
        - Single title + multiple slices → draw once spanning all slices
        - Multiple titles (one per slice) → draw each in its slice
        - When row_km/row_split is used, R auto-generates per-slice titles
        """
        # Auto-generate split titles if row_title is None but we have multiple slices.
        # R uses the split factor labels as per-slice titles (Heatmap-class.R).
        n_row_slices = len(self._row_order_list) if self._row_order_list else 1
        row_title = self.row_title
        if row_title is None and n_row_slices > 1:
            if self._row_split_labels is not None:
                row_title = [str(lbl) for lbl in self._row_split_labels]
            else:
                row_title = [str(i + 1) for i in range(n_row_slices)]
        if row_title is None:
            return

        comp = f"row_title_{self.row_title_side}"
        if comp not in col_components:
            return

        title_col = col_components.index(comp) + 1
        body_row = row_components.index("heatmap_body") + 1

        from .heatmap_list import _register_component
        vp = grid_py.Viewport(
            layout_pos_row=body_row,
            layout_pos_col=title_col,
            name=f"{self.name}_row_title",
        )
        grid_py.push_viewport(vp)

        gp = self.row_title_gp
        if isinstance(gp, dict):
            gp = grid_py.Gpar(**gp)

        titles = row_title if isinstance(row_title, list) else [row_title]

        if len(titles) == 1 or n_row_slices <= 1:
            # Single title centered in the row_title column
            grid_py.grid_text(
                label=titles[0],
                x=0.5, y=0.5, default_units="npc",
                rot=self.row_title_rot, gp=gp,
                name=f"{self.name}_row_title_text",
            )
        else:
            # Per-slice titles (R Heatmap-layout.R:145-147)
            slice_layout = self._layout["slice"]
            slice_y = slice_layout["y"]
            slice_height = slice_layout["height"]
            for i in range(min(len(titles), n_row_slices)):
                slice_vp = grid_py.Viewport(
                    x=grid_py.Unit(0, "npc"),
                    y=slice_y[i],
                    width=grid_py.Unit(1, "npc"),
                    height=slice_height[i],
                    just=["left", "top"],
                    name=f"{self.name}_row_title_{i+1}",
                )
                grid_py.push_viewport(slice_vp)
                _register_component(
                    f"row_title_{self.name}_{i + 1}",
                    f"{self.name}_row_title_{i+1}")
                grid_py.grid_text(
                    label=titles[i], x=0.5, y=0.5,
                    default_units="npc",
                    rot=self.row_title_rot, gp=gp,
                )
                grid_py.up_viewport()

        grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Annotation drawing
    # ------------------------------------------------------------------

    def _draw_annotations_grid(
        self,
        row_components: List[str],
        col_components: List[str],
        body_row: int,
        body_col: int,
    ) -> None:
        """Draw annotations by delegating to HeatmapAnnotation.draw().

        Mirrors R's annotation drawing which passes the observation index
        (row order for row annotations, column order for column annotations)
        to each HeatmapAnnotation's draw method.
        """
        from .heatmap_list import _register_component

        # Compute full observation indices for annotations
        # For unsplit heatmaps, concatenate all slices into one index
        row_index = np.concatenate(self._row_order_list) if self._row_order_list else np.arange(self.matrix.shape[0])
        col_index = np.concatenate(self._column_order_list) if self._column_order_list else np.arange(self.matrix.shape[1])

        _col_pad = grid_py.Unit(float(ht_opt("COLUMN_ANNO_PADDING")), "mm")
        _row_pad = grid_py.Unit(float(ht_opt("ROW_ANNO_PADDING")), "mm")

        for anno_attr, comp_list, comp_name, pos_row, pos_col, index in [
            ("top_annotation", row_components, "top_annotation", None, body_col, col_index),
            ("bottom_annotation", row_components, "bottom_annotation", None, body_col, col_index),
            ("left_annotation", col_components, "left_annotation", body_row, None, row_index),
            ("right_annotation", col_components, "right_annotation", body_row, None, row_index),
        ]:
            ha = getattr(self, anno_attr, None)
            if ha is None or comp_name not in comp_list:
                continue

            # Inject heatmap reference for anno_summary (R: parent.frame(7))
            if hasattr(ha, 'anno_list'):
                for sa in ha.anno_list.values():
                    af = getattr(sa, '_anno_fun', None)
                    if af is not None and getattr(af, '_needs_ht_ref', False):
                        af.var_env["_ht_ref"] = self

            if pos_row is None:
                pos_row = comp_list.index(comp_name) + 1
            if pos_col is None:
                pos_col = comp_list.index(comp_name) + 1

            # Push layout-cell viewport
            vp_name = f"{self.name}_{comp_name}"
            vp = grid_py.Viewport(
                layout_pos_row=pos_row,
                layout_pos_col=pos_col,
                clip=False,
                name=vp_name,
            )
            grid_py.push_viewport(vp)

            # Push inner viewport with padding gap (R Heatmap-layout.R:354-417)
            # The gap separates the annotation from the heatmap body.
            if comp_name == "top_annotation":
                inner_vp = grid_py.Viewport(
                    y=_col_pad, height=grid_py.Unit(1, "npc") - _col_pad,
                    just=["center", "bottom"], clip=False,
                    name=f"{vp_name}_inner",
                )
            elif comp_name == "bottom_annotation":
                inner_vp = grid_py.Viewport(
                    y=grid_py.Unit(0, "npc"),
                    height=grid_py.Unit(1, "npc") - _col_pad,
                    just=["center", "bottom"], clip=False,
                    name=f"{vp_name}_inner",
                )
            elif comp_name == "left_annotation":
                inner_vp = grid_py.Viewport(
                    x=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(1, "npc") - _row_pad,
                    just=["left", "center"], clip=False,
                    name=f"{vp_name}_inner",
                )
            else:  # right_annotation
                inner_vp = grid_py.Viewport(
                    x=_row_pad,
                    width=grid_py.Unit(1, "npc") - _row_pad,
                    just=["left", "center"], clip=False,
                    name=f"{vp_name}_inner",
                )
            grid_py.push_viewport(inner_vp)
            if hasattr(ha, 'draw'):
                ha.draw(index=index, k=1, n=1)

            # Register each individual annotation for decorate_annotation()
            anno_list = getattr(ha, 'anno_list', {})
            for anno_name in anno_list:
                _register_component(f"annotation_{anno_name}", vp_name)

            grid_py.up_viewport()  # inner_vp
            grid_py.up_viewport()  # layout cell vp

    # ------------------------------------------------------------------
    # make_row_cluster / make_column_cluster (R-compatible aliases)
    # ------------------------------------------------------------------

    def make_row_cluster(self) -> None:
        """Compute row clustering (alias for internal _compute_row_layout)."""
        self._compute_row_layout()

    def make_column_cluster(self) -> None:
        """Compute column clustering (alias for internal _compute_column_layout)."""
        self._compute_column_layout()

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Heatmap(name={self.name!r}, nrow={self.nrow}, ncol={self.ncol}, "
            f"layout_computed={self._layout_computed})"
        )
