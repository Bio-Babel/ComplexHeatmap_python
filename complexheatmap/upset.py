"""UpSet plot implementation.

Provides :func:`make_comb_mat` for creating combination matrices from
sets, and :func:`UpSet` for rendering UpSet plots as ``Heatmap``
objects compatible with the :class:`~complexheatmap.heatmap_list.HeatmapList`
drawing pipeline.

This mirrors the ``UpSet`` / ``make_comb_mat`` functionality in the R
*ComplexHeatmap* package.

Examples
--------
>>> from complexheatmap.upset import make_comb_mat, UpSet
>>> sets = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}}
>>> m = make_comb_mat(sets)
>>> comb_size(m)
array([...])
"""

from __future__ import annotations

import itertools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import grid_py

__all__ = [
    "make_comb_mat",
    "CombMat",
    "comb_degree",
    "comb_name",
    "comb_size",
    "set_name",
    "set_size",
    "extract_comb",
    "normalize_comb_mat",
    "UpSet",
    "upset_top_annotation",
    "upset_right_annotation",
    "upset_left_annotation",
]


# ======================================================================
# CombMat container
# ======================================================================

class CombMat:
    """Combination matrix container for UpSet plots.

    Stores the binary set membership matrix and precomputed combination
    information produced by :func:`make_comb_mat`.

    Attributes
    ----------
    set_names : list of str
        Names of the input sets.
    comb_mat : numpy.ndarray
        Binary matrix of shape ``(n_sets, n_combinations)`` where each
        column encodes which sets participate in that combination.
    comb_sizes : numpy.ndarray
        Number of elements in each combination.
    set_sizes : numpy.ndarray
        Total size of each input set.
    mode : str
        Combination mode: ``"distinct"``, ``"intersect"``, or ``"union"``.
    elements : dict
        Mapping from combination index to the set of element names.
    """

    def __init__(
        self,
        set_names: List[str],
        comb_mat: np.ndarray,
        comb_sizes: np.ndarray,
        set_sizes: np.ndarray,
        mode: str,
        elements: Optional[Dict[int, set]] = None,
    ) -> None:
        self.set_names: List[str] = list(set_names)
        self.comb_mat: np.ndarray = np.asarray(comb_mat, dtype=int)
        _cs = np.asarray(comb_sizes)
        self.comb_sizes: np.ndarray = _cs if np.issubdtype(_cs.dtype, np.floating) else _cs.astype(int)
        self.set_sizes: np.ndarray = np.asarray(set_sizes, dtype=int)
        self.mode: str = mode
        self.elements: Dict[int, set] = elements if elements is not None else {}

    @property
    def n_sets(self) -> int:
        """Number of input sets."""
        return len(self.set_names)

    @property
    def n_comb(self) -> int:
        """Number of combinations."""
        return self.comb_mat.shape[1]

    def __getitem__(self, key: Any) -> "CombMat":
        """Subset combinations (columns) by boolean mask or integer indices.

        Mirrors R's ``[.comb_mat`` which subsets along the combination axis.
        """
        idx = np.asarray(key)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        new_comb_mat = self.comb_mat[:, idx]
        new_comb_sizes = self.comb_sizes[idx]
        new_elements = {i: self.elements.get(int(old_i), set())
                        for i, old_i in enumerate(idx)}
        return CombMat(
            set_names=self.set_names,
            comb_mat=new_comb_mat,
            comb_sizes=new_comb_sizes,
            set_sizes=self.set_sizes,
            mode=self.mode,
            elements=new_elements,
        )

    def __repr__(self) -> str:
        return (
            f"CombMat(n_sets={self.n_sets}, n_comb={self.n_comb}, "
            f"mode={self.mode!r})"
        )


# ======================================================================
# make_comb_mat
# ======================================================================

def make_comb_mat(
    x: Union[Dict[str, set], np.ndarray],
    mode: str = "distinct",
    top_n_sets: Optional[int] = None,
    min_set_size: int = 0,
    universal_set: Optional[set] = None,
    complement_size: Optional[int] = None,
    value_fun: Optional[Callable[[set], int]] = None,
) -> CombMat:
    """Create a combination matrix for an UpSet plot.

    Parameters
    ----------
    x : dict of set, or numpy.ndarray
        If a dict, keys are set names and values are Python sets of
        element names.  If a 2-D binary numpy array, rows are elements
        and columns are sets.
    mode : str
        Combination mode:

        - ``"distinct"`` (default): elements that belong to *exactly*
          the indicated sets.
        - ``"intersect"``: elements in the intersection of the
          indicated sets (may also belong to others).
        - ``"union"``: elements in the union of the indicated sets.
    top_n_sets : int, optional
        Keep only the *top_n_sets* largest sets before computing
        combinations.
    min_set_size : int
        Remove sets smaller than this threshold.
    universal_set : set, optional
        Element universe.  Defaults to the union of all sets.
    complement_size : int, optional
        If given, include the complement (elements in no set) with this
        size.
    value_fun : callable, optional
        Custom function to compute combination size from the element
        set.  Defaults to ``len``.

    Returns
    -------
    CombMat
        A :class:`CombMat` container.

    Raises
    ------
    ValueError
        If *mode* is not one of the recognised values, or if *x* is
        neither a dict nor a 2-D array.
    """
    if mode not in ("distinct", "intersect", "union"):
        raise ValueError(
            f"`mode` must be 'distinct', 'intersect', or 'union', got {mode!r}"
        )

    if value_fun is None:
        value_fun = len

    # ------------------------------------------------------------------
    # Convert input to dict of sets
    # ------------------------------------------------------------------
    if isinstance(x, np.ndarray):
        if x.ndim != 2:
            raise ValueError("Array input must be 2-D (elements x sets).")
        n_elem, n_sets = x.shape
        set_names = [f"set{i + 1}" for i in range(n_sets)]
        sets_dict: Dict[str, set] = {}
        for ci in range(n_sets):
            sets_dict[set_names[ci]] = {
                ri for ri in range(n_elem) if x[ri, ci] != 0
            }
    elif isinstance(x, dict):
        sets_dict = {k: set(v) for k, v in x.items()}
        set_names = list(sets_dict.keys())
    else:
        raise ValueError("`x` must be a dict of sets or a 2-D numpy array.")

    # ------------------------------------------------------------------
    # Filter sets
    # ------------------------------------------------------------------
    if min_set_size > 0:
        sets_dict = {k: v for k, v in sets_dict.items() if len(v) >= min_set_size}
        set_names = list(sets_dict.keys())

    if top_n_sets is not None and top_n_sets < len(set_names):
        sorted_names = sorted(set_names, key=lambda k: len(sets_dict[k]), reverse=True)
        set_names = sorted_names[:top_n_sets]
        sets_dict = {k: sets_dict[k] for k in set_names}

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    if universal_set is not None:
        universe = set(universal_set)
    else:
        universe: set = set()
        for s in sets_dict.values():
            universe |= s

    n_sets_final = len(set_names)
    set_sizes = np.array([len(sets_dict[k]) for k in set_names], dtype=int)

    # ------------------------------------------------------------------
    # Enumerate combinations (all non-empty subsets of sets)
    # ------------------------------------------------------------------
    comb_cols: List[np.ndarray] = []
    comb_sizes_list: List[int] = []
    comb_elements: Dict[int, set] = {}

    for r in range(1, n_sets_final + 1):
        for combo in itertools.combinations(range(n_sets_final), r):
            indicator = np.zeros(n_sets_final, dtype=int)
            for ci in combo:
                indicator[ci] = 1

            included_sets = [sets_dict[set_names[ci]] for ci in combo]
            excluded_sets = [
                sets_dict[set_names[ci]]
                for ci in range(n_sets_final)
                if ci not in combo
            ]

            if mode == "distinct":
                elems = set.intersection(*included_sets) if included_sets else set()
                for ex in excluded_sets:
                    elems -= ex
            elif mode == "intersect":
                elems = set.intersection(*included_sets) if included_sets else set()
            elif mode == "union":
                elems = set.union(*included_sets) if included_sets else set()
            else:
                elems = set()

            elems &= universe

            size = value_fun(elems)
            comb_idx = len(comb_cols)
            comb_cols.append(indicator)
            comb_sizes_list.append(size)
            comb_elements[comb_idx] = elems

    # ------------------------------------------------------------------
    # Filter out empty combinations (matching R default behaviour)
    # ------------------------------------------------------------------
    non_empty = [i for i, s in enumerate(comb_sizes_list) if s > 0]
    comb_cols = [comb_cols[i] for i in non_empty]
    comb_sizes_list = [comb_sizes_list[i] for i in non_empty]
    comb_elements = {new: comb_elements[old] for new, old in enumerate(non_empty)}

    # ------------------------------------------------------------------
    # Optional complement
    # ------------------------------------------------------------------
    if complement_size is not None:
        indicator = np.zeros(n_sets_final, dtype=int)
        comb_cols.append(indicator)
        comb_sizes_list.append(complement_size)
        comb_elements[len(comb_cols) - 1] = set()

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    if comb_cols:
        comb_matrix = np.column_stack(comb_cols)
    else:
        comb_matrix = np.zeros((n_sets_final, 0), dtype=int)

    comb_sizes_arr = np.array(comb_sizes_list, dtype=int)

    return CombMat(
        set_names=set_names,
        comb_mat=comb_matrix,
        comb_sizes=comb_sizes_arr,
        set_sizes=set_sizes,
        mode=mode,
        elements=comb_elements,
    )


# ======================================================================
# Accessors
# ======================================================================

def comb_degree(m: CombMat) -> np.ndarray:
    """Degree (number of sets) for each combination.

    Parameters
    ----------
    m : CombMat
        Combination matrix.

    Returns
    -------
    numpy.ndarray
        1-D integer array of length ``n_comb``.
    """
    return m.comb_mat.sum(axis=0)


def comb_name(m: CombMat) -> List[str]:
    """Binary-string name for each combination.

    Each name is a binary string of ``0`` and ``1`` characters, one per
    set, matching the R convention (e.g. ``"110"`` means sets 1 and 2
    participate).

    Parameters
    ----------
    m : CombMat
        Combination matrix.

    Returns
    -------
    list of str
    """
    names: List[str] = []
    for ci in range(m.n_comb):
        names.append("".join(str(m.comb_mat[ri, ci]) for ri in range(m.n_sets)))
    return names


def comb_size(m: CombMat) -> np.ndarray:
    """Size (element count) for each combination.

    Parameters
    ----------
    m : CombMat
        Combination matrix.

    Returns
    -------
    numpy.ndarray
        1-D integer array of length ``n_comb``.
    """
    return m.comb_sizes.copy()


def set_name(m: CombMat) -> List[str]:
    """Names of the input sets.

    Parameters
    ----------
    m : CombMat
        Combination matrix.

    Returns
    -------
    list of str
    """
    return list(m.set_names)


def set_size(m: CombMat) -> np.ndarray:
    """Sizes of the input sets.

    Parameters
    ----------
    m : CombMat
        Combination matrix.

    Returns
    -------
    numpy.ndarray
        1-D integer array of length ``n_sets``.
    """
    return m.set_sizes.copy()


def extract_comb(m: CombMat, comb_name_str: str) -> set:
    """Extract the elements belonging to a named combination.

    Parameters
    ----------
    m : CombMat
        Combination matrix.
    comb_name_str : str
        Combination name as returned by :func:`comb_name` (e.g.
        ``"110"``).

    Returns
    -------
    set
        Element set.

    Raises
    ------
    KeyError
        If the combination name is not found.
    """
    names = comb_name(m)
    for idx, cn in enumerate(names):
        if cn == comb_name_str:
            return set(m.elements.get(idx, set()))
    raise KeyError(f"Combination {comb_name_str!r} not found.")


def normalize_comb_mat(m: CombMat, full: bool = False) -> CombMat:
    """Normalise combination sizes to fractions.

    Parameters
    ----------
    m : CombMat
        Combination matrix.
    full : bool
        If ``True``, normalise by the universe size (sum of all
        combination sizes).  If ``False``, normalise each combination
        by the minimum participating set size.

    Returns
    -------
    CombMat
        A new ``CombMat`` with fractional ``comb_sizes``.
    """
    if full:
        total = float(m.comb_sizes.sum()) or 1.0
        normed = m.comb_sizes.astype(float) / total
    else:
        normed = np.empty(m.n_comb, dtype=float)
        for ci in range(m.n_comb):
            participating = [
                m.set_sizes[ri]
                for ri in range(m.n_sets)
                if m.comb_mat[ri, ci]
            ]
            denom = float(min(participating)) if participating else 1.0
            normed[ci] = m.comb_sizes[ci] / max(denom, 1.0)

    return CombMat(
        set_names=list(m.set_names),
        comb_mat=m.comb_mat.copy(),
        comb_sizes=normed,
        set_sizes=m.set_sizes.copy(),
        mode=m.mode,
        elements=dict(m.elements),
    )


# ======================================================================
# Annotation helpers
# ======================================================================

def upset_top_annotation(
    m: CombMat,
    *,
    bar_width: float = 0.6,
    gp: Optional[Dict[str, Any]] = None,
    height: Optional[Any] = None,
    show_annotation_name: bool = True,
    annotation_name: str = "Intersection\nsize",
    ylim: Optional[Tuple[float, float]] = None,
    axis_param: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Create the top barplot annotation for an UpSet plot.

    Shows the combination (intersection) sizes as bars above the
    dot-matrix.

    Parameters
    ----------
    m : CombMat
        Combination matrix.
    bar_width : float
        Relative bar width (0 to 1).
    gp : dict, optional
        Graphical parameters for the bars (``fill``, ``col``, etc.).
    height : object, optional
        Annotation height (grid_py.Unit or numeric mm).
    show_annotation_name : bool
        Whether to show the annotation name label.
    annotation_name : str
        Label text.
    ylim : tuple of float, optional
        Y-axis limits ``(min, max)``.
    axis_param : dict, optional
        Additional axis parameters.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`~complexheatmap.annotation_functions.anno_barplot`.

    Returns
    -------
    HeatmapAnnotation
        A column annotation containing a barplot of combination sizes.
    """
    from .heatmap_annotation import HeatmapAnnotation
    from .annotation_functions import anno_barplot

    sizes = m.comb_sizes.copy().astype(float)
    fill_col = "black"
    if gp is not None:
        fill_col = gp.get("fill", gp.get("facecolor", "black"))

    _height = height
    if _height is None:
        _height = grid_py.Unit(3, "cm")
    elif not isinstance(_height, grid_py.Unit):
        _height = grid_py.Unit(float(_height), "mm")

    barplot_gp = {"fill": fill_col, "col": fill_col}

    anno = anno_barplot(
        x=sizes,
        which="column",
        bar_width=bar_width,
        gp=barplot_gp,
        ylim=ylim,
        axis_param=axis_param or {"side": "left"},
        height=_height,
        **kwargs,
    )

    ha = HeatmapAnnotation(
        **{"Intersection\nsize": anno},
        which="column",
        show_annotation_name=show_annotation_name,
        annotation_name_side="left",
    )
    return ha


def upset_right_annotation(
    m: CombMat,
    *,
    bar_width: float = 0.6,
    gp: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    show_annotation_name: bool = True,
    annotation_name: str = "Set size",
    xlim: Optional[Tuple[float, float]] = None,
    axis_param: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Create the right barplot annotation for an UpSet plot.

    Shows the set sizes as horizontal bars to the right of the
    dot-matrix.

    Parameters
    ----------
    m : CombMat
        Combination matrix.
    bar_width : float
        Relative bar width.
    gp : dict, optional
        Graphical parameters for the bars.
    width : object, optional
        Annotation width (grid_py.Unit or numeric mm).
    show_annotation_name : bool
        Whether to show the annotation name label.
    annotation_name : str
        Label text.
    xlim : tuple of float, optional
        X-axis limits ``(min, max)``.
    axis_param : dict, optional
        Additional axis parameters.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    HeatmapAnnotation
        A row annotation with set size barplots.
    """
    from .heatmap_annotation import HeatmapAnnotation
    from .annotation_functions import anno_barplot

    sizes = m.set_sizes.copy().astype(float)
    fill_col = "black"
    if gp is not None:
        fill_col = gp.get("fill", gp.get("facecolor", "black"))

    _width = width
    if _width is None:
        _width = grid_py.Unit(3, "cm")
    elif not isinstance(_width, grid_py.Unit):
        _width = grid_py.Unit(float(_width), "mm")

    barplot_gp = {"fill": fill_col, "col": fill_col}

    anno = anno_barplot(
        x=sizes,
        which="row",
        bar_width=bar_width,
        gp=barplot_gp,
        ylim=xlim,
        axis_param=axis_param or {"side": "bottom"},
        width=_width,
        **kwargs,
    )

    ha = HeatmapAnnotation(
        **{"Set size": anno},
        which="row",
        show_annotation_name=show_annotation_name,
        annotation_name_side="bottom",
    )
    return ha


def upset_left_annotation(
    m: CombMat,
    *,
    bar_width: float = 0.6,
    gp: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    show_annotation_name: bool = True,
    annotation_name: str = "Set size",
    xlim: Optional[Tuple[float, float]] = None,
    axis_param: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Create a left-side barplot annotation for an UpSet plot.

    Identical to :func:`upset_right_annotation` but positioned on the
    left of the dot-matrix.

    Parameters
    ----------
    m : CombMat
        Combination matrix.
    bar_width : float
        Relative bar width.
    gp : dict, optional
        Graphical parameters.
    width : object, optional
        Annotation width (grid_py.Unit or numeric mm).
    show_annotation_name : bool
        Whether to show the annotation name.
    annotation_name : str
        Label text.
    xlim : tuple of float, optional
        X-axis limits.
    axis_param : dict, optional
        Additional axis parameters.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    HeatmapAnnotation
        A row annotation with set size barplots (for left side).
    """
    from .heatmap_annotation import HeatmapAnnotation
    from .annotation_functions import anno_barplot

    sizes = m.set_sizes.copy().astype(float)
    fill_col = "black"
    if gp is not None:
        fill_col = gp.get("fill", gp.get("facecolor", "black"))

    _width = width
    if _width is None:
        _width = grid_py.Unit(3, "cm")
    elif not isinstance(_width, grid_py.Unit):
        _width = grid_py.Unit(float(_width), "mm")

    barplot_gp = {"fill": fill_col, "col": fill_col}

    anno = anno_barplot(
        x=sizes,
        which="row",
        bar_width=bar_width,
        gp=barplot_gp,
        ylim=xlim,
        axis_param=axis_param or {"side": "bottom"},
        width=_width,
        **kwargs,
    )

    ha = HeatmapAnnotation(
        **{"Set size": anno},
        which="row",
        show_annotation_name=show_annotation_name,
        annotation_name_side="bottom",
    )
    return ha


# ======================================================================
# UpSet (main entry point)
# ======================================================================

def UpSet(
    m: CombMat,
    *,
    comb_order: Optional[Union[np.ndarray, Sequence[int]]] = None,
    set_order: Optional[Union[np.ndarray, Sequence[int]]] = None,
    top_annotation: Optional[Any] = None,
    right_annotation: Optional[Any] = None,
    left_annotation: Optional[Any] = None,
    row_names_side: str = "left",
    show_row_names: bool = True,
    row_names_gp: Optional[grid_py.Gpar] = None,
    comb_col: Optional[Union[str, Sequence[str]]] = None,
    pt_size: Optional[grid_py.Unit] = None,
    lwd: float = 2.0,
    bg_col: str = "#F0F0F0",
    bg_pt_col: str = "#CCCCCC",
    set_name_gp: Optional[grid_py.Gpar] = None,
    comb_name_rot: float = 0,
    show_comb_name: bool = False,
    column_title: Optional[str] = None,
    row_title: Optional[str] = None,
    name: str = "upset",
    **kwargs: Any,
) -> Any:
    """Create an UpSet plot as a Heatmap-based object.

    The UpSet plot consists of a binary dot-matrix (sets x combinations)
    with optional barplot annotations showing combination and set sizes.
    It is implemented using :class:`~complexheatmap.heatmap.Heatmap`
    with a custom ``layer_fun`` for the dots-and-connectors rendering.

    Parameters
    ----------
    m : CombMat
        Combination matrix produced by :func:`make_comb_mat`.
    comb_order : array-like of int, optional
        Custom ordering of combinations (0-based column indices into
        ``m.comb_mat``).  Default: sorted by decreasing combination
        size.
    set_order : array-like of int, optional
        Custom ordering of sets (0-based row indices).  Default:
        sorted by decreasing set size.
    top_annotation : HeatmapAnnotation, optional
        Annotation above the matrix.  Defaults to
        :func:`upset_top_annotation`.
    right_annotation : HeatmapAnnotation, optional
        Annotation to the right.  Defaults to
        :func:`upset_right_annotation`.
    left_annotation : HeatmapAnnotation, optional
        Annotation to the left.
    row_names_side : str
        ``"left"`` or ``"right"``.
    show_row_names : bool
        Whether to display set names.
    row_names_gp : grid_py.Gpar, optional
        Graphical parameters for set name labels.
    comb_col : str or sequence of str, optional
        Colour(s) for the combination dots and connectors.
    pt_size : grid_py.Unit, optional
        Dot size. Default ``Unit(3, "mm")``.
    lwd : float
        Connector line width in points.
    bg_col : str
        Background colour for the matrix panel.
    bg_pt_col : str
        Colour for inactive dots.
    set_name_gp : grid_py.Gpar, optional
        Graphical parameters for set name labels (alias for
        ``row_names_gp``).
    comb_name_rot : float
        Rotation angle for combination name labels.
    show_comb_name : bool
        Whether to show combination names.
    column_title : str, optional
        Title above the plot.
    row_title : str, optional
        Title beside the plot.
    name : str
        Heatmap name for the component registry.
    **kwargs
        Forwarded to :class:`~complexheatmap.heatmap.Heatmap`.

    Returns
    -------
    Heatmap
        A Heatmap object that can be added to a HeatmapList.
    """
    from .heatmap import Heatmap

    # Default ordering
    if comb_order is None:
        comb_order = np.argsort(-m.comb_sizes)
    else:
        comb_order = np.asarray(comb_order)

    if set_order is None:
        set_order = np.argsort(-m.set_sizes)
    else:
        set_order = np.asarray(set_order)
        # Convert string names to integer indices (R compatibility)
        if set_order.dtype.kind in ('U', 'S', 'O'):
            name_to_idx = {name: i for i, name in enumerate(m.set_names)}
            set_order = np.array([name_to_idx[str(s)] for s in set_order], dtype=int)

    # Default annotations
    if top_annotation is None:
        top_annotation = upset_top_annotation(m)
    if right_annotation is None:
        right_annotation = upset_right_annotation(m)

    # Default colours
    if comb_col is None:
        comb_col_arr: List[str] = ["black"] * m.n_comb
    elif isinstance(comb_col, str):
        comb_col_arr = [comb_col] * m.n_comb
    else:
        comb_col_arr = list(comb_col)

    if pt_size is None:
        pt_size = grid_py.Unit(3, "mm")

    if row_names_gp is None and set_name_gp is not None:
        row_names_gp = set_name_gp

    # Reorder the combination matrix
    ordered_mat = m.comb_mat[:, comb_order][set_order, :]
    n_sets = len(set_order)
    n_comb = len(comb_order)

    # Build the numeric matrix for the Heatmap (sets x combinations).
    # Value encodes: 0 = inactive, 1 = active.
    heatmap_mat = ordered_mat.astype(float)

    # Row labels = set names in order
    row_labels = [m.set_names[i] for i in set_order]

    # Column labels = combination binary names in order
    all_cnames = comb_name(m)
    col_labels = [all_cnames[i] for i in comb_order]

    # Colours for the ordered combinations
    ordered_comb_col = [comb_col_arr[i] for i in comb_order]

    # Capture closured variables for the layer_fun
    _ordered_mat = ordered_mat
    _ordered_comb_col = ordered_comb_col
    _pt_size = pt_size
    _lwd = lwd
    _bg_pt_col = bg_pt_col

    def _upset_layer_fun(
        j: np.ndarray,
        i: np.ndarray,
        x: Any,
        y: Any,
        w: Any,
        h: Any,
        fill: Any,
    ) -> None:
        """Draw UpSet dots and connecting lines via grid_py."""
        n_cells = len(j)
        if n_cells == 0:
            return

        # Determine local grid dimensions
        unique_j = sorted(set(int(jj) for jj in j))
        unique_i = sorted(set(int(ii) for ii in i))
        n_local_cols = len(unique_j)
        n_local_rows = len(unique_i)

        # Build a lookup from (col_idx, row_idx) -> position index
        pos_map: Dict[Tuple[int, int], int] = {}
        for idx in range(n_cells):
            pos_map[(int(j[idx]), int(i[idx]))] = idx

        # Helper: extract float values from Unit objects for grid_points
        def _units_to_floats(units):
            """Convert a list of Unit objects to a list of floats (npc)."""
            return [float(u._values[0]) if hasattr(u, '_values') else float(u)
                    for u in units]

        # Draw background dots for all cells
        all_x_f = _units_to_floats(x)
        all_y_f = _units_to_floats(y)

        if all_x_f:
            grid_py.grid_points(
                x=all_x_f,
                y=all_y_f,
                default_units="npc",
                pch=16,
                size=_pt_size,
                gp=grid_py.Gpar(col=_bg_pt_col, fill=_bg_pt_col),
                name="upset_bg_dots",
            )

        # Draw active dots and connectors per column
        for col_idx in unique_j:
            # Find active rows in this column
            active_positions = []
            for row_idx in unique_i:
                key = (col_idx, row_idx)
                if key in pos_map:
                    pidx = pos_map[key]
                    # Check if this cell is active in the ordered_mat
                    # j values are 0-based column indices into the current slice
                    # i values are 0-based row indices into the current slice
                    if _ordered_mat[row_idx, col_idx]:
                        active_positions.append(pidx)

            # Determine colour for this column
            col_color = "black"
            if col_idx < len(_ordered_comb_col):
                col_color = _ordered_comb_col[col_idx]

            # Draw active dots
            if active_positions:
                active_x = _units_to_floats([x[p] for p in active_positions])
                active_y = _units_to_floats([y[p] for p in active_positions])
                grid_py.grid_points(
                    x=active_x,
                    y=active_y,
                    default_units="npc",
                    pch=16,
                    size=_pt_size,
                    gp=grid_py.Gpar(col=col_color, fill=col_color),
                    name=f"upset_active_dots_{col_idx}",
                )

                # Draw connector line between first and last active dot
                if len(active_positions) >= 2:
                    conn_x = active_x[0]
                    grid_py.grid_segments(
                        x0=grid_py.Unit(conn_x, "npc"),
                        y0=grid_py.Unit(active_y[0], "npc"),
                        x1=grid_py.Unit(conn_x, "npc"),
                        y1=grid_py.Unit(active_y[-1], "npc"),
                        gp=grid_py.Gpar(col=col_color, lwd=_lwd),
                        name=f"upset_connector_{col_idx}",
                    )

    # Build colour mapping: all cells get the background colour;
    # the layer_fun handles the actual visual encoding.
    col_map = {0.0: bg_col, 1.0: bg_col}

    ht = Heatmap(
        matrix=heatmap_mat,
        col=col_map,
        name=name,
        row_labels=row_labels,
        column_labels=col_labels if show_comb_name else None,
        show_row_names=show_row_names,
        row_names_side=row_names_side,
        row_names_gp=row_names_gp,
        show_column_names=show_comb_name,
        column_names_rot=comb_name_rot,
        cluster_rows=False,
        cluster_columns=False,
        row_order=list(range(n_sets)),
        column_order=list(range(n_comb)),
        top_annotation=top_annotation,
        right_annotation=right_annotation,
        left_annotation=left_annotation,
        rect_gp=grid_py.Gpar(col=bg_col, fill=bg_col),
        layer_fun=_upset_layer_fun,
        show_heatmap_legend=False,
        column_title=column_title,
        row_title=row_title,
        border=False,
        **kwargs,
    )

    return ht
