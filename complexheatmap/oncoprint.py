"""OncoPrint visualisation for cancer genomics data.

Provides :func:`oncoPrint` for creating OncoPrint visualisations
compatible with the :class:`~complexheatmap.heatmap_list.HeatmapList`
drawing pipeline, along with helper functions for alteration graphics.

This mirrors the ``oncoPrint`` function and associated helpers in the R
*ComplexHeatmap* package.

Examples
--------
>>> import numpy as np
>>> from complexheatmap.oncoprint import oncoPrint
>>> mat = {
...     "snv": np.array([[1, 0, 1], [0, 1, 0]]),
...     "del": np.array([[0, 1, 0], [1, 0, 0]]),
... }
>>> op = oncoPrint(mat, col={"snv": "red", "del": "blue"})
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import grid_py

__all__ = [
    "oncoPrint",
    "alter_graphic",
    "test_alter_fun",
]


# ======================================================================
# Default alteration drawing functions
# ======================================================================

_DEFAULT_COLORS: Dict[str, str] = {
    "snv": "#008000",
    "missense": "#008000",
    "nonsense": "#000000",
    "frameshift": "#8B4513",
    "splice": "#FF8C00",
    "inframe": "#9370DB",
    "amp": "red",
    "homdel": "blue",
    "del": "blue",
    "gain": "#FF6666",
    "loss": "#6666FF",
    "fusion": "#FF00FF",
    "mut": "#008000",
}


def _default_get_type(x: str) -> List[str]:
    """Parse alteration types from a semicolon-separated string.

    Parameters
    ----------
    x : str
        Raw alteration string (e.g. ``"snv;amp"``).

    Returns
    -------
    list of str
        Parsed alteration types.
    """
    if not x or not isinstance(x, str):
        return []
    return [t.strip() for t in x.replace(",", ";").split(";") if t.strip()]


def _make_default_alter_fun(
    col: Dict[str, str],
) -> Dict[str, Callable[..., None]]:
    """Build default alteration drawing functions using grid_py.

    Parameters
    ----------
    col : dict
        Alteration type to colour mapping.

    Returns
    -------
    dict
        Mapping from alteration type (and ``"background"``) to drawing
        callable.  Each function has the signature
        ``(x, y, w, h)`` where arguments are grid_py Units.
    """
    funs: Dict[str, Callable[..., None]] = {}

    def _background(x: Any, y: Any, w: Any, h: Any, **kw: Any) -> None:
        grid_py.grid_rect(
            x=x, y=y, width=w, height=h,
            gp=grid_py.Gpar(fill="#CCCCCC", col=None),
        )

    funs["background"] = _background

    for alt_type, color in col.items():
        if alt_type in (
            "snv", "missense", "nonsense", "frameshift",
            "splice", "inframe", "mut",
        ):
            # Small centred rectangle (point mutation style)
            def _draw_snv(
                x: Any, y: Any, w: Any, h: Any,
                _color: str = color,
                **kw: Any,
            ) -> None:
                grid_py.grid_rect(
                    x=x, y=y,
                    width=grid_py.Unit(0.9, "snpc") if not isinstance(w, grid_py.Unit) else w * 0.9,
                    height=grid_py.Unit(0.4, "snpc") if not isinstance(h, grid_py.Unit) else h * 0.4,
                    gp=grid_py.Gpar(fill=_color, col=None),
                )

            funs[alt_type] = _draw_snv
        elif alt_type in ("amp", "gain", "homdel", "del", "loss"):
            # Full-height rectangle
            def _draw_cnv(
                x: Any, y: Any, w: Any, h: Any,
                _color: str = color,
                **kw: Any,
            ) -> None:
                grid_py.grid_rect(
                    x=x, y=y,
                    width=grid_py.Unit(0.9, "snpc") if not isinstance(w, grid_py.Unit) else w * 0.9,
                    height=grid_py.Unit(0.9, "snpc") if not isinstance(h, grid_py.Unit) else h * 0.9,
                    gp=grid_py.Gpar(fill=_color, col=None),
                )

            funs[alt_type] = _draw_cnv
        else:
            # Generic: small rectangle
            def _draw_generic(
                x: Any, y: Any, w: Any, h: Any,
                _color: str = color,
                **kw: Any,
            ) -> None:
                grid_py.grid_rect(
                    x=x, y=y,
                    width=grid_py.Unit(0.9, "snpc") if not isinstance(w, grid_py.Unit) else w * 0.9,
                    height=grid_py.Unit(0.4, "snpc") if not isinstance(h, grid_py.Unit) else h * 0.4,
                    gp=grid_py.Gpar(fill=_color, col=None),
                )

            funs[alt_type] = _draw_generic

    return funs


# ======================================================================
# Public helpers
# ======================================================================

def alter_graphic(
    graphic: str = "rect",
    width: float = 1.0,
    height: float = 1.0,
    horiz_just: str = "center",
    vert_just: str = "center",
    col: str = "black",
    fill: str = "red",
    pch: int = 16,
) -> Callable[..., None]:
    """Create an alteration drawing function for OncoPrint cells.

    Parameters
    ----------
    graphic : str
        Graphic type: ``"rect"`` (rectangle) or ``"point"`` (circle).
    width : float
        Relative width of the graphic (0 to 1).
    height : float
        Relative height of the graphic (0 to 1).
    horiz_just : str
        Horizontal justification: ``"center"``, ``"left"``, ``"right"``.
    vert_just : str
        Vertical justification: ``"center"``, ``"top"``, ``"bottom"``.
    col : str
        Border/outline colour.
    fill : str
        Fill colour.
    pch : int
        Point character (for ``graphic="point"``).

    Returns
    -------
    callable
        Drawing function with signature ``(x, y, w, h)``.
    """
    _width = width
    _height = height

    def _draw(
        x: Any, y: Any, w: Any, h: Any, **kwargs: Any,
    ) -> None:
        if graphic == "rect":
            grid_py.grid_rect(
                x=x, y=y,
                width=w * _width if isinstance(w, grid_py.Unit) else grid_py.Unit(_width, "npc"),
                height=h * _height if isinstance(h, grid_py.Unit) else grid_py.Unit(_height, "npc"),
                just="centre",
                gp=grid_py.Gpar(fill=fill, col=col),
            )
        elif graphic == "point":
            grid_py.grid_points(
                x=x, y=y,
                pch=pch,
                size=grid_py.Unit(min(_width, _height) * 4, "mm"),
                gp=grid_py.Gpar(col=fill, fill=fill),
            )
        else:
            # Fallback: rectangle
            grid_py.grid_rect(
                x=x, y=y,
                width=w * _width if isinstance(w, grid_py.Unit) else grid_py.Unit(_width, "npc"),
                height=h * _height if isinstance(h, grid_py.Unit) else grid_py.Unit(_height, "npc"),
                just="centre",
                gp=grid_py.Gpar(fill=fill, col=col),
            )

    return _draw


def test_alter_fun(
    alter_fun: Dict[str, Callable[..., None]],
    type_names: Optional[List[str]] = None,
) -> None:
    """Test alteration functions by drawing sample cells.

    Renders a grid of sample cells, one per alteration type, so you can
    visually verify the drawing functions.

    Parameters
    ----------
    alter_fun : dict
        Mapping from alteration type name to drawing callable.
    type_names : list of str, optional
        Subset of types to test.  Defaults to all non-background keys
        in *alter_fun*.
    """
    if type_names is None:
        type_names = [k for k in alter_fun.keys() if k != "background"]

    n = len(type_names)
    if n == 0:
        return

    # Create a viewport layout with one cell per type
    grid_py.grid_newpage()
    layout = grid_py.Grid_Layout(
        nrow=1,
        ncol=n,
        widths=[grid_py.Unit(1, "null")] * n,
        heights=[grid_py.Unit(1, "null")],
    )
    grid_py.push_viewport(grid_py.Viewport(layout=layout))

    for idx, tname in enumerate(type_names):
        grid_py.push_viewport(grid_py.Viewport(
            layout_pos_row=0,
            layout_pos_col=idx,
        ))

        # Background
        bg_fun = alter_fun.get("background")
        if bg_fun is not None:
            bg_fun(
                grid_py.Unit(0.5, "npc"),
                grid_py.Unit(0.5, "npc"),
                grid_py.Unit(0.8, "npc"),
                grid_py.Unit(0.8, "npc"),
            )

        # Alteration graphic
        fun = alter_fun.get(tname)
        if fun is not None:
            fun(
                grid_py.Unit(0.5, "npc"),
                grid_py.Unit(0.5, "npc"),
                grid_py.Unit(0.8, "npc"),
                grid_py.Unit(0.8, "npc"),
            )

        # Label
        grid_py.grid_text(
            label=tname,
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.1, "npc"),
            gp=grid_py.Gpar(fontsize=8),
        )

        grid_py.up_viewport()

    grid_py.up_viewport()


# ======================================================================
# oncoPrint (main entry point)
# ======================================================================

_oncoprint_counter: int = 0


def _next_oncoprint_name() -> str:
    """Generate the next unique OncoPrint name."""
    global _oncoprint_counter
    _oncoprint_counter += 1
    return f"oncoPrint_{_oncoprint_counter}"


def oncoPrint(
    mat: Union[Dict[str, np.ndarray], np.ndarray],
    name: Optional[str] = None,
    get_type: Optional[Callable[[str], List[str]]] = None,
    alter_fun: Optional[
        Union[Dict[str, Callable[..., None]], Callable[..., None]]
    ] = None,
    alter_fun_is_vectorized: Optional[bool] = None,
    col: Optional[Dict[str, str]] = None,
    top_annotation: Optional[Any] = None,
    right_annotation: Optional[Any] = None,
    left_annotation: Optional[Any] = None,
    bottom_annotation: Optional[Any] = None,
    show_pct: bool = True,
    pct_gp: Optional[grid_py.Gpar] = None,
    pct_digits: int = 0,
    pct_side: str = "left",
    pct_include: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    show_row_names: bool = True,
    row_names_side: str = "right",
    row_names_gp: Optional[grid_py.Gpar] = None,
    row_split: Optional[Any] = None,
    column_labels: Optional[List[str]] = None,
    column_names_gp: Optional[grid_py.Gpar] = None,
    column_split: Optional[Any] = None,
    row_order: Optional[Union[np.ndarray, Sequence[int]]] = None,
    column_order: Optional[Union[np.ndarray, Sequence[int]]] = None,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    remove_empty_columns: bool = False,
    remove_empty_rows: bool = False,
    show_column_names: bool = False,
    show_heatmap_legend: bool = True,
    heatmap_legend_param: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Create an OncoPrint visualisation.

    The OncoPrint is implemented as a :class:`~complexheatmap.heatmap.Heatmap`
    with a custom ``cell_fun`` that draws alteration-type specific
    graphics in each cell.

    Parameters
    ----------
    mat : dict or numpy.ndarray
        Alteration data.  If a dict, each key is an alteration type
        and the value is a binary matrix of shape ``(genes, samples)``.
        If a 2-D character numpy array, alteration types are parsed
        from cell strings using *get_type*.
    name : str, optional
        Name for the heatmap in the component registry.
    get_type : callable, optional
        Function to extract alteration types from a string cell value.
        Only used when *mat* is a character matrix.  Defaults to
        :func:`_default_get_type`.
    alter_fun : dict or callable, optional
        Drawing functions for each alteration type.  If a dict, keys
        match alteration type names and values are callables with
        signature ``(x, y, w, h)``.  If a single callable, it receives
        ``(x, y, w, h, v)`` where *v* is a boolean dict of active types.
        If ``None``, default drawing functions are generated.
    alter_fun_is_vectorized : bool, optional
        Whether *alter_fun* accepts vectorised input.  If ``None``,
        guessed automatically (dict -> True, function -> False).
    col : dict, optional
        Colours for each alteration type.
    top_annotation : HeatmapAnnotation, optional
        Annotation above the oncoPrint.  Defaults to a stacked barplot
        via :func:`~complexheatmap.annotation_functions.anno_oncoprint_barplot`.
    right_annotation : HeatmapAnnotation, optional
        Annotation to the right.  Defaults to a stacked barplot.
    left_annotation : HeatmapAnnotation, optional
        Annotation to the left.
    bottom_annotation : HeatmapAnnotation, optional
        Annotation below.
    show_pct : bool
        Whether to show alteration percentages on the left.
    pct_gp : grid_py.Gpar, optional
        Graphical parameters for percentage text.
    pct_digits : int
        Decimal digits for percentages.
    pct_side : str
        ``"left"`` or ``"right"`` for percentage labels.
    pct_include : list of str, optional
        Alteration types included for percentage calculation.
    row_labels : list of str, optional
        Gene / row names.
    show_row_names : bool
        Show row names.
    row_names_side : str
        ``"left"`` or ``"right"``.
    row_names_gp : grid_py.Gpar, optional
        Row name graphical parameters.
    row_split : array-like, optional
        Row split specification.
    column_labels : list of str, optional
        Sample / column names.
    column_names_gp : grid_py.Gpar, optional
        Column name graphical parameters.
    column_split : array-like, optional
        Column split specification.
    row_order : array-like of int, optional
        Custom gene ordering.
    column_order : array-like of int, optional
        Custom sample ordering.
    cluster_rows : bool
        Whether to cluster rows.
    cluster_columns : bool
        Whether to cluster columns.
    remove_empty_columns : bool
        Remove samples with no alterations.
    remove_empty_rows : bool
        Remove genes with no alterations.
    show_column_names : bool
        Whether to show sample names.
    show_heatmap_legend : bool
        Whether to show the legend.
    heatmap_legend_param : dict, optional
        Legend parameters.
    **kwargs
        Forwarded to :class:`~complexheatmap.heatmap.Heatmap`.

    Returns
    -------
    Heatmap
        A Heatmap object that can be added to a HeatmapList.
    """
    from .heatmap import Heatmap
    from .heatmap_annotation import HeatmapAnnotation, rowAnnotation
    from .annotation_functions import anno_oncoprint_barplot, anno_text
    from ._utils import pindex, subset_gp, max_text_width

    if get_type is None:
        get_type = _default_get_type

    if name is None:
        name = _next_oncoprint_name()

    # ------------------------------------------------------------------
    # Convert mat to dict of binary matrices (mat_list)
    # ------------------------------------------------------------------
    mat_list: Dict[str, np.ndarray] = {}

    if isinstance(mat, dict):
        # dict of binary matrices
        for k, v in mat.items():
            arr = np.asarray(v)
            if arr.dtype == bool:
                mat_list[k] = arr.astype(int)
            else:
                mat_list[k] = arr.astype(int)
        all_type = list(mat_list.keys())
    elif isinstance(mat, np.ndarray):
        raw = np.asarray(mat)
        if raw.ndim == 3:
            # (types, genes, samples) array
            for i in range(raw.shape[0]):
                mat_list[f"type{i + 1}"] = raw[i].astype(int)
            all_type = list(mat_list.keys())
        elif raw.ndim == 2:
            # Character matrix: parse types from strings
            all_type_set: set = set()
            for val in raw.flat:
                all_type_set.update(get_type(str(val)))
            all_type = sorted(all_type_set)
            for atype in all_type:
                binary = np.zeros(raw.shape, dtype=int)
                for idx in np.ndindex(raw.shape):
                    if atype in get_type(str(raw[idx])):
                        binary[idx] = 1
                mat_list[atype] = binary
        else:
            raise ValueError("Array input must be 2-D or 3-D.")
    else:
        raise ValueError("`mat` must be a dict of matrices or a numpy array.")

    if not all_type:
        raise ValueError("No alteration types found.")

    first_mat = mat_list[all_type[0]]
    n_genes, n_samples = first_mat.shape

    # ------------------------------------------------------------------
    # Build 3-D alteration array: (genes, samples, types)
    # ------------------------------------------------------------------
    arr = np.zeros((n_genes, n_samples, len(all_type)), dtype=bool)
    for ti, atype in enumerate(all_type):
        arr[:, :, ti] = mat_list[atype].astype(bool)

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    if col is None:
        col = {}
        for tn in all_type:
            col[tn] = _DEFAULT_COLORS.get(tn, "#888888")

    # ------------------------------------------------------------------
    # Alteration functions
    # ------------------------------------------------------------------
    if alter_fun is None:
        alter_fun_dict = _make_default_alter_fun(col)
        alter_fun_is_vectorized = True
    elif isinstance(alter_fun, dict):
        alter_fun_dict = dict(alter_fun)
        if "background" not in alter_fun_dict:
            alter_fun_dict["background"] = _make_default_alter_fun(col)["background"]
        if alter_fun_is_vectorized is None:
            alter_fun_is_vectorized = True
    else:
        # Single callable
        alter_fun_single = alter_fun
        alter_fun_dict = {}
        if alter_fun_is_vectorized is None:
            alter_fun_is_vectorized = False

    is_dict_mode = isinstance(alter_fun, dict) or alter_fun is None

    # ------------------------------------------------------------------
    # Ordering helpers (memo sort for columns)
    # ------------------------------------------------------------------
    count_matrix = arr.sum(axis=2)
    n_mut = np.apply_along_axis(lambda row: np.any(row), axis=1, arr=arr.reshape(n_genes, -1))
    n_mut_per_gene = arr.any(axis=2).sum(axis=1)

    def _oncoprint_row_order() -> np.ndarray:
        return np.lexsort((-n_mut_per_gene, -count_matrix.sum(axis=1)))

    def _oncoprint_column_order(row_ord: np.ndarray) -> np.ndarray:
        """Memo-sort: order columns to show mutual exclusivity."""
        cm = count_matrix[row_ord, :]
        scores = np.zeros(n_samples, dtype=float)
        for si in range(n_samples):
            score = 0.0
            for gi in range(len(row_ord)):
                if cm[gi, si] > 0:
                    score += 2 ** (len(row_ord) - gi)
            scores[si] = score
        return np.argsort(-scores)

    # ------------------------------------------------------------------
    # Remove empty rows/columns
    # ------------------------------------------------------------------
    if row_labels is None:
        row_labels = [str(i) for i in range(n_genes)]
    if column_labels is None:
        column_labels = [str(j) for j in range(n_samples)]

    if remove_empty_columns:
        col_mask = arr.any(axis=2).any(axis=0)
        arr = arr[:, col_mask, :]
        column_labels = [column_labels[j] for j in range(n_samples) if col_mask[j]]
        if column_split is not None:
            column_split = np.asarray(column_split)[col_mask]
        n_samples = arr.shape[1]
        count_matrix = arr.sum(axis=2)
        n_mut_per_gene = arr.any(axis=2).sum(axis=1)

    if remove_empty_rows:
        row_mask = arr.any(axis=2).any(axis=1)
        arr = arr[row_mask, :, :]
        row_labels = [row_labels[i] for i in range(n_genes) if row_mask[i]]
        if row_split is not None:
            row_split = np.asarray(row_split)[row_mask]
        n_genes = arr.shape[0]
        count_matrix = arr.sum(axis=2)
        n_mut_per_gene = arr.any(axis=2).sum(axis=1)

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------
    if row_order is None:
        row_order_arr = _oncoprint_row_order()
    else:
        row_order_arr = np.asarray(row_order)

    if column_order is None:
        column_order_arr = _oncoprint_column_order(row_order_arr)
    else:
        column_order_arr = np.asarray(column_order)

    # ------------------------------------------------------------------
    # Percentage labels
    # ------------------------------------------------------------------
    if pct_include is None:
        pct_include = all_type
    pct_arr = arr[:, :, [all_type.index(t) for t in pct_include if t in all_type]]
    pct_num = pct_arr.any(axis=2).sum(axis=1) / max(n_samples, 1)
    pct_text = [f"{round(v * 100, pct_digits):.{pct_digits}f}%" for v in pct_num]

    # ------------------------------------------------------------------
    # Build cell_fun / layer_fun
    # ------------------------------------------------------------------
    _arr = arr
    _all_type = all_type

    if is_dict_mode:
        def _onco_cell_fun(
            j: int, i: int,
            x: Any, y: Any, w: Any, h: Any,
            fill: str,
        ) -> None:
            """Draw one OncoPrint cell with alteration-specific graphics."""
            alter_fun_dict["background"](x, y, w, h)
            for ti, tn in enumerate(_all_type):
                if _arr[i, j, ti]:
                    fun = alter_fun_dict.get(tn)
                    if fun is not None:
                        fun(x, y, w, h)
    else:
        # Single-function mode
        _alter_fun_single = alter_fun  # type: ignore[assignment]

        def _onco_cell_fun(
            j: int, i: int,
            x: Any, y: Any, w: Any, h: Any,
            fill: str,
        ) -> None:
            v = {tn: bool(_arr[i, j, ti]) for ti, tn in enumerate(_all_type)}
            _alter_fun_single(x, y, w, h, v)  # type: ignore[operator]

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    if pct_gp is None:
        pct_gp = grid_py.Gpar(fontsize=10)
    if row_names_gp is None:
        row_names_gp = pct_gp

    # Convert Gpar to dict for anno_text if needed
    _pct_gp_dict: Dict[str, Any] = {}
    if isinstance(pct_gp, grid_py.Gpar):
        if hasattr(pct_gp, '_store'):
            _pct_gp_dict = dict(pct_gp._store)
        elif hasattr(pct_gp, '__dict__'):
            _pct_gp_dict = {k: v for k, v in pct_gp.__dict__.items() if not k.startswith('_')}
    elif isinstance(pct_gp, dict):
        _pct_gp_dict = dict(pct_gp)

    # Left annotation: show pct text
    pct_ha = None
    if show_pct:
        pct_anno = anno_text(
            pct_text,
            which="row",
            gp=_pct_gp_dict,
        )
        pct_ha = HeatmapAnnotation(
            pct=pct_anno,
            which="row",
            show_annotation_name=False,
        )

    # Default top annotation: stacked barplot
    if top_annotation is None:
        top_annotation = HeatmapAnnotation(
            cbar=anno_oncoprint_barplot(which="column"),
            which="column",
        )

    # Default right annotation: stacked barplot
    if right_annotation is None:
        right_annotation = rowAnnotation(
            rbar=anno_oncoprint_barplot(which="row"),
        )

    # Handle left_annotation with pct
    if show_pct and pct_side == "left":
        if left_annotation is None:
            left_annotation = pct_ha
    if show_pct and pct_side == "right":
        # Attach pct to right side (after right_annotation)
        pass

    # ------------------------------------------------------------------
    # Heatmap legend
    # ------------------------------------------------------------------
    if heatmap_legend_param is None:
        heatmap_legend_param = {}
    if "title" not in heatmap_legend_param:
        heatmap_legend_param["title"] = "Alterations"

    # Build a color map: use the count_matrix as the heatmap matrix.
    # The actual graphics are drawn by cell_fun; the colour map is just
    # for the legend.
    legend_col = col

    # ------------------------------------------------------------------
    # Build the Heatmap
    # ------------------------------------------------------------------
    # Use count_matrix as the display matrix (for ordering/legend).
    # Assign uniform colour so the cell rectangles are just background.
    bg_col = "#CCCCCC"
    uniform_col = {i: bg_col for i in range(int(count_matrix.max()) + 2)}
    uniform_col[0] = bg_col

    ht = Heatmap(
        matrix=count_matrix,
        col=uniform_col,
        name=name,
        row_labels=row_labels,
        column_labels=column_labels,
        show_row_names=show_row_names,
        row_names_side=row_names_side,
        row_names_gp=row_names_gp,
        show_column_names=show_column_names,
        column_names_gp=column_names_gp,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
        row_order=list(row_order_arr),
        column_order=list(column_order_arr),
        row_split=row_split,
        column_split=column_split,
        top_annotation=top_annotation,
        right_annotation=right_annotation,
        left_annotation=left_annotation,
        bottom_annotation=bottom_annotation,
        rect_gp=grid_py.Gpar(col=bg_col, fill=bg_col),
        cell_fun=_onco_cell_fun,
        show_heatmap_legend=show_heatmap_legend,
        heatmap_legend_param=heatmap_legend_param,
        border=False,
        **kwargs,
    )

    # Attach metadata for downstream use (e.g. by anno_oncoprint_barplot)
    ht._oncoprint_arr = arr
    ht._oncoprint_types = all_type
    ht._oncoprint_col = col

    return ht
