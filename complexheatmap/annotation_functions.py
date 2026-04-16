"""Built-in annotation drawing functions (``anno_*``).

R source correspondence
-----------------------
``R/AnnotationFunction-function.R`` -- all ``anno_*`` factory functions.

Each public function in this module returns an
:class:`~complexheatmap.annotation_function.AnnotationFunction` whose
internal draw callback renders a specific annotation type using
``grid_py`` (the Python port of R's ``grid`` package).

Convention
----------
* For **column** annotations the x-axis maps to observation indices and
  the y-axis maps to data values.
* For **row** annotations the y-axis maps to observation indices and
  the x-axis maps to data values.

Examples
--------
>>> import numpy as np
>>> from complexheatmap.annotation_functions import anno_barplot
>>> af = anno_barplot(np.random.rand(20), which="column")
>>> af.nobs
20
"""

from __future__ import annotations

import textwrap
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

from .annotation_function import AnnotationFunction

__all__ = [
    "anno_simple",
    "anno_barplot",
    "anno_boxplot",
    "anno_points",
    "anno_lines",
    "anno_text",
    "anno_histogram",
    "anno_density",
    "anno_joyplot",
    "anno_horizon",
    "anno_image",
    "anno_link",
    "anno_mark",
    "anno_block",
    "anno_summary",
    "anno_empty",
    "anno_textbox",
    "anno_customize",
    "anno_numeric",
    "anno_oncoprint_barplot",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SIZE: float = 10.0  # mm  (R default: unit(1, "cm"))

# Built-in discrete colour palettes (tab20/tab10 hex values).
# Avoids matplotlib dependency for auto-colour assignment.
_TAB20 = [
    "#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C",
    "#98DF8A", "#D62728", "#FF9896", "#9467BD", "#C5B0D5",
    "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", "#7F7F7F",
    "#C7C7C7", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5",
]
_TAB10 = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _resolve_gp(gp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise user graphic-parameters into grid_py.Gpar-compatible keys."""
    if gp is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in gp.items():
        if k == "facecolor":
            out["fill"] = v
        elif k == "edgecolor":
            out["col"] = v
        elif k == "linewidth":
            out["lwd"] = v
        elif k == "color":
            out["col"] = v
        elif k == "linestyle":
            out["lty"] = v
        elif k == "fontweight":
            out["fontface"] = v
        else:
            out[k] = v
    return out


def _to_gpar(**kw: Any) -> grid_py.Gpar:
    """Create a grid_py.Gpar from keyword arguments."""
    return grid_py.Gpar(**kw)


def _expand_data_lim(
    data: np.ndarray,
    ylim: Optional[Tuple[float, float]],
    extend: float = 0.05,
    baseline: float = 0.0,
) -> Tuple[float, float]:
    """Compute data-axis limits with optional extension.

    Mirrors R ``AnnotationFunction-function.R:1329-1348``.
    """
    if ylim is not None:
        return tuple(ylim)  # type: ignore[return-value]
    lo = float(np.nanmin(data))
    hi = float(np.nanmax(data))
    if lo == hi:
        hi = lo + 1.0
    # R baseline handling (lines 1339-1348)
    if isinstance(baseline, (int, float)):
        if baseline <= lo:
            # baseline at or below data range
            return (float(baseline), hi + extend * (hi - float(baseline)))
        elif baseline >= hi:
            return (lo - extend * (float(baseline) - lo), float(baseline))
        else:
            rng = hi - lo
            return (lo - extend * rng, hi + extend * rng)
    rng = hi - lo
    return (lo - extend * rng, hi + extend * rng)


def _is_row(which: str) -> bool:
    return which == "row"


def _isnan_safe(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of NaN-like values, safe for object arrays."""
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return np.array([_isnan_scalar(v) for v in x])


def _isnan_scalar(v: Any) -> bool:
    """Check whether a single value is NaN-like."""
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _color_mapping_to_list(
    x: np.ndarray,
    col: Optional[Union[Dict[str, str], Callable[..., Any]]] = None,
    na_col: str = "grey",
) -> List[str]:
    """Map data values to colour strings.

    Parameters
    ----------
    x : np.ndarray
        Data values (categorical or numeric).
    col : dict or callable, optional
        Colour specification.
    na_col : str
        Colour for NaN / missing values.

    Returns
    -------
    list of str
    """
    colors: List[str] = []

    def _rgba_to_hex(c: Any) -> str:
        """Convert an RGBA tuple/list or string to a hex colour string."""
        if isinstance(c, str):
            return c
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
            if len(c) >= 4:
                a = int(c[3] * 255)
                return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
            return f"#{r:02x}{g:02x}{b:02x}"
        return str(c)

    # col_fn: optional callable for continuous data without explicit col
    col_fn: Any = None

    if col is None:
        unique_vals = np.unique(x[~_isnan_safe(x)])
        if len(unique_vals) <= 20:
            col_map: Optional[Dict] = {
                v: _TAB20[i % len(_TAB20)]
                for i, v in enumerate(unique_vals)
            }
        else:
            # Continuous data: auto-generate a colour ramp
            from ._color import color_ramp2
            vmin, vmax = float(unique_vals.min()), float(unique_vals.max())
            col_map = None
            mid = (vmin + vmax) / 2
            _auto_ramp = color_ramp2([vmin, mid, vmax],
                                     ["#313695", "#FFFFBF", "#A50026"])
            col_fn = lambda v: _auto_ramp(float(v))
    elif callable(col) and not isinstance(col, dict):
        col_fn = col
        col_map = None
    else:
        # User-supplied dict: ensure values are hex strings
        if isinstance(col, dict):
            col_map = {k: _rgba_to_hex(v) for k, v in col.items()}
        else:
            col_map = col  # type: ignore[assignment]

    for v in x:
        if _isnan_scalar(v):
            colors.append(na_col)
        elif col_map is not None:
            colors.append(col_map.get(v, na_col))  # type: ignore[union-attr]
        elif col_fn is not None:
            colors.append(_rgba_to_hex(col_fn(v)))
        else:
            colors.append(na_col)
    return colors


def _default_width_height(
    which: str,
    width: Optional[Any],
    height: Optional[Any],
    default_mm: float = _DEFAULT_SIZE,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Apply R-like defaults: fixed dimension for the 'short' axis, npc(1)
    for the 'long' axis."""
    if which == "column":
        if height is None:
            height = default_mm
        if width is None:
            width = grid_py.Unit(1, "npc")
    else:
        if width is None:
            width = default_mm
        if height is None:
            height = grid_py.Unit(1, "npc")
    return width, height


# =========================================================================
# anno_simple
# =========================================================================


def anno_simple(
    x: ArrayLike,
    col: Optional[Union[Dict[str, str], Callable[..., Any]]] = None,
    na_col: str = "grey",
    which: str = "column",
    border: bool = False,
    gp: Optional[Dict[str, Any]] = None,
    pch: Optional[Any] = None,
    pt_size: Optional[float] = None,
    pt_gp: Optional[Dict[str, Any]] = None,
    simple_anno_size: Optional[float] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Simple colour-bar annotation.

    Parameters
    ----------
    x : array-like
        Categorical or numeric values.  Can be a 1-D vector or 2-D matrix.
    col : dict or callable, optional
        Colour mapping.
    na_col : str
        Colour for NaN / missing values.
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Whether to draw cell borders.
    gp : dict, optional
        Graphic parameters forwarded to rectangle grobs.
    pch : array-like, optional
        Point character overlay on each cell.
    pt_size : float, optional
        Marker size when *pch* is set.
    pt_gp : dict, optional
        Graphic parameters for point markers.
    simple_anno_size : float, optional
        Size in mm (overrides default).
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x)
    input_is_matrix = x_arr.ndim == 2
    if input_is_matrix and x_arr.shape[1] == 1:
        x_arr = x_arr[:, 0]
        input_is_matrix = False

    n = x_arr.shape[0]
    nc = x_arr.shape[1] if input_is_matrix else 1

    # R: simple_anno_size defaults to ht_opt$simple_anno_size = 5mm
    # _DEFAULT_SIZE (10mm) is for barplot/points, not simple annotations
    if simple_anno_size is not None:
        anno_size = simple_anno_size
    else:
        from ._globals import ht_opt as _ht_opt
        anno_size = float(_ht_opt("simple_anno_size"))
    w, h = _default_width_height(which, width, height, anno_size * nc)
    merged_gp = _resolve_gp(gp)

    # Capture closure variables
    _x = x_arr
    _col = col
    _na_col = na_col
    _border = border
    _gp = merged_gp
    _pch = pch
    _pt_size = pt_size
    _pt_gp = _resolve_gp(pt_gp)
    _which = which
    _input_is_matrix = input_is_matrix

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]

        if _which == "column":
            if _input_is_matrix:
                nc_local = subset.shape[1]
                for j in range(nc_local):
                    col_vals = subset[:, j]
                    colors = _color_mapping_to_list(col_vals, col=_col, na_col=_na_col)
                    x_pos = [(i + 0.5) / ni for i in range(ni)]
                    y_pos = (nc_local - j - 0.5) / nc_local
                    for ci, (xp, c) in enumerate(zip(x_pos, colors)):
                        gpar_kw = dict(_gp)
                        gpar_kw["fill"] = c
                        if _border:
                            gpar_kw.setdefault("col", "black")
                        else:
                            gpar_kw["col"] = c
                        grid_py.grid_rect(
                            x=grid_py.Unit(xp, "npc"),
                            y=grid_py.Unit(y_pos, "npc"),
                            width=grid_py.Unit(1.0 / ni, "npc"),
                            height=grid_py.Unit(1.0 / nc_local, "npc"),
                            gp=_to_gpar(**gpar_kw),
                        )
            else:
                colors = _color_mapping_to_list(subset, col=_col, na_col=_na_col)
                for i, c in enumerate(colors):
                    xp = (i + 0.5) / ni
                    gpar_kw = dict(_gp)
                    gpar_kw["fill"] = c
                    if _border:
                        gpar_kw.setdefault("col", "black")
                    else:
                        gpar_kw["col"] = c
                    grid_py.grid_rect(
                        x=grid_py.Unit(xp, "npc"),
                        y=grid_py.Unit(0.5, "npc"),
                        width=grid_py.Unit(1.0 / ni, "npc"),
                        height=grid_py.Unit(1, "npc"),
                        gp=_to_gpar(**gpar_kw),
                    )
            if _border:
                grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))
        else:
            # Row annotation
            if _input_is_matrix:
                nc_local = subset.shape[1]
                for j in range(nc_local):
                    col_vals = subset[:, j]
                    colors = _color_mapping_to_list(col_vals, col=_col, na_col=_na_col)
                    x_pos = (j + 0.5) / nc_local
                    for ri, c in enumerate(colors):
                        yp = (ni - ri - 0.5) / ni
                        gpar_kw = dict(_gp)
                        gpar_kw["fill"] = c
                        if _border:
                            gpar_kw.setdefault("col", "black")
                        else:
                            gpar_kw["col"] = c
                        grid_py.grid_rect(
                            x=grid_py.Unit(x_pos, "npc"),
                            y=grid_py.Unit(yp, "npc"),
                            width=grid_py.Unit(1.0 / nc_local, "npc"),
                            height=grid_py.Unit(1.0 / ni, "npc"),
                            gp=_to_gpar(**gpar_kw),
                        )
            else:
                colors = _color_mapping_to_list(subset, col=_col, na_col=_na_col)
                for i, c in enumerate(colors):
                    yp = (ni - i - 0.5) / ni
                    gpar_kw = dict(_gp)
                    gpar_kw["fill"] = c
                    if _border:
                        gpar_kw.setdefault("col", "black")
                    else:
                        gpar_kw["col"] = c
                    grid_py.grid_rect(
                        x=grid_py.Unit(0.5, "npc"),
                        y=grid_py.Unit(yp, "npc"),
                        width=grid_py.Unit(1, "npc"),
                        height=grid_py.Unit(1.0 / ni, "npc"),
                        gp=_to_gpar(**gpar_kw),
                    )
            if _border:
                grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))

    var_env: Dict[str, Any] = {"x": x_arr, "col": col, "na_col": na_col, "gp": gp}
    subset_rules: Dict[str, Optional[str]] = {
        "x": "matrix_row" if input_is_matrix else "array",
    }

    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_simple",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=(0.5, nc + 0.5),
        subsettable=True,
        subset_rule=subset_rules,
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_barplot
# =========================================================================


def anno_barplot(
    x: ArrayLike,
    baseline: float = 0.0,
    which: str = "column",
    border: bool = True,
    bar_width: float = 0.6,
    beside: bool = False,
    gp: Optional[Dict[str, Any]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    extend: float = 0.05,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Bar-plot annotation.

    Parameters
    ----------
    x : array-like
        Numeric values (1-D) or matrix (2-D for stacked / grouped bars).
    baseline : float
        Baseline value for bars.
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw bar borders.
    bar_width : float
        Relative width of each bar (0-1).
    beside : bool
        If *x* is 2-D, place bars side-by-side instead of stacking.
    gp : dict, optional
        Graphic parameters.
    ylim : tuple of float, optional
        Data axis limits.
    extend : float
        Fraction to extend limits beyond data range.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    is_matrix = x_arr.ndim == 2
    n = x_arr.shape[0]

    # R AnnotationFunction-function.R:1323-1327:
    # For stacked bars (matrix, not beside), data range uses rowSums.
    if is_matrix and not beside:
        _lim_data = np.nansum(x_arr, axis=1, keepdims=True)
    else:
        _lim_data = x_arr
    data_lim = _expand_data_lim(_lim_data, ylim, extend, baseline)
    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _is_matrix = is_matrix
    _baseline = baseline
    _bar_width = bar_width
    _beside = beside
    _border = border
    _gp = merged_gp
    _data_lim = data_lim
    _axis = axis
    _axis_param = axis_param

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]
        d_lo, d_hi = _data_lim
        d_range = d_hi - d_lo if d_hi != d_lo else 1.0

        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0.5, ni + 0.5) if not _is_row(_which) else (_data_lim[0], _data_lim[1]),
            yscale=(_data_lim[0], _data_lim[1]) if not _is_row(_which) else (0.5, ni + 0.5),
        ))

        fill_color = _gp.get("fill", "steelblue")
        border_col = _gp.get("col", "black") if _border else "transparent"

        if _is_matrix and not _beside:
            n_cols = subset.shape[1]
            # Per-column fill colours: R uses gp$fill recycled over columns
            if isinstance(fill_color, (list, tuple, np.ndarray)) and len(fill_color) >= n_cols:
                col_fills = list(fill_color)[:n_cols]
            else:
                col_fills = [_TAB10[j % len(_TAB10)] for j in range(n_cols)]
            for i in range(ni):
                bottom = _baseline
                for j in range(n_cols):
                    val = subset[i, j]
                    fc = col_fills[j % len(col_fills)]
                    if _is_row(_which):
                        grid_py.grid_rect(
                            x=grid_py.Unit(bottom, "native"),
                            y=grid_py.Unit(i + 1, "native"),
                            width=grid_py.Unit(val - _baseline if j == 0 else val, "native"),
                            height=grid_py.Unit(_bar_width, "native"),
                            just="left",
                            gp=_to_gpar(fill=fc, col=border_col),
                        )
                    else:
                        grid_py.grid_rect(
                            x=grid_py.Unit(i + 1, "native"),
                            y=grid_py.Unit(bottom, "native"),
                            width=grid_py.Unit(_bar_width, "native"),
                            height=grid_py.Unit(val - _baseline if j == 0 else val, "native"),
                            just="bottom",
                            gp=_to_gpar(fill=fc, col=border_col),
                        )
                    bottom += (val - _baseline) if j == 0 else val
        else:
            for i in range(ni):
                val = float(subset[i]) if not _is_matrix else float(subset[i, 0])
                bar_h = val - _baseline
                if _is_row(_which):
                    grid_py.grid_rect(
                        x=grid_py.Unit(_baseline, "native"),
                        y=grid_py.Unit(i + 1, "native"),
                        width=grid_py.Unit(bar_h, "native"),
                        height=grid_py.Unit(_bar_width, "native"),
                        just="left",
                        gp=_to_gpar(fill=fill_color, col=border_col),
                    )
                else:
                    grid_py.grid_rect(
                        x=grid_py.Unit(i + 1, "native"),
                        y=grid_py.Unit(_baseline, "native"),
                        width=grid_py.Unit(_bar_width, "native"),
                        height=grid_py.Unit(bar_h, "native"),
                        just="bottom",
                        gp=_to_gpar(fill=fill_color, col=border_col),
                    )

        # Border rect
        grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))

        # Axis — respects axis_param (R: at, labels, side, labels_rot)
        if _axis and k == 1:
            ap = _axis_param or {}
            ticks = ap.get("at", None)
            tick_labels = ap.get("labels", None)
            if ticks is None:
                ticks = grid_py.grid_pretty(_data_lim)
                ticks = [float(t) for t in ticks if _data_lim[0] <= t <= _data_lim[1]]
            else:
                ticks = [float(t) for t in ticks]
            if tick_labels is None:
                tick_labels = [f"{t:g}" for t in ticks]
            axis_gp = grid_py.Gpar(fontsize=ap.get("fontsize", 7))
            if _is_row(_which):
                grid_py.grid_xaxis(at=ticks, label=tick_labels, gp=axis_gp)
            else:
                grid_py.grid_yaxis(at=ticks, label=tick_labels, gp=axis_gp)

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_barplot",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "array" if not is_matrix else "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_boxplot
# =========================================================================


def anno_boxplot(
    x: ArrayLike,
    which: str = "column",
    border: bool = True,
    gp: Optional[Dict[str, Any]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    extend: float = 0.05,
    outline: bool = True,
    box_width: float = 0.6,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Box-plot annotation.

    Parameters
    ----------
    x : array-like
        2-D array (rows = observations, columns = values) or list of arrays.
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw box borders.
    gp : dict, optional
        Graphic parameters.
    ylim : tuple of float, optional
        Data axis limits.
    extend : float
        Fraction to extend limits.
    outline : bool
        Show outlier points.
    box_width : float
        Relative box width.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x_list: List[np.ndarray] = [x[i, :] for i in range(x.shape[0])]
    elif isinstance(x, (list, tuple)):
        x_list = [np.asarray(v, dtype=float) for v in x]
    else:
        x_list = [np.asarray(x, dtype=float)]

    n = len(x_list)
    flat = np.concatenate([np.ravel(a) for a in x_list])
    data_lim = _expand_data_lim(flat, ylim, extend)
    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x_list = x_list
    _which = which
    _data_lim = data_lim
    _gp = merged_gp
    _box_width = box_width
    _border = border
    _outline = outline
    _axis = axis

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        from .grid_extensions import grid_boxplot

        ni = len(index)

        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0.5, ni + 0.5) if not _is_row(_which) else _data_lim,
            yscale=_data_lim if not _is_row(_which) else (0.5, ni + 0.5),
            clip=True,
        ))

        for pos_i, idx in enumerate(index):
            values = _x_list[idx]
            pos = pos_i + 1
            if _is_row(_which):
                bp = grid_boxplot(
                    value=values, pos=pos, direction="x",
                    box_width=_box_width, outline=_outline, gp=_gp,
                    default_units="native",
                )
            else:
                bp = grid_boxplot(
                    value=values, pos=pos, direction="y",
                    box_width=_box_width, outline=_outline, gp=_gp,
                    default_units="native",
                )
            grid_py.grid_draw(bp)

        grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))
        if _axis and k == 1:
            ticks = grid_py.grid_pretty(_data_lim)
            ticks = [float(t) for t in ticks if _data_lim[0] <= t <= _data_lim[1]]
            tick_labels = [f"{t:g}" for t in ticks]
            if _is_row(_which):
                grid_py.grid_xaxis(at=ticks, label=tick_labels,
                                    gp=grid_py.Gpar(fontsize=7))
            else:
                grid_py.grid_yaxis(at=ticks, label=tick_labels,
                                    gp=grid_py.Gpar(fontsize=7))

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x_list": x_list, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_boxplot",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x_list": "array"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_points
# =========================================================================


def anno_points(
    x: ArrayLike,
    which: str = "column",
    border: bool = True,
    gp: Optional[Dict[str, Any]] = None,
    pch: int = 16,
    size: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    extend: float = 0.05,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Scatter-points annotation.

    Parameters
    ----------
    x : array-like
        Numeric values.
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border around the annotation region.
    gp : dict, optional
        Graphic parameters forwarded to ``grid_points``.
    pch : int
        Marker style (R-style pch code).
    size : float, optional
        Marker size in mm.
    ylim : tuple of float, optional
        Data axis limits.
    extend : float
        Fraction to extend limits.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    is_matrix = x_arr.ndim == 2
    n = x_arr.shape[0]
    # anno_points: data range without forced baseline=0
    # (unlike anno_barplot which includes baseline in range)
    if ylim is not None:
        data_lim = tuple(ylim)
    else:
        lo = float(np.nanmin(x_arr))
        hi = float(np.nanmax(x_arr))
        rng = hi - lo if hi != lo else 1.0
        data_lim = (lo - extend * rng, hi + extend * rng)
    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _data_lim = data_lim
    _gp = merged_gp
    _pch = pch
    _size = size
    _axis = axis
    _axis_param = axis_param
    _border = border

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]

        # R viewports default to clip="on" — points at boundaries are clipped
        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0.5, ni + 0.5) if not _is_row(_which) else _data_lim,
            yscale=_data_lim if not _is_row(_which) else (0.5, ni + 0.5),
            clip=True,
        ))

        if is_matrix:
            n_cols = subset.shape[1]
            for j in range(n_cols):
                col_vals = subset[:, j]
                positions = np.arange(1, ni + 1)
                # Per-column gp: R recycles col/fill per column
                # (R: subset_gp(gp, i))
                col_gp = dict(_gp)
                for gp_key in ("col", "fill"):
                    if gp_key in col_gp and isinstance(col_gp[gp_key], (list, tuple)):
                        col_gp[gp_key] = col_gp[gp_key][j % len(col_gp[gp_key])]
                # Per-column pch (R: pch[i])
                col_pch = _pch
                if isinstance(_pch, (list, tuple)) and len(_pch) > 1:
                    col_pch = _pch[j % len(_pch)]
                # Per-column size (R: size[i])
                col_size = _size or 2
                if isinstance(_size, (list, tuple)) and len(_size) > 1:
                    col_size = _size[j % len(_size)]
                if _is_row(_which):
                    grid_py.grid_points(
                        x=grid_py.Unit(col_vals, "native"),
                        y=grid_py.Unit(positions, "native"),
                        pch=col_pch,
                        size=grid_py.Unit(col_size, "mm"),
                        gp=_to_gpar(**col_gp),
                    )
                else:
                    grid_py.grid_points(
                        x=grid_py.Unit(positions, "native"),
                        y=grid_py.Unit(col_vals, "native"),
                        pch=col_pch,
                        size=grid_py.Unit(col_size, "mm"),
                        gp=_to_gpar(**col_gp),
                    )
        else:
            positions = np.arange(1, ni + 1)
            if _is_row(_which):
                grid_py.grid_points(
                    x=grid_py.Unit(subset, "native"),
                    y=grid_py.Unit(positions, "native"),
                    pch=_pch,
                    size=grid_py.Unit(_size or 2, "mm"),
                    gp=_to_gpar(**_gp),
                )
            else:
                grid_py.grid_points(
                    x=grid_py.Unit(positions, "native"),
                    y=grid_py.Unit(subset, "native"),
                    pch=_pch,
                    size=grid_py.Unit(_size or 2, "mm"),
                    gp=_to_gpar(**_gp),
                )

        grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black" if _border else "transparent"))
        # Axis — respects axis_param (R: at, labels, side, labels_rot)
        if _axis and k == 1:
            ap = _axis_param or {}
            ticks = ap.get("at", None)
            tick_labels = ap.get("labels", None)
            if ticks is None:
                ticks = grid_py.grid_pretty(_data_lim)
                ticks = [float(t) for t in ticks if _data_lim[0] <= t <= _data_lim[1]]
            else:
                ticks = [float(t) for t in ticks]
            if tick_labels is None:
                tick_labels = [f"{t:g}" for t in ticks]
            axis_gp = grid_py.Gpar(fontsize=ap.get("fontsize", 7))
            if _is_row(_which):
                grid_py.grid_xaxis(at=ticks, label=tick_labels, gp=axis_gp)
            else:
                grid_py.grid_yaxis(at=ticks, label=tick_labels, gp=axis_gp)

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_points",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "array" if not is_matrix else "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_lines
# =========================================================================


def anno_lines(
    x: ArrayLike,
    which: str = "column",
    border: bool = True,
    gp: Optional[Dict[str, Any]] = None,
    smooth: bool = False,
    add_points: Optional[bool] = None,
    pch: int = 16,
    size: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    extend: float = 0.05,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Line-plot annotation.

    Parameters
    ----------
    x : array-like
        Numeric values (1-D or 2-D with one line per column).
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border.
    gp : dict, optional
        Graphic parameters for lines.
    smooth : bool
        Apply smoothing.
    add_points : bool, optional
        Overlay scatter points (defaults to ``True`` when *smooth* is True).
    pch : int
        Marker style for overlaid points.
    size : float, optional
        Marker size.
    ylim : tuple of float, optional
        Data axis limits.
    extend : float
        Fraction to extend limits.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.shape[0]
    is_matrix = x_arr.ndim == 2
    data_lim = _expand_data_lim(x_arr, ylim, extend)
    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    if add_points is None:
        add_points = smooth

    _x = x_arr
    _which = which
    _data_lim = data_lim
    _gp = merged_gp
    _add_points = add_points
    _axis = axis
    _border = border

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]
        positions = np.arange(1, ni + 1).astype(float)

        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0.5, ni + 0.5) if not _is_row(_which) else _data_lim,
            yscale=_data_lim if not _is_row(_which) else (0.5, ni + 0.5),
        ))

        if is_matrix:
            n_lines = subset.shape[1]
            for j in range(n_lines):
                vals = subset[:, j]
                if _is_row(_which):
                    grid_py.grid_lines(
                        x=grid_py.Unit(vals, "native"),
                        y=grid_py.Unit(positions, "native"),
                        gp=_to_gpar(**_gp),
                    )
                else:
                    grid_py.grid_lines(
                        x=grid_py.Unit(positions, "native"),
                        y=grid_py.Unit(vals, "native"),
                        gp=_to_gpar(**_gp),
                    )
        else:
            if _is_row(_which):
                grid_py.grid_lines(
                    x=grid_py.Unit(subset, "native"),
                    y=grid_py.Unit(positions, "native"),
                    gp=_to_gpar(**_gp),
                )
            else:
                grid_py.grid_lines(
                    x=grid_py.Unit(positions, "native"),
                    y=grid_py.Unit(subset, "native"),
                    gp=_to_gpar(**_gp),
                )

            if _add_points:
                if _is_row(_which):
                    grid_py.grid_points(
                        x=grid_py.Unit(subset, "native"),
                        y=grid_py.Unit(positions, "native"),
                        pch=pch,
                        gp=_to_gpar(**_gp),
                    )
                else:
                    grid_py.grid_points(
                        x=grid_py.Unit(positions, "native"),
                        y=grid_py.Unit(subset, "native"),
                        pch=pch,
                        gp=_to_gpar(**_gp),
                    )

        grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black" if _border else "transparent"))
        if _axis and k == 1:
            ticks = grid_py.grid_pretty(_data_lim)
            ticks = [float(t) for t in ticks if _data_lim[0] <= t <= _data_lim[1]]
            tick_labels = [f"{t:g}" for t in ticks]
            if _is_row(_which):
                grid_py.grid_xaxis(at=ticks, label=tick_labels,
                                    gp=grid_py.Gpar(fontsize=7))
            else:
                grid_py.grid_yaxis(at=ticks, label=tick_labels,
                                    gp=grid_py.Gpar(fontsize=7))

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_lines",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "array" if not is_matrix else "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_text
# =========================================================================


def anno_text(
    x: ArrayLike,
    which: str = "column",
    gp: Optional[Dict[str, Any]] = None,
    rot: Optional[float] = None,
    just: Optional[str] = None,
    location: Optional[float] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    show_name: bool = False,
) -> AnnotationFunction:
    """Text annotation.

    Parameters
    ----------
    x : array-like of str
        Text labels.
    which : str
        ``"column"`` or ``"row"``.
    gp : dict, optional
        Graphic parameters (``fontsize``, ``col``, etc.).
    rot : float, optional
        Rotation angle in degrees.
    just : str, optional
        Text justification.
    location : float, optional
        Position along the non-observation axis (0-1, npc).
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.
    show_name : bool
        Whether to show the annotation name.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x)
    n = len(x_arr)

    if rot is None:
        rot = 90.0 if which == "column" else 0.0
    if just is None:
        just = "center"

    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _rot = rot
    _just = just
    _gp = merged_gp
    _location = location

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]
        loc = _location if _location is not None else 0.5

        for i, txt in enumerate(subset):
            if _which == "column":
                xp = (i + 0.5) / ni
                grid_py.grid_text(
                    label=str(txt),
                    x=grid_py.Unit(xp, "npc"),
                    y=grid_py.Unit(loc, "npc"),
                    rot=_rot,
                    just=_just,
                    gp=_to_gpar(**_gp),
                )
            else:
                yp = (ni - i - 0.5) / ni
                grid_py.grid_text(
                    label=str(txt),
                    x=grid_py.Unit(loc, "npc"),
                    y=grid_py.Unit(yp, "npc"),
                    rot=_rot,
                    just=_just,
                    gp=_to_gpar(**_gp),
                )

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_text",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=(0.0, 1.0),
        subsettable=True,
        subset_rule={"x": "array"},
        show_name=show_name,
        width=w,
        height=h,
    )


# =========================================================================
# anno_histogram
# =========================================================================


def anno_histogram(
    x: ArrayLike,
    which: str = "column",
    n_breaks: int = 11,
    border: bool = False,
    gp: Optional[Dict[str, Any]] = None,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Histogram annotation.

    Parameters
    ----------
    x : array-like
        2-D array -- each row's values are binned into a histogram.
    which : str
        ``"column"`` or ``"row"``.
    n_breaks : int
        Number of histogram bins.
    border : bool
        Draw bin borders.
    gp : dict, optional
        Graphic parameters.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    n = x_arr.shape[0]

    global_min = float(np.nanmin(x_arr))
    global_max = float(np.nanmax(x_arr))
    data_lim = (global_min, global_max)
    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _data_lim = data_lim
    _n_breaks = n_breaks
    _gp = merged_gp
    _border = border
    _axis = axis

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        for pos, idx in enumerate(index):
            row_data = _x[idx, :]
            row_data = row_data[~np.isnan(row_data)]
            if len(row_data) == 0:
                continue

            counts, bin_edges = np.histogram(row_data, bins=_n_breaks, range=_data_lim)
            max_count = counts.max() if counts.max() > 0 else 1
            normed = counts / max_count

            grid_py.push_viewport(grid_py.Viewport(
                x=grid_py.Unit((pos + 0.5) / ni, "npc") if not _is_row(_which) else grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc") if not _is_row(_which) else grid_py.Unit((ni - pos - 0.5) / ni, "npc"),
                width=grid_py.Unit(1.0 / ni, "npc") if not _is_row(_which) else grid_py.Unit(1, "npc"),
                height=grid_py.Unit(1, "npc") if not _is_row(_which) else grid_py.Unit(1.0 / ni, "npc"),
                xscale=_data_lim if not _is_row(_which) else (0, 1),
                yscale=(0, 1) if not _is_row(_which) else _data_lim,
            ))

            fill_col = _gp.get("fill", "steelblue")
            border_col = _gp.get("col", "black") if _border else "transparent"

            bin_widths = np.diff(bin_edges)
            for bi in range(len(counts)):
                if normed[bi] <= 0:
                    continue
                if not _is_row(_which):
                    grid_py.grid_rect(
                        x=grid_py.Unit(bin_edges[bi], "native"),
                        y=grid_py.Unit(0, "npc"),
                        width=grid_py.Unit(bin_widths[bi], "native"),
                        height=grid_py.Unit(normed[bi], "npc"),
                        just=["left", "bottom"],
                        gp=_to_gpar(fill=fill_col, col=border_col),
                    )
                else:
                    grid_py.grid_rect(
                        x=grid_py.Unit(0, "npc"),
                        y=grid_py.Unit(bin_edges[bi], "native"),
                        width=grid_py.Unit(normed[bi], "npc"),
                        height=grid_py.Unit(bin_widths[bi], "native"),
                        just=["left", "bottom"],
                        gp=_to_gpar(fill=fill_col, col=border_col),
                    )

            grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_histogram",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_density
# =========================================================================


def anno_density(
    x: ArrayLike,
    which: str = "column",
    type: str = "lines",
    xlim: Optional[Tuple[float, float]] = None,
    heatmap_colors: Optional[Any] = None,
    joyplot_scale: float = 1.0,
    border: bool = True,
    gp: Optional[Dict[str, Any]] = None,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Density-plot annotation.

    Parameters
    ----------
    x : array-like
        2-D array -- each row is a distribution.
    which : str
        ``"column"`` or ``"row"``.
    type : str
        ``"lines"``, ``"violin"``, or ``"heatmap"``.
    xlim : tuple of float, optional
        Range for the density estimation axis.
    heatmap_colors : colormap, optional
        Colours for heatmap mode.
    joyplot_scale : float
        Vertical scaling for joyplot-style rendering.
    border : bool
        Draw border.
    gp : dict, optional
        Graphic parameters.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    n = x_arr.shape[0]

    global_min = float(np.nanmin(x_arr))
    global_max = float(np.nanmax(x_arr))
    if xlim is not None:
        data_lim: Tuple[float, float] = tuple(xlim)  # type: ignore[assignment]
    else:
        rng = global_max - global_min if global_max != global_min else 1.0
        data_lim = (global_min - 0.1 * rng, global_max + 0.1 * rng)

    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _type = type
    _data_lim = data_lim
    _gp = merged_gp
    _border = border
    _axis = axis

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        for pos, idx in enumerate(index):
            row_data = _x[idx, :]
            row_data = row_data[~np.isnan(row_data)]
            if len(row_data) < 2:
                continue

            # Push per-observation viewport
            grid_py.push_viewport(grid_py.Viewport(
                x=grid_py.Unit((pos + 0.5) / ni, "npc") if not _is_row(_which) else grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc") if not _is_row(_which) else grid_py.Unit((ni - pos - 0.5) / ni, "npc"),
                width=grid_py.Unit(1.0 / ni, "npc") if not _is_row(_which) else grid_py.Unit(1, "npc"),
                height=grid_py.Unit(1, "npc") if not _is_row(_which) else grid_py.Unit(1.0 / ni, "npc"),
                xscale=_data_lim,
                yscale=(0, 1),
            ))

            # Simple density via histogram approximation
            n_grid = 200
            grid_vals = np.linspace(_data_lim[0], _data_lim[1], n_grid)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(row_data)
                density = kde(grid_vals)
            except ImportError:
                counts_arr, edges = np.histogram(row_data, bins=50, range=_data_lim, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                density = np.interp(grid_vals, centers, counts_arr)

            max_d = density.max() if density.max() > 0 else 1.0
            normed = density / max_d

            fill_col = _gp.get("fill", "steelblue")
            border_col = _gp.get("col", "black") if _border else "transparent"

            if _type in ("lines", "violin"):
                # Draw as polygon
                x_poly = np.concatenate([grid_vals, grid_vals[::-1]])
                y_poly = np.concatenate([normed * 0.45 + 0.5, (0.5 - normed * 0.45)[::-1]])
                grid_py.grid_polygon(
                    x=grid_py.Unit(x_poly, "native"),
                    y=grid_py.Unit(y_poly, "npc"),
                    gp=_to_gpar(fill=fill_col, col=border_col),
                )
            # heatmap mode: draw colored rect segments
            elif _type == "heatmap":
                if heatmap_colors is not None:
                    cmap_obj = heatmap_colors
                else:
                    from ._color import color_ramp2
                    _ylor = color_ramp2([0, 0.5, 1],
                                        ["#FFFFCC", "#FD8D3C", "#800026"])
                    cmap_obj = lambda v: _ylor(float(v))
                for gi in range(n_grid - 1):
                    c = cmap_obj(normed[gi])
                    grid_py.grid_rect(
                        x=grid_py.Unit(grid_vals[gi], "native"),
                        y=grid_py.Unit(0, "npc"),
                        width=grid_py.Unit(grid_vals[gi + 1] - grid_vals[gi], "native"),
                        height=grid_py.Unit(1, "npc"),
                        just=["left", "bottom"],
                        gp=_to_gpar(fill=c, col=c),
                    )

            grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_density",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_joyplot
# =========================================================================


def anno_joyplot(
    x: ArrayLike,
    which: str = "column",
    gp: Optional[Dict[str, Any]] = None,
    scale: float = 2.0,
    transparency: float = 0.6,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Joy-plot / ridgeline annotation.

    Parameters
    ----------
    x : array-like
        2-D array -- each row is a distribution.
    which : str
        ``"column"`` or ``"row"``.
    gp : dict, optional
        Graphic parameters.
    scale : float
        Vertical scaling factor (controls overlap between ridges).
    transparency : float
        Alpha transparency for filled areas.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    n = x_arr.shape[0]

    global_min = float(np.nanmin(x_arr))
    global_max = float(np.nanmax(x_arr))
    rng = global_max - global_min if global_max != global_min else 1.0
    data_lim = (global_min - 0.1 * rng, global_max + 0.1 * rng)

    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _data_lim = data_lim
    _gp = merged_gp
    _scale = scale
    _transparency = transparency

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        grid_py.push_viewport(grid_py.Viewport(
            xscale=_data_lim if not _is_row(_which) else (0, ni + _scale),
            yscale=(0, ni + _scale) if not _is_row(_which) else _data_lim,
        ))

        grid_vals = np.linspace(_data_lim[0], _data_lim[1], 200)

        for pos, idx in enumerate(index):
            row_data = _x[idx, :]
            row_data = row_data[~np.isnan(row_data)]
            if len(row_data) < 2:
                continue

            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(row_data)
                density = kde(grid_vals)
            except ImportError:
                counts_arr, edges = np.histogram(row_data, bins=50, range=_data_lim, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                density = np.interp(grid_vals, centers, counts_arr)

            max_d = density.max() if density.max() > 0 else 1.0
            normed = density / max_d * _scale

            fill_col = _gp.get("fill", "steelblue")

            if not _is_row(_which):
                x_poly = np.concatenate([grid_vals, grid_vals[::-1]])
                y_poly = np.concatenate([pos + normed, np.full(len(grid_vals), pos)])
                grid_py.grid_polygon(
                    x=grid_py.Unit(x_poly, "native"),
                    y=grid_py.Unit(y_poly, "native"),
                    gp=_to_gpar(fill=fill_col, col="black", alpha=_transparency),
                )
            else:
                x_poly = np.concatenate([pos + normed, np.full(len(grid_vals), pos)])
                y_poly = np.concatenate([grid_vals, grid_vals[::-1]])
                grid_py.grid_polygon(
                    x=grid_py.Unit(x_poly, "native"),
                    y=grid_py.Unit(y_poly, "native"),
                    gp=_to_gpar(fill=fill_col, col="black", alpha=_transparency),
                )

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_joyplot",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_horizon
# =========================================================================


def anno_horizon(
    x: ArrayLike,
    which: str = "column",
    gp: Optional[Dict[str, Any]] = None,
    n_slice: int = 4,
    negative_from_top: bool = False,
    normalize: bool = True,
    gap: float = 0.0,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Horizon-chart annotation.

    Parameters
    ----------
    x : array-like
        Numeric values (2-D, rows = observations, columns = time points).
    which : str
        ``"column"`` or ``"row"``.
    gp : dict, optional
        Graphic parameters.
    n_slice : int
        Number of colour slices.
    negative_from_top : bool
        If True, negative bands drawn from top.
    normalize : bool
        Normalize each row to [0, 1].
    gap : float
        Gap between rows.
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Extra axis configuration.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    n = x_arr.shape[0]

    data_lim = (0.0, 1.0) if normalize else (
        float(np.nanmin(x_arr)),
        float(np.nanmax(x_arr)),
    )
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _n_slice = n_slice
    _normalize = normalize
    _gap = gap

    def _draw(index: np.ndarray, k: int, n_slices_total: int) -> None:
        ni = len(index)
        from ._color import color_ramp2
        _blues_ramp = color_ramp2([0, 1], ["#F7FBFF", "#08306B"])
        pos_cmap = lambda v: _blues_ramp(float(v))

        for pos, idx in enumerate(index):
            row = _x[idx, :]
            valid = row[~np.isnan(row)]
            if len(valid) == 0:
                continue

            if _normalize:
                rmin, rmax = valid.min(), valid.max()
                r = rmax - rmin if rmax != rmin else 1.0
                vals = (valid - rmin) / r
            else:
                vals = valid

            n_t = len(vals)

            grid_py.push_viewport(grid_py.Viewport(
                x=grid_py.Unit((pos + 0.5) / ni, "npc") if not _is_row(_which) else grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc") if not _is_row(_which) else grid_py.Unit((ni - pos - 0.5) / ni, "npc"),
                width=grid_py.Unit(1.0 / ni, "npc") if not _is_row(_which) else grid_py.Unit(1, "npc"),
                height=grid_py.Unit(1, "npc") if not _is_row(_which) else grid_py.Unit(1.0 / ni, "npc"),
                xscale=(0, n_t - 1),
                yscale=(0, 1),
            ))

            for s in range(_n_slice):
                lo = s / _n_slice
                hi_val = (s + 1) / _n_slice
                band = np.clip(vals - lo, 0, hi_val - lo) / (hi_val - lo)

                fill_col = pos_cmap(s + 1)
                for t in range(n_t - 1):
                    if band[t] <= 0 and band[t + 1] <= 0:
                        continue
                    x_poly = np.array([t, t + 1, t + 1, t])
                    y_poly = np.array([0, 0, band[t + 1], band[t]])
                    grid_py.grid_polygon(
                        x=grid_py.Unit(x_poly, "native"),
                        y=grid_py.Unit(y_poly, "native"),
                        gp=_to_gpar(fill=fill_col, col=fill_col),
                    )

            grid_py.up_viewport()

    var_env: Dict[str, Any] = {"x": x_arr, "gp": gp}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_horizon",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "matrix_row"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_image
# =========================================================================


def anno_image(
    image: Sequence[str],
    which: str = "column",
    border: bool = True,
    gp: Optional[Dict[str, Any]] = None,
    space: float = 1.0,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Image annotation from file paths.

    Parameters
    ----------
    image : sequence of str
        File paths to images (PNG, JPG, etc.).
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border around each image cell.
    gp : dict, optional
        Graphic parameters.
    space : float
        Spacing (mm) around images.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    image_paths = list(image)
    n = len(image_paths)
    w, h = _default_width_height(which, width, height)
    merged_gp = _resolve_gp(gp)

    _paths = image_paths
    _which = which
    _border = border
    _gp = merged_gp

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        for pos, idx in enumerate(index):
            path = _paths[idx]
            if not path or path.strip() == "":
                continue

            grid_py.push_viewport(grid_py.Viewport(
                x=grid_py.Unit((pos + 0.5) / ni, "npc") if not _is_row(_which) else grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc") if not _is_row(_which) else grid_py.Unit((ni - pos - 0.5) / ni, "npc"),
                width=grid_py.Unit(1.0 / ni, "npc") if not _is_row(_which) else grid_py.Unit(1, "npc"),
                height=grid_py.Unit(1, "npc") if not _is_row(_which) else grid_py.Unit(1.0 / ni, "npc"),
            ))

            try:
                from PIL import Image as _PILImage
                img_data = np.asarray(_PILImage.open(path).convert("RGBA")) / 255.0
                grid_py.grid_raster(
                    image=img_data,
                    x=grid_py.Unit(0.5, "npc"),
                    y=grid_py.Unit(0.5, "npc"),
                    width=grid_py.Unit(1, "npc"),
                    height=grid_py.Unit(1, "npc"),
                    interpolate=True,
                )
            except (FileNotFoundError, OSError):
                warnings.warn(f"Cannot read image: {path}", stacklevel=2)

            if _border:
                grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))

            grid_py.up_viewport()

    var_env: Dict[str, Any] = {"image_paths": image_paths}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_image",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=(0.0, 1.0),
        subsettable=True,
        subset_rule={"image_paths": "array"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_link
# =========================================================================


def anno_link(
    align_to: Union[Dict[str, Sequence[int]], Sequence[Sequence[int]]],
    panel_fun: Optional[Callable[..., Any]] = None,
    which: str = "column",
    side: Optional[str] = None,
    size: Optional[float] = None,
    gap: float = 1.0,
    link_width: float = 5.0,
    link_gp: Optional[Dict[str, Any]] = None,
    extend: float = 0.0,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    internal_line: bool = True,
) -> AnnotationFunction:
    """Link / zoom annotation connecting heatmap regions to detail panels.

    Parameters
    ----------
    align_to : dict or list
        Mapping from label to indices, or list of index groups.
    panel_fun : callable, optional
        Drawing function for each linked panel.
    which : str
        ``"column"`` or ``"row"``.
    side : str, optional
        Side to place links.
    size : float, optional
        Size of each detail panel (mm).
    gap : float
        Gap between heatmap and link region (mm).
    link_width : float
        Width of the connecting link region (mm).
    link_gp : dict, optional
        Graphic parameters for connecting lines.
    extend : float
        Extension beyond heatmap region.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.
    internal_line : bool
        Draw internal separator lines.

    Returns
    -------
    AnnotationFunction
    """
    if isinstance(align_to, dict):
        groups = [np.asarray(v, dtype=int) for v in align_to.values()]
        group_labels = list(align_to.keys())
    else:
        groups = [np.asarray(v, dtype=int) for v in align_to]
        group_labels = [str(i) for i in range(len(groups))]

    all_indices = np.concatenate(groups) if groups else np.array([], dtype=int)
    n_val = int(all_indices.max()) + 1 if len(all_indices) > 0 else 0

    link_merged = _resolve_gp(link_gp)

    _groups = groups
    _which = which
    _panel_fun = panel_fun
    _link_gp = link_merged

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        n_groups = len(_groups)

        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0, 1) if _is_row(_which) else (-0.5, ni - 0.5),
            yscale=(-0.5, ni - 0.5) if _is_row(_which) else (0, 1),
        ))

        for g_idx, grp in enumerate(_groups):
            mask = np.isin(index, grp)
            if not np.any(mask):
                continue

            positions = np.where(mask)[0]
            src_lo = float(positions.min()) - 0.5
            src_hi = float(positions.max()) + 0.5

            dst_lo = g_idx / max(n_groups, 1)
            dst_hi = (g_idx + 1) / max(n_groups, 1)

            fill_col = _link_gp.get("fill", "lightgrey")
            border_col = _link_gp.get("col", "grey")

            if not _is_row(_which):
                x_poly = np.array([src_lo, src_hi, dst_hi * ni - 0.5, dst_lo * ni - 0.5])
                y_poly = np.array([0.3, 0.3, 0.7, 0.7])
                grid_py.grid_polygon(
                    x=grid_py.Unit(x_poly, "native"),
                    y=grid_py.Unit(y_poly, "npc"),
                    gp=_to_gpar(fill=fill_col, col=border_col, alpha=0.5),
                )
            else:
                x_poly = np.array([0.3, 0.3, 0.7, 0.7])
                y_poly = np.array([src_lo, src_hi, dst_hi * ni - 0.5, dst_lo * ni - 0.5])
                grid_py.grid_polygon(
                    x=grid_py.Unit(x_poly, "npc"),
                    y=grid_py.Unit(y_poly, "native"),
                    gp=_to_gpar(fill=fill_col, col=border_col, alpha=0.5),
                )

        grid_py.up_viewport()

    var_env: Dict[str, Any] = {"groups": groups, "group_labels": group_labels}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_link",
        which=which,
        var_env=var_env,
        n=n_val,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=False,
        width=width or (link_width + (size or _DEFAULT_SIZE)),
        height=height or (link_width + (size or _DEFAULT_SIZE)),
    )


# =========================================================================
# anno_mark
# =========================================================================


def anno_mark(
    at: Sequence[int],
    labels: Sequence[str],
    which: str = "column",
    side: Optional[str] = None,
    lines_gp: Optional[Dict[str, Any]] = None,
    labels_gp: Optional[Dict[str, Any]] = None,
    labels_rot: Optional[float] = None,
    padding: float = 1.0,
    link_width: float = 5.0,
    link_height: Optional[float] = None,
    link_gp: Optional[Dict[str, Any]] = None,
    extend: float = 0.0,
) -> AnnotationFunction:
    """Mark annotation with labelled connectors.

    Faithfully ports R's ``anno_mark`` (AnnotationFunction-function.R:3132).

    Parameters
    ----------
    at : sequence of int
        Observation indices to mark.
    labels : sequence of str
        Labels corresponding to each *at* position.
    which : str
        ``"column"`` or ``"row"``.
    side : str, optional
        Side to place labels.  Default ``"right"`` for row,
        ``"top"`` for column (matching R).
    lines_gp : dict, optional
        Graphic parameters for connector lines (deprecated alias for
        *link_gp*).
    labels_gp : dict, optional
        Graphic parameters for label text.
    labels_rot : float, optional
        Label rotation angle.  Default ``0`` for row, ``90`` for column.
    padding : float
        Padding around labels (mm).  R default: ``unit(1, "mm")``.
    link_width : float
        Width of connecting-line region (mm).  R default: ``unit(5, "mm")``.
    link_height : float, optional
        Height of connecting-line region (mm).  Defaults to *link_width*.
    link_gp : dict, optional
        Graphic parameters for connecting lines.
    extend : float
        Extension beyond heatmap region (mm).

    Returns
    -------
    AnnotationFunction
    """
    from ._utils import smart_align, max_text_width, max_text_height

    at_arr = np.asarray(at, dtype=int)
    labels_arr = np.asarray(labels, dtype=str)
    n_val = int(at_arr.max()) + 1 if len(at_arr) > 0 else 0

    if link_height is None:
        link_height = link_width

    if link_gp is None:
        link_gp = lines_gp
    link_merged = _resolve_gp(link_gp)
    labels_merged = _resolve_gp(labels_gp)

    if side is None:
        side = "right" if _is_row(which) else "top"

    if labels_rot is None:
        labels_rot = 0.0 if _is_row(which) else 90.0

    # -- R: od = order(at); at = at[od]; labels = labels[od] --------
    od = np.argsort(at_arr)
    at_arr = at_arr[od]
    labels_arr = labels_arr[od]

    # -- Build lookup tables (R: at2index, at2labels) ---------------
    at2index = {int(a): i for i, a in enumerate(at_arr)}
    at2labels = {int(a): str(l) for a, l in zip(at_arr, labels_arr)}

    # -- Compute annotation natural size (R:3174-3180) --------------
    # R: width = link_width + max_text_width(labels, gp=labels_gp, rot=labels_rot)
    # R: height = link_height + max_text_height(labels, gp=labels_gp, rot=labels_rot)
    _link_w_mm = float(link_width)
    _link_h_mm = float(link_height)

    if _is_row(which):
        # R: width = link_width + max_text_width(labels, gp, rot)
        # max_text_width returns max(grobWidth(textGrob(text[i],...))) in mm
        _text_w = max_text_width(list(labels_arr), gp=labels_merged,
                                 rot=labels_rot)
        _tw_mm_arr = grid_py.convert_width(_text_w, "mm", valueOnly=True)
        _tw_mm = float(np.max(_tw_mm_arr))
        anno_width_mm = _link_w_mm + _tw_mm
        anno_width = grid_py.Unit(anno_width_mm, "mm")
        anno_height = None  # 1 npc
    else:
        # R: height = link_height + max_text_height(labels, gp, rot)
        _text_h = max_text_height(list(labels_arr), gp=labels_merged,
                                  rot=labels_rot)
        _th_mm_arr = grid_py.convert_height(_text_h, "mm", valueOnly=True)
        _th_mm = float(np.max(_th_mm_arr))
        anno_height_mm = _link_h_mm + _th_mm
        anno_height = grid_py.Unit(anno_height_mm, "mm")
        anno_width = None  # 1 npc

    # -- Capture variables for the draw closure ----------------------
    _at = at_arr
    _labels = labels_arr
    _which = which
    _side = side
    _link_gp = link_merged
    _labels_gp = labels_merged
    _labels_rot = labels_rot % 360
    _padding_mm = float(padding)
    _extend = np.atleast_1d(np.asarray(extend, dtype=float))
    if len(_extend) == 1:
        _extend = np.repeat(_extend, 2)

    # -- Row draw function (R:3193-3268) -----------------------------
    def _row_draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        # Intersect `at` with current slice index (R:3202-3210)
        cur_at = np.array([a for a in _at if a in index], dtype=int)
        if len(cur_at) == 0:
            return
        # R reverses for right-side: labels = rev(at2labels[...])
        cur_labels = [at2labels[int(a)] for a in reversed(cur_at)]
        cur_link_gp_indices = [at2index[int(a)] for a in reversed(cur_at)]

        # R: if(is.null(.scale)) .scale = c(0.5, n+0.5)
        # When _pos/_scale are injected by HeatmapList (split scenario),
        # use those instead of default positions.
        _injected_pos = var_env.get('_pos', None)
        _injected_scale = var_env.get('_scale', None)

        if _injected_scale is not None:
            scale = tuple(_injected_scale)
        else:
            scale = (0.5, ni + 0.5)

        # R: pushViewport(viewport(xscale = c(0, 1), yscale = .scale))
        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0, 1), yscale=scale,
        ))

        # Extend in native units
        _cv = lambda u, to: float(np.squeeze(
            grid_py.convert_height(u, to, valueOnly=True)))
        ext_native = np.array([
            _cv(grid_py.Unit(_extend[0], "mm"), "native"),
            _cv(grid_py.Unit(_extend[1], "mm"), "native"),
        ])

        # R: text_height = convertHeight(text_height(labels, gp) + padding, "native")
        txt_h_list = []
        for lbl in cur_labels:
            if _labels_rot in (90, 270):
                th = _cv(
                    grid_py.string_width([lbl]) + grid_py.Unit(_padding_mm, "mm"),
                    "native")
            else:
                th = _cv(
                    grid_py.string_height([lbl]) + grid_py.Unit(_padding_mm, "mm"),
                    "native")
            txt_h_list.append(th)
        text_height = np.array(txt_h_list)

        # R: if(is.null(.pos)) { i2 = rev(which(...))); pos = n-i2+1 }
        #    else { pos = .pos[rev(which(index %in% at))] }
        rev_indices = [np.where(index == a)[0][0] for a in reversed(cur_at)]
        if _injected_pos is not None:
            pos = np.array([_injected_pos[j] for j in rev_indices],
                           dtype=float)
        else:
            i2 = np.array(rev_indices, dtype=float)
            pos = ni - i2

        # R: smartAlign(h1, h2, c(.scale[1] - extend[1], .scale[2] + extend[2]))
        h1 = pos - text_height * 0.5
        h2 = pos + text_height * 0.5
        pos_adjusted = smart_align(
            h1, h2,
            (scale[0] - ext_native[0], scale[1] + ext_native[1]),
        )
        h = (pos_adjusted[:, 0] + pos_adjusted[:, 1]) / 2.0

        n2 = len(cur_labels)

        # -- Determine justification (R:3234-3253) ---
        if _side == "right":
            if _labels_rot == 90:
                just = ["center", "top"]
            elif _labels_rot == 270:
                just = ["center", "bottom"]
            elif 90 < _labels_rot < 270:
                just = ["right", "center"]
            else:
                just = ["left", "center"]
        else:
            if _labels_rot == 90:
                just = ["center", "bottom"]
            elif _labels_rot == 270:
                just = ["center", "top"]
            elif 90 < _labels_rot < 270:
                just = ["left", "center"]
            else:
                just = ["right", "center"]

        lw_unit = grid_py.Unit(_link_w_mm, "mm")

        if _side == "right":
            # R:3255-3260
            for i_m in range(n2):
                gp_i = {k: v for k, v in _labels_gp.items()}
                grid_py.grid_text(
                    label=cur_labels[i_m],
                    x=lw_unit,
                    y=grid_py.Unit(h[i_m], "native"),
                    just=just,
                    rot=_labels_rot,
                    gp=_to_gpar(**gp_i),
                )
            # R: link_width = link_width - unit(1, "mm")
            lw_seg = grid_py.Unit(_link_w_mm - 1.0, "mm")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _link_gp.items()}
                # Segment 1: x=0 → x=link_width*1/3, y=pos
                grid_py.grid_segments(
                    x0=grid_py.Unit(0, "npc"),
                    y0=grid_py.Unit(pos[i_m], "native"),
                    x1=lw_seg * (1.0 / 3),
                    y1=grid_py.Unit(pos[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )
                # Segment 2: link_width*1/3 → 2/3, pos → h
                grid_py.grid_segments(
                    x0=lw_seg * (1.0 / 3),
                    y0=grid_py.Unit(pos[i_m], "native"),
                    x1=lw_seg * (2.0 / 3),
                    y1=grid_py.Unit(h[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )
                # Segment 3: link_width*2/3 → link_width, h → h
                grid_py.grid_segments(
                    x0=lw_seg * (2.0 / 3),
                    y0=grid_py.Unit(h[i_m], "native"),
                    x1=lw_seg,
                    y1=grid_py.Unit(h[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )
        else:
            # R:3262-3266 (side == "left")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _labels_gp.items()}
                grid_py.grid_text(
                    label=cur_labels[i_m],
                    x=grid_py.Unit(1, "npc") - lw_unit,
                    y=grid_py.Unit(h[i_m], "native"),
                    just=just,
                    rot=_labels_rot,
                    gp=_to_gpar(**gp_i),
                )
            lw_seg = grid_py.Unit(_link_w_mm - 1.0, "mm")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _link_gp.items()}
                grid_py.grid_segments(
                    x0=grid_py.Unit(1, "npc"),
                    y0=grid_py.Unit(pos[i_m], "native"),
                    x1=grid_py.Unit(1, "npc") - lw_seg * (1.0 / 3),
                    y1=grid_py.Unit(pos[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(1, "npc") - lw_seg * (1.0 / 3),
                    y0=grid_py.Unit(pos[i_m], "native"),
                    x1=grid_py.Unit(1, "npc") - lw_seg * (2.0 / 3),
                    y1=grid_py.Unit(h[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(1, "npc") - lw_seg * (2.0 / 3),
                    y0=grid_py.Unit(h[i_m], "native"),
                    x1=grid_py.Unit(1, "npc") - lw_seg,
                    y1=grid_py.Unit(h[i_m], "native"),
                    gp=_to_gpar(**gp_i),
                )

        grid_py.up_viewport()

    # -- Column draw function (R:3270-3346) --------------------------
    def _col_draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        cur_at = np.array([a for a in _at if a in index], dtype=int)
        if len(cur_at) == 0:
            return
        # Column: no reversal (R:3284)
        cur_labels = [at2labels[int(a)] for a in cur_at]
        cur_link_gp_indices = [at2index[int(a)] for a in cur_at]

        scale = (0.5, ni + 0.5)
        # R: pushViewport(viewport(yscale = c(0, 1), xscale = .scale))
        grid_py.push_viewport(grid_py.Viewport(
            xscale=scale, yscale=(0, 1),
        ))

        _cvw = lambda u, to: float(np.squeeze(
            grid_py.convert_width(u, to, valueOnly=True)))
        ext_native = np.array([
            _cvw(grid_py.Unit(_extend[0], "mm"), "native"),
            _cvw(grid_py.Unit(_extend[1], "mm"), "native"),
        ])

        txt_h_list = []
        for lbl in cur_labels:
            if _labels_rot in (0, 180):
                th = _cvw(
                    grid_py.string_width([lbl]) + grid_py.Unit(_padding_mm, "mm"),
                    "native")
            else:
                th = _cvw(
                    grid_py.string_height([lbl]) + grid_py.Unit(_padding_mm, "mm"),
                    "native")
            txt_h_list.append(th)
        text_height = np.array(txt_h_list)

        # R: i2 = which(index %in% at); pos = i2
        i2 = np.array([np.where(index == a)[0][0] for a in cur_at],
                       dtype=float)
        pos = i2 + 1  # 1-based in R

        h1 = pos - text_height * 0.5
        h2 = pos + text_height * 0.5
        pos_adjusted = smart_align(
            h1, h2,
            (scale[0] - ext_native[0], scale[1] + ext_native[1]),
        )
        h = (pos_adjusted[:, 0] + pos_adjusted[:, 1]) / 2.0

        n2 = len(cur_labels)

        # Justification (R:3311-3330)
        if _side == "top":
            if _labels_rot == 0:
                just = ["center", "bottom"]
            elif _labels_rot == 180:
                just = ["center", "top"]
            elif 0 < _labels_rot < 180:
                just = ["left", "center"]
            else:
                just = ["right", "center"]
        else:
            if _labels_rot == 0:
                just = ["center", "top"]
            elif _labels_rot == 180:
                just = ["center", "bottom"]
            elif 0 < _labels_rot < 180:
                just = ["right", "center"]
            else:
                just = ["left", "center"]

        lh_unit = grid_py.Unit(_link_h_mm, "mm")

        if _side == "top":
            # R:3332-3337
            for i_m in range(n2):
                gp_i = {k: v for k, v in _labels_gp.items()}
                grid_py.grid_text(
                    label=cur_labels[i_m],
                    x=grid_py.Unit(h[i_m], "native"),
                    y=lh_unit,
                    just=just,
                    rot=_labels_rot,
                    gp=_to_gpar(**gp_i),
                )
            lh_seg = grid_py.Unit(_link_h_mm - 1.0, "mm")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _link_gp.items()}
                grid_py.grid_segments(
                    x0=grid_py.Unit(pos[i_m], "native"),
                    y0=grid_py.Unit(0, "npc"),
                    x1=grid_py.Unit(pos[i_m], "native"),
                    y1=lh_seg * (1.0 / 3),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(pos[i_m], "native"),
                    y0=lh_seg * (1.0 / 3),
                    x1=grid_py.Unit(h[i_m], "native"),
                    y1=lh_seg * (2.0 / 3),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(h[i_m], "native"),
                    y0=lh_seg * (2.0 / 3),
                    x1=grid_py.Unit(h[i_m], "native"),
                    y1=lh_seg,
                    gp=_to_gpar(**gp_i),
                )
        else:
            # R:3338-3344 (side == "bottom")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _labels_gp.items()}
                grid_py.grid_text(
                    label=cur_labels[i_m],
                    x=grid_py.Unit(h[i_m], "native"),
                    y=grid_py.Unit(1, "npc") - lh_unit,
                    just=just,
                    rot=_labels_rot,
                    gp=_to_gpar(**gp_i),
                )
            lh_seg = grid_py.Unit(_link_h_mm - 1.0, "mm")
            for i_m in range(n2):
                gp_i = {k: v for k, v in _link_gp.items()}
                grid_py.grid_segments(
                    x0=grid_py.Unit(pos[i_m], "native"),
                    y0=grid_py.Unit(1, "npc"),
                    x1=grid_py.Unit(pos[i_m], "native"),
                    y1=grid_py.Unit(1, "npc") - lh_seg * (1.0 / 3),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(pos[i_m], "native"),
                    y0=grid_py.Unit(1, "npc") - lh_seg * (1.0 / 3),
                    x1=grid_py.Unit(h[i_m], "native"),
                    y1=grid_py.Unit(1, "npc") - lh_seg * (2.0 / 3),
                    gp=_to_gpar(**gp_i),
                )
                grid_py.grid_segments(
                    x0=grid_py.Unit(h[i_m], "native"),
                    y0=grid_py.Unit(1, "npc") - lh_seg * (2.0 / 3),
                    x1=grid_py.Unit(h[i_m], "native"),
                    y1=grid_py.Unit(1, "npc") - lh_seg,
                    gp=_to_gpar(**gp_i),
                )

        grid_py.up_viewport()

    # -- Select draw function ----------------------------------------
    if _is_row(which):
        _draw_fn = _row_draw
    else:
        _draw_fn = _col_draw

    var_env: Dict[str, Any] = {"at": at_arr, "labels": labels_arr}
    return AnnotationFunction(
        fun=_draw_fn,
        fun_name="anno_mark",
        which=which,
        var_env=var_env,
        n=n_val,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=False,
        width=anno_width,
        height=anno_height,
    )


# =========================================================================
# anno_empty
# =========================================================================


def anno_empty(
    which: str = "column",
    border: bool = True,
    zoom: bool = False,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    show_name: bool = False,
) -> AnnotationFunction:
    """Empty annotation placeholder.

    Parameters
    ----------
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border around the empty region.
    zoom : bool
        If True and heatmap is split, empty slices have equal size.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.
    show_name : bool
        Whether to show the annotation name.

    Returns
    -------
    AnnotationFunction
    """
    w, h = _default_width_height(which, width, height)
    _border = border

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        if _border:
            grid_py.grid_rect()

    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_empty",
        which=which,
        n=None,
        data_scale=(0.0, 1.0),
        subsettable=True,
        show_name=show_name,
        width=w,
        height=h,
    )


# =========================================================================
# anno_block
# =========================================================================


def anno_block(
    align_to: Optional[Union[Dict[str, Sequence[int]], Sequence[int]]] = None,
    gp: Optional[Dict[str, Any]] = None,
    labels: Optional[Sequence[str]] = None,
    labels_gp: Optional[Dict[str, Any]] = None,
    labels_rot: Optional[float] = None,
    labels_offset: float = 0.5,
    labels_just: str = "center",
    which: str = "column",
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    show_name: bool = False,
    panel_fun: Optional[Callable[..., Any]] = None,
) -> AnnotationFunction:
    """Block annotation for slice grouping.

    Parameters
    ----------
    align_to : dict or array-like, optional
        Group-to-indices mapping or a factor-like integer array.
    gp : dict, optional
        Graphic parameters for block rectangles.
    labels : sequence of str, optional
        Labels for each block.
    labels_gp : dict, optional
        Graphic parameters for label text.
    labels_rot : float, optional
        Label rotation angle.
    labels_offset : float
        Position of labels within block (0-1).
    labels_just : str
        Label justification.
    which : str
        ``"column"`` or ``"row"``.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.
    show_name : bool
        Whether to show the annotation name.
    panel_fun : callable, optional
        Custom drawing function per panel.

    Returns
    -------
    AnnotationFunction
    """
    merged_gp = _resolve_gp(gp)
    labels_merged = _resolve_gp(labels_gp)
    w, h = _default_width_height(which, width, height)

    if labels_rot is None:
        labels_rot = 0.0

    _gp = merged_gp
    _labels = labels
    _labels_gp = labels_merged
    _labels_rot = labels_rot
    _labels_offset = labels_offset
    _labels_just = labels_just
    _which = which
    _panel_fun = panel_fun

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        fill_col = _gp.get("fill", "lightgrey")
        border_col = _gp.get("col", "black")

        # Handle per-slice colours
        if isinstance(fill_col, (list, tuple, np.ndarray)):
            fill_col = fill_col[(k - 1) % len(fill_col)]

        grid_py.grid_rect(gp=_to_gpar(fill=fill_col, col=border_col))

        if _labels is not None and (k - 1) < len(_labels):
            grid_py.grid_text(
                label=_labels[k - 1],
                x=grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc"),
                rot=_labels_rot,
                just=_labels_just,
                gp=_to_gpar(**_labels_gp),
            )

        if _panel_fun is not None:
            _panel_fun(index, k, n_slices)

    var_env: Dict[str, Any] = {"gp": gp, "labels": labels}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_block",
        which=which,
        var_env=var_env,
        n=None,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=show_name,
        width=w,
        height=h,
    )


# =========================================================================
# anno_summary
# =========================================================================


def anno_summary(
    which: str = "column",
    border: bool = True,
    bar_width: float = 0.8,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    extend: float = 0.05,
    outline: bool = True,
    box_width: float = 0.6,
    pch: int = 1,
    size: Any = None,
    gp: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Summary annotation (boxplot for continuous, stacked bars for discrete).

    Faithfully ports R's ``anno_summary``
    (AnnotationFunction-function.R:3447-3579).

    The annotation summarises the data in the parent heatmap across
    k-means (or other) row/column slices.  For continuous data, a
    boxplot is drawn per slice.  For discrete data, stacked proportion
    bars are drawn.

    The parent heatmap's matrix is injected via the ``var_env["_ht_ref"]``
    mechanism — the ``Heatmap`` class populates this before drawing.

    Parameters
    ----------
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border.
    bar_width : float
        Relative bar width (discrete mode).
    axis : bool
        Show data axis.
    axis_param : dict, optional
        Custom axis ticks/labels (``at``, ``labels``).
    ylim : tuple, optional
        Data axis limits.
    extend : float
        Fraction to extend limits.
    outline : bool
        Show outliers in boxplot mode.
    box_width : float
        Relative box width (boxplot mode).
    pch : int
        Point character for outliers.
    size : object, optional
        Outlier point size.
    gp : dict, optional
        Graphic parameters (``fill`` colors per slice).
    width, height : object, optional
        Annotation dimensions.

    Returns
    -------
    AnnotationFunction
    """
    from .grid_extensions import grid_boxplot as _grid_boxplot

    merged_gp = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height)
    _which = which
    _border = border
    _bar_width = bar_width
    _axis = axis
    _axis_param = axis_param
    _ylim = ylim
    _extend = extend
    _outline = outline
    _box_width = box_width
    _pch = pch
    _size = size
    _gp = merged_gp

    # var_env holds the heatmap reference — populated by Heatmap before drawing
    var_env: Dict[str, Any] = {"gp": gp, "_ht_ref": None}

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ht_ref = var_env.get("_ht_ref", None)
        if ht_ref is None:
            # Fallback: no heatmap data available
            grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))
            return

        mat = ht_ref.matrix
        is_discrete = not np.issubdtype(mat.dtype, np.floating)

        # Get the order list for the split dimension
        if _is_row(_which):
            # column annotation → summarise across row slices
            order_list = ht_ref._row_order_list or [np.arange(mat.shape[0])]
        else:
            order_list = ht_ref._column_order_list or [np.arange(mat.shape[1])]

        ng = len(order_list)
        if ng == 0:
            return

        if is_discrete:
            # --- DISCRETE: stacked proportion bars ---
            # (R AnnotationFunction-function.R:3508-3519)
            from collections import Counter

            if _is_row(_which):
                tl = []
                for od in order_list:
                    vals = mat[od, 0] if mat.ndim == 2 else mat[od]
                    tl.append(Counter(vals))
            else:
                tl = []
                for od in order_list:
                    vals = mat[0, od] if mat.ndim == 2 else mat[od]
                    tl.append(Counter(vals))

            # Get all categories and their colors from gp
            all_cats = sorted(set().union(*[c.keys() for c in tl]))

            grid_py.push_viewport(grid_py.Viewport(
                xscale=(0.5, ng + 0.5), yscale=(0, 1)))

            for i in range(ng):
                total = sum(tl[i].values()) or 1
                props = {c: tl[i].get(c, 0) / total for c in all_cats}
                cum = 0
                for ci, cat in enumerate(all_cats):
                    p = props[cat]
                    fill_col = "grey"
                    fill_list = _gp.get("fill", None)
                    if isinstance(fill_list, (list, tuple)):
                        fill_col = fill_list[ci % len(fill_list)]
                    elif isinstance(fill_list, str):
                        fill_col = fill_list
                    grid_py.grid_rect(
                        x=grid_py.Unit(i + 1, "native"),
                        y=grid_py.Unit(cum + p, "native"),
                        width=grid_py.Unit(_bar_width, "native"),
                        height=grid_py.Unit(p, "native"),
                        just="top",
                        gp=grid_py.Gpar(fill=fill_col, col="white"),
                    )
                    cum += p

            grid_py.up_viewport()

        else:
            # --- CONTINUOUS: boxplots ---
            # (R AnnotationFunction-function.R:3521-3554)
            if not _is_row(_which):
                # Column annotation → summarise rows per slice
                # Each vl[i] contains all values in the slice
                vl = [mat[od].ravel().astype(float) for od in order_list]
            else:
                # Row annotation → summarise columns per slice
                vl = [mat[:, od].ravel().astype(float) for od in order_list]

            # Compute data scale
            if _ylim is not None:
                data_scale = list(_ylim)
            else:
                all_vals = np.concatenate([v[np.isfinite(v)] for v in vl
                                           if len(v[np.isfinite(v)]) > 0])
                if len(all_vals) == 0:
                    data_scale = [0.0, 1.0]
                else:
                    data_scale = [float(np.min(all_vals)), float(np.max(all_vals))]
            rng = data_scale[1] - data_scale[0]
            if rng == 0:
                rng = max(abs(data_scale[0]), 1.0)
            data_scale[0] -= _extend * rng
            data_scale[1] += _extend * rng

            grid_py.push_viewport(grid_py.Viewport(
                xscale=(0.5, ng + 0.5), yscale=tuple(data_scale)))

            for i in range(ng):
                v = vl[i]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue

                # Per-slice fill color (R: recycle_gp + subset_gp)
                fill_col = "#CCCCCC"
                fill_list = _gp.get("fill", None)
                if isinstance(fill_list, (list, tuple)):
                    fill_col = fill_list[i % len(fill_list)]
                elif isinstance(fill_list, str):
                    fill_col = fill_list

                bp_grob = _grid_boxplot(
                    value=v, pos=float(i + 1),
                    outline=_outline, box_width=_box_width,
                    gp={"fill": fill_col, "col": "black"},
                    direction="vertical",
                )
                grid_py.grid_draw(bp_grob)

            # Axis
            if _axis:
                ap = _axis_param or {}
                ticks = ap.get("at", None)
                tick_labels = ap.get("labels", None)
                if ticks is None:
                    ticks = grid_py.grid_pretty(data_scale)
                    ticks = [float(t) for t in ticks
                             if data_scale[0] <= t <= data_scale[1]]
                else:
                    ticks = [float(t) for t in ticks]
                if tick_labels is None:
                    tick_labels = [f"{t:g}" for t in ticks]
                grid_py.grid_yaxis(at=ticks, label=tick_labels,
                                   gp=grid_py.Gpar(fontsize=7))

            grid_py.up_viewport()

        if _border:
            grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))

    af = AnnotationFunction(
        fun=_draw,
        fun_name="anno_summary",
        which=which,
        var_env=var_env,
        n=None,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=True,
        width=w,
        height=h,
    )
    # Mark as needing heatmap reference injection
    af._needs_ht_ref = True
    return af


# =========================================================================
# anno_textbox
# =========================================================================


def anno_textbox(
    align_to: Union[Dict[str, Sequence[int]], Sequence[Sequence[int]]],
    text: Union[Dict[str, str], Sequence[str]],
    which: str = "column",
    background_gp: Optional[Dict[str, Any]] = None,
    gp: Optional[Dict[str, Any]] = None,
    max_width: Optional[float] = None,
    word_wrap: bool = False,
    add_new_line: bool = False,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Text-box annotation.

    Parameters
    ----------
    align_to : dict or list
        Mapping from label to indices, or list of index groups.
    text : dict or list
        Text content per group.
    which : str
        ``"column"`` or ``"row"``.
    background_gp : dict, optional
        Graphic parameters for background box.
    gp : dict, optional
        Graphic parameters for text.
    max_width : float, optional
        Maximum width for word wrapping (mm).
    word_wrap : bool
        Enable automatic word wrapping.
    add_new_line : bool
        Insert newlines between text items.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    if isinstance(align_to, dict):
        groups = [np.asarray(v, dtype=int) for v in align_to.values()]
        group_labels = list(align_to.keys())
    else:
        groups = [np.asarray(v, dtype=int) for v in align_to]
        group_labels = [str(i) for i in range(len(groups))]

    if isinstance(text, dict):
        texts = [text[k] for k in group_labels] if isinstance(align_to, dict) else list(text.values())
    else:
        texts = list(text)

    all_indices = np.concatenate(groups) if groups else np.array([], dtype=int)
    n_val = int(all_indices.max()) + 1 if len(all_indices) > 0 else 0

    bg_merged = _resolve_gp(background_gp)
    text_merged = _resolve_gp(gp)
    w, h = _default_width_height(which, width, height, _DEFAULT_SIZE * 2)

    _groups = groups
    _texts = texts
    _which = which
    _bg_gp = bg_merged
    _text_gp = text_merged
    _word_wrap = word_wrap
    _max_width = max_width

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)

        for g_idx, (grp, txt) in enumerate(zip(_groups, _texts)):
            mask = np.isin(index, grp)
            if not np.any(mask):
                continue

            positions = np.where(mask)[0]
            lo = float(positions.min())
            hi = float(positions.max())
            mid_frac = ((lo + hi) / 2 + 0.5) / ni

            display_text = str(txt)
            if _word_wrap and _max_width is not None:
                display_text = textwrap.fill(display_text, width=int(_max_width / 2))

            bg_fill = _bg_gp.get("fill", "lightyellow")
            bg_col = _bg_gp.get("col", "grey")

            if _which == "column":
                lo_frac = lo / ni
                width_frac = (hi - lo + 1) / ni
                grid_py.grid_rect(
                    x=grid_py.Unit(lo_frac, "npc"),
                    y=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(width_frac, "npc"),
                    height=grid_py.Unit(1, "npc"),
                    just=["left", "bottom"],
                    gp=_to_gpar(fill=bg_fill, col=bg_col),
                )
                grid_py.grid_text(
                    label=display_text,
                    x=grid_py.Unit(mid_frac, "npc"),
                    y=grid_py.Unit(0.5, "npc"),
                    gp=_to_gpar(**_text_gp),
                )
            else:
                lo_frac = (ni - hi - 1) / ni
                height_frac = (hi - lo + 1) / ni
                grid_py.grid_rect(
                    x=grid_py.Unit(0, "npc"),
                    y=grid_py.Unit(lo_frac, "npc"),
                    width=grid_py.Unit(1, "npc"),
                    height=grid_py.Unit(height_frac, "npc"),
                    just=["left", "bottom"],
                    gp=_to_gpar(fill=bg_fill, col=bg_col),
                )
                grid_py.grid_text(
                    label=display_text,
                    x=grid_py.Unit(0.5, "npc"),
                    y=grid_py.Unit(lo_frac + height_frac / 2, "npc"),
                    gp=_to_gpar(**_text_gp),
                )

    var_env: Dict[str, Any] = {"groups": groups, "texts": texts}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_textbox",
        which=which,
        var_env=var_env,
        n=n_val,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=False,
        width=w,
        height=h,
    )


# =========================================================================
# anno_customize
# =========================================================================


def anno_customize(
    x: ArrayLike,
    graphics: Optional[Callable[..., Any]] = None,
    which: str = "column",
    border: bool = True,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
) -> AnnotationFunction:
    """Custom graphics annotation.

    Parameters
    ----------
    x : array-like
        Data values passed to the *graphics* callback.
    graphics : callable, optional
        Drawing function with signature ``graphics(x_i, index_i)``
        called once per observation.
    which : str
        ``"column"`` or ``"row"``.
    border : bool
        Draw border around the annotation region.
    width : object, optional
        Annotation width.
    height : object, optional
        Annotation height.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x)
    n = len(x_arr)
    w, h = _default_width_height(which, width, height)

    _x = x_arr
    _which = which
    _graphics = graphics
    _border = border

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]

        if _graphics is not None:
            for pos, val in enumerate(subset):
                if _which == "column":
                    grid_py.push_viewport(grid_py.Viewport(
                        x=grid_py.Unit((pos + 0.5) / ni, "npc"),
                        y=grid_py.Unit(0.5, "npc"),
                        width=grid_py.Unit(1.0 / ni, "npc"),
                        height=grid_py.Unit(1, "npc"),
                    ))
                else:
                    grid_py.push_viewport(grid_py.Viewport(
                        x=grid_py.Unit(0.5, "npc"),
                        y=grid_py.Unit((ni - pos - 0.5) / ni, "npc"),
                        width=grid_py.Unit(1, "npc"),
                        height=grid_py.Unit(1.0 / ni, "npc"),
                    ))
                _graphics(val, index[pos])
                grid_py.up_viewport()

        if _border:
            grid_py.grid_rect(gp=_to_gpar(fill="transparent", col="black"))

    var_env: Dict[str, Any] = {"x": x_arr}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_customize",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=(0.0, 1.0),
        subsettable=True,
        subset_rule={"x": "array"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_numeric
# =========================================================================


def anno_numeric(
    x: ArrayLike,
    rg: Optional[Tuple[float, float]] = None,
    labels_gp: Optional[Dict[str, Any]] = None,
    x_convert: Optional[Callable[[float], str]] = None,
    labels_format: Optional[str] = None,
    bg_gp: Optional[Dict[str, Any]] = None,
    bar_width: Optional[float] = None,
    round_corners: bool = True,
    which: str = "row",
    align_to: str = "left",
    width: Optional[Any] = None,
) -> AnnotationFunction:
    """Numeric bar annotation with labels.

    Parameters
    ----------
    x : array-like
        Numeric values.
    rg : tuple of float, optional
        Data range ``(min, max)``.
    labels_gp : dict, optional
        Graphic parameters for value labels.
    x_convert : callable, optional
        Function to convert numeric values to label strings.
    labels_format : str, optional
        Format string for labels (e.g. ``"{:.2f}"``).
    bg_gp : dict, optional
        Graphic parameters for background bar.
    bar_width : float, optional
        Bar width as a fraction (0-1).
    round_corners : bool
        Whether to draw rounded bar ends.
    which : str
        ``"column"`` or ``"row"`` (defaults to ``"row"``).
    align_to : str
        Bar alignment -- ``"left"`` or ``"right"``.
    width : object, optional
        Annotation width.

    Returns
    -------
    AnnotationFunction
    """
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)

    if rg is None:
        rg = (float(np.nanmin(x_arr)), float(np.nanmax(x_arr)))
    data_lim = rg

    labels_merged = _resolve_gp(labels_gp)
    bg_merged = _resolve_gp(bg_gp) if bg_gp else {"fill": "lightgrey", "col": "transparent"}
    bw = bar_width if bar_width is not None else 0.7

    w, h = _default_width_height(which, width, None, _DEFAULT_SIZE * 2)

    _x = x_arr
    _which = which
    _data_lim = data_lim
    _labels_gp = labels_merged
    _bg_gp = bg_merged
    _bw = bw
    _x_convert = x_convert
    _labels_format = labels_format
    _align_to = align_to

    def _draw(index: np.ndarray, k: int, n_slices: int) -> None:
        ni = len(index)
        subset = _x[index]
        lo, hi = _data_lim
        rng = hi - lo if hi != lo else 1.0

        for i, val in enumerate(subset):
            frac = float(np.clip((val - lo) / rng, 0, 1))

            if _which == "row":
                yp = (ni - i - 0.5) / ni
                # Background bar
                grid_py.grid_rect(
                    x=grid_py.Unit(0, "npc"),
                    y=grid_py.Unit(yp, "npc"),
                    width=grid_py.Unit(1, "npc"),
                    height=grid_py.Unit(_bw / ni, "npc"),
                    just="left",
                    gp=_to_gpar(**_bg_gp),
                )
                # Value bar
                bar_fill = _labels_gp.get("fill", "steelblue")
                if _align_to == "left":
                    grid_py.grid_rect(
                        x=grid_py.Unit(0, "npc"),
                        y=grid_py.Unit(yp, "npc"),
                        width=grid_py.Unit(frac, "npc"),
                        height=grid_py.Unit(_bw / ni, "npc"),
                        just="left",
                        gp=_to_gpar(fill=bar_fill, col="transparent"),
                    )
                else:
                    grid_py.grid_rect(
                        x=grid_py.Unit(1 - frac, "npc"),
                        y=grid_py.Unit(yp, "npc"),
                        width=grid_py.Unit(frac, "npc"),
                        height=grid_py.Unit(_bw / ni, "npc"),
                        just="left",
                        gp=_to_gpar(fill=bar_fill, col="transparent"),
                    )

                # Label
                if _x_convert is not None:
                    lbl = _x_convert(val)
                elif _labels_format is not None:
                    lbl = _labels_format.format(val)
                else:
                    lbl = f"{val:.1f}"

                grid_py.grid_text(
                    label=lbl,
                    x=grid_py.Unit(frac + 0.02, "npc") if _align_to == "left" else grid_py.Unit(1.0 - frac - 0.02, "npc"),
                    y=grid_py.Unit(yp, "npc"),
                    just="left" if _align_to == "left" else "right",
                    gp=_to_gpar(**{k: v for k, v in _labels_gp.items() if k not in ("fill",)}),
                )
            else:
                xp = (i + 0.5) / ni
                grid_py.grid_rect(
                    x=grid_py.Unit(xp, "npc"),
                    y=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(_bw / ni, "npc"),
                    height=grid_py.Unit(1, "npc"),
                    just="bottom",
                    gp=_to_gpar(**_bg_gp),
                )
                bar_fill = _labels_gp.get("fill", "steelblue")
                grid_py.grid_rect(
                    x=grid_py.Unit(xp, "npc"),
                    y=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(_bw / ni, "npc"),
                    height=grid_py.Unit(frac, "npc"),
                    just="bottom",
                    gp=_to_gpar(fill=bar_fill, col="transparent"),
                )

    var_env: Dict[str, Any] = {"x": x_arr}
    return AnnotationFunction(
        fun=_draw,
        fun_name="anno_numeric",
        which=which,
        var_env=var_env,
        n=n,
        data_scale=data_lim,
        subsettable=True,
        subset_rule={"x": "array"},
        show_name=True,
        width=w,
        height=h,
    )


# =========================================================================
# anno_oncoprint_barplot
# =========================================================================


def anno_oncoprint_barplot(
    type: Optional[Union[str, Sequence[str]]] = None,
    which: str = "column",
    border: bool = False,
    bar_width: float = 0.6,
    beside: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    show_fraction: bool = False,
    axis: bool = True,
    axis_param: Optional[Dict[str, Any]] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    # Python-specific: explicit data passing (replaces R's parent.frame)
    _oncoprint_arr: Optional[np.ndarray] = None,
    _oncoprint_col: Optional[Dict[str, str]] = None,
    _oncoprint_types: Optional[List[str]] = None,
) -> AnnotationFunction:
    """OncoPrint-specific stacked bar-plot annotation.

    Port of R ``oncoPrint.R:737-839``.  Shows per-sample (column) or
    per-gene (row) counts of each alteration type as a stacked barplot
    with colours matching the oncoPrint colour mapping.

    Parameters
    ----------
    type : str or list of str, optional
        Mutation types to include.  ``None`` means all types.
    which : str
        ``"column"`` (per-sample) or ``"row"`` (per-gene).
    border : bool
        Draw bar borders.
    bar_width : float
        Relative bar width.
    beside : bool
        Side-by-side instead of stacked.
    ylim : tuple of float, optional
        Data axis limits.
    show_fraction : bool
        Show fractions instead of counts.
    axis : bool
        Show axis.
    axis_param : dict, optional
        Extra axis configuration.
    width, height : object, optional
        Annotation size.
    _oncoprint_arr : numpy.ndarray, optional
        3-D boolean array ``(genes, samples, types)`` from oncoPrint.
    _oncoprint_col : dict, optional
        ``{type_name: colour}`` mapping from oncoPrint.
    _oncoprint_types : list of str, optional
        Alteration type names from oncoPrint.

    Returns
    -------
    AnnotationFunction
    """
    w, h = _default_width_height(which, width, height)
    # R default: 2cm for oncoPrint barplot annotation
    if which == "column" and h is None:
        h = grid_py.Unit(2, "cm")
    if which == "row" and w is None:
        w = grid_py.Unit(2, "cm")

    # If data is available, pre-compute the stacked barplot matrix
    # and delegate to anno_barplot (matching R oncoPrint.R:750-803).
    if _oncoprint_arr is not None and _oncoprint_col is not None and _oncoprint_types is not None:
        arr = _oncoprint_arr
        all_type = list(_oncoprint_types)
        col = dict(_oncoprint_col)

        # Filter to requested types
        if type is not None:
            req = [type] if isinstance(type, str) else list(type)
            all_type = [t for t in all_type if t in req]
        if not all_type:
            all_type = list(_oncoprint_types)

        # Select type indices
        type_indices = [list(_oncoprint_types).index(t)
                        for t in all_type if t in list(_oncoprint_types)]
        arr_sub = arr[:, :, type_indices]

        # Compute stacked barplot values
        # R column_fun: v = apply(arr, c(2,3), sum) → (samples, types)
        # R row_fun:    v = apply(arr, c(1,3), sum) → (genes, types)
        if which == "column":
            v = arr_sub.sum(axis=0)  # (samples, types)
        else:
            v = arr_sub.sum(axis=1)  # (genes, types)
        v = v.astype(float)

        if show_fraction:
            denom = arr.shape[0] if which == "column" else arr.shape[1]
            v = v / max(denom, 1)

        # Build colour vector matching type order
        fill_colors = [col.get(t, "#888888") for t in all_type]

        # R: anno_barplot(v, gp=gpar(fill=col, col=NA), ...)
        if axis_param is None:
            if which == "column":
                axis_param = {"side": "left"}
            else:
                axis_param = {"side": "top", "labels_rot": 0}

        # R oncoPrint.R:821: anno@show_name = FALSE
        anno = anno_barplot(
            x=v,
            which=which,
            bar_width=bar_width,
            beside=beside,
            gp={"fill": fill_colors, "col": "transparent"},
            ylim=ylim,
            axis=axis,
            axis_param=axis_param,
            border=border,
            width=w,
            height=h,
        )
        anno.show_name = False
        return anno

    # Fallback: no data available yet (placeholder)
    _which = which

    def _draw_placeholder(index: np.ndarray, k: int, n_slices: int) -> None:
        grid_py.grid_text(
            label="No OncoPrint data",
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.5, "npc"),
            gp=_to_gpar(col="grey", fontsize=8),
        )

    return AnnotationFunction(
        fun=_draw_placeholder,
        fun_name="anno_oncoprint_barplot",
        which=which,
        var_env={"types": type},
        n=None,
        data_scale=(0.0, 1.0),
        subsettable=False,
        show_name=False,
        width=w,
        height=h,
    )
