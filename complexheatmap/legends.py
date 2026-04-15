"""Legend creation and packing for ComplexHeatmap.

R source correspondence
-----------------------
``R/grid.Legend.R`` -- ``Legend``, ``Legends-class``, ``packLegend``,
``discrete_legend_body``, ``vertical_continuous_legend_body``, and
``horizontal_continuous_legend_body``.

Provides :func:`Legend` (factory that returns a :class:`Legends` object),
:class:`Legends` (a wrapper around a grid_py GTree grob), and
:func:`pack_legend` for arranging multiple legends.

All drawing uses ``grid_py`` (the Python port of R's grid package).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import grid_py

from ._globals import ht_opt

__all__ = [
    "Legend",
    "Legends",
    "pack_legend",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _SizedGTree(grid_py.GTree):
    """A GTree that reports explicit width/height for grobwidth/grobheight units.

    This allows the grid layout system to measure legend grobs and allocate
    the correct amount of space, eliminating hardcoded legend dimensions.
    """

    def __init__(self, *, width_mm: float, height_mm: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._explicit_width_mm = width_mm
        self._explicit_height_mm = height_mm

    def width_details(self) -> Any:
        return grid_py.Unit(self._explicit_width_mm, "mm")

    def height_details(self) -> Any:
        return grid_py.Unit(self._explicit_height_mm, "mm")


_legend_id: int = 0


def _next_legend_name() -> str:
    global _legend_id
    _legend_id += 1
    return f"legend_{_legend_id}"


def _str_width_mm(text: str) -> float:
    """Measure the width of *text* in mm using grid_py's Cairo metrics."""
    try:
        u = grid_py.Unit(1, "strwidth", data=text)
        result = grid_py.convert_width(u, "mm")
        # result may be a Unit or a float
        if hasattr(result, '_values'):
            return float(result._values[0])
        return float(result)
    except (TypeError, ValueError, AttributeError, RuntimeError):
        # Fallback if no renderer is active yet
        return len(text) * 2.5


def _str_height_mm(text: str = "X") -> float:
    """Measure the height of *text* in mm using grid_py's Cairo metrics."""
    try:
        u = grid_py.Unit(1, "strheight", data=text)
        result = grid_py.convert_height(u, "mm")
        if hasattr(result, '_values'):
            return float(result._values[0])
        return float(result)
    except (TypeError, ValueError, AttributeError, RuntimeError):
        return 3.5


def _resolve_gp(
    gp: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge *gp* over *defaults*, returning a new dict."""
    base: Dict[str, Any] = dict(defaults) if defaults else {}
    if gp:
        base.update(gp)
    return base


def _dict_to_gpar(d: Dict[str, Any]) -> grid_py.Gpar:
    """Convert a plain dict of graphical parameters to a grid_py Gpar.

    Translates R-style keys (``col``, ``fill``, ``lwd``, ``lty``,
    ``fontface``) to grid_py-compatible keys.
    """
    mapped: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "col":
            mapped["col"] = v
        elif k == "color":
            mapped["col"] = v
        elif k == "fill":
            mapped["fill"] = v
        elif k == "lwd":
            mapped["lwd"] = v
        elif k == "linewidth":
            mapped["lwd"] = v
        elif k == "lty":
            mapped["lty"] = v
        elif k == "linestyle":
            mapped["lty"] = v
        elif k == "fontsize":
            mapped["fontsize"] = v
        elif k == "fontfamily":
            mapped["fontfamily"] = v
        elif k == "fontweight":
            mapped["fontface"] = v
        elif k == "fontface":
            mapped["fontface"] = v
        else:
            mapped[k] = v
    return grid_py.Gpar(**mapped)


def _resolve_colors(
    at: List[Any],
    legend_gp: Dict[str, Any],
    col_fun: Optional[Callable[..., Any]],
) -> List[str]:
    """Return a list of colours, one per entry in *at*."""
    n = len(at)
    # Try legend_gp first
    for key in ("fill", "facecolor", "color", "col"):
        val = legend_gp.get(key)
        if val is not None:
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)[:n]
            elif isinstance(val, str):
                return [val] * n

    # Fallback to col_fun
    if col_fun is not None:
        result: List[str] = []
        for v in at:
            try:
                c = col_fun(float(v))
                result.append(c if isinstance(c, str) else str(c))
            except (TypeError, ValueError):
                result.append("#CCCCCC")
        return result

    return ["#CCCCCC"] * n


# ---------------------------------------------------------------------------
# Legends class (wrapper around a GTree grob)
# ---------------------------------------------------------------------------

class Legends:
    """Container for a legend grob, mirroring R's ``Legends`` S4 class.

    Parameters
    ----------
    grob : grid_py.GTree or grid_py.Grob, optional
        The grid grob representing this legend.
    name : str, optional
        Legend name (used internally).
    type : str
        One of ``"single_legend"``, ``"single_legend_no_title"``,
        or ``"pack_legend"``.
    n : int
        Number of individual legends contained.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    """

    def __init__(
        self,
        grob: Optional[Any] = None,
        name: Optional[str] = None,
        type: str = "single_legend",
        n: int = 1,
        direction: str = "vertical",
    ) -> None:
        self.grob = grob
        self.name = name
        self.type = type
        self.n = n
        self.direction = direction

    def draw(self, **kwargs: Any) -> None:
        """Draw the legend by calling ``grid_py.grid_draw``.

        Parameters
        ----------
        **kwargs
            Passed through to ``grid_py.grid_draw``.
        """
        if self.grob is not None:
            grid_py.grid_draw(self.grob)

    def _repr_png_(self) -> bytes:
        """Jupyter notebook PNG display of the legend."""
        if self.grob is None:
            return b""
        # Measure legend size
        w_mm = getattr(self.grob, '_explicit_width_mm', 30.0)
        h_mm = getattr(self.grob, '_explicit_height_mm', 40.0)
        # Add margin
        w_in = (w_mm + 10) / 25.4
        h_in = (h_mm + 10) / 25.4
        w_in = max(w_in, 1.5)
        h_in = max(h_in, 1.5)

        grid_py.grid_newpage(width=w_in, height=h_in, dpi=150)
        grid_py.push_viewport(grid_py.Viewport(
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.5, "npc"),
            width=grid_py.Unit(w_mm, "mm"),
            height=grid_py.Unit(h_mm, "mm"),
            name="legend_preview",
        ))
        self.draw()
        grid_py.up_viewport()
        renderer = grid_py.get_state().get_renderer()
        return renderer.to_png_bytes()

    def __repr__(self) -> str:
        if self.type == "single_legend":
            return "A single legend"
        elif self.type == "single_legend_no_title":
            return "A single legend without title"
        else:
            return f"A pack of {self.n} legends"


# ---------------------------------------------------------------------------
# Legend factory function
# ---------------------------------------------------------------------------

def Legend(
    at: Optional[List[Any]] = None,
    labels: Optional[List[str]] = None,
    col_fun: Optional[Callable[..., Any]] = None,
    name: Optional[str] = None,
    grob: Optional[Any] = None,
    break_dist: Optional[Union[float, List[float]]] = None,
    nrow: Optional[int] = None,
    ncol: int = 1,
    by_row: bool = False,
    grid_height: float = 4.0,
    grid_width: float = 4.0,
    tick_length: float = 0.8,
    gap: float = 2.0,
    column_gap: Optional[float] = None,
    row_gap: float = 0.0,
    labels_gp: Optional[Dict[str, Any]] = None,
    labels_rot: float = 0,
    border: Optional[Union[bool, str]] = None,
    background: str = "#EEEEEE",
    type: str = "grid",
    graphics: Optional[List[Callable[..., Any]]] = None,
    legend_gp: Optional[Dict[str, Any]] = None,
    pch: int = 16,
    size: float = 2.0,
    legend_height: Optional[float] = None,
    legend_width: Optional[float] = None,
    direction: str = "vertical",
    title: str = "",
    title_gp: Optional[Dict[str, Any]] = None,
    title_position: str = "topleft",
    title_gap: float = 2.0,
) -> Legends:
    """Create a single legend, returned as a :class:`Legends` object.

    This is the Python equivalent of R's ``Legend()`` function from
    ``grid.Legend.R``.  It builds a ``grid_py.GTree`` grob containing
    colored rectangles / color bar, text labels, and an optional title.

    Parameters
    ----------
    at : list, optional
        Break values for the legend entries.  For discrete legends these
        are the data values; for continuous legends they are tick positions.
    labels : list of str, optional
        Text labels.  Defaults to ``str(v)`` for each value in *at*.
    col_fun : callable, optional
        Color mapping function (e.g. from ``color_ramp2``).  When provided,
        a continuous color-bar legend is created.
    name : str, optional
        Internal legend name.
    grob : grid_py grob, optional
        Pre-built legend body grob.
    nrow : int, optional
        Number of rows to arrange discrete legend grids.
    ncol : int
        Number of columns for discrete legend grids.
    by_row : bool
        Arrange legend grids by row (True) or by column (False).
    grid_height : float
        Height of each legend grid cell in mm.
    grid_width : float
        Width of each legend grid cell in mm.
    tick_length : float
        Tick length in mm for continuous legends.
    gap : float
        Gap between columns in mm (alias for column_gap).
    column_gap : float, optional
        Gap between columns in mm.
    row_gap : float
        Gap between rows in mm.
    labels_gp : dict, optional
        Graphical parameters for label text.
    labels_rot : float
        Rotation of labels in degrees.
    border : bool or str, optional
        Border color for legend grids.  ``True`` uses ``"black"``.
    background : str
        Background color for legend grids (used with points/lines type).
    type : str
        ``"grid"``, ``"points"``, ``"lines"``, or ``"boxplot"``.
    graphics : list of callable, optional
        Custom drawing functions, one per legend entry.
    legend_gp : dict, optional
        Graphical parameters for legend grids (use ``fill`` for colors).
    pch : int
        Point character type when *type* is ``"points"``.
    size : float
        Point size in mm when *type* is ``"points"``.
    legend_height : float, optional
        Total height of a continuous color bar in mm.
    legend_width : float, optional
        Total width of a continuous color bar in mm.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    title : str
        Legend title text.
    title_gp : dict, optional
        Graphical parameters for the title.
    title_position : str
        Position of the title: ``"topleft"``, ``"topcenter"``,
        ``"leftcenter"``, ``"lefttop"``, ``"leftcenter-rot"``,
        ``"lefttop-rot"``.
    title_gap : float
        Gap between title and legend body in mm.

    Returns
    -------
    Legends
        A :class:`Legends` object wrapping the legend grob.
    """
    if name is None:
        name = _next_legend_name()

    if column_gap is None:
        column_gap = gap

    # Resolve gp dicts
    _labels_gp = _resolve_gp(labels_gp, ht_opt("legend_labels_gp"))
    if "fontsize" not in _labels_gp:
        _labels_gp["fontsize"] = 10
    _title_gp = _resolve_gp(title_gp, ht_opt("legend_title_gp"))
    _legend_gp = legend_gp if legend_gp is not None else {}

    # Default labels from at
    if at is None and labels is not None:
        at = list(range(1, len(labels) + 1))
    if at is None:
        at = []
    if labels is None:
        labels = [str(v) for v in at]

    # ----- Pre-built grob body -----
    if grob is not None:
        legend_body = grob
    elif col_fun is None:
        # ----- Discrete legend body -----
        if border is None:
            border = "white"
        legend_body = _discrete_legend_body(
            at=at, labels=labels, nrow=nrow, ncol=ncol,
            by_row=by_row, grid_height=grid_height, grid_width=grid_width,
            gap=gap, column_gap=column_gap, row_gap=row_gap,
            labels_gp=_labels_gp, labels_rot=labels_rot,
            border=border, background=background,
            type=type, graphics=graphics, legend_gp=_legend_gp,
            pch=pch, size=size,
        )
    else:
        # ----- Continuous legend body -----
        if at is not None and len(at) == 0:
            # Auto-detect breaks from col_fun
            breaks = getattr(col_fun, "breaks", None)
            if breaks is None:
                raise ValueError(
                    "You should provide 'at' for the color mapping function."
                )
            at_arr = grid_py.grid_pretty(
                [float(breaks[0]), float(breaks[-1])]
            ).tolist()
            at = at_arr
            labels = [str(v) for v in at]

        if direction == "vertical":
            legend_body = _vertical_continuous_legend_body(
                at=at, labels=labels, col_fun=col_fun,
                break_dist=break_dist, grid_height=grid_height,
                grid_width=grid_width, tick_length=tick_length,
                legend_height=legend_height, labels_gp=_labels_gp,
                border=border, legend_gp=_legend_gp,
            )
        else:
            legend_body = _horizontal_continuous_legend_body(
                at=at, labels=labels, col_fun=col_fun,
                break_dist=break_dist, grid_height=grid_height,
                grid_width=grid_width, tick_length=tick_length,
                legend_width=legend_width, labels_gp=_labels_gp,
                labels_rot=labels_rot, border=border, legend_gp=_legend_gp,
            )

    # ----- Handle no title -----
    if title is None or (isinstance(title, str) and title.strip() == ""):
        obj = Legends(
            grob=legend_body,
            name=name,
            type="single_legend_no_title",
            n=1,
            direction=direction,
        )
        return obj

    # ----- Assemble title + body -----
    # Port of R grid.Legend.R:238-264 (title_position="topleft"):
    #   total_height = title_height + title_padding + legend_height
    #   title at y=1npc (top), just=c("left","top")
    #   body  at y=0npc (bottom), just=c("left","bottom")
    #   Both in ONE viewport of size total_width × total_height
    title_gpar = _dict_to_gpar(_title_gp)
    title_h_mm = _str_height_mm(title)
    title_padding = title_gap  # R: ht_opt$LEGEND_TITLE_PADDING, default ~2.5mm

    body_w = getattr(legend_body, '_explicit_width_mm', 20.0)
    body_h = getattr(legend_body, '_explicit_height_mm', 30.0)

    title_w = _str_width_mm(title)
    total_w = max(body_w, title_w)
    total_h = title_h_mm + title_padding + body_h

    # Title: at the top of the combined viewport
    # R: textGrob(title, x=title_x, y=unit(1,"npc"), just=c("left","top"))
    if title_position in ("topleft",):
        title_x = grid_py.Unit(0, "npc")
        title_just = ["left", "top"]
    else:  # topcenter
        title_x = grid_py.Unit(0.5, "npc")
        title_just = "top"

    title_grob = grid_py.text_grob(
        title,
        x=title_x,
        y=grid_py.Unit(total_h, "mm"),
        just=title_just,
        gp=title_gpar,
        name="legend_title",
    )

    # Body: at the bottom of the combined viewport
    # R: edit_vp_in_legend_grob(legend_body, x=0, y=0, just=c(0,0))
    # legend_body's internal coordinates start from (0,0) at bottom-left,
    # so we just place it at y=0mm in the combined viewport.
    # No extra sub-viewport needed — body grob's coordinates are in mm
    # relative to its own origin.

    # Create a wrapper viewport for the legend body that positions it
    # at the bottom of the combined space
    body_vp = grid_py.Viewport(
        x=grid_py.Unit(0, "npc"),
        y=grid_py.Unit(0, "npc"),
        width=grid_py.Unit(body_w, "mm"),
        height=grid_py.Unit(body_h, "mm"),
        just=["left", "bottom"],
        name=f"legend_body_vp_{name}",
    )
    body_tree = grid_py.GTree(
        children=grid_py.GList(legend_body),
        name=f"legend_body_tree_{name}",
        vp=body_vp,
    )

    children = grid_py.GList()
    children.append(title_grob)
    children.append(body_tree)

    legend_vp = grid_py.Viewport(
        width=grid_py.Unit(total_w, "mm"),
        height=grid_py.Unit(total_h, "mm"),
        name=f"legend_vp_{name}",
    )

    gf = _SizedGTree(
        children=children,
        name=name,
        vp=legend_vp,
        width_mm=total_w,
        height_mm=total_h,
    )

    obj = Legends(
        grob=gf,
        name=name,
        type="single_legend",
        n=1,
        direction=direction,
    )
    return obj


# ---------------------------------------------------------------------------
# Discrete legend body builder
# ---------------------------------------------------------------------------

def _discrete_legend_body(
    at: List[Any],
    labels: List[str],
    nrow: Optional[int],
    ncol: int,
    by_row: bool,
    grid_height: float,
    grid_width: float,
    gap: float,
    column_gap: float,
    row_gap: float,
    labels_gp: Dict[str, Any],
    labels_rot: float,
    border: Union[bool, str],
    background: str,
    type: str,
    graphics: Optional[List[Callable[..., Any]]],
    legend_gp: Dict[str, Any],
    pch: int,
    size: float,
) -> grid_py.GTree:
    """Build the grid grob for a discrete legend body.

    Parameters
    ----------
    at : list
        Break values.
    labels : list of str
        Labels for each entry.
    nrow, ncol : int
        Grid layout dimensions.
    by_row : bool
        Fill order.
    grid_height, grid_width : float
        Grid cell size in mm.
    gap, column_gap, row_gap : float
        Gaps in mm.
    labels_gp : dict
        Label graphical parameters.
    labels_rot : float
        Label rotation in degrees.
    border : bool or str
        Border color.
    background : str
        Background color for points/lines type.
    type : str
        ``"grid"``, ``"points"``, or ``"lines"``.
    graphics : list of callable, optional
        Custom drawing functions.
    legend_gp : dict
        Grid cell graphical parameters.
    pch : int
        Point character.
    size : float
        Point size in mm.

    Returns
    -------
    grid_py.GTree
    """
    n_labels = len(labels)
    if n_labels == 0:
        return grid_py.GTree(name="empty_legend_body")

    # Resolve layout
    if nrow is None:
        actual_nrow = max(1, (n_labels + ncol - 1) // ncol)
    else:
        actual_nrow = nrow
        ncol = max(1, (n_labels + actual_nrow - 1) // actual_nrow)

    if n_labels == 1:
        actual_nrow = 1
        ncol = 1
    ncol = min(ncol, n_labels)

    colors = _resolve_colors(at, legend_gp, col_fun=None)

    # Build index matrix (nrow x ncol)
    index_mat = np.full((actual_nrow, ncol), -1, dtype=int)
    for idx in range(n_labels):
        if by_row:
            r, c = divmod(idx, ncol)
        else:
            c, r = divmod(idx, actual_nrow)
        if r < actual_nrow and c < ncol:
            index_mat[r, c] = idx

    # Determine border color
    if border is True:
        border_col = "black"
    elif isinstance(border, str):
        border_col = border
    else:
        border_col = None

    labels_padding_left = 1.0  # mm
    labels_gpar = _dict_to_gpar(labels_gp)

    # Build grob children
    children = grid_py.GList()

    # Positions: we lay out from top-left
    y_offset = 0.0  # mm from top
    for r in range(actual_nrow):
        x_offset = 0.0  # mm from left
        for c in range(ncol):
            idx = index_mat[r, c]
            if idx < 0:
                continue

            color = colors[idx] if idx < len(colors) else background

            # Cell centre position (mm units, y measured from top)
            cell_cx = grid_py.Unit(x_offset + grid_width / 2, "mm")
            cell_cy = (
                grid_py.Unit(1, "npc")
                - grid_py.Unit(y_offset + grid_height / 2, "mm")
            )
            cell_w = grid_py.Unit(grid_width, "mm")
            cell_h = grid_py.Unit(grid_height, "mm")

            # --- Custom graphics functions (R grid.Legend.R:587-598) ---
            if graphics is not None and idx < len(graphics):
                _fn = graphics[idx]
                # Capture the function's drawing output as a GTree,
                # exactly like R's grid.grabExpr(fl[[k]](x, y, w, h)).
                _cx, _cy, _cw, _ch = cell_cx, cell_cy, cell_w, cell_h
                grabbed = grid_py.grid_grab_expr(
                    lambda _f=_fn, _x=_cx, _y=_cy, _w=_cw, _h=_ch: _f(
                        _x, _y, _w, _h
                    ),
                    width=grid_width / 25.4,   # mm → inches
                    height=grid_height / 25.4,
                )
                if grabbed is not None:
                    grabbed.name = f"legend_graphic_{idx}"
                    children.append(grabbed)

            # --- Standard grid/points/lines types ---
            elif type == "grid":
                cell_gp_kw: Dict[str, Any] = {"fill": color}
                if border_col:
                    cell_gp_kw["col"] = border_col
                    cell_gp_kw["lwd"] = 0.5
                else:
                    cell_gp_kw["col"] = color  # no visible border

                cell_grob = grid_py.rect_grob(
                    x=cell_cx, y=cell_cy,
                    width=cell_w, height=cell_h,
                    gp=grid_py.Gpar(**cell_gp_kw),
                    name=f"legend_grid_{idx}",
                )
                children.append(cell_grob)

            elif type == "points":
                # Background rect
                bg_grob = grid_py.rect_grob(
                    x=cell_cx, y=cell_cy,
                    width=cell_w, height=cell_h,
                    gp=grid_py.Gpar(fill=background, col=background),
                    name=f"legend_bg_{idx}",
                )
                children.append(bg_grob)
                # Point
                pt_grob = grid_py.points_grob(
                    x=cell_cx, y=cell_cy,
                    pch=pch,
                    size=grid_py.Unit(size, "mm"),
                    gp=grid_py.Gpar(col=color, fill=color),
                    name=f"legend_point_{idx}",
                )
                children.append(pt_grob)

            elif type == "lines":
                # Background rect
                bg_grob = grid_py.rect_grob(
                    x=cell_cx, y=cell_cy,
                    width=cell_w, height=cell_h,
                    gp=grid_py.Gpar(fill=background, col=background),
                    name=f"legend_bg_{idx}",
                )
                children.append(bg_grob)
                # Line segment
                line_gp_kw: Dict[str, Any] = {"col": color}
                lw = legend_gp.get("lwd", legend_gp.get("linewidth", 1.5))
                line_gp_kw["lwd"] = lw
                line_grob = grid_py.segments_grob(
                    x0=grid_py.Unit(x_offset, "mm"),
                    y0=cell_cy,
                    x1=grid_py.Unit(x_offset + grid_width, "mm"),
                    y1=cell_cy,
                    gp=grid_py.Gpar(**line_gp_kw),
                    name=f"legend_line_{idx}",
                )
                children.append(line_grob)

            # Label text
            label = labels[idx] if idx < len(labels) else ""
            label_x = x_offset + grid_width + labels_padding_left
            label_y = y_offset + grid_height / 2

            label_grob = grid_py.text_grob(
                label,
                x=grid_py.Unit(label_x, "mm"),
                y=grid_py.Unit(1, "npc") - grid_py.Unit(label_y, "mm"),
                just="left",
                rot=labels_rot,
                gp=labels_gpar,
                name=f"legend_label_{idx}",
            )
            children.append(label_grob)

            x_offset += grid_width + labels_padding_left + 12.0 + column_gap

        y_offset += grid_height + row_gap

    # Compute total size from layout parameters
    body_w = ncol * (grid_width + labels_padding_left + 12.0 + column_gap)
    text_half_h = _str_height_mm() / 2  # bottom label descender padding
    body_h = actual_nrow * (grid_height + row_gap) + text_half_h
    return _SizedGTree(
        children=children, name="discrete_legend_body",
        width_mm=body_w, height_mm=body_h,
    )


# ---------------------------------------------------------------------------
# Continuous legend body builders
# ---------------------------------------------------------------------------

def _vertical_continuous_legend_body(
    at: List[Any],
    labels: List[str],
    col_fun: Callable[..., Any],
    break_dist: Optional[Union[float, List[float]]],
    grid_height: float,
    grid_width: float,
    tick_length: float,
    legend_height: Optional[float],
    labels_gp: Dict[str, Any],
    border: Optional[Union[bool, str]],
    legend_gp: Dict[str, Any],
) -> grid_py.GTree:
    """Build a vertical continuous color-bar legend body.

    Parameters
    ----------
    at : list
        Tick positions.
    labels : list of str
        Tick labels.
    col_fun : callable
        Color mapping function.
    break_dist : float or list of float, optional
        Zooming factor for inter-break distances.
    grid_height, grid_width : float
        Dimensions in mm.
    tick_length : float
        Tick length in mm.
    legend_height : float, optional
        Total bar height in mm.
    labels_gp : dict
        Label graphical parameters.
    border : bool or str, optional
        Border color.
    legend_gp : dict
        Bar graphical parameters.

    Returns
    -------
    grid_py.GTree
    """
    if legend_height is None:
        legend_height = 40.0

    at_vals = [float(v) for v in at]
    if not at_vals:
        return grid_py.GTree(name="empty_colorbar")

    vmin = min(at_vals)
    vmax = max(at_vals)
    if vmax == vmin:
        vmax = vmin + 1.0

    # Build gradient as many thin raster strips
    n_steps = 256
    step_values = np.linspace(vmin, vmax, n_steps)
    gradient_colors_raw = col_fun(step_values)
    if isinstance(gradient_colors_raw, str):
        gradient_colors = [gradient_colors_raw] * n_steps
    else:
        gradient_colors = list(gradient_colors_raw)

    children = grid_py.GList()

    # Pad top and bottom to leave room for tick label text that extends
    # beyond the color bar edges (half a text line on each side).
    # R's legend body grob accounts for this via grobHeight which
    # includes the label text extent.
    _text_h = _str_height_mm()
    _y_pad_bottom = _text_h / 2  # space for bottom label descender
    _y_pad_top = _text_h / 2     # space for top label ascender

    # Draw gradient as stacked thin rectangles (bottom = vmin, top = vmax)
    step_h = legend_height / n_steps
    for i, color in enumerate(gradient_colors):
        y_pos = i * step_h + _y_pad_bottom
        r = grid_py.rect_grob(
            x=grid_py.Unit(0, "mm"),
            y=grid_py.Unit(y_pos, "mm"),
            width=grid_py.Unit(grid_width, "mm"),
            height=grid_py.Unit(step_h + 0.1, "mm"),
            just=["left", "bottom"],
            gp=grid_py.Gpar(fill=color, col=color, lwd=0),
            name=f"colorbar_strip_{i}",
        )
        children.append(r)

    # Border around the entire bar
    if border is not None and border is not False:
        border_col = border if isinstance(border, str) else "black"
        border_grob = grid_py.rect_grob(
            x=grid_py.Unit(0, "mm"),
            y=grid_py.Unit(_y_pad_bottom, "mm"),
            width=grid_py.Unit(grid_width, "mm"),
            height=grid_py.Unit(legend_height, "mm"),
            just=["left", "bottom"],
            gp=grid_py.Gpar(col=border_col, fill="transparent", lwd=0.8),
            name="colorbar_border",
        )
        children.append(border_grob)

    # Tick marks and labels on the right side
    labels_gpar = _dict_to_gpar(labels_gp)
    for v, label in zip(at_vals, labels):
        frac = (v - vmin) / (vmax - vmin)
        y_pos = frac * legend_height + _y_pad_bottom

        # Tick mark
        tick = grid_py.segments_grob(
            x0=grid_py.Unit(grid_width, "mm"),
            y0=grid_py.Unit(y_pos, "mm"),
            x1=grid_py.Unit(grid_width + tick_length, "mm"),
            y1=grid_py.Unit(y_pos, "mm"),
            gp=grid_py.Gpar(col="black", lwd=0.5),
            name=f"colorbar_tick_{label}",
        )
        children.append(tick)

        # Label
        lbl = grid_py.text_grob(
            str(label),
            x=grid_py.Unit(grid_width + tick_length + 1.0, "mm"),
            y=grid_py.Unit(y_pos, "mm"),
            just="left",
            gp=labels_gpar,
            name=f"colorbar_label_{label}",
        )
        children.append(lbl)

    # Width = color bar + tick + gap + max label width
    label_w = max((_str_width_mm(str(l)) for l in labels), default=8.0)
    body_w = grid_width + tick_length + 1.0 + label_w
    # Height = bottom pad + color bar + top pad
    body_h = _y_pad_bottom + legend_height + _y_pad_top
    return _SizedGTree(
        children=children, name="continuous_legend_body",
        width_mm=body_w, height_mm=body_h,
    )


def _horizontal_continuous_legend_body(
    at: List[Any],
    labels: List[str],
    col_fun: Callable[..., Any],
    break_dist: Optional[Union[float, List[float]]],
    grid_height: float,
    grid_width: float,
    tick_length: float,
    legend_width: Optional[float],
    labels_gp: Dict[str, Any],
    labels_rot: float,
    border: Optional[Union[bool, str]],
    legend_gp: Dict[str, Any],
) -> grid_py.GTree:
    """Build a horizontal continuous color-bar legend body.

    Parameters
    ----------
    at : list
        Tick positions.
    labels : list of str
        Tick labels.
    col_fun : callable
        Color mapping function.
    break_dist : float or list of float, optional
        Zooming factor.
    grid_height, grid_width : float
        Dimensions in mm.
    tick_length : float
        Tick length in mm.
    legend_width : float, optional
        Total bar width in mm.
    labels_gp : dict
        Label graphical parameters.
    labels_rot : float
        Label rotation.
    border : bool or str, optional
        Border color.
    legend_gp : dict
        Bar graphical parameters.

    Returns
    -------
    grid_py.GTree
    """
    if legend_width is None:
        legend_width = 40.0

    at_vals = [float(v) for v in at]
    if not at_vals:
        return grid_py.GTree(name="empty_colorbar")

    vmin = min(at_vals)
    vmax = max(at_vals)
    if vmax == vmin:
        vmax = vmin + 1.0

    n_steps = 256
    step_values = np.linspace(vmin, vmax, n_steps)
    gradient_colors_raw = col_fun(step_values)
    if isinstance(gradient_colors_raw, str):
        gradient_colors = [gradient_colors_raw] * n_steps
    else:
        gradient_colors = list(gradient_colors_raw)

    children = grid_py.GList()

    step_w = legend_width / n_steps
    for i, color in enumerate(gradient_colors):
        x_pos = i * step_w
        r = grid_py.rect_grob(
            x=grid_py.Unit(x_pos, "mm"),
            y=grid_py.Unit(0, "mm"),
            width=grid_py.Unit(step_w + 0.05, "mm"),
            height=grid_py.Unit(grid_height, "mm"),
            just=["left", "bottom"],
            gp=grid_py.Gpar(fill=color, col=color, lwd=0),
            name=f"colorbar_strip_{i}",
        )
        children.append(r)

    # Border
    if border is not None and border is not False:
        border_col = border if isinstance(border, str) else "black"
        border_grob = grid_py.rect_grob(
            x=grid_py.Unit(0, "mm"),
            y=grid_py.Unit(0, "mm"),
            width=grid_py.Unit(legend_width, "mm"),
            height=grid_py.Unit(grid_height, "mm"),
            just=["left", "bottom"],
            gp=grid_py.Gpar(col=border_col, fill="transparent", lwd=0.8),
            name="colorbar_border",
        )
        children.append(border_grob)

    # Ticks and labels below
    labels_gpar = _dict_to_gpar(labels_gp)
    for v, label in zip(at_vals, labels):
        frac = (v - vmin) / (vmax - vmin)
        x_pos = frac * legend_width

        tick = grid_py.segments_grob(
            x0=grid_py.Unit(x_pos, "mm"),
            y0=grid_py.Unit(0, "mm"),
            x1=grid_py.Unit(x_pos, "mm"),
            y1=grid_py.Unit(-tick_length, "mm"),
            gp=grid_py.Gpar(col="black", lwd=0.5),
            name=f"colorbar_tick_{label}",
        )
        children.append(tick)

        lbl = grid_py.text_grob(
            str(label),
            x=grid_py.Unit(x_pos, "mm"),
            y=grid_py.Unit(-tick_length - 1.0, "mm"),
            just="top",
            rot=labels_rot,
            gp=labels_gpar,
            name=f"colorbar_label_{label}",
        )
        children.append(lbl)

    label_h = _str_height_mm()
    body_w = legend_width
    body_h = grid_height + tick_length + 1.0 + label_h
    return _SizedGTree(
        children=children, name="continuous_legend_body",
        width_mm=body_w, height_mm=body_h,
    )


# ---------------------------------------------------------------------------
# pack_legend
# ---------------------------------------------------------------------------

def pack_legend(
    *legends: Union[Legends, Any],
    direction: str = "vertical",
    max_height: Optional[float] = None,
    max_width: Optional[float] = None,
    column_gap: float = 2.0,
    row_gap: float = 2.0,
) -> Legends:
    """Pack multiple legends into a single :class:`Legends` object.

    Corresponds to R's ``packLegend()``.

    Parameters
    ----------
    *legends : Legends
        Legend objects to pack.
    direction : str
        ``"vertical"`` or ``"horizontal"``.
    max_height : float, optional
        Maximum height in mm before wrapping to a new column (vertical).
    max_width : float, optional
        Maximum width in mm before wrapping to a new row (horizontal).
    column_gap : float
        Gap between columns in mm.
    row_gap : float
        Gap between rows in mm.

    Returns
    -------
    Legends
        A combined :class:`Legends` object.
    """
    # Flatten inputs
    flat_grobs: List[Any] = []
    for item in legends:
        if isinstance(item, Legends):
            if item.grob is not None:
                flat_grobs.append(item.grob)
        elif grid_py.is_grob(item):
            flat_grobs.append(item)
        else:
            raise TypeError(
                f"Expected Legends or grob, got {type(item)!r}"
            )

    if not flat_grobs:
        return Legends(
            grob=grid_py.GTree(name="empty_packed_legend"),
            type="pack_legend",
            n=0,
            direction=direction,
        )

    # Compute sizes and wrap each grob in a positioned viewport
    gap_mm = row_gap if direction == "vertical" else column_gap
    sizes: List[Tuple[float, float]] = []
    for g in flat_grobs:
        gw = getattr(g, '_explicit_width_mm', 20.0)
        gh = getattr(g, '_explicit_height_mm', 20.0)
        sizes.append((gw, gh))

    total_w = 0.0
    total_h = 0.0
    if direction == "vertical":
        total_w = max((s[0] for s in sizes), default=0)
        total_h = sum(s[1] for s in sizes) + gap_mm * max(len(sizes) - 1, 0)
    else:
        total_w = sum(s[0] for s in sizes) + gap_mm * max(len(sizes) - 1, 0)
        total_h = max((s[1] for s in sizes), default=0)

    # Wrap each legend in a viewport with correct offset
    children = grid_py.GList()
    offset = 0.0  # running offset in mm
    for i, g in enumerate(flat_grobs):
        gw, gh = sizes[i]
        if direction == "vertical":
            vp = grid_py.Viewport(
                x=grid_py.Unit(0, "mm"),
                y=grid_py.Unit(total_h - offset, "mm"),
                width=grid_py.Unit(gw, "mm"),
                height=grid_py.Unit(gh, "mm"),
                just=["left", "top"],
                name=f"packed_legend_{i}",
            )
            offset += gh + gap_mm
        else:
            vp = grid_py.Viewport(
                x=grid_py.Unit(offset, "mm"),
                y=grid_py.Unit(total_h, "mm"),
                width=grid_py.Unit(gw, "mm"),
                height=grid_py.Unit(gh, "mm"),
                just=["left", "top"],
                name=f"packed_legend_{i}",
            )
            offset += gw + gap_mm

        wrapped = grid_py.GTree(
            children=grid_py.GList(g),
            name=f"packed_legend_wrap_{i}",
            vp=vp,
        )
        children.append(wrapped)

    packed_grob = _SizedGTree(
        children=children,
        name="packed_legends",
        width_mm=total_w,
        height_mm=total_h,
    )

    obj = Legends(
        grob=packed_grob,
        type="pack_legend",
        n=len(flat_grobs),
        direction=direction,
    )
    # Attach layout metadata
    obj._max_height = max_height  # type: ignore[attr-defined]
    obj._max_width = max_width  # type: ignore[attr-defined]
    obj._column_gap = column_gap  # type: ignore[attr-defined]
    obj._row_gap = row_gap  # type: ignore[attr-defined]

    return obj
