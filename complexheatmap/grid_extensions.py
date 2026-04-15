"""Grid graphics extensions (textbox, boxplot grobs, gt_render, annotation axis).

R source correspondence
-----------------------
``R/grid.Legend.R`` (annotation_axis_grob), various ComplexHeatmap helper
functions for cell-level decorations using R's ``grid`` package.

All drawing uses ``grid_py`` (the Python port of R's grid).
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import grid_py

__all__ = [
    "grid_boxplot",
    "grid_textbox",
    "textbox_grob",
    "gt_render",
    "annotation_axis_grob",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_gp(gp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise a *gp* (graphical parameters) dict.

    The ``gp`` dict follows the R naming convention:

    * ``col`` / ``color`` -- line / text colour
    * ``fill`` -- fill colour
    * ``lwd`` / ``linewidth`` -- line width
    * ``lty`` / ``linestyle`` -- line style
    * ``fontsize`` -- font size in points
    * ``fontfamily`` -- font family name
    * ``fontface`` / ``fontweight`` -- font weight

    Returns a dict with grid_py Gpar-compatible keys.
    """
    if gp is None:
        return {}

    out: Dict[str, Any] = {}
    for k, v in gp.items():
        if k == "color":
            out["col"] = v
        elif k == "linewidth":
            out["lwd"] = v
        elif k == "linestyle":
            out["lty"] = v
        elif k == "fontweight":
            out["fontface"] = v
        elif k == "facecolor":
            out["fill"] = v
        else:
            out[k] = v
    return out


def _to_gpar(d: Dict[str, Any]) -> grid_py.Gpar:
    """Convert a normalised dict to a grid_py.Gpar object."""
    return grid_py.Gpar(**d)


# ---------------------------------------------------------------------------
# grid_boxplot
# ---------------------------------------------------------------------------

def grid_boxplot(
    value: np.ndarray,
    pos: float = 0.5,
    outline: bool = True,
    box_width: float = 0.6,
    gp: Optional[Dict[str, Any]] = None,
    direction: str = "vertical",
) -> grid_py.GTree:
    """Create a boxplot grob (analogous to R's ``grid.boxplot``).

    Parameters
    ----------
    value : numpy.ndarray
        1-D array of numeric values.
    pos : float
        Position along the non-value axis in npc (0-1).
    outline : bool
        Whether to include outlier points.
    box_width : float
        Width of the box in npc units.
    gp : dict, optional
        Graphical parameters (``col``, ``fill``, ``lwd``, etc.).
    direction : str
        ``"vertical"`` or ``"horizontal"``.

    Returns
    -------
    grid_py.GTree
        A GTree grob containing the boxplot elements.
    """
    value = np.asarray(value, dtype=float)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return grid_py.GTree(name="empty_boxplot")

    mgp = _resolve_gp(gp)
    line_color = mgp.get("col", "black")
    fill_color = mgp.get("fill", "white")
    lw = mgp.get("lwd", 1.0)

    q1, med, q3 = float(np.percentile(value, 25)), float(np.percentile(value, 50)), float(np.percentile(value, 75))
    iqr = q3 - q1
    whisker_lo = max(float(np.min(value)), q1 - 1.5 * iqr)
    whisker_hi = min(float(np.max(value)), q3 + 1.5 * iqr)
    outliers = value[(value < whisker_lo) | (value > whisker_hi)] if outline else np.array([])

    half = box_width / 2.0
    children = grid_py.GList()

    if direction == "vertical":
        # Box
        box = grid_py.rect_grob(
            x=pos, y=(q1 + q3) / 2,
            width=box_width, height=iqr,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, fill=fill_color, lwd=lw),
            name="box",
        )
        children.append(box)

        # Median line
        median_line = grid_py.segments_grob(
            x0=pos - half, y0=med,
            x1=pos + half, y1=med,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="median",
        )
        children.append(median_line)

        # Lower whisker
        children.append(grid_py.segments_grob(
            x0=pos, y0=q1, x1=pos, y1=whisker_lo,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="whisker_lo",
        ))
        # Upper whisker
        children.append(grid_py.segments_grob(
            x0=pos, y0=q3, x1=pos, y1=whisker_hi,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="whisker_hi",
        ))

        # Caps
        cap = half * 0.5
        children.append(grid_py.segments_grob(
            x0=pos - cap, y0=whisker_lo,
            x1=pos + cap, y1=whisker_lo,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="cap_lo",
        ))
        children.append(grid_py.segments_grob(
            x0=pos - cap, y0=whisker_hi,
            x1=pos + cap, y1=whisker_hi,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="cap_hi",
        ))

        # Outliers
        if len(outliers) > 0:
            children.append(grid_py.points_grob(
                x=[pos] * len(outliers),
                y=outliers.tolist(),
                default_units="npc",
                pch=1,
                size=grid_py.Unit(2, "mm"),
                gp=grid_py.Gpar(col=line_color),
                name="outliers",
            ))
    else:
        # Horizontal boxplot
        box = grid_py.rect_grob(
            x=(q1 + q3) / 2, y=pos,
            width=iqr, height=box_width,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, fill=fill_color, lwd=lw),
            name="box",
        )
        children.append(box)

        median_line = grid_py.segments_grob(
            x0=med, y0=pos - half,
            x1=med, y1=pos + half,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="median",
        )
        children.append(median_line)

        children.append(grid_py.segments_grob(
            x0=whisker_lo, y0=pos, x1=q1, y1=pos,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="whisker_lo",
        ))
        children.append(grid_py.segments_grob(
            x0=q3, y0=pos, x1=whisker_hi, y1=pos,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="whisker_hi",
        ))

        cap = half * 0.5
        children.append(grid_py.segments_grob(
            x0=whisker_lo, y0=pos - cap,
            x1=whisker_lo, y1=pos + cap,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="cap_lo",
        ))
        children.append(grid_py.segments_grob(
            x0=whisker_hi, y0=pos - cap,
            x1=whisker_hi, y1=pos + cap,
            default_units="npc",
            gp=grid_py.Gpar(col=line_color, lwd=lw),
            name="cap_hi",
        ))

        if len(outliers) > 0:
            children.append(grid_py.points_grob(
                x=outliers.tolist(),
                y=[pos] * len(outliers),
                default_units="npc",
                pch=1,
                size=grid_py.Unit(2, "mm"),
                gp=grid_py.Gpar(col=line_color),
                name="outliers",
            ))

    return grid_py.GTree(children=children, name="boxplot")


# ---------------------------------------------------------------------------
# textbox_grob / grid_textbox
# ---------------------------------------------------------------------------

def textbox_grob(
    text: Union[str, List[str]],
    gp: Optional[Dict[str, Any]] = None,
    background_gp: Optional[Dict[str, Any]] = None,
    max_width: Optional[int] = None,
    word_wrap: bool = True,
) -> grid_py.GTree:
    """Create a textbox grob with optional background.

    Parameters
    ----------
    text : str or list of str
        The text content.  A list is joined with newlines.
    gp : dict, optional
        Graphical parameters for the text (``fontsize``, ``col``, ...).
    background_gp : dict, optional
        Graphical parameters for the background rectangle (``fill``,
        ``col``, ``lwd``, ...).
    max_width : int, optional
        Maximum width in characters for word wrapping.
    word_wrap : bool
        Whether to wrap text at *max_width*.

    Returns
    -------
    grid_py.GTree
        A GTree containing a background rectangle and text grob.
    """
    if isinstance(text, list):
        text = "\n".join(text)

    if word_wrap and max_width is not None:
        text = textwrap.fill(text, width=int(max_width))

    children = grid_py.GList()

    # Background rectangle
    if background_gp is not None:
        bgp = _resolve_gp(background_gp)
        bg_rect = grid_py.rect_grob(
            x=0.5, y=0.5,
            width=1, height=1,
            default_units="npc",
            gp=_to_gpar(bgp),
            name="textbox_bg",
        )
        children.append(bg_rect)

    # Text
    text_gp = _resolve_gp(gp)
    text_gp.setdefault("col", "black")
    text_gp.setdefault("fontsize", 10)

    text_g = grid_py.text_grob(
        text,
        x=0.5, y=0.5,
        default_units="npc",
        just="centre",
        gp=_to_gpar(text_gp),
        name="textbox_text",
    )
    children.append(text_g)

    return grid_py.GTree(children=children, name="textbox_grob")


def grid_textbox(
    text: Union[str, List[str]],
    x: float = 0.5,
    y: float = 0.5,
    gp: Optional[Dict[str, Any]] = None,
    max_width: Optional[int] = None,
    background_gp: Optional[Dict[str, Any]] = None,
    word_wrap: bool = True,
    ha: str = "center",
    va: str = "center",
    padding: float = 0.02,
) -> grid_py.GTree:
    """Draw a text box with optional background.

    Parameters
    ----------
    text : str or list of str
        The text content.
    x, y : float
        Position in npc coordinates.
    gp : dict, optional
        Text graphical parameters.
    max_width : int, optional
        Maximum width (in characters) for word wrapping.
    background_gp : dict, optional
        Background-rectangle graphical parameters.
    word_wrap : bool
        Whether to wrap long lines.
    ha, va : str
        Horizontal / vertical alignment.
    padding : float
        Padding around text in npc units.

    Returns
    -------
    grid_py.GTree
        A GTree containing the textbox elements.
    """
    if isinstance(text, list):
        text = "\n".join(text)

    if word_wrap and max_width is not None:
        text = textwrap.fill(text, width=int(max_width))

    children = grid_py.GList()

    # Map alignment
    just_h = {"center": "centre", "left": "left", "right": "right"}.get(ha, ha)
    just_v = {"center": "centre", "top": "top", "bottom": "bottom"}.get(va, va)
    just = [just_h, just_v]

    # Background rectangle
    if background_gp is not None:
        bgp = _resolve_gp(background_gp)
        bg_rect = grid_py.rect_grob(
            x=x, y=y,
            width=1, height=1,
            default_units="npc",
            just=just,
            gp=_to_gpar(bgp),
            name="grid_textbox_bg",
        )
        children.append(bg_rect)

    # Text grob
    text_gp = _resolve_gp(gp)
    text_gp.setdefault("col", "black")
    text_gp.setdefault("fontsize", 10)

    text_g = grid_py.text_grob(
        text,
        x=x, y=y,
        default_units="npc",
        just=just,
        gp=_to_gpar(text_gp),
        name="grid_textbox_text",
    )
    children.append(text_g)

    return grid_py.GTree(children=children, name="grid_textbox")


# ---------------------------------------------------------------------------
# gt_render
# ---------------------------------------------------------------------------

def gt_render(
    text: str,
    gp: Optional[Dict[str, Any]] = None,
    padding: float = 0,
) -> Dict[str, Any]:
    """Create a rich-text grob specification with basic markup.

    Supports a minimal subset of HTML-like tags:

    * ``<b>bold</b>``
    * ``<i>italic</i>``
    * ``<br>`` or ``<br/>`` for line breaks

    Parameters
    ----------
    text : str
        Input text with optional ``<b>``, ``<i>``, ``<br>`` tags.
    gp : dict, optional
        Baseline graphical parameters.
    padding : float
        Extra padding around rendered text (in points).

    Returns
    -------
    dict
        A grob specification with keys ``type``, ``segments``, ``gp``,
        ``padding``.  ``segments`` is a list of dicts, each with ``text``,
        ``bold`` (bool), ``italic`` (bool).
    """
    segments: List[Dict[str, Any]] = []
    remaining = text

    # Normalise <br> variants
    remaining = re.sub(r"<br\s*/?>", "\n", remaining)

    # Tokenise bold/italic spans
    pattern = re.compile(r"<(b|i)>(.*?)</\1>", re.DOTALL)
    pos = 0
    for m in pattern.finditer(remaining):
        # plain text before the match
        if m.start() > pos:
            segments.append({
                "text": remaining[pos:m.start()],
                "bold": False,
                "italic": False,
            })
        tag = m.group(1)
        segments.append({
            "text": m.group(2),
            "bold": tag == "b",
            "italic": tag == "i",
        })
        pos = m.end()

    # trailing plain text
    if pos < len(remaining):
        segments.append({
            "text": remaining[pos:],
            "bold": False,
            "italic": False,
        })

    return {
        "type": "gt_render",
        "segments": segments,
        "gp": gp,
        "padding": padding,
    }


# ---------------------------------------------------------------------------
# annotation_axis_grob
# ---------------------------------------------------------------------------

def annotation_axis_grob(
    at: Optional[Sequence[float]] = None,
    labels: Optional[Union[bool, Sequence[str]]] = True,
    labels_rot: float = 0,
    gp: Optional[Dict[str, Any]] = None,
    side: str = "bottom",
    facing: str = "outside",
) -> grid_py.GTree:
    """Create an axis grob for heatmap annotations.

    Corresponds to R's ``annotation_axis_grob`` helper used in
    ``ComplexHeatmap`` for row/column annotation axes.

    Parameters
    ----------
    at : sequence of float, optional
        Tick positions in data coordinates (0-1 native scale).
    labels : bool or sequence of str, optional
        ``True`` to use ``at`` values as labels, ``False`` for no labels,
        or a list of label strings.
    labels_rot : float
        Rotation angle for tick labels in degrees.
    gp : dict, optional
        Graphical parameters for the axis (``fontsize``, ``col``, ...).
    side : str
        Which side the axis is on: ``"bottom"``, ``"top"``, ``"left"``,
        ``"right"``.
    facing : str
        ``"outside"`` (ticks point away from plot) or ``"inside"``.

    Returns
    -------
    grid_py.GTree
        A GTree grob representing the annotation axis.
    """
    if at is None:
        at = [0.0, 0.25, 0.5, 0.75, 1.0]

    at_vals = [float(v) for v in at]

    # Resolve labels
    if labels is True:
        label_strs = [str(v) for v in at_vals]
    elif labels is False or labels is None:
        label_strs = None
    else:
        label_strs = [str(lbl) for lbl in labels]

    axis_gp = _resolve_gp(gp)
    axis_gp.setdefault("col", "black")
    axis_gp.setdefault("fontsize", 8)
    axis_gp.setdefault("lwd", 0.5)

    gpar = _to_gpar(axis_gp)
    tick_len = 2.0  # mm

    children = grid_py.GList()

    if side in ("bottom", "top"):
        # Horizontal axis
        is_top = side == "top"
        outside = facing == "outside"
        # Tick direction
        if (is_top and outside) or (not is_top and not outside):
            tick_dir = 1  # ticks go up
        else:
            tick_dir = -1  # ticks go down

        y_base = 1.0 if is_top else 0.0

        # Main axis line
        axis_line = grid_py.segments_grob(
            x0=0.0, y0=y_base, x1=1.0, y1=y_base,
            default_units="npc",
            gp=gpar,
            name="axis_line",
        )
        children.append(axis_line)

        for i, v in enumerate(at_vals):
            # Tick mark
            tick = grid_py.segments_grob(
                x0=v, y0=y_base,
                x1=v,
                y1=y_base,
                default_units="npc",
                gp=gpar,
                name=f"tick_{i}",
            )
            # Manually set y1 with mm offset -- use Unit arithmetic
            children.append(grid_py.segments_grob(
                x0=grid_py.Unit(v, "npc"),
                y0=grid_py.Unit(y_base, "npc"),
                x1=grid_py.Unit(v, "npc"),
                y1=grid_py.Unit(y_base, "npc") + grid_py.Unit(
                    tick_dir * tick_len, "mm"
                ),
                gp=gpar,
                name=f"tick_{i}",
            ))

            # Label
            if label_strs is not None:
                label_just = "top" if tick_dir < 0 else "bottom"
                lbl = grid_py.text_grob(
                    label_strs[i],
                    x=grid_py.Unit(v, "npc"),
                    y=grid_py.Unit(y_base, "npc") + grid_py.Unit(
                        tick_dir * (tick_len + 1.0), "mm"
                    ),
                    just=label_just,
                    rot=labels_rot,
                    gp=gpar,
                    name=f"label_{i}",
                )
                children.append(lbl)

    else:
        # Vertical axis (left / right)
        is_right = side == "right"
        outside = facing == "outside"

        if (is_right and outside) or (not is_right and not outside):
            tick_dir = 1  # ticks go right
        else:
            tick_dir = -1  # ticks go left

        x_base = 1.0 if is_right else 0.0

        axis_line = grid_py.segments_grob(
            x0=x_base, y0=0.0, x1=x_base, y1=1.0,
            default_units="npc",
            gp=gpar,
            name="axis_line",
        )
        children.append(axis_line)

        for i, v in enumerate(at_vals):
            children.append(grid_py.segments_grob(
                x0=grid_py.Unit(x_base, "npc"),
                y0=grid_py.Unit(v, "npc"),
                x1=grid_py.Unit(x_base, "npc") + grid_py.Unit(
                    tick_dir * tick_len, "mm"
                ),
                y1=grid_py.Unit(v, "npc"),
                gp=gpar,
                name=f"tick_{i}",
            ))

            if label_strs is not None:
                label_just = "left" if tick_dir > 0 else "right"
                lbl = grid_py.text_grob(
                    label_strs[i],
                    x=grid_py.Unit(x_base, "npc") + grid_py.Unit(
                        tick_dir * (tick_len + 1.0), "mm"
                    ),
                    y=grid_py.Unit(v, "npc"),
                    just=label_just,
                    rot=labels_rot,
                    gp=gpar,
                    name=f"label_{i}",
                )
                children.append(lbl)

    return grid_py.GTree(children=children, name="annotation_axis")
