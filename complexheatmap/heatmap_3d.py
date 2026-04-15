"""3-D heatmap and bar chart visualization using grid_py.

Provides ``Heatmap3D`` and ``bar3D`` for rendering a numeric matrix as a
three-dimensional bar chart where bar height and colour both encode the cell
value.  All drawing is performed via ``grid_py`` (the Python port of R's
*grid* graphics system) using oblique-projection polygons.
"""

__all__ = ["Heatmap3D", "bar3D"]

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import grid_py


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_color_func() -> Callable[[float], str]:
    """Return a blue-white-red diverging colour ramp (callable).

    Returns
    -------
    callable
        A function mapping a value in [0, 1] to a hex colour string.
    """
    from ._color import color_ramp2

    breaks = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ["#3B4CC0", "#8CAFFE", "#F7F7F7", "#F08060", "#B40426"]
    return color_ramp2(breaks, colors)


def _value_to_hex(
    values: np.ndarray,
    col: Optional[Callable[..., Any]] = None,
) -> List[str]:
    """Map a flat array of numeric values to hex colour strings.

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values.
    col : callable, optional
        A colour-mapping function.  When *None* the default blue-white-red
        ramp is used.  The callable should accept a single numeric value
        and return a hex string.

    Returns
    -------
    list of str
        One hex colour per input value.
    """
    finite_mask = np.isfinite(values)
    finite_vals = values[finite_mask]
    if len(finite_vals) == 0:
        return ["#CCCCCC"] * len(values)

    vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    span = vmax - vmin if vmax > vmin else 1.0

    # Replace NaN with vmin for colour mapping purposes
    safe_values = np.where(np.isfinite(values), values, vmin)

    if col is not None:
        # Try raw-value mapping first (color_ramp2 signature)
        try:
            sample = col(safe_values[0])
            if isinstance(sample, str):
                return [col(v) for v in safe_values]
        except Exception:
            pass
        # Fall back to normalised [0, 1] mapping
        normed = (safe_values - vmin) / span
        return [col(float(n)) for n in normed]

    cfunc = _default_color_func()
    normed = (safe_values - vmin) / span
    return [cfunc(float(n)) for n in normed]


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Return a darkened version of *hex_color*.

    Parameters
    ----------
    hex_color : str
        ``#RRGGBB`` hex string.
    factor : float
        Darkening factor (0 = black, 1 = original).

    Returns
    -------
    str
        Darkened ``#RRGGBB`` string.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) < 6:
        hex_color = hex_color.ljust(6, "0")
    r = int(int(hex_color[0:2], 16) * factor)
    g = int(int(hex_color[2:4], 16) * factor)
    b = int(int(hex_color[4:6], 16) * factor)
    return f"#{min(r,255):02X}{min(g,255):02X}{min(b,255):02X}"


def _lighten(hex_color: str, factor: float = 0.3) -> str:
    """Return a lightened version of *hex_color*.

    Parameters
    ----------
    hex_color : str
        ``#RRGGBB`` hex string.
    factor : float
        Amount of white to blend in (0 = original, 1 = white).

    Returns
    -------
    str
        Lightened ``#RRGGBB`` string.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) < 6:
        hex_color = hex_color.ljust(6, "0")
    r = int(int(hex_color[0:2], 16) + (255 - int(hex_color[0:2], 16)) * factor)
    g = int(int(hex_color[2:4], 16) + (255 - int(hex_color[2:4], 16)) * factor)
    b = int(int(hex_color[4:6], 16) + (255 - int(hex_color[4:6], 16)) * factor)
    return f"#{min(r,255):02X}{min(g,255):02X}{min(b,255):02X}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Heatmap3D:
    """A 3-D perspective heatmap rendered via ``grid_py`` polygons.

    Uses a simple oblique (cavalier) projection to render each matrix cell
    as a 3-D bar whose height and colour both encode the value.

    Parameters
    ----------
    matrix : np.ndarray
        2-D numeric matrix of shape ``(n_rows, n_cols)``.
    col : callable, optional
        Colour-mapping function.  Accepts a numeric value and returns a
        hex string, e.g. the output of ``color_ramp2``.  When *None* the
        default blue-white-red ramp is used.
    bar_height_scale : float
        Multiplicative scale factor for bar heights.
    row_names : list of str, optional
        Row labels.  Defaults to ``R1, R2, ...``.
    column_names : list of str, optional
        Column labels.  Defaults to ``C1, C2, ...``.
    row_title : str
        Label for the row (y) axis.
    column_title : str
        Label for the column (x) axis.
    show_row_names : bool
        Whether to show row tick labels.
    show_column_names : bool
        Whether to show column tick labels.
    title : str
        Overall title.
    bar_width : float
        Width of each bar along the x-axis (0 < bar_width <= 1).
    bar_depth : float
        Depth of each bar along the y-axis (0 < bar_depth <= 1).
    theta : float
        Oblique projection angle in degrees (0–90).
    name : str, optional
        Name for the top-level ``grid_py.GTree``.

    Attributes
    ----------
    matrix : np.ndarray
    grob : grid_py.GTree
        The assembled graphic object, built lazily on first call to
        :meth:`draw`.

    Examples
    --------
    >>> import numpy as np
    >>> from complexheatmap import Heatmap3D
    >>> mat = np.random.rand(4, 5)
    >>> h3d = Heatmap3D(mat, title="demo")
    >>> h3d.draw()
    """

    def __init__(
        self,
        matrix: np.ndarray,
        col: Optional[Callable[..., Any]] = None,
        bar_height_scale: float = 1.0,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        row_title: str = "",
        column_title: str = "",
        show_row_names: bool = True,
        show_column_names: bool = True,
        title: str = "",
        bar_width: float = 0.8,
        bar_depth: float = 0.8,
        theta: float = 60.0,
        name: Optional[str] = None,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.col = col
        self.bar_height_scale = bar_height_scale
        self.n_rows, self.n_cols = self.matrix.shape
        self.row_names = row_names or [f"R{i + 1}" for i in range(self.n_rows)]
        self.column_names = column_names or [f"C{j + 1}" for j in range(self.n_cols)]
        self.row_title = row_title
        self.column_title = column_title
        self.show_row_names = show_row_names
        self.show_column_names = show_column_names
        self.title = title
        self.bar_width = bar_width
        self.bar_depth = bar_depth
        self.theta = theta
        self.name = name or "heatmap_3d"
        self.grob: Optional[grid_py.GTree] = None

    # ------------------------------------------------------------------
    # Building the grob
    # ------------------------------------------------------------------

    def _build_grob(self) -> grid_py.GTree:
        """Construct the ``grid_py.GTree`` for this 3-D heatmap.

        Returns
        -------
        grid_py.GTree
            A tree of polygon grobs representing the 3-D bars.
        """
        mat = self.matrix
        nr, nc = self.n_rows, self.n_cols

        # Normalise heights to [0, 1] range then scale
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        span = vmax - vmin if vmax > vmin else 1.0
        heights = (mat - vmin) / span * self.bar_height_scale

        # Colours
        flat_values = mat.ravel()
        colors = _value_to_hex(flat_values, col=self.col)

        # Oblique projection parameters
        theta_rad = math.radians(self.theta)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        # Each cell occupies a 1x1 square in the base grid.
        # We place bars in NPC-like coordinates within a dedicated viewport.
        # To fit everything, compute total extent.
        max_h = float(np.nanmax(heights)) if np.any(np.isfinite(heights)) else 0.5
        # Projection offset for the tallest bar
        dx_proj = max_h * cos_t
        dy_proj = max_h * sin_t

        total_w = nc + dx_proj + 0.5  # extra margin
        total_h = nr + dy_proj + 0.5

        children: list = []
        idx = 0

        # Draw back-to-front: iterate rows from top (far) to bottom (near),
        # columns left to right.
        for i in range(nr):
            for j in range(nc):
                h = float(heights[i, j]) if np.isfinite(heights[i, j]) else 0.0
                fill_hex = colors[i * nc + j]

                # Base rectangle corners (bottom-left of cell)
                bx = j
                by = (nr - 1 - i)  # flip so row 0 is at top/back

                bw = self.bar_width
                bd = self.bar_depth

                # The 3 visible faces of an oblique-projected bar:
                # front, top, right-side

                # Projection offsets for this bar's height
                pdx = h * cos_t
                pdy = h * sin_t

                # --- Front face (rectangle) ---
                fx = [bx, bx + bw, bx + bw, bx]
                fy = [by, by, by + bd, by + bd]

                children.append(grid_py.polygon_grob(
                    x=grid_py.Unit([v / total_w for v in fx], "npc"),
                    y=grid_py.Unit([v / total_h for v in fy], "npc"),
                    gp=grid_py.Gpar(fill=fill_hex, col="black", lwd=0.5),
                    name=f"front_{i}_{j}",
                ))

                if h > 0.001:
                    # --- Top face (parallelogram) ---
                    tx = [bx, bx + bw, bx + bw + pdx, bx + pdx]
                    ty = [by + bd, by + bd, by + bd + pdy, by + bd + pdy]

                    children.append(grid_py.polygon_grob(
                        x=grid_py.Unit([v / total_w for v in tx], "npc"),
                        y=grid_py.Unit([v / total_h for v in ty], "npc"),
                        gp=grid_py.Gpar(fill=_lighten(fill_hex, 0.3), col="black", lwd=0.5),
                        name=f"top_{i}_{j}",
                    ))

                    # --- Right face (parallelogram) ---
                    rx = [bx + bw, bx + bw + pdx, bx + bw + pdx, bx + bw]
                    ry = [by, by + pdy, by + bd + pdy, by + bd]

                    children.append(grid_py.polygon_grob(
                        x=grid_py.Unit([v / total_w for v in rx], "npc"),
                        y=grid_py.Unit([v / total_h for v in ry], "npc"),
                        gp=grid_py.Gpar(fill=_darken(fill_hex, 0.7), col="black", lwd=0.5),
                        name=f"right_{i}_{j}",
                    ))

                idx += 1

        # Optional title
        if self.title:
            children.append(grid_py.text_grob(
                label=self.title,
                x=grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.97, "npc"),
                gp=grid_py.Gpar(fontsize=14),
                name="title",
            ))

        tree = grid_py.GTree(children=grid_py.GList(*children), name=self.name)
        self.grob = tree
        return tree

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self) -> grid_py.GTree:
        """Build and draw the 3-D heatmap.

        Returns
        -------
        grid_py.GTree
            The rendered grob tree.
        """
        grob = self._build_grob()
        grid_py.grid_newpage()
        grid_py.grid_draw(grob)
        return grob


def bar3D(
    matrix: np.ndarray,
    col: Optional[Callable[..., Any]] = None,
    bar_height_scale: float = 1.0,
    row_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    title: str = "",
    theta: float = 60.0,
    **kwargs: Any,
) -> grid_py.GTree:
    """Create a 3-D bar chart representation of a matrix.

    Convenience function that creates a :class:`Heatmap3D` instance and
    draws it.  Mirrors the R ``bar3D`` helper.

    Parameters
    ----------
    matrix : np.ndarray
        2-D numeric matrix.
    col : callable, optional
        Colour-mapping function.
    bar_height_scale : float
        Multiplicative scale factor for bar heights.
    row_names : list of str, optional
        Row labels.
    column_names : list of str, optional
        Column labels.
    title : str
        Overall title.
    theta : float
        Oblique projection angle in degrees (0–90).
    **kwargs
        Forwarded to :class:`Heatmap3D`.

    Returns
    -------
    grid_py.GTree
        The rendered grob tree.

    Examples
    --------
    >>> import numpy as np
    >>> from complexheatmap import bar3D
    >>> mat = np.random.rand(3, 4)
    >>> grob = bar3D(mat, title="bars")
    """
    h3d = Heatmap3D(
        matrix,
        col=col,
        bar_height_scale=bar_height_scale,
        row_names=row_names,
        column_names=column_names,
        title=title,
        theta=theta,
        **kwargs,
    )
    return h3d.draw()
