"""Density and frequency heatmap functions.

Provides ``density_heatmap`` and ``frequency_heatmap`` for visualising
per-column distributions as colour-encoded density or frequency profiles.

Faithfully ports R's ``ComplexHeatmap::densityHeatmap`` and
``ComplexHeatmap::frequencyHeatmap`` (densityHeatmap.R).

Uses ``grid_py`` as the rendering backend (via the returned
:class:`~complexheatmap.heatmap.Heatmap` object).
"""

__all__ = ["density_heatmap", "frequency_heatmap"]

import string
import random
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import grid_py

# Save builtins before any parameter shadowing
_builtin_range = range

# ---------------------------------------------------------------------------
# R's rev(brewer.pal(11, "Spectral"))
# ---------------------------------------------------------------------------
_SPECTRAL_11_REV = [
    "#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598",
    "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142",
]


def _random_str(n: int = 8) -> str:
    """Generate a random alphanumeric string (matching R's random_str)."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=n))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_density_list(
    data_list: List[np.ndarray],
    density_param: Optional[Dict[str, Any]] = None,
) -> list:
    """Compute density for each element using scipy gaussian_kde.

    Returns a list of (x, y) tuples matching R's density() output.
    """
    result = []
    for vals in data_list:
        finite = vals[np.isfinite(vals)]
        if len(finite) < 2:
            result.append((np.array([0.0]), np.array([0.0])))
            continue
        try:
            kde = gaussian_kde(finite)
            # Use 512 points (R's default) for density estimation
            lo, hi = float(np.min(finite)), float(np.max(finite))
            pad = (hi - lo) * 0.1 if hi > lo else 1.0
            x = np.linspace(lo - pad, hi + pad, 512)
            y = kde(x)
            result.append((x, y))
        except np.linalg.LinAlgError:
            result.append((np.array([0.0]), np.array([0.0])))
    return result


from ._utils import smart_align as _smart_align


# ---------------------------------------------------------------------------
# KS distance (matching R's ks_dist)
# ---------------------------------------------------------------------------

def _hist_breaks(vals: np.ndarray, breaks: Union[str, int] = "Sturges") -> np.ndarray:
    """Compute histogram break points matching R's hist().

    R's hist() with breaks="Sturges" computes Sturges nclass then uses
    pretty() to generate nice round bin edges that cover the data range.
    """
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5])

    if isinstance(breaks, str) and breaks.lower() == "sturges":
        nclass = int(np.ceil(np.log2(len(vals)))) + 1
    elif isinstance(breaks, (int, np.integer)):
        nclass = int(breaks)
    else:
        return np.asarray(breaks, dtype=float)

    # Mimic R's pretty() for hist break computation
    data_range = hi - lo
    step = data_range / nclass
    # Round step to a "nice" number (1, 2, 5 * 10^k)
    magnitude = 10 ** np.floor(np.log10(step))
    residual = step / magnitude
    if residual <= 1.5:
        nice_step = 1 * magnitude
    elif residual <= 3.5:
        nice_step = 2 * magnitude
    elif residual <= 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    nice_lo = np.floor(lo / nice_step) * nice_step
    nice_hi = np.ceil(hi / nice_step) * nice_step
    edges = np.arange(nice_lo, nice_hi + nice_step * 0.5, nice_step)
    # Ensure last edge covers data max
    if edges[-1] < hi:
        edges = np.append(edges, edges[-1] + nice_step)
    return edges


def _ks_dist_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between two samples.

    Port of R's ks_dist_pair (densityHeatmap.R:272-282).
    """
    n_x = float(len(x))
    n_y = float(len(y))
    w = np.concatenate([x, y])
    order = np.argsort(w, kind="mergesort")
    z = np.cumsum(np.where(order < n_x, 1.0 / n_x, -1.0 / n_y))
    return float(np.max(np.abs(z)))


def _ks_dist_matrix(data_list: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise KS distance matrix.

    Port of R's ks_dist (densityHeatmap.R:285-326).
    Returns a condensed distance vector suitable for scipy.
    """
    from scipy.spatial.distance import squareform
    nc = len(data_list)
    d = np.zeros((nc, nc))
    for i in range(1, nc):
        for j in range(i):
            d[i, j] = _ks_dist_pair(data_list[i], data_list[j])
            d[j, i] = d[i, j]
    return squareform(d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def density_heatmap(
    data: Union[np.ndarray, List[np.ndarray], list],
    density_param: Optional[Dict[str, Any]] = None,
    col: Optional[Union[List[str], Callable[..., Any]]] = None,
    color_space: str = "LAB",
    ylab: str = "Density",
    column_title: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    range: Optional[tuple] = None,
    title_gp: Optional[Dict[str, Any]] = None,
    ylab_gp: Optional[Dict[str, Any]] = None,
    tick_label_gp: Optional[Dict[str, Any]] = None,
    quantile_gp: Optional[Dict[str, Any]] = None,
    show_quantiles: bool = True,
    column_order: Optional[Any] = None,
    column_names_side: str = "bottom",
    show_column_names: bool = True,
    column_names_max_height: Optional[Any] = None,
    column_names_gp: Optional[Dict[str, Any]] = None,
    column_names_rot: float = 90,
    cluster_columns: bool = False,
    clustering_distance_columns: str = "ks",
    clustering_method_columns: str = "complete",
    **kwargs: Any,
) -> "Heatmap":
    """Create a density heatmap from a matrix or list of distributions.

    Faithfully ports R's ``ComplexHeatmap::densityHeatmap``
    (densityHeatmap.R:63-269).

    Each column is shown as a colour-encoded density profile.
    Quantile lines (0%, 25%, 50%, 75%, 100%) and a mean line (dark red)
    are overlaid, with smart-aligned labels on the right.
    A y-axis with tick marks is drawn on the left.
    """
    from .heatmap import Heatmap
    from .heatmap_annotation import rowAnnotation
    from .annotation_function import AnnotationFunction
    from ._color import color_ramp2

    # `range` parameter shadows Python builtin — restore it locally
    _range = _builtin_range

    # --- resolve aliases (R: range = ylim) --------------------------------
    if range is not None and ylim is None:
        ylim = range
    if title is not None and column_title is None:
        column_title = title
    if column_title is None:
        column_title = "Density heatmap"

    # --- default gpar values (matching R defaults) -------------------------
    if title_gp is None:
        title_gp = {"fontsize": 14}
    if ylab_gp is None:
        ylab_gp = {"fontsize": 12}
    if tick_label_gp is None:
        tick_label_gp = {"fontsize": 10}
    if quantile_gp is None:
        quantile_gp = {"fontsize": 10}
    if column_names_gp is None:
        column_names_gp = {"fontsize": 12}
    if column_names_max_height is None:
        column_names_max_height = grid_py.Unit(6, "cm")

    # --- normalise input (R: densityHeatmap.R:112-119) --------------------
    if isinstance(data, list):
        data_list = [np.asarray(d, dtype=float) for d in data]
        nm = [f"V{j + 1}" for j in _range(len(data_list))]
    elif isinstance(data, np.ndarray):
        data_list = [data[:, j] for j in _range(data.shape[1])]
        nm = [f"V{j + 1}" for j in _range(data.shape[1])]
    else:
        raise ValueError("data must be a matrix or list")

    n_cols = len(data_list)

    # Determine column names
    if hasattr(data, 'dtype') and isinstance(data, np.ndarray):
        pass  # nm already set
    if "column_names" in kwargs:
        nm = list(kwargs.pop("column_names"))
    elif "column_labels" in kwargs:
        nm = list(kwargs.pop("column_labels"))

    # --- compute density (R: densityHeatmap.R:120-122) --------------------
    density_list = _compute_density_list(data_list, density_param)
    quantile_list = np.zeros((5, n_cols))
    mean_value = np.zeros(n_cols)
    for j in _range(n_cols):
        finite = data_list[j][np.isfinite(data_list[j])]
        if len(finite) > 0:
            quantile_list[:, j] = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
            mean_value[j] = np.mean(finite)

    # --- determine y-range (R: densityHeatmap.R:127-133) ------------------
    all_density_x = np.concatenate([d[0] for d in density_list])
    max_x = float(np.quantile(all_density_x, 0.99))
    min_x = float(np.quantile(all_density_x, 0.01))

    if ylim is not None:
        min_x = ylim[0]
        max_x = ylim[1]

    # --- build density matrix (R: densityHeatmap.R:135-144) ---------------
    # R: x = seq(min_x, max_x, length.out = 500)
    x_grid = np.linspace(min_x, max_x, 500)

    # R: f = approxfun(r$x, r$y); res = f(x); res[is.na(res)] = 0; rev(res)
    mat_cols = []
    for d_x, d_y in density_list:
        if len(d_x) < 2:
            mat_cols.append(np.zeros(500))
            continue
        f = interp1d(d_x, d_y, bounds_error=False, fill_value=0.0)
        res = f(x_grid)
        res = np.where(np.isnan(res), 0.0, res)
        mat_cols.append(res[::-1])  # R: rev(res)

    mat = np.column_stack(mat_cols)
    # mat shape: (500, n_cols)

    # --- column clustering (R: densityHeatmap.R:146-154) ------------------
    column_dend = None
    if cluster_columns:
        if clustering_distance_columns == "ks":
            from scipy.cluster.hierarchy import linkage, to_tree
            from scipy.spatial.distance import squareform
            # KS distance on original data
            finite_lists = [d[np.isfinite(d)] for d in data_list]
            d_condensed = _ks_dist_matrix(finite_lists)
            Z = linkage(d_condensed, method=clustering_method_columns)
            # R: dend = reorder(dend, colMeans(mat))
            # Pass dendrogram linkage; Heatmap handles reordering
            cluster_columns = Z

    # --- colour mapping (R: densityHeatmap.R:156-160) ---------------------
    if col is not None:
        if callable(col):
            palette = col
        else:
            palette = col
    else:
        # R: col = rev(brewer.pal(11, "Spectral"))
        # R: col = colorRamp2(seq(0, quantile(mat, 0.99), length.out=11), col)
        q99 = float(np.quantile(mat, 0.99))
        if q99 <= 0:
            q99 = 1.0
        breaks = np.linspace(0, q99, len(_SPECTRAL_11_REV)).tolist()
        palette = color_ramp2(breaks, _SPECTRAL_11_REV, space=color_space)

    # --- y-axis tick values (R: bb = grid.pretty(c(min_x, max_x))) --------
    bb = grid_py.grid_pretty([min_x, max_x])
    bb_labels = [f"{v:g}" for v in bb]

    # --- left annotation width (R: densityHeatmap.R:177-178) ----
    # R: width = grobHeight(textGrob(ylab, gp=ylab_gp))*2
    #          + max_text_width(bb, gp=tick_label_gp) + unit(4, "mm")
    # Estimate: ylab height (rotated 90 = its string width) * 2 + tick widths + 4mm
    ylab_fontsize = ylab_gp.get("fontsize", 12)
    tick_fontsize = tick_label_gp.get("fontsize", 10)
    # Rough estimate: ylab width when drawn rotated ≈ char height * 2
    ylab_height_mm = ylab_fontsize * 0.3528 * 2  # approx grobHeight * 2
    tick_width_mm = max(len(s) for s in bb_labels) * tick_fontsize * 0.2
    left_width_mm = ylab_height_mm + tick_width_mm + 4.0

    # --- right annotation width (R: densityHeatmap.R:180-182) ----
    # R: width = grobWidth(textGrob("100%", gp=quantile_gp)) + unit(6, "mm")
    quantile_fontsize = quantile_gp.get("fontsize", 10)
    right_width_mm = len("100%") * quantile_fontsize * 0.22 + 6.0

    # --- custom annotation functions (draw content directly) ----------------
    # Instead of R's anno_empty + post_fun/decorate approach, we create
    # custom annotation drawing functions. This avoids clip issues because
    # annotation VPs are not clipped in the same way as body VPs.

    # Left annotation: y-axis label + ticks
    _ylab_c = ylab
    _ylab_gp_c = ylab_gp
    _tick_label_gp_c = tick_label_gp
    _bb_c = bb
    _min_x_c = min_x
    _max_x_c = max_x

    def _draw_left_axis(index: np.ndarray, k: int, n_slices: int) -> None:
        """Draw y-axis label and ticks in left annotation.

        The annotation VP is narrow, so we position the axis line at x=1
        (right edge, adjacent to heatmap body) with ticks going left
        (main=False draws ticks to the right, main=True to the left).
        """
        if k != 1:
            return  # only draw on first slice
        # Push VP with data yscale
        vp = grid_py.Viewport(yscale=(_min_x_c, _max_x_c))
        grid_py.push_viewport(vp)
        # Y-axis at the right edge of annotation (main=False: axis at x=1,
        # ticks to the right — but we want ticks going LEFT from right edge,
        # so use main=True which draws at x=0 with ticks going left.
        # Instead, position axis at right side: main=False gives axis at x=1npc
        # with labels to the right — but that's wrong. We need labels
        # to the left. Let's draw it manually:

        # Draw tick marks and labels
        for val in _bb_c:
            # Tick mark
            grid_py.grid_segments(
                x0=grid_py.Unit(1, "npc"),
                y0=grid_py.Unit(val, "native"),
                x1=grid_py.Unit(1, "npc") - grid_py.Unit(0.5, "lines"),
                y1=grid_py.Unit(val, "native"),
            )
            # Tick label
            grid_py.grid_text(
                f"{val:g}",
                x=grid_py.Unit(1, "npc") - grid_py.Unit(1, "lines"),
                y=grid_py.Unit(val, "native"),
                just="right",
                gp=grid_py.Gpar(**_tick_label_gp_c),
            )
        # Axis line
        grid_py.grid_segments(
            x0=grid_py.Unit(1, "npc"),
            y0=grid_py.Unit(_min_x_c, "native"),
            x1=grid_py.Unit(1, "npc"),
            y1=grid_py.Unit(_max_x_c, "native"),
        )

        # Y-label (rotated 90) to the far left
        grid_py.grid_text(
            _ylab_c,
            x=grid_py.Unit(0.5, "lines"),
            rot=90,
            gp=grid_py.Gpar(**_ylab_gp_c),
        )
        grid_py.up_viewport()

    left_anno_fun = AnnotationFunction(
        fun=_draw_left_axis,
        fun_name="density_yaxis",
        which="row",
        n=None,
        data_scale=(min_x, max_x),
        subsettable=True,
        show_name=False,
        width=grid_py.Unit(left_width_mm, "mm"),
    )

    # Right annotation: quantile labels with connector lines
    _quantile_list_c = quantile_list
    _mean_value_c = mean_value
    _quantile_gp_c = quantile_gp

    def _draw_right_quantile(index: np.ndarray, k: int, n_slices: int) -> None:
        """Draw quantile labels with smart alignment."""
        if k != n_slices:
            return  # only draw on last slice

        # Determine which quantiles are within range
        lq = []
        for qi in _range(5):
            row = _quantile_list_c[qi, :]
            if np.all(row > _max_x_c) or np.all(row < _min_x_c):
                lq.append(False)
            else:
                lq.append(True)
        if np.all(_mean_value_c > _max_x_c) or np.all(_mean_value_c < _min_x_c):
            lq.append(False)
        else:
            lq.append(True)

        all_labels = ["0%", "25%", "50%", "75%", "100%", "mean"]
        # Use median of all columns for label positioning
        all_y = np.concatenate([
            np.median(_quantile_list_c, axis=1),
            [np.median(_mean_value_c)],
        ])

        labels = [l for l, q in zip(all_labels, lq) if q]
        y_vals = all_y[np.array(lq)]
        if len(labels) == 0:
            return

        od = np.argsort(y_vals)
        y_vals = y_vals[od]
        labels = [labels[i] for i in od]

        vp = grid_py.Viewport(yscale=(_min_x_c, _max_x_c))
        grid_py.push_viewport(vp)

        # Smart-align
        total_range = _max_x_c - _min_x_c
        text_h_native = total_range * 0.05
        h1 = y_vals - text_h_native * 0.5
        h2 = y_vals + text_h_native * 0.5
        aligned = _smart_align(h1, h2, (_min_x_c, _max_x_c))
        h_centers = (aligned[:, 0] + aligned[:, 1]) / 2.0

        for i in _range(len(labels)):
            grid_py.grid_text(
                labels[i],
                x=grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(h_centers[i], "native"),
                gp=grid_py.Gpar(**_quantile_gp_c),
            )
        grid_py.up_viewport()

    right_anno_fun = None
    if show_quantiles:
        right_anno_fun = AnnotationFunction(
            fun=_draw_right_quantile,
            fun_name="density_quantile",
            which="row",
            n=None,
            data_scale=(min_x, max_x),
            subsettable=True,
            show_name=False,
            width=grid_py.Unit(right_width_mm, "mm"),
        )

    left_anno = rowAnnotation(
        axis=left_anno_fun,
        show_annotation_name=False,
    )
    right_anno = None
    if right_anno_fun is not None:
        right_anno = rowAnnotation(
            quantile=right_anno_fun,
            show_annotation_name=False,
        )

    # --- random string for unique names (R: densityHeatmap.R:186) ---------
    rstr = _random_str(8)

    # --- layer_fun for quantile + mean lines (R: post_fun body decoration) -
    _quantile_list_lf = quantile_list
    _mean_value_lf = mean_value
    _min_x_lf = min_x
    _max_x_lf = max_x

    def _density_layer_fun(j_arr, i_arr, x_arr, y_arr, w_arr, h_arr, fill_arr):
        """Draw quantile lines and mean line on the body.

        Mirrors R's post_fun body decoration (densityHeatmap.R:204-215),
        but runs inside the slice VP via layer_fun.
        """
        if not show_quantiles:
            return
        # j_arr has the column order; extract unique columns in order
        seen = set()
        col_indices = []
        for j in j_arr:
            if j not in seen:
                seen.add(j)
                col_indices.append(j)
        nc = len(col_indices)
        if nc == 0:
            return

        # Push VP with data coordinates
        vp = grid_py.Viewport(
            xscale=(0.5, nc + 0.5),
            yscale=(_min_x_lf, _max_x_lf),
        )
        grid_py.push_viewport(vp)

        # Draw 5 quantile lines
        x_pos = list(np.arange(1, nc + 1))
        for qi in _range(5):
            y_vals = [float(_quantile_list_lf[qi, ci]) for ci in col_indices]
            grid_py.grid_lines(
                x=x_pos, y=y_vals,
                default_units="native",
                gp=grid_py.Gpar(lty="dashed"),
            )

        # Mean line (dark red)
        y_mean = [float(_mean_value_lf[ci]) for ci in col_indices]
        grid_py.grid_lines(
            x=x_pos, y=y_mean,
            default_units="native",
            gp=grid_py.Gpar(lty="dashed", col="darkred"),
        )

        grid_py.up_viewport()

    # --- build Heatmap (R: densityHeatmap.R:163-184) ----------------------
    user_left = kwargs.pop("left_annotation", None)
    user_right = kwargs.pop("right_annotation", None)

    hm_name = f"density_{rstr}"

    hm_kwargs: Dict[str, Any] = dict(
        name="density",  # legend display name (R: name = "density")
        column_title=column_title,
        column_title_gp=grid_py.Gpar(**title_gp),
        cluster_rows=False,  # R line 166: cluster_rows = FALSE
        cluster_columns=cluster_columns,
        clustering_distance_columns=clustering_distance_columns,
        clustering_method_columns=clustering_method_columns,
        column_names_side=column_names_side,
        show_column_names=show_column_names,
        column_names_max_height=column_names_max_height,
        column_names_gp=grid_py.Gpar(**column_names_gp),
        column_names_rot=column_names_rot,
        show_row_names=False,
        left_annotation=user_left if user_left is not None else left_anno,
        col=palette,
        layer_fun=_density_layer_fun,
    )
    if column_order is not None:
        hm_kwargs["column_order"] = column_order
    if right_anno is not None or user_right is not None:
        hm_kwargs["right_annotation"] = (
            user_right if user_right is not None else right_anno
        )
    hm_kwargs.update(kwargs)

    hm = Heatmap(
        mat,
        column_labels=nm,
        **hm_kwargs,
    )

    # R: ht@name = paste0(ht@name, "_", random_str) — unique VP names
    hm.name = hm_name

    return hm


def frequency_heatmap(
    data: Union[np.ndarray, List[np.ndarray], list],
    breaks: Union[int, str, np.ndarray] = "Sturges",
    stat: str = "count",
    col: Optional[Union[List[str], Callable[..., Any]]] = None,
    color_space: str = "LAB",
    ylab: str = "Frequency",
    column_title: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    range: Optional[tuple] = None,
    title_gp: Optional[Dict[str, Any]] = None,
    ylab_gp: Optional[Dict[str, Any]] = None,
    tick_label_gp: Optional[Dict[str, Any]] = None,
    column_order: Optional[Any] = None,
    column_names_side: str = "bottom",
    show_column_names: bool = True,
    column_names_max_height: Optional[Any] = None,
    column_names_gp: Optional[Dict[str, Any]] = None,
    column_names_rot: float = 90,
    cluster_columns: bool = False,
    column_names: Optional[List[str]] = None,
    use_3d: bool = False,
    **kwargs: Any,
) -> "Heatmap":
    """Create a frequency heatmap (histogram-based) from column data.

    Faithfully ports R's ``ComplexHeatmap::frequencyHeatmap``
    (densityHeatmap.R:392-546).

    Parameters
    ----------
    use_3d : bool
        When *True*, visualise the frequencies as a 3-D heatmap via
        :func:`~.heatmap_3d.Heatmap3D` instead of a flat ``Heatmap``.
        Matches R's ``frequencyHeatmap(use_3d = TRUE)``
        (densityHeatmap.R:416,481-497).
    """
    from .heatmap import Heatmap
    from .heatmap_annotation import rowAnnotation
    from .annotation_function import AnnotationFunction
    from ._color import color_ramp2

    _range = _builtin_range  # restore builtin shadowed by parameter

    # --- resolve aliases ---------------------------------------------------
    if range is not None and ylim is None:
        ylim = range
    if title is not None and column_title is None:
        column_title = title
    if column_title is None:
        column_title = "Frequency heatmap"

    # --- default gpar -------------------------------------------------------
    if title_gp is None:
        title_gp = {"fontsize": 14}
    if ylab_gp is None:
        ylab_gp = {"fontsize": 12}
    if tick_label_gp is None:
        tick_label_gp = {"fontsize": 10}
    if column_names_gp is None:
        column_names_gp = {"fontsize": 12}
    if column_names_max_height is None:
        column_names_max_height = grid_py.Unit(6, "cm")

    # --- normalise input ---------------------------------------------------
    if isinstance(data, list):
        data_list = [np.asarray(d, dtype=float) for d in data]
    elif isinstance(data, np.ndarray):
        data_list = [data[:, j] for j in _range(data.shape[1])]
    else:
        raise ValueError("data must be a matrix or list")

    n_cols = len(data_list)
    if column_names is None:
        column_names = [f"V{j + 1}" for j in _range(n_cols)]

    # --- common bins (R: frequencyHeatmap.R:444-448) -----------------------
    # R: h = hist(unlist(data), breaks = breaks, plot = FALSE)
    #    breaks = h$breaks
    # R computes nice round bin edges from ALL pooled data, then reuses
    # the same edges per-column.
    all_vals = np.concatenate([d[np.isfinite(d)] for d in data_list])
    bin_edges = _hist_breaks(all_vals, breaks)

    if ylim is not None:
        bin_edges = bin_edges[(bin_edges >= ylim[0]) & (bin_edges <= ylim[1])]
        if len(bin_edges) < 2:
            bin_edges = _hist_breaks(all_vals[
                (all_vals >= ylim[0]) & (all_vals <= ylim[1])], breaks)

    n_bins = len(bin_edges) - 1
    min_x = float(bin_edges[0])
    max_x = float(bin_edges[-1])

    # --- compute frequency matrix (R: frequencyHeatmap.R:450-468) ----------
    freq_matrix = np.zeros((n_bins, n_cols), dtype=float)
    for j in _range(n_cols):
        col_vals = data_list[j][np.isfinite(data_list[j])]
        if len(col_vals) > 0:
            counts, _ = np.histogram(col_vals, bins=bin_edges)
            if stat == "count":
                freq_matrix[:, j] = counts[::-1]  # R: rev()
            elif stat == "proportion":
                s = float(np.sum(counts))
                freq_matrix[:, j] = (counts / s if s > 0 else counts)[::-1]
            elif stat == "density":
                widths = np.diff(bin_edges)
                s = float(np.sum(counts))
                dens = counts / (s * widths) if s > 0 else counts
                freq_matrix[:, j] = dens[::-1]

    # --- colour palette (R: frequencyHeatmap.R:472-476) --------------------
    if col is not None:
        if callable(col):
            palette = col
        else:
            palette = col
    else:
        # R: col = brewer.pal(9, "Blues")
        blues_9 = [
            "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
            "#4292C6", "#2171B5", "#08519C", "#08306B",
        ]
        q99 = float(np.quantile(freq_matrix, 0.99))
        if q99 <= 0:
            q99 = 1.0
        fbreaks = np.linspace(0, q99, len(blues_9)).tolist()
        palette = color_ramp2(fbreaks, blues_9, space=color_space)

    # --- y-axis ticks (R: bb = grid.pretty(c(min_x, max_x))) ---------------
    bb = grid_py.grid_pretty([min_x, max_x])
    bb_labels = [f"{v:g}" for v in bb]

    # --- annotation widths --------------------------------------------------
    ylab_fontsize = ylab_gp.get("fontsize", 12)
    tick_fontsize = tick_label_gp.get("fontsize", 10)
    ylab_height_mm = ylab_fontsize * 0.3528 * 2
    tick_width_mm = max(len(s) for s in bb_labels) * tick_fontsize * 0.2
    left_width_mm = ylab_height_mm + tick_width_mm + 4.0

    # Custom left annotation with y-axis
    _ylab_f = ylab
    _ylab_gp_f = ylab_gp
    _tick_label_gp_f = tick_label_gp
    _bb_f = bb
    _min_x_f = min_x
    _max_x_f = max_x

    def _draw_freq_left(index: np.ndarray, k: int, n_slices: int) -> None:
        if k != 1:
            return
        vp = grid_py.Viewport(yscale=(_min_x_f, _max_x_f))
        grid_py.push_viewport(vp)
        for val in _bb_f:
            grid_py.grid_segments(
                x0=grid_py.Unit(1, "npc"),
                y0=grid_py.Unit(val, "native"),
                x1=grid_py.Unit(1, "npc") - grid_py.Unit(0.5, "lines"),
                y1=grid_py.Unit(val, "native"),
            )
            grid_py.grid_text(
                f"{val:g}",
                x=grid_py.Unit(1, "npc") - grid_py.Unit(1, "lines"),
                y=grid_py.Unit(val, "native"),
                just="right",
                gp=grid_py.Gpar(**_tick_label_gp_f),
            )
        grid_py.grid_segments(
            x0=grid_py.Unit(1, "npc"),
            y0=grid_py.Unit(_min_x_f, "native"),
            x1=grid_py.Unit(1, "npc"),
            y1=grid_py.Unit(_max_x_f, "native"),
        )
        grid_py.grid_text(
            _ylab_f,
            x=grid_py.Unit(0.5, "lines"),
            rot=90,
            gp=grid_py.Gpar(**_ylab_gp_f),
        )
        grid_py.up_viewport()

    left_anno_fun = AnnotationFunction(
        fun=_draw_freq_left,
        fun_name="freq_yaxis",
        which="row",
        n=None,
        data_scale=(min_x, max_x),
        subsettable=True,
        show_name=False,
        width=grid_py.Unit(left_width_mm, "mm"),
    )

    left_anno = rowAnnotation(
        axis=left_anno_fun,
        show_annotation_name=False,
    )

    rstr = _random_str(8)
    stat_name = stat if stat in ("count", "density", "proportion") else "count"
    hm_name = f"{stat_name}_{rstr}"

    user_left = kwargs.pop("left_annotation", None)

    hm_kwargs: Dict[str, Any] = dict(
        name=stat_name,  # legend display name
        column_title=column_title,
        column_title_gp=grid_py.Gpar(**title_gp),
        cluster_rows=False,
        cluster_columns=cluster_columns,
        column_names_side=column_names_side,
        show_column_names=show_column_names,
        column_names_max_height=column_names_max_height,
        column_names_gp=grid_py.Gpar(**column_names_gp),
        column_names_rot=column_names_rot,
        show_row_names=False,
        left_annotation=user_left if user_left is not None else left_anno,
        col=palette,
    )
    if column_order is not None:
        hm_kwargs["column_order"] = column_order
    hm_kwargs.update(kwargs)

    if use_3d:
        from .heatmap_3d import Heatmap3D
        hm = Heatmap3D(
            freq_matrix,
            column_labels=column_names,
            **hm_kwargs,
        )
    else:
        hm = Heatmap(
            freq_matrix,
            column_labels=column_names,
            **hm_kwargs,
        )

    hm.name = hm_name

    return hm
