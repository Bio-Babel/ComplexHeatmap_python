"""Density and frequency heatmap functions.

Provides ``density_heatmap`` and ``frequency_heatmap`` for visualising
per-column distributions as colour-encoded density or frequency profiles.

Uses ``grid_py`` as the rendering backend (via the returned
:class:`~complexheatmap.heatmap.Heatmap` object).
"""

__all__ = ["density_heatmap", "frequency_heatmap"]

import numpy as np
from scipy.stats import gaussian_kde
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_palette(
    col: Optional[Union[List[str], Callable[..., Any]]],
    n: int = 256,
) -> List[str]:
    """Return a list of *n* hex colours from *col* or a default YlOrRd ramp.

    Parameters
    ----------
    col : list of str or callable, optional
        Explicit colour list or a ``color_ramp2``-style callable.
    n : int
        Number of colours when building a default ramp.

    Returns
    -------
    list of str
        Hex colour strings.
    """
    if col is not None:
        if callable(col):
            # Assume color_ramp2-style; sample n evenly-spaced values later
            return col  # type: ignore[return-value]
        return list(col)

    # Build a YlOrRd-like ramp using the package's own color_ramp2
    from ._color import color_ramp2

    breaks = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ["#FFFFCC", "#FED976", "#FD8D3C", "#E31A1C", "#800026"]
    return color_ramp2(breaks, colors)


def _compute_density_matrix(
    data: np.ndarray,
    n_grid: int = 256,
    ylim: Optional[tuple] = None,
) -> tuple:
    """Compute a density matrix from *data* (samples x observations).

    Parameters
    ----------
    data : np.ndarray
        2-D array where each column is a distribution.
    n_grid : int
        Number of grid points for the density evaluation.
    ylim : tuple, optional
        ``(lo, hi)`` for the common evaluation grid.  When *None* the data
        range (plus 5 % padding) is used.

    Returns
    -------
    grid : np.ndarray of shape ``(n_grid,)``
        The evaluation grid.
    density_matrix : np.ndarray of shape ``(n_grid, n_cols)``
        Density values, one column per input column.
    """
    n_cols = data.shape[1]
    col_data = [data[:, j][np.isfinite(data[:, j])] for j in range(n_cols)]

    all_vals = np.concatenate(col_data)
    if ylim is None:
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        lo -= pad
        hi += pad
    else:
        lo, hi = ylim

    grid = np.linspace(lo, hi, n_grid)
    density_matrix = np.zeros((n_grid, n_cols), dtype=float)

    for j, vals in enumerate(col_data):
        if len(vals) < 2:
            continue
        try:
            kde = gaussian_kde(vals)
            density_matrix[:, j] = kde(grid)
        except np.linalg.LinAlgError:
            # Singular covariance — fall back to zeros
            pass

    return grid, density_matrix


def _quantile_indices(
    grid: np.ndarray,
    data_col: np.ndarray,
    quantiles: List[float],
) -> List[int]:
    """Return grid indices closest to the requested quantiles of *data_col*.

    Parameters
    ----------
    grid : np.ndarray
        1-D evaluation grid.
    data_col : np.ndarray
        1-D data vector (finite values only).
    quantiles : list of float
        Quantile probabilities, e.g. ``[0.25, 0.5, 0.75]``.

    Returns
    -------
    list of int
        Grid indices for each quantile value.
    """
    qvals = np.nanquantile(data_col, quantiles)
    return [int(np.argmin(np.abs(grid - qv))) for qv in qvals]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def density_heatmap(
    data: Union[np.ndarray, List[np.ndarray], list],
    title: str = "",
    col: Optional[Union[List[str], Callable[..., Any]]] = None,
    column_title: Optional[str] = None,
    column_names: Optional[List[str]] = None,
    column_names_rot: float = 90,
    show_column_names: bool = True,
    cluster_columns: bool = True,
    clustering_distance_columns: str = "euclidean",
    clustering_method_columns: str = "complete",
    mc_cores: int = 1,
    ylab: str = "Density",
    ylim: Optional[tuple] = None,
    show_quantiles: bool = True,
    quantile_values: Optional[List[float]] = None,
    n_grid: int = 256,
    **kwargs: Any,
) -> "Heatmap":
    """Create a density heatmap from a matrix or list of distributions.

    Each column is shown as a colour-encoded density profile instead of raw
    values.  This is useful for comparing distributions across samples.
    The returned object is a :class:`~complexheatmap.heatmap.Heatmap` that
    renders via ``grid_py``.

    Parameters
    ----------
    data : array-like
        2-D array whose columns are distributions, **or** a list of 1-D
        arrays (which may have different lengths).
    title : str
        Heatmap title.
    col : list of str or callable, optional
        Colour palette for density encoding.  Accepts a list of hex
        strings or a ``color_ramp2``-style callable.  When *None* a
        built-in YlOrRd ramp is used.
    column_title : str, optional
        Title for the column axis.
    column_names : list of str, optional
        Column labels.  When *None*, ``V1, V2, ...`` is used.
    column_names_rot : float
        Rotation angle (degrees) for column labels.
    show_column_names : bool
        Whether to display column labels.
    cluster_columns : bool
        Whether to cluster columns by their density profiles.
    clustering_distance_columns : str
        Distance metric passed to ``scipy.spatial.distance.pdist``.
    clustering_method_columns : str
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
    mc_cores : int
        Reserved for future parallel KDE computation.
    ylab : str
        Label for the y-axis (grid / density axis).
    ylim : tuple, optional
        ``(lo, hi)`` limits for the evaluation grid.
    show_quantiles : bool
        If *True*, attach quantile metadata for downstream rendering.
    quantile_values : list of float, optional
        Quantile probabilities to show (default ``[0.25, 0.5, 0.75]``).
    n_grid : int
        Number of grid points for KDE evaluation (default 256).
    **kwargs
        Forwarded to the ``Heatmap`` constructor.

    Returns
    -------
    Heatmap
        A :class:`~complexheatmap.heatmap.Heatmap` object that can be drawn
        or combined with other heatmaps via ``+`` / ``%v%``.

    Examples
    --------
    >>> import numpy as np
    >>> from complexheatmap import density_heatmap
    >>> mat = np.random.randn(100, 5)
    >>> hm = density_heatmap(mat, title="example")
    """
    from .heatmap import Heatmap  # deferred to avoid circular imports

    # --- normalise input ------------------------------------------------
    if isinstance(data, list):
        max_len = max(len(d) for d in data)
        mat = np.full((max_len, len(data)), np.nan)
        for j, d in enumerate(data):
            arr = np.asarray(d, dtype=float)
            mat[: len(arr), j] = arr
    else:
        mat = np.asarray(data, dtype=float)

    n_cols = mat.shape[1]

    if column_names is None:
        column_names = [f"V{j + 1}" for j in range(n_cols)]

    if quantile_values is None:
        quantile_values = [0.25, 0.5, 0.75]

    # --- compute density matrix -----------------------------------------
    grid, density_matrix = _compute_density_matrix(mat, n_grid=n_grid, ylim=ylim)

    # --- build colour palette -------------------------------------------
    palette = _resolve_palette(col, n=256)

    # --- construct Heatmap ----------------------------------------------
    hm_kwargs: Dict[str, Any] = dict(
        name=title if title else "density",
        column_title=column_title or "",
        show_column_names=show_column_names,
        cluster_columns=cluster_columns,
        clustering_distance_columns=clustering_distance_columns,
        clustering_method_columns=clustering_method_columns,
    )
    # If palette is a callable (color_ramp2), pass it as col
    if callable(palette):
        hm_kwargs["col"] = palette
    hm_kwargs.update(kwargs)

    hm = Heatmap(
        density_matrix,
        row_labels=[f"{grid[i]:.3g}" for i in range(n_grid)],
        column_labels=column_names,
        **hm_kwargs,
    )

    # Attach quantile metadata for downstream rendering
    if show_quantiles:
        q_data: Dict[int, List[int]] = {}
        for j in range(n_cols):
            col_vals = mat[:, j][np.isfinite(mat[:, j])]
            if len(col_vals) > 0:
                q_data[j] = _quantile_indices(grid, col_vals, quantile_values)
            else:
                q_data[j] = []
        hm._density_quantile_indices = q_data  # type: ignore[attr-defined]
        hm._density_quantile_values = quantile_values  # type: ignore[attr-defined]
        hm._density_grid = grid  # type: ignore[attr-defined]

    return hm


def frequency_heatmap(
    data: Union[np.ndarray, List[np.ndarray], list],
    breaks: Union[int, np.ndarray] = 20,
    title: str = "",
    col: Optional[Union[List[str], Callable[..., Any]]] = None,
    column_names: Optional[List[str]] = None,
    column_names_rot: float = 90,
    show_column_names: bool = True,
    cluster_columns: bool = True,
    clustering_distance_columns: str = "euclidean",
    clustering_method_columns: str = "complete",
    ylab: str = "Frequency",
    ylim: Optional[tuple] = None,
    **kwargs: Any,
) -> "Heatmap":
    """Create a frequency heatmap (histogram-based) from column data.

    Instead of a smooth KDE, values are binned and coloured by count /
    frequency.  The returned object is a :class:`~complexheatmap.heatmap.Heatmap`
    that renders via ``grid_py``.

    Parameters
    ----------
    data : array-like
        2-D array whose columns are distributions, **or** a list of 1-D
        arrays.
    breaks : int or np.ndarray
        Number of bins (int) or explicit bin edges (array).
    title : str
        Heatmap title.
    col : list of str or callable, optional
        Colour palette for frequency encoding.
    column_names : list of str, optional
        Column labels.
    column_names_rot : float
        Rotation angle (degrees) for column labels.
    show_column_names : bool
        Whether to display column labels.
    cluster_columns : bool
        Whether to cluster columns.
    clustering_distance_columns : str
        Distance metric for clustering.
    clustering_method_columns : str
        Linkage method for clustering.
    ylab : str
        Label for the y-axis.
    ylim : tuple, optional
        ``(lo, hi)`` limits.
    **kwargs
        Forwarded to the ``Heatmap`` constructor.

    Returns
    -------
    Heatmap
        A :class:`~complexheatmap.heatmap.Heatmap` object.

    Examples
    --------
    >>> import numpy as np
    >>> from complexheatmap import frequency_heatmap
    >>> mat = np.random.randn(100, 5)
    >>> hm = frequency_heatmap(mat, breaks=15)
    """
    from .heatmap import Heatmap

    # --- normalise input ------------------------------------------------
    if isinstance(data, list):
        max_len = max(len(d) for d in data)
        mat = np.full((max_len, len(data)), np.nan)
        for j, d in enumerate(data):
            arr = np.asarray(d, dtype=float)
            mat[: len(arr), j] = arr
    else:
        mat = np.asarray(data, dtype=float)

    n_cols = mat.shape[1]
    if column_names is None:
        column_names = [f"V{j + 1}" for j in range(n_cols)]

    # --- common bins ----------------------------------------------------
    all_vals = mat[np.isfinite(mat)]
    if ylim is not None:
        lo, hi = ylim
    else:
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))

    if isinstance(breaks, (int, np.integer)):
        bin_edges = np.linspace(lo, hi, int(breaks) + 1)
    else:
        bin_edges = np.asarray(breaks, dtype=float)

    n_bins = len(bin_edges) - 1

    freq_matrix = np.zeros((n_bins, n_cols), dtype=float)
    for j in range(n_cols):
        col_vals = mat[:, j][np.isfinite(mat[:, j])]
        if len(col_vals) > 0:
            counts, _ = np.histogram(col_vals, bins=bin_edges)
            freq_matrix[:, j] = counts

    # --- build colour palette -------------------------------------------
    palette = _resolve_palette(col, n=256)

    # --- build Heatmap ---------------------------------------------------
    bin_labels = [
        f"{bin_edges[i]:.2g}-{bin_edges[i + 1]:.2g}" for i in range(n_bins)
    ]

    hm_kwargs: Dict[str, Any] = dict(
        name=title if title else "frequency",
        show_column_names=show_column_names,
        cluster_columns=cluster_columns,
        clustering_distance_columns=clustering_distance_columns,
        clustering_method_columns=clustering_method_columns,
    )
    if callable(palette):
        hm_kwargs["col"] = palette
    hm_kwargs.update(kwargs)

    hm = Heatmap(
        freq_matrix,
        row_labels=bin_labels,
        column_labels=column_names,
        **hm_kwargs,
    )
    hm._frequency_bin_edges = bin_edges  # type: ignore[attr-defined]
    return hm
