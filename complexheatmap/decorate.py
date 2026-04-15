"""Decoration functions for post-draw heatmap modification.

After a :class:`~complexheatmap.heatmap_list.HeatmapList` has been drawn,
these functions look up named viewports from the global registry and call
a user-supplied drawing callback inside the appropriate viewport using
``grid_py.seek_viewport``.

This mirrors the ``decorate_*`` family of functions in the R
*ComplexHeatmap* package.

Examples
--------
>>> from complexheatmap.decorate import decorate_heatmap_body
>>> import grid_py
>>> def add_circle(vp_name):
...     grid_py.grid_circle(
...         x=grid_py.Unit(0.5, "npc"),
...         y=grid_py.Unit(0.5, "npc"),
...         r=grid_py.Unit(0.3, "npc"),
...         gp=grid_py.Gpar(fill="#FF000080"),
...     )
>>> decorate_heatmap_body("mat1", add_circle)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import grid_py

__all__ = [
    "decorate_heatmap_body",
    "decorate_annotation",
    "decorate_column_dend",
    "decorate_row_dend",
    "decorate_row_names",
    "decorate_column_names",
    "decorate_row_title",
    "decorate_column_title",
    "decorate_dimnames",
    "list_components",
]


# ---------------------------------------------------------------------------
# Internal registry access
# ---------------------------------------------------------------------------


def _get_registry() -> Dict[str, str]:
    """Retrieve the current component registry.

    Returns
    -------
    dict
        Mapping of component names to grid_py viewport names.
    """
    from .heatmap_list import _COMPONENT_REGISTRY
    return _COMPONENT_REGISTRY


def _lookup_component(name: str) -> str:
    """Look up a single component by exact name.

    Parameters
    ----------
    name : str
        The full component name.

    Returns
    -------
    str
        The grid_py viewport name.

    Raises
    ------
    KeyError
        If the component is not found.
    """
    reg = _get_registry()
    if name not in reg:
        available = ", ".join(sorted(reg.keys())) or "(none)"
        raise KeyError(
            f"Component {name!r} not found in the drawn heatmap. "
            f"Available: {available}"
        )
    return reg[name]


def _find_component(pattern: str) -> str:
    """Find a component by partial name match.

    Tries exact match first, then prefix/substring match.

    Parameters
    ----------
    pattern : str
        Partial or full component name.

    Returns
    -------
    str
        The grid_py viewport name.

    Raises
    ------
    KeyError
        If no match is found.
    ValueError
        If multiple ambiguous matches are found.
    """
    reg = _get_registry()

    # Exact match
    if pattern in reg:
        return reg[pattern]

    # Prefix / substring match
    matches = [k for k in reg if pattern in k]
    if len(matches) == 1:
        return reg[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous component pattern {pattern!r}: matches {matches}"
        )

    available = ", ".join(sorted(reg.keys())) or "(none)"
    raise KeyError(
        f"No component matching {pattern!r}. Available: {available}"
    )


def _seek_and_call(viewport_name: str, fun: Callable[..., Any]) -> None:
    """Navigate to a viewport, run the callback, then return.

    Parameters
    ----------
    viewport_name : str
        Name of the grid_py viewport to seek.
    fun : callable
        User callback.  Called with no arguments; drawing operations
        inside the callback target the sought viewport.
    """
    # Remember current viewport
    current_vp = grid_py.current_viewport()
    current_name = getattr(current_vp, "name", None) or "ROOT"

    try:
        grid_py.seek_viewport(viewport_name)
    except (LookupError, TypeError) as exc:
        raise KeyError(
            f"Cannot navigate to viewport {viewport_name!r}: {exc}"
        ) from exc

    try:
        fun()
    finally:
        # Navigate back
        if current_name not in (None, "ROOT"):
            grid_py.seek_viewport(current_name)


# ---------------------------------------------------------------------------
# Public decoration functions
# ---------------------------------------------------------------------------


def decorate_heatmap_body(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
    row_slice: Optional[int] = None,
    column_slice: int = 1,
    # Legacy aliases
    slice_row: Optional[int] = None,
    slice_col: Optional[int] = None,
) -> None:
    """Add graphics to a heatmap body after drawing.

    Mirrors R's ``decorate_heatmap_body(heatmap, code, slice, row_slice, column_slice)``.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap to decorate.
    fun : callable
        Function that is called with no arguments inside the target
        viewport.  Use ``grid_py`` drawing functions within the callback.
    slice : int
        Shorthand for *row_slice* (default 1).
    row_slice : int, optional
        Row slice index (1-based). Defaults to *slice*.
    column_slice : int
        Column slice index (1-based).
    slice_row : int, optional
        Legacy alias for *row_slice*.
    slice_col : int, optional
        Legacy alias for *column_slice*.

    Raises
    ------
    KeyError
        If the named heatmap body component is not found.
    """
    # Resolve aliases: R uses slice -> row_slice, column_slice
    if row_slice is None:
        row_slice = slice_row if slice_row is not None else slice
    if slice_col is not None:
        column_slice = slice_col
    component_name = f"heatmap_body_{heatmap}_{row_slice}_{column_slice}"
    vp_name = _find_component(component_name)
    _seek_and_call(vp_name, fun)


def decorate_annotation(
    annotation: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to an annotation track after drawing.

    Parameters
    ----------
    annotation : str
        Name of the annotation to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).

    Raises
    ------
    KeyError
        If the named annotation component is not found.
    """
    component_name = f"annotation_{annotation}_{slice}"
    try:
        vp_name = _find_component(component_name)
    except (KeyError, ValueError):
        vp_name = _find_component(f"annotation_{annotation}")
    _seek_and_call(vp_name, fun)


def decorate_column_dend(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a column dendrogram after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose column dendrogram to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    component_name = f"column_dend_{heatmap}_{slice}"
    vp_name = _find_component(component_name)
    _seek_and_call(vp_name, fun)


def decorate_row_dend(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a row dendrogram after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose row dendrogram to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    component_name = f"row_dend_{heatmap}_{slice}"
    vp_name = _find_component(component_name)
    _seek_and_call(vp_name, fun)


def decorate_row_names(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a row names area after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose row names to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    component_name = f"row_names_{heatmap}_{slice}"
    vp_name = _find_component(component_name)
    _seek_and_call(vp_name, fun)


def decorate_column_names(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a column names area after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose column names to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    component_name = f"column_names_{heatmap}_{slice}"
    vp_name = _find_component(component_name)
    _seek_and_call(vp_name, fun)


def decorate_row_title(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a row title area after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose row title to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    try:
        component_name = f"row_title_{heatmap}_{slice}"
        vp_name = _find_component(component_name)
    except (KeyError, ValueError):
        vp_name = _find_component("global_row_title")
    _seek_and_call(vp_name, fun)


def decorate_column_title(
    heatmap: str,
    fun: Callable[..., Any],
    slice: int = 1,
) -> None:
    """Add graphics to a column title area after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap whose column title to decorate.
    fun : callable
        Drawing callback (no arguments).
    slice : int
        Slice index (1-based).
    """
    try:
        component_name = f"column_title_{heatmap}_{slice}"
        vp_name = _find_component(component_name)
    except (KeyError, ValueError):
        vp_name = _find_component("global_column_title")
    _seek_and_call(vp_name, fun)


def decorate_dimnames(
    heatmap: str,
    fun: Callable[..., Any],
    which: str = "row",
    slice: int = 1,
) -> None:
    """Add graphics to row or column names area after drawing.

    Parameters
    ----------
    heatmap : str
        Name of the heatmap.
    fun : callable
        Drawing callback (no arguments).
    which : str
        ``"row"`` or ``"column"``.
    slice : int
        Slice index (1-based).

    Raises
    ------
    ValueError
        If ``which`` is not ``"row"`` or ``"column"``.
    """
    if which == "row":
        decorate_row_names(heatmap, fun, slice)
    elif which == "column":
        decorate_column_names(heatmap, fun, slice)
    else:
        raise ValueError(f"`which` must be 'row' or 'column', got {which!r}")


def list_components(ht_list: Optional[Any] = None) -> List[str]:
    """List all viewport/component names in the last drawn heatmap.

    Parameters
    ----------
    ht_list : HeatmapList, optional
        Unused; retained for API compatibility with R.  The global
        registry from the most recent ``draw()`` call is always used.

    Returns
    -------
    list of str
        Sorted list of component names.
    """
    return sorted(_get_registry().keys())
