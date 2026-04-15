"""HeatmapAnnotation class and convenience constructors.

R source correspondence
-----------------------
``R/HeatmapAnnotation-class.R`` -- S4 class composing multiple
``SingleAnnotation`` objects into one annotation group that can be
attached to a heatmap.

All drawing uses ``grid_py`` (the Python port of R's ``grid`` package).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import grid_py

from .annotation_function import AnnotationFunction
from .single_annotation import SingleAnnotation
from .color_mapping import ColorMapping
from ._globals import ht_opt

__all__ = [
    "HeatmapAnnotation",
    "rowAnnotation",
    "columnAnnotation",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_absolute_unit(u: Any) -> bool:
    """Check if *u* is an absolute grid_py.Unit (mm, cm, inches, points)."""
    if isinstance(u, grid_py.Unit):
        return u.is_absolute()
    # Numeric values are treated as mm (absolute)
    try:
        float(u)
        return True
    except (TypeError, ValueError):
        return False


def _unit_to_mm(u: Any) -> float:
    """Extract a numeric mm value from a unit-like object."""
    if isinstance(u, grid_py.Unit):
        # For absolute units, values[0] is the numeric amount
        # Common absolute units: mm, cm, inches, points
        vals = u.values
        units = u.units_list
        if units and units[0] == "mm":
            return float(vals[0])
        elif units and units[0] == "cm":
            return float(vals[0]) * 10.0
        elif units and units[0] in ("inches", "in"):
            return float(vals[0]) * 25.4
        elif units and units[0] in ("points", "pt"):
            return float(vals[0]) * 25.4 / 72.0
        # Fallback: return raw value
        return float(vals[0])
    return float(u)


# ---------------------------------------------------------------------------
# HeatmapAnnotation
# ---------------------------------------------------------------------------

class HeatmapAnnotation:
    """Collection of annotation tracks for a heatmap.

    Each named keyword argument (or column in *df*) becomes a
    :class:`SingleAnnotation`.  The ``+`` operator is supported for
    combining with :class:`Heatmap` objects.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        DataFrame whose columns are turned into simple annotations.
    name : str, optional
        Name for this annotation group.
    col : dict, optional
        Dict mapping annotation names to colour specifications.
    na_col : str
        Colour for NA / missing values.
    which : str
        ``"column"`` or ``"row"``.
    gp : dict, optional
        Shared graphical parameters for all annotations.
    border : bool
        Whether to draw borders.
    gap : float
        Gap between annotation tracks in mm.
    show_annotation_name : bool or list of bool
        Whether to show annotation names.
    annotation_name_gp : dict, optional
        Text parameters for annotation names.
    annotation_name_side : str, optional
        Side for names.
    annotation_height : list of float, optional
        Per-annotation heights in mm (column annotations).
    annotation_width : list of float, optional
        Per-annotation widths in mm (row annotations).
    height : float, optional
        Total height in mm (column annotations).
    width : float, optional
        Total width in mm (row annotations).
    show_legend : bool or list of bool
        Whether to show legends for each annotation.
    **annotations
        Named annotation arguments.
    """

    def __init__(
        self,
        *,
        df: Optional[Any] = None,
        name: Optional[str] = None,
        col: Optional[Dict[str, Any]] = None,
        na_col: str = "grey",
        which: str = "column",
        gp: Optional[Dict[str, Any]] = None,
        border: bool = False,
        gap: float = 1.0,
        show_annotation_name: Union[bool, List[bool]] = True,
        annotation_name_gp: Optional[Dict[str, Any]] = None,
        annotation_name_side: Optional[str] = None,
        annotation_height: Optional[List[float]] = None,
        annotation_width: Optional[List[float]] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        show_legend: Union[bool, List[bool]] = True,
        **annotations: Any,
    ) -> None:
        if which not in ("column", "row"):
            raise ValueError(f"`which` must be 'column' or 'row', got {which!r}")

        self._which: str = which
        self._name: str = name if name is not None else (
            "column_annotation" if which == "column" else "row_annotation"
        )
        self.na_col: str = na_col
        self.gp: Dict[str, Any] = gp if gp is not None else {}
        self.border: bool = border
        self.gap: float = gap
        self.annotation_name_gp: Dict[str, Any] = (
            annotation_name_gp if annotation_name_gp is not None else {}
        )
        self.annotation_name_side: str = (
            annotation_name_side or ("right" if which == "column" else "bottom")
        )
        self._height: Optional[float] = height
        self._width: Optional[float] = width

        col_map: Dict[str, Any] = col if col is not None else {}

        # Extract annotation_legend_param before processing
        self.annotation_legend_param: Dict[str, Any] = annotations.pop(
            "annotation_legend_param", {}
        )

        # ------------------------------------------------------------------
        # Collect annotation sources in insertion order
        # ------------------------------------------------------------------
        anno_sources: OrderedDict[str, Any] = OrderedDict()

        # 1. DataFrame columns
        if df is not None:
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"`df` must be a pandas DataFrame, got {type(df)!r}")
            for c in df.columns:
                anno_sources[str(c)] = df[c].values

        # 2. Keyword annotations (override df columns with same name)
        for k, v in annotations.items():
            anno_sources[k] = v

        n_annos = len(anno_sources)

        # Broadcast scalar show_annotation_name / show_legend
        if isinstance(show_annotation_name, bool):
            show_name_list = [show_annotation_name] * n_annos
        else:
            show_name_list = list(show_annotation_name)

        if isinstance(show_legend, bool):
            show_legend_list = [show_legend] * n_annos
        else:
            show_legend_list = list(show_legend)

        # Per-annotation sizes
        ann_heights = annotation_height or [None] * n_annos  # type: ignore[list-item]
        ann_widths = annotation_width or [None] * n_annos  # type: ignore[list-item]

        # ------------------------------------------------------------------
        # Build SingleAnnotation objects
        # ------------------------------------------------------------------
        self.anno_list: OrderedDict[str, SingleAnnotation] = OrderedDict()

        for idx, (anno_name, anno_val) in enumerate(anno_sources.items()):
            ann_col = col_map.get(anno_name)

            s_height = ann_heights[idx] if idx < len(ann_heights) else None
            s_width = ann_widths[idx] if idx < len(ann_widths) else None
            s_show_name = show_name_list[idx] if idx < len(show_name_list) else True
            s_show_legend = show_legend_list[idx] if idx < len(show_legend_list) else True

            if isinstance(anno_val, AnnotationFunction):
                sa = SingleAnnotation(
                    name=anno_name,
                    fun=anno_val,
                    which=which,
                    na_col=na_col,
                    gp=gp,
                    border=border,
                    show_name=s_show_name,
                    show_legend=s_show_legend,
                    name_gp=self.annotation_name_gp,
                    name_side=self.annotation_name_side,
                    width=s_width,
                    height=s_height,
                )
            elif isinstance(anno_val, SingleAnnotation):
                sa = anno_val
                sa.name = anno_name
                sa.which = which
            else:
                # Treat as simple value-based annotation
                sa = SingleAnnotation(
                    name=anno_name,
                    value=anno_val,
                    col=ann_col,
                    which=which,
                    na_col=na_col,
                    gp=gp,
                    border=border,
                    show_name=s_show_name,
                    show_legend=s_show_legend,
                    name_gp=self.annotation_name_gp,
                    name_side=self.annotation_name_side,
                    width=s_width,
                    height=s_height,
                )

            self.anno_list[anno_name] = sa

        # ------------------------------------------------------------------
        # Distribute total height / width across annotations
        # ------------------------------------------------------------------
        if which == "column" and height is not None and n_annos > 0:
            total_gap = self.gap * max(n_annos - 1, 0)
            per_anno = (height - total_gap) / n_annos
            for sa in self.anno_list.values():
                if sa.height is None:
                    sa.height = per_anno

        if which == "row" and width is not None and n_annos > 0:
            total_gap = self.gap * max(n_annos - 1, 0)
            per_anno = (width - total_gap) / n_annos
            for sa in self.anno_list.values():
                if sa.width is None:
                    sa.width = per_anno

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def which(self) -> str:
        """Orientation: ``"column"`` or ``"row"``."""
        return self._which

    @property
    def names(self) -> List[str]:
        """Ordered list of annotation names."""
        return list(self.anno_list.keys())

    @property
    def nobs(self) -> Optional[int]:
        """Number of observations (from the first annotation with known nobs)."""
        for sa in self.anno_list.values():
            if sa.nobs is not None:
                return sa.nobs
        return None

    @property
    def width(self) -> Optional[Any]:
        """Total width.

        For row annotations this is the sum of annotation widths plus gaps.
        For column annotations, returns the stored value.
        """
        if self._width is not None:
            return self._width
        if self._which == "row" and self.anno_list:
            widths = [sa.width for sa in self.anno_list.values() if sa.width is not None]
            abs_widths = [w for w in widths if _is_absolute_unit(w)]
            if abs_widths:
                total = sum(_unit_to_mm(w) for w in abs_widths)
                return total + self.gap * max(len(abs_widths) - 1, 0)
        return self._width

    @width.setter
    def width(self, value: Optional[Any]) -> None:
        self._width = value

    @property
    def height(self) -> Optional[Any]:
        """Total height.

        For column annotations this is the sum of annotation heights plus gaps.
        For row annotations, returns the stored value.
        """
        if self._height is not None:
            return self._height
        if self._which == "column" and self.anno_list:
            heights = [sa.height for sa in self.anno_list.values() if sa.height is not None]
            abs_heights = [h for h in heights if _is_absolute_unit(h)]
            if abs_heights:
                total = sum(_unit_to_mm(h) for h in abs_heights)
                return total + self.gap * max(len(abs_heights) - 1, 0)
        return self._height

    @height.setter
    def height(self, value: Optional[Any]) -> None:
        self._height = value

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        index: Union[np.ndarray, Sequence[int]],
        k: int = 1,
        n: int = 1,
    ) -> None:
        """Draw all annotation tracks using grid_py viewports.

        Parameters
        ----------
        index : array-like of int
            Observation indices (0-based).
        k : int
            Current slice index (1-based).
        n : int
            Total number of slices.
        """
        anno_items = list(self.anno_list.values())
        n_annos = len(anno_items)
        if n_annos == 0:
            return

        for i, sa in enumerate(anno_items):
            # Create a viewport for each annotation track
            if self._which == "column":
                # Stack vertically
                grid_py.push_viewport(grid_py.Viewport(
                    x=grid_py.Unit(0, "npc"),
                    y=grid_py.Unit(1 - (i + 1) / n_annos, "npc"),
                    width=grid_py.Unit(1, "npc"),
                    height=grid_py.Unit(1.0 / n_annos, "npc"),
                    just=["left", "bottom"],
                ))
            else:
                # Stack horizontally
                grid_py.push_viewport(grid_py.Viewport(
                    x=grid_py.Unit(i / n_annos, "npc"),
                    y=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(1.0 / n_annos, "npc"),
                    height=grid_py.Unit(1, "npc"),
                    just=["left", "bottom"],
                ))

            sa.draw(index, k=k, n=n)
            grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, indices: Union[np.ndarray, Sequence[int]]) -> "HeatmapAnnotation":
        """Return a new :class:`HeatmapAnnotation` with subsetted data.

        Parameters
        ----------
        indices : array-like of int
            0-based observation indices to keep.

        Returns
        -------
        HeatmapAnnotation
        """
        new = copy.copy(self)
        new.anno_list = OrderedDict()
        for anno_name, sa in self.anno_list.items():
            new.anno_list[anno_name] = sa.subset(indices)
        return new

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.anno_list)

    def __getitem__(self, key: str) -> SingleAnnotation:
        return self.anno_list[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.anno_list)

    def __contains__(self, key: str) -> bool:
        return key in self.anno_list

    # ------------------------------------------------------------------
    # Addition (for combining with Heatmap)
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> Any:
        """Support ``HeatmapAnnotation + Heatmap`` via AdditivUnit protocol."""
        if hasattr(other, "__radd__"):
            return NotImplemented
        return NotImplemented

    def __radd__(self, other: Any) -> Any:
        if other == 0:
            return self
        return NotImplemented

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names = ", ".join(self.anno_list.keys())
        return (
            f"HeatmapAnnotation(which={self._which!r}, "
            f"n_annotations={len(self.anno_list)}, names=[{names}])"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

_CONSTRUCTOR_KEYS = frozenset({
    "df", "name", "col", "na_col", "gp", "border", "gap",
    "show_annotation_name", "annotation_name_gp", "annotation_name_side",
    "annotation_height", "annotation_width", "height", "width",
    "show_legend",
})


def rowAnnotation(**kwargs: Any) -> HeatmapAnnotation:
    """Create a row annotation.

    Shorthand for ``HeatmapAnnotation(which='row', **kwargs)``.

    Parameters
    ----------
    **kwargs
        Passed to :class:`HeatmapAnnotation`.

    Returns
    -------
    HeatmapAnnotation
    """
    ctor_kwargs: Dict[str, Any] = {}
    anno_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _CONSTRUCTOR_KEYS:
            ctor_kwargs[k] = v
        else:
            anno_kwargs[k] = v
    return HeatmapAnnotation(which="row", **ctor_kwargs, **anno_kwargs)


def columnAnnotation(**kwargs: Any) -> HeatmapAnnotation:
    """Create a column annotation.

    Shorthand for ``HeatmapAnnotation(which='column', **kwargs)``.

    Parameters
    ----------
    **kwargs
        Passed to :class:`HeatmapAnnotation`.

    Returns
    -------
    HeatmapAnnotation
    """
    ctor_kwargs: Dict[str, Any] = {}
    anno_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _CONSTRUCTOR_KEYS:
            ctor_kwargs[k] = v
        else:
            anno_kwargs[k] = v
    return HeatmapAnnotation(which="column", **ctor_kwargs, **anno_kwargs)
