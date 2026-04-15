"""SingleAnnotation class -- a single annotation track.

R source correspondence
-----------------------
``R/SingleAnnotation-class.R`` -- S4 class wrapping an AnnotationFunction
with a name, colour mapping, legend parameters, and display options.

All drawing uses ``grid_py`` (the Python port of R's ``grid`` package).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import grid_py

from .annotation_function import AnnotationFunction
from .annotation_functions import anno_simple
from .color_mapping import ColorMapping
from ._globals import ht_opt

__all__ = [
    "SingleAnnotation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_color_mapping(
    value: np.ndarray,
    col: Union[Dict[str, str], Callable[..., Any], ColorMapping, None],
    name: str,
    na_col: str,
) -> Optional[ColorMapping]:
    """Build a :class:`ColorMapping` from *col* and *value*.

    Parameters
    ----------
    value : numpy.ndarray
        Data values for the annotation.
    col : dict, callable, ColorMapping, or None
        User-supplied colour specification.
    name : str
        Name used when auto-generating a ``ColorMapping``.
    na_col : str
        Colour for missing / NA values.

    Returns
    -------
    ColorMapping or None
    """
    if col is None:
        return None
    if isinstance(col, ColorMapping):
        return col
    if isinstance(col, dict):
        return ColorMapping(name=name, colors=col, na_col=na_col)
    if callable(col):
        return ColorMapping(name=name, col_fun=col, na_col=na_col)
    return None


def _default_anno_size(which: str) -> float:
    """Return the default annotation size in mm from global options."""
    return float(ht_opt("simple_anno_size"))


# ---------------------------------------------------------------------------
# SingleAnnotation
# ---------------------------------------------------------------------------

class SingleAnnotation:
    """Single annotation track for a heatmap.

    Wraps either a simple colour bar (from *value* + *col*) or a custom
    :class:`~complexheatmap.annotation_function.AnnotationFunction`.

    Parameters
    ----------
    name : str
        Annotation name / identifier.
    value : array-like, optional
        Data values for a simple (colour-bar) annotation.
    col : dict or callable or ColorMapping, optional
        Colour mapping.
    fun : AnnotationFunction or callable, optional
        Custom annotation function.
    label : str, optional
        Display label (defaults to *name*).
    na_col : str
        Colour for NA / missing values.
    which : str
        ``"column"`` or ``"row"``.
    show_legend : bool
        Whether to include this annotation in the legend.
    gp : dict, optional
        Graphical parameters forwarded to the drawing function.
    border : bool
        Whether to draw a border around each cell.
    legend_param : dict, optional
        Additional legend customisation parameters.
    show_name : bool
        Whether to display the annotation name alongside the track.
    name_gp : dict, optional
        Text parameters for the annotation name.
    name_side : str, optional
        Side on which to draw the name.
    name_rot : float, optional
        Rotation angle for the annotation name text.
    width : object, optional
        Width of the annotation track.
    height : object, optional
        Height of the annotation track.
    """

    def __init__(
        self,
        name: str,
        value: Optional[Any] = None,
        col: Optional[Union[Dict[str, str], Callable[..., Any], ColorMapping]] = None,
        fun: Optional[Union[AnnotationFunction, Callable[..., Any]]] = None,
        label: Optional[str] = None,
        na_col: str = "grey",
        which: str = "column",
        show_legend: bool = True,
        gp: Optional[Dict[str, Any]] = None,
        border: bool = False,
        legend_param: Optional[Dict[str, Any]] = None,
        show_name: bool = True,
        name_gp: Optional[Dict[str, Any]] = None,
        name_side: Optional[str] = None,
        name_rot: Optional[float] = None,
        width: Optional[Any] = None,
        height: Optional[Any] = None,
    ) -> None:
        if which not in ("column", "row"):
            raise ValueError(f"`which` must be 'column' or 'row', got {which!r}")

        self.name: str = name
        self.label: str = label if label is not None else name
        self.na_col: str = na_col
        self.which: str = which
        self.show_legend: bool = show_legend
        self.gp: Dict[str, Any] = gp if gp is not None else {}
        self.border: bool = border
        self.legend_param: Dict[str, Any] = legend_param if legend_param is not None else {}
        self.show_name: bool = show_name
        self.name_gp: Dict[str, Any] = name_gp if name_gp is not None else {}
        self.name_rot: Optional[float] = name_rot
        self._color_mapping: Optional[ColorMapping] = None
        self._is_anno_matrix: bool = False

        # Resolve name_side defaults
        if name_side is not None:
            self.name_side = name_side
        else:
            self.name_side = "right" if which == "column" else "bottom"

        # ------------------------------------------------------------------
        # Build internal AnnotationFunction
        # ------------------------------------------------------------------
        if fun is not None:
            # Custom annotation function provided
            if isinstance(fun, AnnotationFunction):
                self._anno_fun: AnnotationFunction = fun
            elif callable(fun):
                self._anno_fun = AnnotationFunction(
                    fun=fun,
                    fun_name=name,
                    which=which,
                    n=len(value) if value is not None else None,
                    width=width,
                    height=height,
                )
            else:
                raise TypeError(
                    f"`fun` must be an AnnotationFunction or callable, got {type(fun)!r}"
                )
            self._value: Optional[np.ndarray] = (
                np.asarray(value) if value is not None else None
            )
        elif value is not None:
            # Simple value-based annotation
            self._value = np.asarray(value)
            self._color_mapping = _infer_color_mapping(
                self._value, col, name, na_col
            )
            self._is_anno_matrix = self._value.ndim == 2

            # Extract col for anno_simple
            col_arg = col
            if isinstance(col, ColorMapping):
                col_arg = col.color_map if col.is_discrete else col._col_fun

            self._anno_fun = anno_simple(
                x=self._value,
                col=col_arg,
                na_col=na_col,
                which=which,
                border=border,
                gp=gp,
                width=width,
                height=height,
            )
        else:
            raise ValueError(
                "Either `value` or `fun` must be provided to SingleAnnotation."
            )

        # Override dimensions if explicitly set
        if width is not None:
            self._anno_fun.width = width
        if height is not None:
            self._anno_fun.height = height

        # Apply default sizes when not set
        if self.which == "column" and self._anno_fun.height is None:
            self._anno_fun.height = _default_anno_size(which)
        if self.which == "row" and self._anno_fun.width is None:
            self._anno_fun.width = _default_anno_size(which)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def color_mapping(self) -> Optional[ColorMapping]:
        """The :class:`ColorMapping` for this annotation, or ``None``."""
        return self._color_mapping

    @property
    def is_anno_matrix(self) -> bool:
        """``True`` when the annotation value is a 2-D matrix."""
        return self._is_anno_matrix

    @property
    def nobs(self) -> Optional[int]:
        """Number of observations."""
        return self._anno_fun.nobs

    @property
    def width(self) -> Optional[Any]:
        """Width of the annotation track."""
        return self._anno_fun.width

    @width.setter
    def width(self, value: Optional[Any]) -> None:
        self._anno_fun.width = value

    @property
    def height(self) -> Optional[Any]:
        """Height of the annotation track."""
        return self._anno_fun.height

    @height.setter
    def height(self, value: Optional[Any]) -> None:
        self._anno_fun.height = value

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        index: Union[np.ndarray, Sequence[int]],
        k: int = 1,
        n: int = 1,
    ) -> None:
        """Draw the annotation track.

        Parameters
        ----------
        index : array-like of int
            Observation indices (0-based).
        k : int
            Current slice index (1-based).
        n : int
            Total number of slices.
        """
        self._anno_fun.draw(index, k=k, n=n)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, indices: Union[np.ndarray, Sequence[int]]) -> "SingleAnnotation":
        """Return a new :class:`SingleAnnotation` with subsetted data.

        Parameters
        ----------
        indices : array-like of int
            0-based observation indices to keep.

        Returns
        -------
        SingleAnnotation
        """
        new = copy.copy(self)
        new._anno_fun = self._anno_fun.subset(indices)
        if self._value is not None:
            idx = np.asarray(indices, dtype=int)
            if self._value.ndim == 2:
                new._value = self._value[idx, :]
            else:
                new._value = self._value[idx]
        return new

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SingleAnnotation(name={self.name!r}, which={self.which!r}, "
            f"nobs={self.nobs})"
        )
