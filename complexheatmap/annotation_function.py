"""Core :class:`AnnotationFunction` class.

R source correspondence
-----------------------
``R/AnnotationFunction-class.R`` -- S4 class definition and constructor.

An ``AnnotationFunction`` wraps a user- or library-supplied drawing
callback so it can be used as a heatmap annotation in the same way as
the R *ComplexHeatmap* ``AnnotationFunction`` S4 class.

All drawing is performed via ``grid_py`` (the Python port of R's
``grid`` package).

Examples
--------
>>> import numpy as np
>>> from complexheatmap.annotation_function import AnnotationFunction
>>> def my_draw(index, k, n):
...     import grid_py
...     grid_py.grid_rect()
>>> af = AnnotationFunction(fun=my_draw, fun_name="my_anno", which="column", n=10)
>>> af.nobs
10
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import grid_py

__all__ = [
    "AnnotationFunction",
]


class AnnotationFunction:
    """Wraps a drawing function for use as a heatmap annotation.

    Parameters
    ----------
    fun : callable
        Drawing function with signature ``fun(index, k, n)`` where
        *index* is an array of observation indices, *k* is the current
        slice index (1-based), and *n* is the total number of slices.
        The function should use ``grid_py`` viewport+grob operations
        for rendering.
    fun_name : str
        Human-readable name of the annotation function
        (e.g. ``"anno_barplot"``).
    which : str
        ``"column"`` or ``"row"`` -- orientation of the annotation
        relative to the heatmap body.
    var_env : dict, optional
        Dictionary of variables captured by *fun* (data arrays, graphic
        parameters, etc.).  Used for subsetting.
    n : int, optional
        Number of observations the annotation spans.
    data_scale : tuple of float
        ``(min, max)`` range for the data axis.  Defaults to ``(0, 1)``.
    subsettable : bool
        Whether the annotation supports subsetting.
    subset_rule : dict
        Per-key rules for subsetting variables in *var_env*.  Each value
        is one of ``"array"`` (subset by index), ``"matrix_row"``,
        ``"matrix_col"``, or ``None`` (leave unchanged).
    show_name : bool
        Whether to display the annotation name alongside the drawing.
    width : object, optional
        Width of the annotation region.  Can be a ``grid_py.Unit`` or
        a numeric value in mm.
    height : object, optional
        Height of the annotation region.  Can be a ``grid_py.Unit`` or
        a numeric value in mm.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        fun: Callable[..., Any],
        fun_name: str = "",
        which: str = "column",
        var_env: Optional[Dict[str, Any]] = None,
        n: Optional[int] = None,
        data_scale: Tuple[float, float] = (0.0, 1.0),
        subsettable: bool = False,
        subset_rule: Optional[Dict[str, Optional[str]]] = None,
        show_name: bool = True,
        width: Optional[Any] = None,
        height: Optional[Any] = None,
    ) -> None:
        if which not in ("column", "row"):
            raise ValueError(f"`which` must be 'column' or 'row', got {which!r}")

        self.fun: Callable[..., Any] = fun
        self.fun_name: str = fun_name
        self.which: str = which
        self.var_env: Dict[str, Any] = var_env if var_env is not None else {}
        self._n: Optional[int] = n
        self.data_scale: Tuple[float, float] = tuple(data_scale)  # type: ignore[assignment]
        self.subsettable: bool = subsettable
        self.subset_rule: Dict[str, Optional[str]] = (
            subset_rule if subset_rule is not None else {}
        )
        self.show_name: bool = show_name

        # Normalise width/height: accept numeric (mm) or grid_py.Unit
        self._width: Optional[Any] = _ensure_unit(width, "mm") if width is not None else None
        self._height: Optional[Any] = _ensure_unit(height, "mm") if height is not None else None

        # Extended space around the annotation (bottom, left, top, right)
        self.extended: Any = grid_py.Unit([0, 0, 0, 0], "mm")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nobs(self) -> Optional[int]:
        """Number of observations, or ``None`` if unknown."""
        return self._n

    @nobs.setter
    def nobs(self, value: Optional[int]) -> None:
        self._n = value

    @property
    def width(self) -> Optional[Any]:
        """Width of the annotation region."""
        return self._width

    @width.setter
    def width(self, value: Optional[Any]) -> None:
        self._width = _ensure_unit(value, "mm") if value is not None else None

    @property
    def height(self) -> Optional[Any]:
        """Height of the annotation region."""
        return self._height

    @height.setter
    def height(self, value: Optional[Any]) -> None:
        self._height = _ensure_unit(value, "mm") if value is not None else None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        index: Union[np.ndarray, Sequence[int]],
        k: int = 1,
        n: int = 1,
    ) -> None:
        """Invoke the wrapped drawing function.

        Parameters
        ----------
        index : array-like of int
            Indices of observations to draw (0-based).
        k : int
            Current slice index (1-based).
        n : int
            Total number of slices.
        """
        index = np.asarray(index, dtype=int)
        self.fun(index, k, n)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, indices: Union[np.ndarray, Sequence[int]]) -> "AnnotationFunction":
        """Return a new :class:`AnnotationFunction` with subsetted data.

        Parameters
        ----------
        indices : array-like of int
            0-based observation indices to keep.

        Returns
        -------
        AnnotationFunction
            A deep copy with *var_env* entries subsetted according to
            *subset_rule* and *n* updated.

        Raises
        ------
        RuntimeError
            If the annotation is not subsettable.
        """
        if not self.subsettable:
            raise RuntimeError(
                f"AnnotationFunction '{self.fun_name}' is not subsettable."
            )

        indices = np.asarray(indices, dtype=int)
        new = self.copy()
        new._n = len(indices)

        for key, rule in self.subset_rule.items():
            if key not in new.var_env:
                continue
            val = new.var_env[key]
            if rule == "array":
                new.var_env[key] = np.asarray(val)[indices]
            elif rule == "matrix_row":
                new.var_env[key] = np.asarray(val)[indices, :]
            elif rule == "matrix_col":
                new.var_env[key] = np.asarray(val)[:, indices]
            # rule is None -> leave unchanged

        return new

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def copy(self) -> "AnnotationFunction":
        """Return a deep copy of this annotation function.

        Returns
        -------
        AnnotationFunction
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        name = self.fun_name or "<anonymous>"
        return (
            f"AnnotationFunction(fun_name={name!r}, which={self.which!r}, "
            f"n={self._n}, data_scale={self.data_scale})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_unit(value: Any, default_unit: str = "mm") -> Any:
    """Return *value* as a ``grid_py.Unit`` if it is not already one.

    Numeric values are interpreted as being in *default_unit*.
    """
    if isinstance(value, grid_py.Unit):
        return value
    try:
        return grid_py.Unit(float(value), default_unit)
    except (TypeError, ValueError):
        return value
