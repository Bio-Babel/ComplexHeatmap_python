"""Color mapping from data values to colors.

R source correspondence
-----------------------
``R/ColorMapping-class.R`` -- the ``ColorMapping`` S4 class that maps
discrete (categorical) or continuous (numeric) data values to colors.

This module provides the :class:`ColorMapping` class, the Python
equivalent of the R ``ColorMapping`` S4 class in *ComplexHeatmap*.
Uses :func:`~complexheatmap._color.color_ramp2` for continuous
color mapping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["ColorMapping"]

# Module-level counter for auto-generated names.
_auto_id: int = 0


def _next_name() -> str:
    """Return an auto-generated color-mapping name."""
    global _auto_id
    _auto_id += 1
    return f"color_mapping_{_auto_id}"


def _is_nan(value: Any) -> bool:
    """Check whether *value* is NaN-like (works for floats, numpy scalars, None)."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


class ColorMapping:
    """Maps data values to colors for heatmap rendering.

    Supports both **discrete** (categorical) and **continuous** (numeric)
    color mapping.

    Parameters
    ----------
    name : str, optional
        Human-readable name for this mapping (shown in legends).  An
        auto-generated name is used when *None*.
    colors : dict or list, optional
        For discrete mapping.  If a *dict*, keys are category levels and
        values are color strings.  If a *list* of color strings, *levels*
        must also be provided.
    levels : list, optional
        Explicit category levels.  Required when *colors* is a list rather
        than a dict.
    col_fun : callable, optional
        For continuous mapping.  A function that accepts a numeric value (or
        array) and returns one (or an array of) color strings.  Typically
        produced by :func:`color_ramp2`.
    breaks : array-like, optional
        Breakpoints for continuous mapping.  When *col_fun* was created by
        ``color_ramp2`` the breaks are read from ``col_fun.breaks``
        automatically.
    na_col : str
        Color used for ``NaN`` / ``None`` values.  Default ``"#FFFFFF"``.

    Raises
    ------
    ValueError
        If neither *colors* nor *col_fun* is provided, or if *colors* is a
        list but *levels* is missing / length-mismatched.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: Optional[str] = None,
        colors: Optional[Union[Dict[str, str], List[str]]] = None,
        levels: Optional[List[str]] = None,
        col_fun: Optional[Callable[..., Any]] = None,
        breaks: Optional[Sequence[float]] = None,
        na_col: str = "#FFFFFF",
    ) -> None:
        if colors is None and col_fun is None:
            raise ValueError(
                "Either 'colors' (for discrete mapping) or 'col_fun' "
                "(for continuous mapping) must be provided."
            )

        self.name: str = name if name is not None else _next_name()
        self.na_col: str = na_col

        # --- Discrete mode ---
        if colors is not None:
            self._type = "discrete"
            self._col_fun: Optional[Callable[..., Any]] = None
            self._breaks: Optional[np.ndarray] = None

            if isinstance(colors, dict):
                # Normalize keys to strings for consistent lookup
                self._levels: List[str] = [str(k) for k in colors.keys()]
                self._color_map: Dict[str, str] = {
                    str(k): v for k, v in colors.items()
                }
            else:
                # colors is a list; levels is mandatory
                if levels is None:
                    raise ValueError(
                        "'levels' must be provided when 'colors' is a list."
                    )
                if len(levels) != len(colors):
                    raise ValueError(
                        f"Length of 'levels' ({len(levels)}) must equal "
                        f"length of 'colors' ({len(colors)})."
                    )
                self._levels = [str(lv) for lv in levels]
                self._color_map = {str(lv): c for lv, c in zip(levels, colors)}

        # --- Continuous mode ---
        else:
            assert col_fun is not None  # ensured by guard above
            self._type = "continuous"
            self._col_fun = col_fun
            self._levels = []
            self._color_map = {}

            if breaks is not None:
                self._breaks = np.asarray(breaks, dtype=float)
            elif hasattr(col_fun, "breaks"):
                self._breaks = np.asarray(col_fun.breaks, dtype=float)
            else:
                self._breaks = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_discrete(self) -> bool:
        """``True`` when the mapping is discrete (categorical)."""
        return self._type == "discrete"

    @property
    def is_continuous(self) -> bool:
        """``True`` when the mapping is continuous (numeric)."""
        return self._type == "continuous"

    @property
    def levels(self) -> List[str]:
        """Category levels (empty list for continuous mappings)."""
        return list(self._levels)

    @property
    def breaks(self) -> Optional[np.ndarray]:
        """Breakpoints for continuous mapping, or ``None``."""
        return self._breaks.copy() if self._breaks is not None else None

    @property
    def color_map(self) -> Dict[str, str]:
        """Level-to-color dict (empty for continuous mappings)."""
        return dict(self._color_map)

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------

    def map_to_colors(
        self, x: Any
    ) -> Union[str, List[str], np.ndarray]:
        """Map data values to color strings.

        Parameters
        ----------
        x : scalar, list, numpy.ndarray, or pandas.Series
            Values to map.

        Returns
        -------
        str or numpy.ndarray
            A single color string when *x* is scalar, or a numpy array of
            color strings with the same shape as the input otherwise.

        Notes
        -----
        ``NaN`` and ``None`` values are mapped to :attr:`na_col`.
        """
        # --- Unwrap pandas Series ---
        try:
            import pandas as pd

            if isinstance(x, pd.Series):
                x = x.to_numpy()
        except ImportError:
            pass

        scalar_input = np.ndim(x) == 0 and not isinstance(x, np.ndarray)

        arr = np.asarray(x, dtype=object) if self.is_discrete else np.asarray(x)
        flat = arr.ravel()

        if self.is_discrete:
            out = np.array(
                [self._map_discrete_scalar(v) for v in flat], dtype=object
            )
        else:
            assert self._col_fun is not None
            out = np.array(
                [self._map_continuous_scalar(v) for v in flat], dtype=object
            )

        if scalar_input:
            return str(out.flat[0])

        return out.reshape(arr.shape)

    # --- private helpers ------------------------------------------------

    def _map_discrete_scalar(self, value: Any) -> str:
        if _is_nan(value):
            return self.na_col
        key = str(value)
        result = self._color_map.get(key)
        if result is not None:
            return result
        # For float values that are whole numbers, try int form
        # e.g., matrix value 1.0 should match dict key "1"
        try:
            fval = float(value)
            if fval == int(fval):
                result = self._color_map.get(str(int(fval)))
                if result is not None:
                    return result
        except (TypeError, ValueError, OverflowError):
            pass
        return self.na_col

    def _map_continuous_scalar(self, value: Any) -> str:
        assert self._col_fun is not None
        if _is_nan(value):
            return self.na_col
        try:
            result = self._col_fun(float(value))
        except (TypeError, ValueError):
            return self.na_col
        # col_fun may return a single string or a 1-element array/list.
        if isinstance(result, (list, np.ndarray)):
            return str(result[0])
        return str(result)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge(*mappings: "ColorMapping") -> "ColorMapping":
        """Merge multiple discrete color mappings.

        Levels and their associated colors are concatenated.  If the same
        level appears in more than one mapping, only the first occurrence is
        kept.

        Parameters
        ----------
        *mappings : ColorMapping
            Two or more discrete ``ColorMapping`` instances.

        Returns
        -------
        ColorMapping
            A new discrete mapping containing the union of levels.

        Raises
        ------
        ValueError
            If any mapping is not discrete.
        """
        if not mappings:
            raise ValueError("At least one ColorMapping must be provided.")

        merged_colors: Dict[str, str] = {}
        names: List[str] = []
        na_col: str = mappings[0].na_col

        for m in mappings:
            if not m.is_discrete:
                raise ValueError(
                    f"Cannot merge continuous mapping '{m.name}'. "
                    "Only discrete mappings can be merged."
                )
            names.append(m.name)
            for level in m._levels:
                if level not in merged_colors:
                    merged_colors[level] = m._color_map[level]

        merged_name = "+".join(names)
        return ColorMapping(name=merged_name, colors=merged_colors, na_col=na_col)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_discrete:
            return (
                f"ColorMapping(name={self.name!r}, type='discrete', "
                f"n_levels={len(self._levels)})"
            )
        n_breaks = len(self._breaks) if self._breaks is not None else 0
        return (
            f"ColorMapping(name={self.name!r}, type='continuous', "
            f"n_breaks={n_breaks})"
        )
