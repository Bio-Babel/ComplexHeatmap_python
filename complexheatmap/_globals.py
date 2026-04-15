"""Global options system for ComplexHeatmap.

R source correspondence
-----------------------
``R/global.R`` -- ``ht_opt`` / ``ht_global_opt`` / ``reset`` functions.

Provides a dict-based option management system analogous to the R
``GlobalOptions`` / ``ht_opt`` interface.  Options can be queried, set,
and temporarily overridden via a context manager.

Examples
--------
>>> from complexheatmap._globals import ht_opt
>>> ht_opt("verbose")          # get a single option
False
>>> ht_opt(verbose=True)       # set option(s)
>>> with ht_opt(verbose=True): # temporary override
...     print(ht_opt("verbose"))
True
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

__all__ = ["ht_opt", "reset_ht_opt"]

# ---------------------------------------------------------------------------
# Default values  (mirrors R ComplexHeatmap::ht_opt defaults)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # graphical-parameter dicts (gpar-like)
    "heatmap_row_names_gp": {},
    "heatmap_column_names_gp": {},
    "heatmap_row_title_gp": {},
    "heatmap_column_title_gp": {},
    "heatmap_border": None,
    # legend
    "legend_title_gp": {"fontsize": 10, "fontweight": "bold"},
    "legend_labels_gp": {"fontsize": 10},
    "legend_grid_width": 4,       # mm
    "legend_grid_height": 4,      # mm
    "legend_border": None,
    "legend_gap": 2,              # mm
    "merge_legends": False,
    # annotation
    "annotation_border": None,
    "simple_anno_size": 5,        # mm
    # padding (mm)
    "DENDROGRAM_PADDING": 0.5,
    "DIMNAME_PADDING": 1,
    "TITLE_PADDING": 2.5,
    "COLUMN_ANNO_PADDING": 0.5,
    "ROW_ANNO_PADDING": 0.5,
    "HEATMAP_LEGEND_PADDING": 2,
    "ANNOTATION_LEGEND_PADDING": 2,
    # misc
    "fast_hclust": False,
    "show_parent_dend_line": True,
    "verbose": False,
    "COLOR": ["blue", "#EEEEEE", "red"],
}

_OPTIONS: Dict[str, Any] = copy.deepcopy(_DEFAULTS)


# ---------------------------------------------------------------------------
# Context-manager helper
# ---------------------------------------------------------------------------

class _CtxHelper:
    """Returned by ``ht_opt(**kw)`` to support ``with`` usage.

    Options are set immediately when ``ht_opt`` is called (so bare-statement
    usage works).  If the caller enters the ``with`` block, the original
    values are restored on ``__exit__``.  If not used as a context manager,
    the changes persist.
    """

    def __init__(self, saved: Dict[str, Any]) -> None:
        self._saved = saved

    def __enter__(self) -> "_CtxHelper":
        return self

    def __exit__(self, *exc: Any) -> None:
        _OPTIONS.update(self._saved)

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<ht_opt context>"


# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------

class _HtOpt:
    """Callable + context-manager facade for heatmap global options."""

    def __call__(
        self,
        __key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Query or set global heatmap options.

        Parameters
        ----------
        __key : str, optional
            If provided (and no *kwargs*), return the value of this option.
        **kwargs
            Option-name / value pairs to set.

        Returns
        -------
        Any
            * No arguments -- deep copy of the full options dict.
            * Single positional string -- value of that option.
            * Keyword arguments -- a ``_CtxHelper`` that can be used with
              ``with`` to restore original values on exit.

        Raises
        ------
        KeyError
            If *__key* or any key in *kwargs* is not a recognised option.
        TypeError
            If positional key and keyword arguments are mixed.
        """
        if __key is not None and kwargs:
            raise TypeError(
                "Cannot mix positional key lookup with keyword setting"
            )

        # --- get single option ---
        if __key is not None:
            if __key not in _OPTIONS:
                raise KeyError(f"Unknown option: {__key!r}")
            return copy.deepcopy(_OPTIONS[__key])

        # --- return all options ---
        if not kwargs:
            return copy.deepcopy(_OPTIONS)

        # --- set options ---
        for k in kwargs:
            if k not in _OPTIONS:
                raise KeyError(f"Unknown option: {k!r}")

        saved = {k: copy.deepcopy(_OPTIONS[k]) for k in kwargs}
        _OPTIONS.update(kwargs)
        return _CtxHelper(saved)

    def __repr__(self) -> str:
        return f"ht_opt({_OPTIONS!r})"


ht_opt = _HtOpt()


def reset_ht_opt() -> None:
    """Reset all options to their defaults.

    Returns
    -------
    None
    """
    _OPTIONS.clear()
    _OPTIONS.update(copy.deepcopy(_DEFAULTS))
