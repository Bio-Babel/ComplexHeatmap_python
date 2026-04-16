"""HeatmapList: container for combined heatmap visualisation.

A :class:`HeatmapList` holds one or more ``Heatmap`` or
``HeatmapAnnotation`` objects and renders them together in a shared
figure, optionally sharing row/column orderings and legends.

This is the Python equivalent of the R ``ComplexHeatmap::HeatmapList``
S4 class.  It uses ``grid_py`` for all rendering.

Examples
--------
>>> import numpy as np
>>> from complexheatmap import Heatmap
>>> ht1 = Heatmap(np.random.randn(10, 8), name="mat1")
>>> ht2 = Heatmap(np.random.randn(10, 6), name="mat2")
>>> ht_list = ht1 + ht2
>>> ht_list.draw()
"""

from __future__ import annotations

import copy
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import grid_py

from .legends import Legend, Legends, pack_legend
from .color_mapping import ColorMapping
from ._globals import ht_opt

__all__ = [
    "HeatmapList",
]

# ---------------------------------------------------------------------------
# Global registry for drawn viewports (used by decorate_* functions)
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: Dict[str, str] = {}
"""Maps component name -> grid_py viewport name for the last drawn HeatmapList."""


def _register_component(name: str, viewport_name: str) -> None:
    """Register a drawn component's viewport for later decoration.

    Parameters
    ----------
    name : str
        Unique component name (e.g. ``"heatmap_body_mat1_1_1"``).
    viewport_name : str
        The grid_py viewport name associated with this component.
    """
    _COMPONENT_REGISTRY[name] = viewport_name


def _clear_registry() -> None:
    """Clear the component registry."""
    _COMPONENT_REGISTRY.clear()


def _get_registry() -> Dict[str, str]:
    """Return the current component registry.

    Returns
    -------
    dict
        Mapping of component names to grid_py viewport names.
    """
    return dict(_COMPONENT_REGISTRY)


# ---------------------------------------------------------------------------
# HeatmapList
# ---------------------------------------------------------------------------


class HeatmapList:
    """A list of heatmaps and annotations for combined visualisation.

    Created by adding ``Heatmap`` objects with ``+`` (horizontal
    concatenation) or by calling ``draw()`` on a single ``Heatmap``.

    Parameters
    ----------
    ht_list : list, optional
        Initial list of ``Heatmap`` / ``HeatmapAnnotation`` objects.
    direction : str
        ``"horizontal"`` or ``"vertical"``.  Default ``"horizontal"``.

    Attributes
    ----------
    ht_list : list
        The list of ``Heatmap`` / ``HeatmapAnnotation`` objects.
    direction : str
        Layout direction: ``"horizontal"`` or ``"vertical"``.
    row_title : str or None
        Global row title text.
    column_title : str or None
        Global column title text.
    """

    def __init__(
        self,
        ht_list: Optional[List[Any]] = None,
        direction: str = "horizontal",
    ) -> None:
        if direction not in ("horizontal", "vertical"):
            raise ValueError(
                f"`direction` must be 'horizontal' or 'vertical', got {direction!r}"
            )
        self.ht_list: List[Any] = list(ht_list or [])
        self.direction: str = direction
        self._layout: Optional[Dict[str, Any]] = None
        self.row_title: Optional[str] = None
        self.column_title: Optional[str] = None
        self.row_title_gp: Dict[str, Any] = {}
        self.column_title_gp: Dict[str, Any] = {}
        self._main_heatmap_index: Optional[int] = None
        self._drawn: bool = False

    # ------------------------------------------------------------------
    # Adding heatmaps
    # ------------------------------------------------------------------

    def add_heatmap(self, ht: Any, direction: Optional[str] = None) -> "HeatmapList":
        """Add a ``Heatmap`` or ``HeatmapAnnotation`` to the list.

        Parameters
        ----------
        ht : Heatmap, HeatmapAnnotation, or HeatmapList
            Object to append.
        direction : str, optional
            If given, must match the current list direction.

        Returns
        -------
        HeatmapList
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If ``direction`` does not match the existing direction, or
            if row/column counts are inconsistent.
        """
        if direction is not None and direction != self.direction:
            raise ValueError(
                f"Cannot add with direction={direction!r} to a "
                f"{self.direction!r} HeatmapList."
            )

        if self._drawn:
            raise RuntimeError(
                "Cannot add heatmaps to a HeatmapList that has already "
                "been drawn."
            )

        if hasattr(ht, "ht_list"):
            # It's another HeatmapList
            self.ht_list.extend(ht.ht_list)
        else:
            self.ht_list.append(ht)

        # Warn on duplicate names
        names = [getattr(h, "name", None) for h in self.ht_list]
        seen = set()
        for n in names:
            if n is not None and n in seen:
                warnings.warn(f"Duplicate heatmap/annotation name: {n!r}")
            if n is not None:
                seen.add(n)

        return self

    # Alias for convenience
    add = add_heatmap

    def __add__(self, other: Any) -> "HeatmapList":
        """Horizontal concatenation via the ``+`` operator.

        Parameters
        ----------
        other : Heatmap, HeatmapAnnotation, or HeatmapList
            Object(s) to concatenate.

        Returns
        -------
        HeatmapList
            A new ``HeatmapList`` containing all heatmaps from both
            operands.
        """
        new = HeatmapList(list(self.ht_list), self.direction)
        new.row_title = self.row_title
        new.column_title = self.column_title
        new.row_title_gp = dict(self.row_title_gp)
        new.column_title_gp = dict(self.column_title_gp)
        if hasattr(other, "ht_list"):
            new.ht_list.extend(other.ht_list)
        else:
            new.ht_list.append(other)
        return new

    def __iadd__(self, other: Any) -> "HeatmapList":
        """In-place horizontal concatenation via ``+=``.

        Parameters
        ----------
        other : Heatmap, HeatmapAnnotation, or HeatmapList
            Object(s) to concatenate.

        Returns
        -------
        HeatmapList
            ``self``.
        """
        if hasattr(other, "ht_list"):
            self.ht_list.extend(other.ht_list)
        else:
            self.ht_list.append(other)
        return self

    def __len__(self) -> int:
        return len(self.ht_list)

    def __getitem__(self, index: int) -> Any:
        return self.ht_list[index]

    def __repr__(self) -> str:
        names = [getattr(ht, "name", "?") for ht in self.ht_list]
        return f"HeatmapList({names}, direction={self.direction!r})"

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def _find_main_heatmap(
        self, main_heatmap: Optional[Union[int, str]] = None
    ) -> int:
        """Identify the main heatmap (controls row/column ordering).

        Parameters
        ----------
        main_heatmap : int or str, optional
            Explicit index or name.  ``None`` selects the first
            ``Heatmap`` with a matrix.

        Returns
        -------
        int
            0-based index into :attr:`ht_list`.
        """
        if main_heatmap is not None:
            if isinstance(main_heatmap, int):
                return main_heatmap
            for idx, ht in enumerate(self.ht_list):
                if getattr(ht, "name", None) == main_heatmap:
                    return idx
            raise ValueError(f"Heatmap named {main_heatmap!r} not found.")

        for idx, ht in enumerate(self.ht_list):
            if hasattr(ht, "matrix") and getattr(ht, "matrix", None) is not None:
                return idx
        return 0

    def _compute_width_ratios(self) -> List[float]:
        """Compute relative width for each heatmap in horizontal mode.

        Returns
        -------
        list of float
            One width-ratio per element in :attr:`ht_list`.
        """
        ratios: List[float] = []
        for ht in self.ht_list:
            if hasattr(ht, "matrix") and ht.matrix is not None:
                ratios.append(float(ht.matrix.shape[1]))
            elif hasattr(ht, "width") and ht.width is not None:
                ratios.append(float(ht.width))
            else:
                ratios.append(1.0)
        return ratios

    def _compute_height_ratios(self) -> List[float]:
        """Compute relative height for each heatmap in vertical mode.

        Returns
        -------
        list of float
            One height-ratio per element in :attr:`ht_list`.
        """
        ratios: List[float] = []
        for ht in self.ht_list:
            if hasattr(ht, "matrix") and ht.matrix is not None:
                ratios.append(float(ht.matrix.shape[0]))
            elif hasattr(ht, "height") and ht.height is not None:
                ratios.append(float(ht.height))
            else:
                ratios.append(1.0)
        return ratios

    def make_layout(self, **kwargs: Any) -> None:
        """Compute layout for all heatmaps.

        This method:

        1. Identifies the main heatmap.
        2. Calls ``make_layout()`` on each individual heatmap (triggering
           clustering and ordering).
        3. Propagates the row ordering from the main heatmap to all
           others (horizontal mode) or column ordering (vertical mode).

        Parameters
        ----------
        **kwargs
            Forwarded to individual heatmap ``make_layout`` calls.
            Recognised keys include ``main_heatmap``.
        """
        if len(self.ht_list) == 0:
            self._layout = {"main_heatmap_index": None}
            return

        main_idx = self._find_main_heatmap(kwargs.pop("main_heatmap", None))
        self._main_heatmap_index = main_idx

        # Filter out HeatmapList-specific keys before forwarding to Heatmap
        _list_only_keys = {
            "main_heatmap", "ht_gap", "merge_legends",
            "heatmap_legend_side", "annotation_legend_side",
            "show_heatmap_legend", "show_annotation_legend",
        }
        ht_kwargs = {k: v for k, v in kwargs.items() if k not in _list_only_keys}

        # R supports row_km/row_split/column_km/column_split in draw() —
        # forward to the main heatmap (R HeatmapList-draw.R:28-45)
        main_ht = self.ht_list[main_idx]
        for split_key in ("row_km", "row_split", "column_km", "column_split"):
            if split_key in ht_kwargs and hasattr(main_ht, f"_{split_key}_input"):
                val = ht_kwargs.pop(split_key)
                if split_key == "row_km":
                    main_ht._row_split_k = int(val)
                elif split_key == "row_split":
                    import pandas as _pd
                    if isinstance(getattr(val, 'dtype', None), _pd.CategoricalDtype):
                        main_ht._row_split_input = _pd.array(val)
                    else:
                        main_ht._row_split_input = np.asarray(val)
                elif split_key == "column_km":
                    main_ht._column_split_k = int(val)
                elif split_key == "column_split":
                    import pandas as _pd
                    if isinstance(getattr(val, 'dtype', None), _pd.CategoricalDtype):
                        main_ht._column_split_input = _pd.array(val)
                    else:
                        main_ht._column_split_input = np.asarray(val)

        # Step 1: layout the main heatmap first
        if hasattr(main_ht, "make_layout"):
            main_ht.make_layout(**ht_kwargs)

        # Step 2: propagate ordering to other heatmaps
        for idx, ht in enumerate(self.ht_list):
            if idx == main_idx:
                continue
            if not hasattr(ht, "make_layout"):
                continue

            if self.direction == "horizontal":
                main_ro = None
                if hasattr(main_ht, "get_row_order"):
                    main_ro = main_ht.get_row_order()
                elif hasattr(main_ht, "_row_order"):
                    main_ro = main_ht._row_order
                if main_ro is not None and hasattr(ht, "_row_order"):
                    ht._row_order = main_ro
                    ht.cluster_rows = False
            else:
                main_co = None
                if hasattr(main_ht, "get_column_order"):
                    main_co = main_ht.get_column_order()
                elif hasattr(main_ht, "_column_order"):
                    main_co = main_ht._column_order
                if main_co is not None and hasattr(ht, "_column_order"):
                    ht._column_order = main_co
                    ht.cluster_columns = False

            ht.make_layout()

        self._layout = {"main_heatmap_index": main_idx}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_row_order(self) -> Dict[str, Any]:
        """Return row ordering per heatmap.

        Returns
        -------
        dict
            Keys are heatmap names, values are row orderings
            (``np.ndarray`` or list of ``np.ndarray`` for sliced heatmaps).
        """
        result: Dict[str, Any] = {}
        for ht in self.ht_list:
            name = getattr(ht, "name", None)
            if name is None:
                continue
            if hasattr(ht, "get_row_order"):
                result[name] = ht.get_row_order()
        return result

    def get_column_order(self) -> Dict[str, Any]:
        """Return column ordering per heatmap.

        Returns
        -------
        dict
            Keys are heatmap names, values are column orderings.
        """
        result: Dict[str, Any] = {}
        for ht in self.ht_list:
            name = getattr(ht, "name", None)
            if name is None:
                continue
            if hasattr(ht, "get_column_order"):
                result[name] = ht.get_column_order()
        return result

    def get_row_dend(self) -> Dict[str, Any]:
        """Return row dendrogram data per heatmap.

        Returns
        -------
        dict
            Keys are heatmap names, values are dendrogram data.
        """
        result: Dict[str, Any] = {}
        for ht in self.ht_list:
            name = getattr(ht, "name", None)
            if name is None:
                continue
            if hasattr(ht, "get_row_dend"):
                result[name] = ht.get_row_dend()
        return result

    def get_column_dend(self) -> Dict[str, Any]:
        """Return column dendrogram data per heatmap.

        Returns
        -------
        dict
            Keys are heatmap names, values are dendrogram data.
        """
        result: Dict[str, Any] = {}
        for ht in self.ht_list:
            name = getattr(ht, "name", None)
            if name is None:
                continue
            if hasattr(ht, "get_column_dend"):
                result[name] = ht.get_column_dend()
        return result

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        *,
        show: bool = True,
        row_title: Optional[str] = None,
        column_title: Optional[str] = None,
        row_title_gp: Optional[Dict[str, Any]] = None,
        column_title_gp: Optional[Dict[str, Any]] = None,
        merge_legends: bool = False,
        heatmap_legend_side: str = "right",
        annotation_legend_side: str = "right",
        show_heatmap_legend: bool = True,
        show_annotation_legend: bool = True,
        annotation_legend_list: Optional[List[Any]] = None,
        ht_gap: Union[float, grid_py.Unit] = 2.0,
        main_heatmap: Optional[Union[int, str]] = None,
        padding: Optional[Union[float, List[float]]] = None,
        width: float = 7.0,
        height: float = 7.0,
        dpi: float = 150.0,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> "HeatmapList":
        """Draw all heatmaps in a combined figure using grid_py.

        Parameters
        ----------
        show : bool
            Whether to display after drawing.
        row_title : str, optional
            Global row title text.
        column_title : str, optional
            Global column title text.
        row_title_gp : dict, optional
            Graphical parameters for the row title.
        column_title_gp : dict, optional
            Graphical parameters for the column title.
        merge_legends : bool
            Whether to merge all legends into one block.
        heatmap_legend_side : str
            Side for heatmap legends.
        annotation_legend_side : str
            Side for annotation legends.
        show_heatmap_legend : bool
            Whether to show heatmap colour legends.
        show_annotation_legend : bool
            Whether to show annotation legends.
        ht_gap : float or Unit
            Gap between heatmaps in mm (if float) or as a Unit.
        main_heatmap : int or str, optional
            Which heatmap controls row order.
        width : float
            Device width in inches.
        height : float
            Device height in inches.
        dpi : float
            Dots per inch.
        filename : str, optional
            If provided, save output to this file.
        **kwargs
            Additional keyword arguments forwarded to ``make_layout``.

        Returns
        -------
        HeatmapList
            ``self``, for chaining.
        """
        # Apply title overrides
        if row_title is not None:
            self.row_title = row_title
        if column_title is not None:
            self.column_title = column_title
        if row_title_gp is not None:
            self.row_title_gp = row_title_gp
        if column_title_gp is not None:
            self.column_title_gp = column_title_gp

        # Compute layout — only forward layout-relevant kwargs
        _layout_kwargs: Dict[str, Any] = {}
        if main_heatmap is not None:
            _layout_kwargs["main_heatmap"] = main_heatmap
        self.make_layout(**_layout_kwargs)

        # Clear global registry
        _clear_registry()

        n = len(self.ht_list)
        if n == 0:
            grid_py.grid_newpage(width=width, height=height, dpi=dpi)
            self._drawn = True
            self._layout = {"main_heatmap_index": None}
            return self

        # Convert ht_gap to Unit if needed
        if isinstance(ht_gap, (int, float)):
            gap_unit = grid_py.Unit(ht_gap, "mm")
        else:
            gap_unit = ht_gap

        # --- Set up renderer ---
        grid_py.grid_newpage(width=width, height=height, dpi=dpi)

        # --- Build top-level layout ---
        has_col_title = self.column_title is not None
        has_row_title = self.row_title is not None

        # Pre-collect legends and build the packed grob for size measurement
        _legend_side = heatmap_legend_side
        _all_legends: List[Legends] = []
        if show_heatmap_legend:
            _all_legends.extend(self._collect_heatmap_legends(kwargs))
        if show_annotation_legend:
            _all_legends.extend(self._collect_annotation_legends(kwargs))
        if annotation_legend_list:
            for lgd in annotation_legend_list:
                if isinstance(lgd, Legends):
                    _all_legends.append(lgd)
        _has_legends = len(_all_legends) > 0

        # Build the packed legend grob and measure its natural size
        _packed_legend: Optional[Legends] = None
        _legend_w_mm = 0.0
        _legend_h_mm = 0.0
        if _has_legends:
            pack_dir = "vertical" if _legend_side in ("left", "right") else "horizontal"
            _packed_legend = pack_legend(*_all_legends, direction=pack_dir)
            # Query the grob's natural size via width_details/height_details
            w_unit = _packed_legend.grob.width_details()
            h_unit = _packed_legend.grob.height_details()
            # Extract mm value via unit-safe conversion
            if w_unit is not None and hasattr(w_unit, '_values'):
                try:
                    w_mm = grid_py.convert_width(w_unit, "mm", valueOnly=True)
                    _legend_w_mm = float(w_mm[0]) if hasattr(w_mm, '__getitem__') else float(w_mm)
                except Exception:
                    _legend_w_mm = float(w_unit._values[0]) if w_unit._units[0] == "mm" else 20.0
            if h_unit is not None and hasattr(h_unit, '_values'):
                try:
                    h_mm = grid_py.convert_height(h_unit, "mm", valueOnly=True)
                    _legend_h_mm = float(h_mm[0]) if hasattr(h_mm, '__getitem__') else float(h_mm)
                except Exception:
                    _legend_h_mm = float(h_unit._values[0]) if h_unit._units[0] == "mm" else 30.0
            # R: HEATMAP_LEGEND_PADDING controls space between legend and body.
            _hlp = float(ht_opt("HEATMAP_LEGEND_PADDING"))
            _legend_w_mm += 2.0 + _hlp * 2  # content_pad + padding both sides
            _legend_h_mm += 2.0 + _hlp * 2

        # Rows: optional column_title_top, [legend_top], heatmap_panel, [legend_bottom]
        row_heights_list: List[grid_py.Unit] = []
        row_names: List[str] = []

        if has_col_title:
            # R: column_title_height = text_height + sum(TITLE_PADDING)
            # TITLE_PADDING can be scalar (mm) or list [bottom, top] (mm).
            _tp = ht_opt("TITLE_PADDING")
            if isinstance(_tp, (list, tuple)):
                _tp_mm = sum(_tp)
            else:
                _tp_mm = float(_tp) * 2  # symmetric
            row_heights_list.append(
                grid_py.Unit(_tp_mm, "mm") + grid_py.Unit(1, "lines")
            )
            row_names.append("column_title")

        if _has_legends and _legend_side == "top":
            row_heights_list.append(grid_py.Unit(_legend_h_mm, "mm"))
            row_names.append("legend_top")

        row_heights_list.append(grid_py.Unit(1, "null"))
        row_names.append("heatmap_panel")

        if _has_legends and _legend_side == "bottom":
            row_heights_list.append(grid_py.Unit(_legend_h_mm, "mm"))
            row_names.append("legend_bottom")

        # Columns: optional row_title_left, [legend_left], heatmap_panel, [legend_right]
        col_widths_list: List[grid_py.Unit] = []
        col_names: List[str] = []

        if has_row_title:
            col_widths_list.append(
                grid_py.Unit(5, "mm") + grid_py.Unit(1, "lines")
            )
            col_names.append("row_title")

        if _has_legends and _legend_side == "left":
            col_widths_list.append(grid_py.Unit(_legend_w_mm, "mm"))
            col_names.append("legend_left")

        col_widths_list.append(grid_py.Unit(1, "null"))
        col_names.append("heatmap_panel")

        if _has_legends and _legend_side == "right":
            col_widths_list.append(grid_py.Unit(_legend_w_mm, "mm"))
            col_names.append("legend_right")

        outer_layout = grid_py.GridLayout(
            nrow=len(row_heights_list),
            ncol=len(col_widths_list),
            heights=grid_py.unit_c(*row_heights_list),
            widths=grid_py.unit_c(*col_widths_list),
        )

        # Apply padding (R: GLOBAL_PADDING = unit(5.5, "points") ≈ 1.94mm each side)
        # R HeatmapList-layout.R:623-632
        if padding is None:
            _pad = [1.94] * 4  # bottom, left, top, right in mm
        elif isinstance(padding, (int, float)):
            _pad = [float(padding)] * 4
        elif len(padding) == 2:
            _pad = list(padding) * 2
        elif len(padding) == 4:
            _pad = list(padding)
        else:
            _pad = [1.94] * 4

        outer_vp = grid_py.Viewport(
            x=grid_py.Unit(_pad[1], "mm"),
            y=grid_py.Unit(_pad[0], "mm"),
            width=grid_py.Unit(1, "npc") - grid_py.Unit(_pad[1] + _pad[3], "mm"),
            height=grid_py.Unit(1, "npc") - grid_py.Unit(_pad[0] + _pad[2], "mm"),
            just=["left", "bottom"],
            name="global",
            layout=outer_layout,
        )
        grid_py.push_viewport(outer_vp)

        # --- Draw column title ---
        if has_col_title:
            ct_row = row_names.index("column_title") + 1
            ct_col = col_names.index("heatmap_panel") + 1
            ct_vp = grid_py.Viewport(
                layout_pos_row=ct_row,
                layout_pos_col=ct_col,
                name="global_column_title",
            )
            grid_py.push_viewport(ct_vp)
            gp_args = {"fontsize": 14}
            gp_args.update(self.column_title_gp)
            # R: grid.text(title, y = title_padding[1], just = "bottom")
            # The bottom padding reserves space for Heatmap3D bar
            # projections that overflow above the body boundary.
            _tp = ht_opt("TITLE_PADDING")
            if isinstance(_tp, (list, tuple)):
                _tp_bottom_mm = float(_tp[0])
            else:
                _tp_bottom_mm = float(_tp)
            grid_py.grid_text(
                self.column_title,
                x=grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(_tp_bottom_mm, "mm"),
                just="bottom",
                gp=grid_py.Gpar(**gp_args),
            )
            _register_component("global_column_title", "global_column_title")
            grid_py.up_viewport()

        # --- Draw row title ---
        if has_row_title:
            rt_row = row_names.index("heatmap_panel") + 1
            rt_col = col_names.index("row_title") + 1
            rt_vp = grid_py.Viewport(
                layout_pos_row=rt_row,
                layout_pos_col=rt_col,
                name="global_row_title",
            )
            grid_py.push_viewport(rt_vp)
            gp_args = {"fontsize": 14}
            gp_args.update(self.row_title_gp)
            grid_py.grid_text(
                self.row_title,
                x=grid_py.Unit(0.5, "npc"),
                y=grid_py.Unit(0.5, "npc"),
                rot=90,
                gp=grid_py.Gpar(**gp_args),
            )
            _register_component("global_row_title", "global_row_title")
            grid_py.up_viewport()

        # --- Heatmap panel ---
        panel_row = row_names.index("heatmap_panel") + 1
        panel_col = col_names.index("heatmap_panel") + 1

        # Push panel viewport (layout-positioned within outer layout)
        panel_vp = grid_py.Viewport(
            layout_pos_row=panel_row,
            layout_pos_col=panel_col,
            name="heatmap_list_panel",
        )
        grid_py.push_viewport(panel_vp)

        # --- Compute per-heatmap slot widths ---
        # Port of R HeatmapList-draw_component.R:128-146:
        #   heatmap_width[i] = sum(component_width(non-body)) + body_null * unit_per_null
        # This ensures each slot is wide enough for its non-body components.
        # We resolve to absolute mm at the Python level after the panel
        # viewport is pushed (so we can query its actual width).
        if self.direction == "horizontal":
            from complexheatmap.heatmap import Heatmap as _HeatmapCls
            _BODY_EXCL = [
                "row_title_left", "row_dend_left", "row_names_left",
                "left_annotation", "right_annotation",
                "row_names_right", "row_dend_right", "row_title_right",
            ]

            # R HeatmapList-draw_component.R:97-146:
            # When width=NULL (default), R sets heatmap_param$width = unit(1,"npc"),
            # making all heatmaps equal width (1/n of panel).
            # When width is set, it's used as the body null proportion.
            #
            # In either case, each slot must be wide enough for its
            # non-body fixed components. R ensures this by:
            #   slot_width[i] = fixed_components + body_share
            try:
                _g = grid_py.convert_width(gap_unit, "mm", valueOnly=True)
                gap_mm = float(_g[0]) if hasattr(_g, '__getitem__') else float(_g)
            except Exception:
                gap_mm = 2.0

            # Get panel width
            renderer = grid_py.get_state().get_renderer()
            panel_vtr = renderer._vp_transform_stack[-1]
            panel_w_mm = panel_vtr.width_cm * 10
            total_gap_mm = gap_mm * max(n - 1, 0)

            # Compute fixed (non-body) component widths per heatmap
            fixed_widths_mm = []
            is_pure_fixed = []  # True for HeatmapAnnotation (no null body)
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls):
                    fw = 0.0
                    for comp in _BODY_EXCL:
                        u = ht.component_width(comp)
                        try:
                            mm = grid_py.convert_width(u, "mm", valueOnly=True)
                            fw += float(mm[0]) if hasattr(mm, '__getitem__') else float(mm)
                        except Exception:
                            if hasattr(u, '_values'):
                                fw += float(u._values[0])
                    fixed_widths_mm.append(fw)
                    is_pure_fixed.append(False)
                else:
                    # HeatmapAnnotation: entirely fixed width (R line 110: size(ht))
                    ha_w = getattr(ht, 'width', None)
                    if ha_w is not None and isinstance(ha_w, (int, float)):
                        fixed_widths_mm.append(float(ha_w))
                    elif ha_w is not None and grid_py.is_unit(ha_w):
                        try:
                            mm = grid_py.convert_width(ha_w, "mm", valueOnly=True)
                            fixed_widths_mm.append(
                                float(mm[0]) if hasattr(mm, '__getitem__') else float(mm))
                        except Exception:
                            fixed_widths_mm.append(float(ha_w._values[0]))
                    else:
                        fixed_widths_mm.append(15.0)  # reasonable default
                    is_pure_fixed.append(True)

            total_fixed_mm = sum(fixed_widths_mm) + total_gap_mm
            remaining_mm = max(panel_w_mm - total_fixed_mm, 0.0)

            # Determine body proportions:
            # - width=NULL → equal share (R: unit(1,"npc") for each)
            # - width=numeric → use as null proportion (R: ncol-based)
            null_values = []
            for idx_ht, ht in enumerate(self.ht_list):
                if is_pure_fixed[idx_ht]:
                    # HeatmapAnnotation: no null body (all fixed)
                    null_values.append(0.0)
                elif isinstance(ht, _HeatmapCls):
                    user_w = getattr(ht, 'width', None)
                    if user_w is None:
                        # R Heatmap-class.R:966-967:
                        # if(is.null(width)) width = unit(ncol(matrix), "null")
                        null_values.append(float(ht.matrix.shape[1]))
                    elif isinstance(user_w, (int, float)):
                        # R line 968-969: numeric → unit(width, "null")
                        null_values.append(float(user_w))
                    else:
                        null_values.append(1.0)
                else:
                    null_values.append(1.0)

            total_null = sum(null_values)
            mm_per_null = remaining_mm / total_null if total_null > 0 else 0.0

            heatmap_widths_mm = []
            for i in range(n):
                w = fixed_widths_mm[i] + null_values[i] * mm_per_null
                heatmap_widths_mm.append(w)

        # --- Vertical direction: compute per-heatmap slot heights ---
        # Port of R HeatmapList-draw_component.R:300-346
        if self.direction == "vertical":
            from complexheatmap.heatmap import Heatmap as _HeatmapCls
            _BODY_EXCL_V = [
                "column_title_top", "column_dend_top", "column_names_top",
                "top_annotation", "bottom_annotation",
                "column_names_bottom", "column_dend_bottom", "column_title_bottom",
            ]
            try:
                _g = grid_py.convert_height(gap_unit, "mm", valueOnly=True)
                gap_mm = float(_g[0]) if hasattr(_g, '__getitem__') else float(_g)
            except Exception:
                gap_mm = 2.0

            renderer = grid_py.get_state().get_renderer()
            panel_vtr = renderer._vp_transform_stack[-1]
            panel_h_mm = panel_vtr.height_cm * 10
            total_gap_mm = gap_mm * max(n - 1, 0)

            fixed_heights_mm = []
            is_pure_fixed = []
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls):
                    fh = 0.0
                    for comp in _BODY_EXCL_V:
                        u = ht.component_height(comp)
                        try:
                            mm = grid_py.convert_height(u, "mm", valueOnly=True)
                            fh += float(mm[0]) if hasattr(mm, '__getitem__') else float(mm)
                        except Exception:
                            pass
                    fixed_heights_mm.append(fh)
                    is_pure_fixed.append(False)
                else:
                    ha_h = getattr(ht, 'height', None)
                    if ha_h is not None and grid_py.is_unit(ha_h):
                        try:
                            mm = grid_py.convert_height(ha_h, "mm", valueOnly=True)
                            fixed_heights_mm.append(
                                float(mm[0]) if hasattr(mm, '__getitem__') else float(mm))
                        except Exception:
                            fixed_heights_mm.append(15.0)
                    elif ha_h is not None and isinstance(ha_h, (int, float)):
                        fixed_heights_mm.append(float(ha_h))
                    else:
                        fixed_heights_mm.append(15.0)
                    is_pure_fixed.append(True)

            total_fixed_mm = sum(fixed_heights_mm) + total_gap_mm
            remaining_mm = max(panel_h_mm - total_fixed_mm, 0.0)

            null_values = []
            for idx_ht, ht in enumerate(self.ht_list):
                if is_pure_fixed[idx_ht]:
                    null_values.append(0.0)
                elif isinstance(ht, _HeatmapCls):
                    null_values.append(float(ht.matrix.shape[0]))
                else:
                    null_values.append(1.0)

            total_null = sum(null_values)
            mm_per_null = remaining_mm / total_null if total_null > 0 else 0.0

            heatmap_heights_mm = []
            for i in range(n):
                h = fixed_heights_mm[i] + null_values[i] * mm_per_null
                heatmap_heights_mm.append(h)

        # --- Cross-heatmap alignment (horizontal: vertical body align) ---
        # --- Cross-heatmap alignment (vertical: horizontal body align) ---
        # Port of R HeatmapList-draw_component.R:255-297
        if self.direction == "vertical":
            from complexheatmap.heatmap import Heatmap as _HeatmapCls2v

            _left_components = ("row_dend_left", "row_names_left",
                                "left_annotation")
            _right_components = ("row_dend_right", "row_names_right",
                                 "right_annotation")

            _title_left_mm = 0.0
            _title_right_mm = 0.0
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2v):
                    tl = ht.component_width("row_title_left")
                    tr = ht.component_width("row_title_right")
                    tl_mm = float(np.squeeze(
                        grid_py.convert_width(tl, "mm", valueOnly=True)))
                    tr_mm = float(np.squeeze(
                        grid_py.convert_width(tr, "mm", valueOnly=True)))
                    _title_left_mm = max(_title_left_mm, tl_mm)
                    _title_right_mm = max(_title_right_mm, tr_mm)

            _max_left_mm = 0.0
            _max_right_mm = 0.0
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2v):
                    left_mm = sum(
                        float(np.squeeze(grid_py.convert_width(
                            ht.component_width(c), "mm", valueOnly=True)))
                        for c in _left_components
                    )
                    right_mm = sum(
                        float(np.squeeze(grid_py.convert_width(
                            ht.component_width(c), "mm", valueOnly=True)))
                        for c in _right_components
                    )
                    _max_left_mm = max(_max_left_mm, left_mm)
                    _max_right_mm = max(_max_right_mm, right_mm)

            # Pad dendrogram widths so all heatmaps have same horizontal layout
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2v):
                    ht._override_widths = getattr(ht, '_override_widths', {})
                    ht._override_widths["row_title_left"] = _title_left_mm
                    ht._override_widths["row_title_right"] = _title_right_mm

                    this_names_l = float(np.squeeze(
                        grid_py.convert_width(
                            ht.component_width("row_names_left"),
                            "mm", valueOnly=True)))
                    this_anno_l = float(np.squeeze(
                        grid_py.convert_width(
                            ht.component_width("left_annotation"),
                            "mm", valueOnly=True)))
                    ht._override_widths["row_dend_left"] = max(
                        0, _max_left_mm - this_names_l - this_anno_l)

                    this_names_r = float(np.squeeze(
                        grid_py.convert_width(
                            ht.component_width("row_names_right"),
                            "mm", valueOnly=True)))
                    this_anno_r = float(np.squeeze(
                        grid_py.convert_width(
                            ht.component_width("right_annotation"),
                            "mm", valueOnly=True)))
                    ht._override_widths["row_dend_right"] = max(
                        0, _max_right_mm - this_names_r - this_anno_r)
        # Port of R HeatmapList-draw_component.R:55-86
        # Compute max top/bottom component heights across all Heatmaps,
        # then adjust each Heatmap's component heights so bodies align.
        if self.direction == "horizontal":
            from complexheatmap.heatmap import Heatmap as _HeatmapCls2

            _top_components = ("column_dend_top", "column_names_top",
                               "top_annotation")
            _bottom_components = ("column_dend_bottom", "column_names_bottom",
                                  "bottom_annotation")

            # R: max_title_component_height
            _title_top_mm = 0.0
            _title_bot_mm = 0.0
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2):
                    tt = ht.component_height("column_title_top")
                    tb = ht.component_height("column_title_bottom")
                    tt_mm = float(np.squeeze(
                        grid_py.convert_height(tt, "mm", valueOnly=True)))
                    tb_mm = float(np.squeeze(
                        grid_py.convert_height(tb, "mm", valueOnly=True)))
                    _title_top_mm = max(_title_top_mm, tt_mm)
                    _title_bot_mm = max(_title_bot_mm, tb_mm)

            # R: max_top/bottom_component_height (dend + names + anno)
            _max_top_mm = 0.0
            _max_bot_mm = 0.0
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2):
                    top_mm = sum(
                        float(np.squeeze(grid_py.convert_height(
                            ht.component_height(c), "mm", valueOnly=True)))
                        for c in _top_components
                    )
                    bot_mm = sum(
                        float(np.squeeze(grid_py.convert_height(
                            ht.component_height(c), "mm", valueOnly=True)))
                        for c in _bottom_components
                    )
                    _max_top_mm = max(_max_top_mm, top_mm)
                    _max_bot_mm = max(_max_bot_mm, bot_mm)

            # R: set_component_height for each heatmap (lines 79-86)
            # Pad dendrogram height so all heatmaps have same total
            # top/bottom component height.
            for ht in self.ht_list:
                if isinstance(ht, _HeatmapCls2):
                    # Set title heights to max
                    ht._override_heights = getattr(ht, '_override_heights', {})
                    ht._override_heights["column_title_top"] = _title_top_mm
                    ht._override_heights["column_title_bottom"] = _title_bot_mm

                    # Pad dend = max_top - (names + anno for this ht)
                    this_names_top_mm = float(np.squeeze(
                        grid_py.convert_height(
                            ht.component_height("column_names_top"),
                            "mm", valueOnly=True)))
                    this_anno_top_mm = float(np.squeeze(
                        grid_py.convert_height(
                            ht.component_height("top_annotation"),
                            "mm", valueOnly=True)))
                    ht._override_heights["column_dend_top"] = max(
                        0, _max_top_mm - this_names_top_mm - this_anno_top_mm)

                    this_names_bot_mm = float(np.squeeze(
                        grid_py.convert_height(
                            ht.component_height("column_names_bottom"),
                            "mm", valueOnly=True)))
                    this_anno_bot_mm = float(np.squeeze(
                        grid_py.convert_height(
                            ht.component_height("bottom_annotation"),
                            "mm", valueOnly=True)))
                    ht._override_heights["column_dend_bottom"] = max(
                        0, _max_bot_mm - this_names_bot_mm - this_anno_bot_mm)

            # Store for HeatmapAnnotation body alignment
            self._max_top_component_mm = _title_top_mm + _max_top_mm
            self._max_bottom_component_mm = _title_bot_mm + _max_bot_mm

        # --- Draw each heatmap using absolute-positioned viewports ---
        # Port of R HeatmapList-draw_component.R:622-631
        for idx, ht in enumerate(self.ht_list):
            ht_name = getattr(ht, "name", f"heatmap_{idx + 1}")

            if self.direction == "horizontal":
                x_mm = sum(heatmap_widths_mm[:idx]) + gap_mm * idx

            # Create an absolute-positioned viewport for this heatmap slot.
            # Port of R HeatmapList-draw_component.R:631
            if self.direction == "horizontal":
                slot_vp = grid_py.Viewport(
                    x=grid_py.Unit(x_mm, "mm"),
                    y=grid_py.Unit(0, "npc"),
                    width=grid_py.Unit(heatmap_widths_mm[idx], "mm"),
                    height=grid_py.Unit(1, "npc"),
                    just=["left", "bottom"],
                    name=f"heatmap_list_slot_{ht_name}",
                )
            else:
                # Vertical: stack top-to-bottom
                # R: y = unit(1,"npc") - sum(heights[1..i-1]) - sum(gaps[1..i-1])
                y_mm = sum(heatmap_heights_mm[:idx]) + gap_mm * idx
                slot_vp = grid_py.Viewport(
                    x=grid_py.Unit(0, "npc"),
                    y=grid_py.Unit(1, "npc") - grid_py.Unit(y_mm, "mm"),
                    width=grid_py.Unit(1, "npc"),
                    height=grid_py.Unit(heatmap_heights_mm[idx], "mm"),
                    just=["left", "top"],
                    name=f"heatmap_list_slot_{ht_name}",
                )
            grid_py.push_viewport(slot_vp)

            if hasattr(ht, "_draw_into_viewport"):
                # Heatmap objects can draw themselves into the current viewport
                ht._draw_into_viewport()
            elif hasattr(ht, "matrix") and ht.matrix is not None:
                self._draw_single_heatmap(ht, ht_name)
            elif hasattr(ht, "draw") and hasattr(ht, "anno_list"):
                # HeatmapAnnotation: draw annotation tracks using the
                # main heatmap's row order (R HeatmapList-draw_component.R:634)
                import numpy as _np
                main_idx = getattr(self, '_main_heatmap_index', 0)
                main_ht = self.ht_list[main_idx] if main_idx < len(self.ht_list) else None
                ro_lt = (main_ht._row_order_list
                         if main_ht and hasattr(main_ht, '_row_order_list')
                            and main_ht._row_order_list
                         else None)

                n_slice = len(ro_lt) if ro_lt else 1
                index = (_np.concatenate(ro_lt) if ro_lt
                         else _np.arange(10))

                # R HeatmapList-draw_component.R:668:
                # pushViewport(viewport(y=max_bottom, height=1npc-max_top-max_bottom, just="bottom"))
                # This aligns the annotation body with the heatmap bodies.
                _max_top = getattr(self, '_max_top_component_mm', 0)
                _max_bot = getattr(self, '_max_bottom_component_mm', 0)
                _body_vp_pushed = False
                if _max_top > 0 or _max_bot > 0:
                    body_align_vp = grid_py.Viewport(
                        y=grid_py.Unit(_max_bot, "mm"),
                        height=(grid_py.Unit(1, "npc")
                                - grid_py.Unit(_max_top, "mm")
                                - grid_py.Unit(_max_bot, "mm")),
                        just=["center", "bottom"],
                        clip=False,
                        name=f"body_align_{ht_name}",
                    )
                    grid_py.push_viewport(body_align_vp)
                    _body_vp_pushed = True

                # R HeatmapList-draw_component.R:637-664:
                # For anno_mark in a split scenario, compute physical
                # positions accounting for slice gaps.
                has_anno_mark = any(
                    getattr(sa, '_anno_fun', None) is not None
                    and getattr(getattr(sa, '_anno_fun', None),
                                'fun_name', '') in ('anno_mark', 'anno_zoom')
                    for sa in ht.anno_list.values()
                )

                if n_slice > 1 and has_anno_mark and ro_lt is not None:
                    sl = getattr(main_ht, '_layout', {}).get('slice', None)
                    if sl is not None:
                        _slice_y = sl["y"]
                        _slice_h = sl["height"]

                        # Compute .pos: physical position of each row
                        # in NPC coords of the BODY viewport (not slot).
                        _pos_list = []
                        for si in range(n_slice):
                            ni_s = len(ro_lt[si])
                            sy = grid_py.convert_height(_slice_y[si], "npc",
                                                        valueOnly=True)
                            sh = grid_py.convert_height(_slice_h[si], "npc",
                                                        valueOnly=True)
                            sy_val = float(_np.squeeze(sy))
                            sh_val = float(_np.squeeze(sh))
                            for j in range(ni_s):
                                p = sy_val - (j + 0.5) / ni_s * sh_val
                                _pos_list.append(p)

                        _pos = _np.array(_pos_list)
                        _scale = (0.0, 1.0)

                        for sa in ht.anno_list.values():
                            af = getattr(sa, '_anno_fun', None)
                            if (af is not None
                                    and getattr(af, 'fun_name', '')
                                    in ('anno_mark', 'anno_zoom')):
                                af.var_env['_pos'] = _pos
                                af.var_env['_scale'] = _scale

                ht.draw(index)

                if _body_vp_pushed:
                    grid_py.up_viewport()
            else:
                _register_component(f"unknown_{ht_name}", f"heatmap_list_slot_{ht_name}")

            grid_py.up_viewport()

        grid_py.up_viewport()  # heatmap_list_panel

        # --- Draw legends ---
        if _has_legends and _packed_legend is not None:
            self._draw_packed_legend(
                _packed_legend, _legend_side,
                row_names, col_names,
            )

        grid_py.up_viewport()  # global (outer_vp)

        self._drawn = True

        # --- Save if requested ---
        state = grid_py.get_state()
        rend = state.get_renderer()
        if filename is not None:
            if rend is not None and hasattr(rend, "write_to_png"):
                rend.write_to_png(filename)

        # --- Display inline in Jupyter if requested ---
        if show and rend is not None and hasattr(rend, "to_png_bytes"):
            try:
                from IPython.display import display, Image
                display(Image(data=rend.to_png_bytes()))
            except ImportError:
                pass

        return self

    # ------------------------------------------------------------------
    # Legend helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _color_mapping_to_legend(
        cm: "ColorMapping",
        param: Dict[str, Any],
    ) -> Legends:
        """Convert a :class:`ColorMapping` to a :class:`Legends` object.

        Mirrors R's ``color_mapping_legend()`` in ``ColorMapping-class.R``.
        """
        title = param.get("title", cm.name)
        title_gp = param.get("title_gp", {"fontsize": 9, "fontface": "bold"})
        labels_gp = param.get("labels_gp", {"fontsize": 8})
        direction = param.get("direction", "vertical")

        # --- Custom graphics legend (R oncoPrint / rect_gp=gpar(type="none"))
        # When heatmap_legend_param carries ``graphics``, the legend icons
        # are drawn by those functions instead of coloured rectangles.
        # This is the mechanism R uses for oncoPrint legends.
        graphics = param.get("graphics")
        if graphics is not None:
            at = param.get("at", [])
            labels = param.get("labels", at)
            grid_height = param.get("grid_height", 4.0)
            grid_width = param.get("grid_width", 4.0)
            ncol = param.get("ncol", 1)
            nrow = param.get("nrow", None)
            return Legend(
                at=at,
                labels=labels,
                title=title,
                title_gp=title_gp,
                labels_gp=labels_gp,
                graphics=graphics,
                grid_height=grid_height,
                grid_width=grid_width,
                ncol=ncol,
                nrow=nrow,
                direction=direction,
            )

        if cm.is_discrete:
            levels = cm.levels
            cmap = cm.color_map
            at = param.get("at", list(range(1, len(levels) + 1)))
            labels = param.get("labels", levels)
            fill_colors = [cmap.get(lv, "#CCCCCC") for lv in levels]
            grid_height = param.get("grid_height", 4.0)
            grid_width = param.get("grid_width", 4.0)
            ncol = param.get("ncol", 1)
            nrow = param.get("nrow", None)
            return Legend(
                at=at,
                labels=labels,
                title=title,
                title_gp=title_gp,
                labels_gp=labels_gp,
                legend_gp={"fill": fill_colors},
                grid_height=grid_height,
                grid_width=grid_width,
                ncol=ncol,
                nrow=nrow,
                direction=direction,
            )
        else:
            # Continuous
            col_fun = cm._col_fun
            breaks = cm.breaks
            if "at" in param:
                at = param["at"]
            elif breaks is not None:
                # Use grid_pretty to generate nice tick values (like R)
                try:
                    at = list(grid_py.grid_pretty(
                        [float(breaks[0]), float(breaks[-1])]
                    ))
                except (TypeError, ValueError, AttributeError):
                    at = list(breaks)
            else:
                at = [0, 1]
            labels = param.get("labels", [
                str(int(v)) if v == int(v) else f"{v:g}" for v in at
            ])
            legend_height = param.get("legend_height", 30.0)
            legend_width = param.get("legend_width", None)
            grid_width = param.get("grid_width", 4.0)
            return Legend(
                col_fun=col_fun,
                at=at,
                labels=labels,
                title=title,
                title_gp=title_gp,
                labels_gp=labels_gp,
                direction=direction,
                legend_height=legend_height,
                legend_width=legend_width,
                grid_width=grid_width,
            )

    def _collect_heatmap_legends(
        self, draw_kwargs: Dict[str, Any],
    ) -> List[Legends]:
        """Collect legend objects from heatmap color mappings.

        Mirrors R's legend collection in ``HeatmapList-legends.R``.
        """
        legends: List[Legends] = []
        seen_names: set = set()

        for ht in self.ht_list:
            if not hasattr(ht, "_color_mapping"):
                continue
            cm = ht._color_mapping
            if cm is None:
                continue
            if not getattr(ht, "show_heatmap_legend", True):
                continue
            # Deduplicate by legend name
            legend_name = getattr(ht, "heatmap_legend_param", {}).get(
                "title", cm.name
            )
            if legend_name in seen_names:
                continue
            seen_names.add(legend_name)

            param = dict(getattr(ht, "heatmap_legend_param", {}))
            lgd = self._color_mapping_to_legend(cm, param)
            legends.append(lgd)

        return legends

    def _collect_annotation_legends(
        self, draw_kwargs: Dict[str, Any],
    ) -> List[Legends]:
        """Collect legend objects from annotation color mappings.

        Mirrors R's annotation legend collection logic.
        """
        legends: List[Legends] = []
        seen_names: set = set()

        for ht in self.ht_list:
            for anno_attr in (
                "top_annotation", "bottom_annotation",
                "left_annotation", "right_annotation",
            ):
                ha = getattr(ht, anno_attr, None)
                if ha is None:
                    continue
                anno_list = getattr(ha, "anno_list", {})
                anno_legend_param = getattr(ha, "annotation_legend_param", {})

                for anno_name, sa in anno_list.items():
                    if not getattr(sa, "show_legend", True):
                        continue
                    cm = getattr(sa, "_color_mapping", None)
                    if cm is None:
                        continue
                    legend_name = anno_legend_param.get(
                        anno_name, {}
                    ).get("title", cm.name)
                    if legend_name in seen_names:
                        continue
                    seen_names.add(legend_name)

                    param = dict(anno_legend_param.get(anno_name, {}))
                    if "title" not in param:
                        param["title"] = cm.name
                    lgd = self._color_mapping_to_legend(cm, param)
                    legends.append(lgd)

        return legends

    def _draw_packed_legend(
        self,
        packed: Legends,
        side: str,
        row_names: List[str],
        col_names: List[str],
    ) -> None:
        """Draw a pre-packed legend into its layout slot.

        The layout already has the correct size allocated via
        ``grobwidth`` / ``grobheight`` units, so we just push a
        viewport at the slot position and draw.
        """
        slot_name = f"legend_{side}"
        if side in ("left", "right") and slot_name in col_names:
            legend_row = row_names.index("heatmap_panel") + 1
            legend_col = col_names.index(slot_name) + 1
        elif side in ("top", "bottom") and slot_name in row_names:
            legend_row = row_names.index(slot_name) + 1
            legend_col = col_names.index("heatmap_panel") + 1
        else:
            return

        vp = grid_py.Viewport(
            layout_pos_row=legend_row,
            layout_pos_col=legend_col,
            name=f"global_{slot_name}",
        )
        grid_py.push_viewport(vp)

        # Position within the legend slot.
        # R (HeatmapList-legends.R:132-206): default is "global_center" or
        # "heatmap_center" — legend viewport centered within its slot.
        # For left/right: x=0.5npc, y=0.5npc, just=center
        # For top/bottom: x=0.5npc, y=0.5npc, just=center
        x = grid_py.Unit(0.5, "npc")
        y = grid_py.Unit(0.5, "npc")
        just = "centre"

        # Compute legend natural size for the inner viewport
        _lw = getattr(packed, '_legend_w_mm', None)
        _lh = getattr(packed, '_legend_h_mm', None)
        if _lw is None or _lh is None:
            g = packed.grob if hasattr(packed, 'grob') else packed
            _w = g.width_details() if hasattr(g, 'width_details') else None
            _h = g.height_details() if hasattr(g, 'height_details') else None
            if _w is not None and grid_py.is_unit(_w):
                try:
                    _lw = float(np.squeeze(grid_py.convert_width(_w, "mm", valueOnly=True)))
                except Exception:
                    _lw = 20.0
            else:
                _lw = 20.0
            if _h is not None and grid_py.is_unit(_h):
                try:
                    _lh = float(np.squeeze(grid_py.convert_height(_h, "mm", valueOnly=True)))
                except Exception:
                    _lh = 30.0
            else:
                _lh = 30.0

        legend_vp = grid_py.Viewport(
            x=x, y=y, just=just,
            width=grid_py.Unit(_lw, "mm"),
            height=grid_py.Unit(_lh, "mm"),
            name=f"legend_draw_{side}",
        )
        grid_py.push_viewport(legend_vp)
        packed.draw()
        grid_py.up_viewport()

        _register_component(f"global_{slot_name}", f"global_{slot_name}")
        grid_py.up_viewport()

    # ------------------------------------------------------------------
    # Internal: draw a single heatmap into the current viewport
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_single_heatmap(ht: Any, name: str) -> None:
        """Render a single heatmap into the current viewport.

        Parameters
        ----------
        ht : Heatmap
            The heatmap object to draw.
        name : str
            Registry name prefix.
        """
        matrix = np.asarray(ht.matrix)
        nr, nc = matrix.shape

        if nr == 0 or nc == 0:
            return

        # Ensure layout is computed
        if hasattr(ht, "_layout_computed") and not ht._layout_computed:
            ht.make_layout()

        # Get row/column component lists and build a sub-layout
        row_components = [
            "column_title_top",
            "column_dend_top",
            "column_names_top",
            "top_annotation",
            "heatmap_body",
            "bottom_annotation",
            "column_names_bottom",
            "column_dend_bottom",
            "column_title_bottom",
        ]
        col_components = [
            "row_title_left",
            "left_annotation",
            "row_names_left",
            "row_dend_left",
            "heatmap_body",
            "row_dend_right",
            "row_names_right",
            "right_annotation",
            "row_title_right",
        ]

        row_heights = [ht.component_height(c) for c in row_components]
        col_widths = [ht.component_width(c) for c in col_components]

        layout = grid_py.GridLayout(
            nrow=len(row_components),
            ncol=len(col_components),
            heights=grid_py.unit_c(*row_heights),
            widths=grid_py.unit_c(*col_widths),
        )

        ht_vp = grid_py.Viewport(
            name=f"{name}_main",
            layout=layout,
        )
        grid_py.push_viewport(ht_vp)

        body_row = row_components.index("heatmap_body") + 1
        body_col = col_components.index("heatmap_body") + 1

        # Get slice info
        n_row_slices = 1
        n_col_slices = 1
        row_order_list = None
        col_order_list = None

        if hasattr(ht, "_row_order_list") and ht._row_order_list is not None:
            row_order_list = ht._row_order_list
            n_row_slices = len(row_order_list)
        if hasattr(ht, "_column_order_list") and ht._column_order_list is not None:
            col_order_list = ht._column_order_list
            n_col_slices = len(col_order_list)

        # Draw heatmap body
        body_vp = grid_py.Viewport(
            layout_pos_row=body_row,
            layout_pos_col=body_col,
            name=f"{name}_heatmap_body_wrap",
        )
        grid_py.push_viewport(body_vp)

        if row_order_list is not None and col_order_list is not None:
            row_sizes = [len(ro) for ro in row_order_list]
            col_sizes = [len(co) for co in col_order_list]
            total_rows = sum(row_sizes)
            total_cols = sum(col_sizes)

            for ri in range(n_row_slices):
                for ci in range(n_col_slices):
                    row_ord = row_order_list[ri]
                    col_ord = col_order_list[ci]
                    sub_mat = matrix[np.ix_(row_ord, col_ord)]
                    snr, snc = sub_mat.shape
                    if snr == 0 or snc == 0:
                        continue

                    x_start = sum(col_sizes[:ci]) / total_cols if total_cols > 0 else 0
                    x_width = col_sizes[ci] / total_cols if total_cols > 0 else 1
                    y_start_top = sum(row_sizes[:ri]) / total_rows if total_rows > 0 else 0
                    y_height = row_sizes[ri] / total_rows if total_rows > 0 else 1

                    slice_vp_name = f"{name}_heatmap_body_{ri + 1}_{ci + 1}"
                    slice_vp = grid_py.Viewport(
                        x=grid_py.Unit(x_start, "npc"),
                        y=grid_py.Unit(1.0 - y_start_top - y_height, "npc"),
                        width=grid_py.Unit(x_width, "npc"),
                        height=grid_py.Unit(y_height, "npc"),
                        just=["left", "bottom"],
                        name=slice_vp_name,
                    )
                    grid_py.push_viewport(slice_vp)

                    # Draw cells
                    if hasattr(ht, "_map_to_colors"):
                        col_matrix = ht._map_to_colors(sub_mat)
                    else:
                        col_matrix = np.full(sub_mat.shape, "#CCCCCC", dtype=object)

                    all_x = []
                    all_y = []
                    all_fill = []
                    for i_r in range(snr):
                        for j_c in range(snc):
                            all_x.append((j_c + 0.5) / snc)
                            all_y.append(1.0 - (i_r + 0.5) / snr)
                            all_fill.append(col_matrix[i_r, j_c])

                    grid_py.grid_rect(
                        x=grid_py.Unit(all_x, "npc"),
                        y=grid_py.Unit(all_y, "npc"),
                        width=grid_py.Unit(1.0 / snc, "npc"),
                        height=grid_py.Unit(1.0 / snr, "npc"),
                        gp=grid_py.Gpar(fill=all_fill, col="transparent"),
                    )

                    _register_component(
                        f"heatmap_body_{name}_{ri + 1}_{ci + 1}",
                        slice_vp_name,
                    )
                    grid_py.up_viewport()
        else:
            # No slicing: draw entire matrix
            slice_vp_name = f"{name}_heatmap_body_1_1"
            slice_vp = grid_py.Viewport(name=slice_vp_name)
            grid_py.push_viewport(slice_vp)

            if hasattr(ht, "_map_to_colors"):
                col_matrix = ht._map_to_colors(matrix)
            else:
                col_matrix = np.full(matrix.shape, "#CCCCCC", dtype=object)

            all_x = []
            all_y = []
            all_fill = []
            for i_r in range(nr):
                for j_c in range(nc):
                    all_x.append((j_c + 0.5) / nc)
                    all_y.append(1.0 - (i_r + 0.5) / nr)
                    all_fill.append(col_matrix[i_r, j_c])

            grid_py.grid_rect(
                x=grid_py.Unit(all_x, "npc"),
                y=grid_py.Unit(all_y, "npc"),
                width=grid_py.Unit(1.0 / nc, "npc"),
                height=grid_py.Unit(1.0 / nr, "npc"),
                gp=grid_py.Gpar(fill=all_fill, col="transparent"),
            )

            _register_component(
                f"heatmap_body_{name}_1_1",
                slice_vp_name,
            )
            grid_py.up_viewport()

        grid_py.up_viewport()  # body_wrap
        grid_py.up_viewport()  # ht_vp (name_main)


# ---------------------------------------------------------------------------
# Top-level convenience functions (R: row_order, column_order)
# ---------------------------------------------------------------------------


def row_order(ht_list: "HeatmapList") -> list:
    """Return the row order of the main heatmap as a list of arrays.

    Port of R ``row_order(ht_list)`` which returns a list of integer
    vectors (one per row slice) from the main heatmap.

    Parameters
    ----------
    ht_list : HeatmapList
        A drawn or laid-out heatmap list.

    Returns
    -------
    list of numpy.ndarray
        One array per row slice, containing 0-based row indices.
    """
    main_idx = getattr(ht_list, "_main_heatmap_index", 0)
    main_ht = ht_list.ht_list[main_idx] if main_idx < len(ht_list.ht_list) else None
    if main_ht is not None and hasattr(main_ht, "_row_order_list") and main_ht._row_order_list:
        return list(main_ht._row_order_list)
    return []


def column_order(ht_list: "HeatmapList") -> dict:
    """Return column orders for all heatmaps.

    Port of R ``column_order(ht_list)`` which returns a named list
    with one entry per heatmap, each containing a list of column
    index vectors (one per column slice).

    Parameters
    ----------
    ht_list : HeatmapList
        A drawn or laid-out heatmap list.

    Returns
    -------
    dict
        Keys are heatmap names, values are lists of numpy.ndarray.
    """
    return ht_list.get_column_order()
