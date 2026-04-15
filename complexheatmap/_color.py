"""Color utilities for ComplexHeatmap.

R source correspondence
-----------------------
``circlize::colorRamp2``, ``ComplexHeatmap:::add_transparency``,
and ``ComplexHeatmap:::rand_color``.

Provides color interpolation, transparency handling, and random color
generation.  Uses the ``scales`` Python package (port of R ``scales``)
for colour-space interpolation where possible, falling back to a
self-contained CIE-LAB implementation.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np

__all__ = ["color_ramp2", "add_transparency", "rand_color"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(color: str) -> np.ndarray:
    """Convert a hex color string or named color to an RGB array in [0, 1].

    Parameters
    ----------
    color : str
        A hex string (``"#RRGGBB"`` or ``"#RRGGBBAA"``) or a CSS4 named
        color recognised by matplotlib.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` with values in [0, 1].
    """
    import matplotlib.colors as mcolors

    rgba = mcolors.to_rgba(color)
    return np.array(rgba[:3], dtype=float)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert an RGB array in [0, 1] to a ``#RRGGBB`` hex string.

    Parameters
    ----------
    rgb : numpy.ndarray
        Shape ``(3,)`` with values in [0, 1].

    Returns
    -------
    str
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to CIE-LAB via XYZ (D65 illuminant).

    Parameters
    ----------
    rgb : numpy.ndarray
        Shape ``(N, 3)`` or ``(3,)`` with values in [0, 1].

    Returns
    -------
    numpy.ndarray
        Same shape, CIE-LAB values.
    """
    squeeze = rgb.ndim == 1
    rgb = np.atleast_2d(rgb).copy()

    # sRGB -> linear
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # linear RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ M.T

    # Normalise by D65 white point
    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz /= ref

    # XYZ -> LAB
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    mask = xyz > eps
    xyz_f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * xyz_f[:, 1] - 16.0
    a = 500.0 * (xyz_f[:, 0] - xyz_f[:, 1])
    b = 200.0 * (xyz_f[:, 1] - xyz_f[:, 2])
    lab = np.column_stack([L, a, b])
    return lab.squeeze() if squeeze else lab


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE-LAB to sRGB [0,1] via XYZ (D65 illuminant).

    Parameters
    ----------
    lab : numpy.ndarray
        Shape ``(N, 3)`` or ``(3,)`` CIE-LAB values.

    Returns
    -------
    numpy.ndarray
        Same shape, sRGB values clipped to [0, 1].
    """
    squeeze = lab.ndim == 1
    lab = np.atleast_2d(lab).copy()

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)

    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz = np.column_stack([xr, yr, zr]) * ref

    # XYZ -> linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    rgb_linear = xyz @ M_inv.T

    # linear -> sRGB gamma
    mask = rgb_linear > 0.0031308
    rgb_linear[mask] = 1.055 * np.power(rgb_linear[mask], 1.0 / 2.4) - 0.055
    rgb_linear[~mask] = 12.92 * rgb_linear[~mask]

    result = np.clip(rgb_linear, 0.0, 1.0)
    return result.squeeze() if squeeze else result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def color_ramp2(
    breaks: Sequence[float],
    colors: Sequence[str],
    space: str = "LAB",
) -> Callable[[Union[float, Sequence[float], np.ndarray]], Union[str, List[str]]]:
    """Create a colour interpolation function.

    Analogous to ``circlize::colorRamp2`` in R.  Builds piece-wise linear
    interpolators in the chosen colour space, returning a callable that maps
    numeric values to hex colour strings.

    Parameters
    ----------
    breaks : sequence of float
        Sorted numeric break-points (length *n*).
    colors : sequence of str
        Colour specifications of the same length as *breaks*.  Any format
        accepted by :func:`matplotlib.colors.to_rgba`.
    space : str, optional
        Colour space for interpolation.  ``"LAB"`` (default) or ``"RGB"``.

    Returns
    -------
    callable
        A function ``f(x)`` that accepts a scalar, list, or ndarray of
        numeric values and returns a single hex string or list of hex
        strings.

    Raises
    ------
    ValueError
        If *breaks* and *colors* differ in length, or if *breaks* are not
        sorted.
    """
    breaks_arr = np.asarray(breaks, dtype=float)
    if len(breaks_arr) != len(colors):
        raise ValueError("breaks and colors must have the same length")
    if not np.all(np.diff(breaks_arr) > 0):
        raise ValueError("breaks must be strictly increasing")

    # Convert colours to the working space
    rgb_arr = np.array([_hex_to_rgb(c) for c in colors])

    if space.upper() == "LAB":
        coords = _rgb_to_lab(rgb_arr)
    elif space.upper() == "RGB":
        coords = rgb_arr.copy()
    else:
        raise ValueError(f"Unsupported colour space: {space!r}")

    def _map(
        x: Union[float, Sequence[float], np.ndarray],
    ) -> Union[str, List[str]]:
        scalar = np.isscalar(x)
        vals = np.atleast_1d(np.asarray(x, dtype=float))
        out_coords = np.empty((len(vals), 3), dtype=float)

        for ch in range(3):
            out_coords[:, ch] = np.interp(vals, breaks_arr, coords[:, ch])

        if space.upper() == "LAB":
            rgb_out = _lab_to_rgb(out_coords)
        else:
            rgb_out = np.clip(out_coords, 0.0, 1.0)

        hex_list = [_rgb_to_hex(row) for row in rgb_out]
        return hex_list[0] if scalar else hex_list

    # Attach metadata for introspection
    _map.breaks = breaks_arr  # type: ignore[attr-defined]
    _map.colors = list(colors)  # type: ignore[attr-defined]
    _map.space = space  # type: ignore[attr-defined]
    return _map


def add_transparency(
    colors: Union[str, Sequence[str]],
    transparency: float,
) -> Union[str, List[str]]:
    """Add an alpha channel to colours.

    Parameters
    ----------
    colors : str or sequence of str
        One or more colour specifications accepted by matplotlib.
    transparency : float
        Transparency value in [0, 1] where 0 is fully opaque and 1 is
        fully transparent.

    Returns
    -------
    str or list of str
        ``"#RRGGBBAA"`` hex strings.  Returns a single string when a
        single colour is provided, otherwise a list.
    """
    import matplotlib.colors as mcolors

    scalar = isinstance(colors, str)
    if scalar:
        colors = [colors]

    alpha = int(round((1.0 - transparency) * 255))
    result: List[str] = []
    for c in colors:
        rgba = mcolors.to_rgba(c)
        r, g, b = (int(round(v * 255)) for v in rgba[:3])
        result.append(f"#{r:02X}{g:02X}{b:02X}{alpha:02X}")

    return result[0] if scalar else result


def rand_color(
    n: int,
    hue: Optional[Union[str, float]] = None,
    luminosity: Optional[str] = None,
) -> List[str]:
    """Generate *n* random colours.

    Uses HSV sampling to produce visually distinct colours.

    Parameters
    ----------
    n : int
        Number of colours to generate.
    hue : str or float, optional
        Reserved for future use (constraining hue range).
    luminosity : str, optional
        One of ``"bright"``, ``"dark"``, ``"light"``, or *None*.

    Returns
    -------
    list of str
        List of ``"#RRGGBB"`` hex strings.
    """
    import matplotlib.colors as mcolors

    rng = np.random.default_rng()
    result: List[str] = []
    for _ in range(n):
        h = rng.uniform(0, 1)
        s = rng.uniform(0.4, 1.0)
        v = rng.uniform(0.5, 1.0)
        if luminosity == "bright":
            v = rng.uniform(0.7, 1.0)
        elif luminosity == "dark":
            v = rng.uniform(0.2, 0.5)
        elif luminosity == "light":
            s = rng.uniform(0.1, 0.4)
            v = rng.uniform(0.8, 1.0)
        rgb = mcolors.hsv_to_rgb([h, s, v])
        result.append(_rgb_to_hex(np.asarray(rgb)))
    return result
