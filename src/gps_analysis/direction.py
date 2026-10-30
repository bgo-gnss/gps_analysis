"""The direction of a motion over an interval, from robust end levels.

For the rotated plot axes of gps_plot (``--axes motion``): the direction a
station (or the rover of a baseline) moved between two epochs, with its
uncertainty, so it can be fixed in config and reused.

Derivation
----------
Given epochs t (any unit) and N/E/U values y (3, N) with σ (3, N):

1. The level at each end of [start, end] is the per-component MEDIAN of the
   samples within ``window`` of that end (inside the interval), robust to an
   outlier or two. Its σ is the larger of the formal weighted-mean σ
   (1/√Σσ⁻²) and the scatter's (1.2533·1.4826·MAD/√n: the median's
   efficiency times the robust standard deviation), so neither a few
   optimistic formal σ nor a lucky small scatter makes it tight.
2. The displacement d = level(end) − level(start), σ_d² = σ_s² + σ_e².
3. Azimuth α = atan2(d_E, d_N) (clockwise from north, 0…360°), plunge
   p = atan2(−d_U, |d_h|) (positive DOWN, as ``geofunc.axes.tilted_axes``).
   First-order propagation: σ_α = √(d_E²σ_N² + d_N²σ_E²)/|d_h|²,
   σ_p = √(d_U²(d_N²σ_N² + d_E²σ_E²)/|d_h|² + |d_h|²σ_U²)/|d|².

A motion within its own noise has no direction: :attr:`Direction.significant`
says whether |d_h| exceeds three times its σ.

Pure numpy, no I/O, inputs never mutated.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: median efficiency × MAD → σ of a normal sample
_MEDIAN_SIGMA = 1.2533 * 1.4826

#: |d_h| / σ(|d_h|) above which a direction is called significant
SIGNIFICANCE = 3.0


@dataclasses.dataclass(frozen=True)
class Direction:
    """A motion's direction over an interval.

    Attributes:
        azimuth: Horizontal direction [°, 0…360, clockwise from north].
        sigma_azimuth: Its 1σ [°].
        plunge: Angle below the horizontal [°, + down].
        sigma_plunge: Its 1σ [°].
        displacement: d (3,), N/E/U, the caller's unit.
        sigma: σ of d (3,).
        n_start, n_end: Samples behind each end level.
    """

    azimuth: float
    sigma_azimuth: float
    plunge: float
    sigma_plunge: float
    displacement: FloatArray
    sigma: FloatArray
    n_start: int
    n_end: int

    @property
    def horizontal(self) -> float:
        """|d_h|, the horizontal size of the motion."""
        return float(np.hypot(self.displacement[0], self.displacement[1]))

    @property
    def significant(self) -> bool:
        """Whether |d_h| is more than :data:`SIGNIFICANCE` σ (a real direction)."""
        dn, de = self.displacement[:2]
        h = self.horizontal
        if h == 0.0:
            return False
        s = np.sqrt((dn * self.sigma[0]) ** 2 + (de * self.sigma[1]) ** 2) / h
        return bool(h > SIGNIFICANCE * s)


def end_level(y: ArrayLike, sigma: ArrayLike) -> tuple[FloatArray, FloatArray, int]:
    """Robust level (3,), its σ (3,) and the sample count, of (3, n) samples.

    Per component over the finite samples: the median; σ = max(formal
    weighted-mean σ, scatter σ of the median). NaN where none is finite.
    """
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(sigma, dtype=float)
    level = np.full(yy.shape[0], np.nan)
    err = np.full(yy.shape[0], np.nan)
    n = 0
    for c in range(yy.shape[0]):
        ok = np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
        k = int(ok.sum())
        if k == 0:
            continue
        v = yy[c, ok]
        level[c] = float(np.median(v))
        formal = 1.0 / np.sqrt(np.sum(ss[c, ok] ** -2))
        scatter = _MEDIAN_SIGMA * float(np.median(np.abs(v - level[c]))) / np.sqrt(k)
        err[c] = max(float(formal), scatter)
        n = max(n, k)
    return level, err, n


def motion_direction(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    start: float,
    end: float,
    window: float,
) -> Direction:
    """The direction of the motion of ``y`` between ``start`` and ``end``.

    Args:
        t: Epochs (N,), any monotonic unit (fractional years, seconds …).
        y: N/E/U values (3, N), the caller's unit.
        sigma: Their 1σ (3, N).
        start, end: The interval, in ``t``'s unit.
        window: How far into the interval from each end the end levels
            reach, in ``t``'s unit (e.g. 7 days of daily solutions).

    Raises:
        ValueError: ``start`` not before ``end``, a window not positive or
            wider than half the interval, or an end with no finite sample.
    """
    if not start < end:
        raise ValueError(f"start {start} is not before end {end}")
    if not 0 < window <= (end - start) / 2:
        raise ValueError(f"window {window} must be in (0, {(end - start) / 2}]")
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(sigma, dtype=float)
    first = (tt >= start) & (tt <= start + window)
    last = (tt >= end - window) & (tt <= end)
    l0, s0, n0 = end_level(yy[:, first], ss[:, first])
    l1, s1, n1 = end_level(yy[:, last], ss[:, last])
    if not (np.isfinite(l0).all() and np.isfinite(l1).all()):
        raise ValueError("no samples at one end of the interval")
    d = l1 - l0
    sd = np.sqrt(s0**2 + s1**2)
    dn, de, du = (float(x) for x in d)
    sn, se, su = (float(x) for x in sd)
    h2 = dn * dn + de * de
    h = np.sqrt(h2)
    azimuth = float(np.degrees(np.arctan2(de, dn)) % 360.0)
    plunge = float(np.degrees(np.arctan2(-du, h)))
    if h2 > 0:
        s_az = np.degrees(np.sqrt((de * sn) ** 2 + (dn * se) ** 2) / h2)
        r2 = h2 + du * du
        s_pl = np.degrees(
            np.sqrt(du * du * ((dn * sn) ** 2 + (de * se) ** 2) / h2 + h2 * su * su)
            / r2
        )
    else:
        s_az = s_pl = float("inf")
    return Direction(
        azimuth=azimuth,
        sigma_azimuth=float(s_az),
        plunge=plunge,
        sigma_plunge=float(s_pl),
        displacement=d,
        sigma=sd,
        n_start=n0,
        n_end=n1,
    )
