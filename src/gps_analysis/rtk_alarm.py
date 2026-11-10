"""Has an RTK baseline just moved? A window offset and its alarm level.

The real-time warning of the Vöktun page (``PLAN-rtk-trigger.md``, T1)
asks of each RTK baseline, every few minutes: *is the newest window off
the level of the window before it?* The statistic is the difference of two
medians, the short-term against the longer-term level, which is the RTK
form of the seismologist's short-term vs long-term average (Allen 1978)
used on real-time GNSS displacements by RAPiD (Ohta et al. 2012, §2) and
GEONET's REGARD (Kawamoto et al. 2017, §2.1):

    after  = epochs in (end − W, end]                 the newest window
    before = epochs in (end − W − G − R, end − W − G] the reference window

    Δ   = median(after) − median(before)              per component
    s   = 1.4826 · MAD(before)                        per component
    σ_Δ = 1.2533 · s · √(1/n_before + 1/n_after)

``s`` is the reference window's own robust scatter: on one-minute RTK
medians the noise is strongly correlated in time (multipath repeats,
atmosphere), so a first-difference noise (``recent_step.daily_noise``)
would be far too small; the scatter over the reference window holds the
wander at that time scale. 1.2533 = √(π/2) is the efficiency of a median
under white noise (:data:`gps_analysis.recent_step.MEDIAN_EFFICIENCY`).
σ_Δ is a guide only: the alarm edges come from the offsets themselves on
quiet data (T4), not from multiples of σ_Δ.

:func:`alarm_level` turns a value into a level by the caller's edges
(green 0, yellow 1, orange 2, red 3). Units are the caller's: ``t`` and
the window lengths in one time unit, ``y`` and the edges in one length.

References: Allen 1978, BSSA 68, 1521–1532; Ohta et al. 2012, JGR 117,
B02311 (doi:10.1029/2011JB008750); Kawamoto et al. 2017, JGR 122,
1324–1349 (doi:10.1002/2016JB013485).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gps_analysis.recent_step import MEDIAN_EFFICIENCY

FloatArray = NDArray[np.float64]

#: 1.4826·MAD = σ for Gaussian noise (Rousseeuw & Croux 1993, §1)
MAD_SIGMA = 1.4826


@dataclasses.dataclass(frozen=True)
class WindowOffset:
    """The newest window's offset from the reference window, per component.

    Attributes:
        offset: Δ = median(after) − median(before), shape (C,) [L].
        sigma: σ_Δ, shape (C,) [L] (a guide, see the module docstring).
        n_after: Epochs in the newest window.
        n_before: Epochs in the reference window.
    """

    offset: FloatArray
    sigma: FloatArray
    n_after: int
    n_before: int


def robust_scatter(y: ArrayLike) -> FloatArray:
    """The robust scatter s = 1.4826 · median(|y − median(y)|) per component.

    Args:
        y: Observations, shape (C, N) or (N,) [L]; NaN left out per component.

    Returns:
        s, shape (C,) [L] (NaN for a component with no finite epoch).

    Reference: Rousseeuw & Croux 1993, JASA 88, §1 (the MAD and its
    consistency factor 1.4826 at the Gaussian).
    """
    a = np.atleast_2d(np.asarray(y, dtype=np.float64))
    out = np.full(a.shape[0], np.nan)
    for c, row in enumerate(a):
        v = row[np.isfinite(row)]
        if v.size:
            out[c] = MAD_SIGMA * np.median(np.abs(v - np.median(v)))
    return out


def window_offset(
    t: ArrayLike,
    y: ArrayLike,
    end: float,
    *,
    window: float,
    reference: float,
    gap: float = 0.0,
    min_after: int = 3,
    min_before: int = 10,
) -> WindowOffset | None:
    """The offset Δ of the window ``(end − window, end]`` from the reference
    window ``(end − window − gap − reference, end − window − gap]``, with
    σ_Δ = 1.2533 · s_before · √(1/n_before + 1/n_after) (module docstring).

    Args:
        t: Epochs, shape (N,) [T], any order.
        y: Observations, shape (C, N) or (N,) [L]; an epoch with any
            component NaN is left out.
        end: The newest window's end [T], included.
        window: W, the newest window's length [T].
        reference: R, the reference window's length [T].
        gap: G, time left out between the two windows [T] (a slow onset
            then does not leak into the reference).
        min_after: Fewest epochs in the newest window.
        min_before: Fewest epochs in the reference window.

    Returns:
        The :class:`WindowOffset`, or None when a window has too few epochs.

    Reference: Ohta et al. 2012, §2 (the RAPiD short-term minus long-term
    level); the median form for robustness to RTK blunders.
    """
    tv = np.asarray(t, dtype=np.float64)
    a = np.atleast_2d(np.asarray(y, dtype=np.float64))
    ok = np.isfinite(tv) & np.isfinite(a).all(axis=0)
    tv, a = tv[ok], a[:, ok]
    start = end - window
    after = (tv > start) & (tv <= end)
    ref_end = start - gap
    before = (tv > ref_end - reference) & (tv <= ref_end)
    n_a, n_b = int(after.sum()), int(before.sum())
    if n_a < min_after or n_b < min_before:
        return None
    offset = np.median(a[:, after], axis=1) - np.median(a[:, before], axis=1)
    sigma = (
        MEDIAN_EFFICIENCY
        * robust_scatter(a[:, before])
        * np.sqrt(1.0 / n_b + 1.0 / n_a)
    )
    return WindowOffset(offset=offset, sigma=sigma, n_after=n_a, n_before=n_b)


def alarm_level(value: float, edges: ArrayLike) -> int:
    """The level of ``|value|`` against increasing ``edges``: the number of
    edges at or below it (0 green, 1 yellow, 2 orange, 3 red for three).

        level = #{k : edges_k ≤ |value|}

    Args:
        value: The statistic [L] (its sign is ignored); NaN gives 0.
        edges: Strictly increasing edges [L].

    Returns:
        The level, 0 … len(edges).

    Raises:
        ValueError: edges not strictly increasing.
    """
    e = np.asarray(edges, dtype=np.float64)
    if e.size > 1 and not np.all(np.diff(e) > 0):
        raise ValueError(f"edges must increase strictly, got {e.tolist()}")
    if not np.isfinite(value):
        return 0
    return int(np.sum(e <= abs(value)))
