"""Rate changes in a series: where a straight line stops fitting.

For ``gpsplot ts --rate-segments auto`` (PLAN-ts-quakes.md Q-T2): the epochs
where the rate of one component (or of one projected direction) changes, so
each span between them can get its own rate from the product estimator. This
module only *places* the breaks; it does not report rates.

Method (the Bárðarbunga/VONC prototype of 2026-11-15)
------------------------------------------------------
1. Bin the samples (default one week) and keep the per-bin median of t and
   y: a few outliers and the daily noise then barely move the fit.
2. Fit a continuous piecewise-linear model (offset, rate, and a hinge
   ``max(t − b, 0)`` per break b) by least squares on the bin medians.
3. Candidate breaks lie on a regular grid (default 1/12 of the time unit,
   about month starts for fractional years), at least ``min_gap`` from both
   ends and from each other.
4. Greedy forward selection: add the candidate that lowers the residual sum
   of squares most, then move every break to its best candidate with the
   others held (coordinate descent, repeated until none moves: a first pick
   between two true breaks is not frozen), and keep the new set only if the
   BIC,
   ``n·ln(RSS/n) + k·ln(n)`` with ``k = 2 + 2·(number of breaks)`` (a break's
   time counts as a parameter), drops by at least ``min_bic_gain``.

The continuous model ties the slopes at the knots, so its rates are biased
next to a sharp change; that is why the rates come from an independent
estimate per span (:func:`segment_edges` gives the spans).

Model (step 2), for breaks b₁…b_k and bin epochs tᵢ::

    ŷ(tᵢ) = a + r·(tᵢ − t₁) + Σⱼ Δrⱼ·max(tᵢ − bⱼ, 0)

a [mm], r [mm/yr] the first span's rate, Δrⱼ [mm/yr] the rate change at bⱼ
[yr]: continuous at every bⱼ (a "broken-stick" / segmented regression,
Hinkley 1971; Muggeo 2003). Selection criterion (step 4)::

    BIC = n·ln(RSS/n) + (2 + 2k)·ln(n)       (Schwarz 1978)

with n bins and RSS the residual sum of squares of the bin medians; each
break adds two parameters (Δrⱼ and bⱼ), the usual count for an estimated
change point (Hinkley 1971).

Numerical notes: least squares by ``numpy.linalg.lstsq`` (SVD), so nearly
collinear hinges near the ends do not blow up; an exact fit is floored at
the smallest float so ln(RSS) stays finite. The search is O(k·m) fits for m
candidates (a few hundred for decades of data, weekly bins: fast).

Pure numpy, no I/O, inputs never mutated, float64. Any time unit works,
but the defaults (``bin_width``, ``candidate_step``, ``min_gap``) are in
years.

References: Schwarz 1978 (BIC); Hinkley 1971 (two-phase regression with a
continuous join); Muggeo 2003 (segmented regression).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: one week in years
WEEK = 1.0 / 52.18
#: one month in years: the candidate grid
MONTH = 1.0 / 12.0
#: a series longer than this (years) gets LONG_GAP between breaks, else SHORT_GAP
SHORT_SPAN = 3.0
LONG_GAP = 0.5
SHORT_GAP = 2.0 / 12.0
#: least BIC drop for a new break
MIN_BIC_GAIN = 10.0
#: most breaks looked for
MAX_BREAKS = 8


@dataclasses.dataclass(frozen=True)
class RateBreaks:
    """Breaks found in a series.

    Attributes:
        breaks: Break epochs, ascending (the caller's time unit).
        bic: BIC of each accepted model: ``bic[0]`` with no break,
            ``bic[i]`` after the i-th break was added (selection order).
        order: The candidates in the order they were added (before the
            refinement moved them).
        min_gap: The gap used (the default depends on the span).
        n_bins: Bins the fit used.
    """

    breaks: tuple[float, ...]
    bic: tuple[float, ...]
    order: tuple[float, ...]
    min_gap: float
    n_bins: int


def default_min_gap(span: float) -> float:
    """``LONG_GAP`` for a span over ``SHORT_SPAN`` years, else ``SHORT_GAP``."""
    return LONG_GAP if span > SHORT_SPAN else SHORT_GAP


def bin_medians(
    t: ArrayLike, y: ArrayLike, width: float = WEEK
) -> tuple[FloatArray, FloatArray]:
    """Per-bin medians of t and y, bins of ``width`` from the first epoch.

    Non-finite samples are dropped first. Returns ``(t_bin, y_bin)``,
    ascending in time.
    """
    if width <= 0:
        raise ValueError(f"bin width must be > 0, got {width}")
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    if tt.shape != yy.shape or tt.ndim != 1:
        raise ValueError("t and y must be 1-D and the same length")
    ok = np.isfinite(tt) & np.isfinite(yy)
    tt, yy = tt[ok], yy[ok]
    if tt.size == 0:
        return np.empty(0), np.empty(0)
    idx = np.floor((tt - tt.min()) / width).astype(np.int64)
    order = np.argsort(idx, kind="stable")
    idx, tt, yy = idx[order], tt[order], yy[order]
    cuts = np.flatnonzero(np.diff(idx)) + 1
    tb = np.array([np.median(a) for a in np.split(tt, cuts)])
    yb = np.array([np.median(a) for a in np.split(yy, cuts)])
    return tb, yb


def _design(breaks: list[float], t: FloatArray, t0: float) -> FloatArray:
    cols = [np.ones_like(t), t - t0] + [np.maximum(t - b, 0.0) for b in breaks]
    return np.column_stack(cols)


def _rss(breaks: list[float], t: FloatArray, y: FloatArray) -> float:
    x = _design(breaks, t, float(t[0]))
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    r = y - x @ coef
    return float(r @ r)


def _bic(rss: float, n: int, n_breaks: int) -> float:
    rss = max(rss, np.finfo(float).tiny * n)  # an exact fit is not -inf
    return float(n * np.log(rss / n) + (2 + 2 * n_breaks) * np.log(n))


def rate_breaks(
    t: ArrayLike,
    y: ArrayLike,
    *,
    min_gap: float | None = None,
    bin_width: float = WEEK,
    candidate_step: float = MONTH,
    min_bic_gain: float = MIN_BIC_GAIN,
    max_breaks: int = MAX_BREAKS,
) -> RateBreaks:
    """Find where the rate of ``y`` changes (see the module docstring).

    Args:
        t: Epochs (N,), e.g. fractional years.
        y: Values (N,), any unit.
        min_gap: Least distance between two breaks and from either end;
            None = :func:`default_min_gap` of the span.
        bin_width: Bin width for the medians.
        candidate_step: Candidate breaks on multiples of this.
        min_bic_gain: Least BIC drop to accept a break.
        max_breaks: Most breaks.

    Returns:
        :class:`RateBreaks`; no breaks when the series is too short for one.

    Raises:
        ValueError: Bad shapes or non-positive widths/steps.
    """
    if candidate_step <= 0:
        raise ValueError(f"candidate_step must be > 0, got {candidate_step}")
    tb, yb = bin_medians(t, y, bin_width)
    n = int(tb.size)
    span = float(tb[-1] - tb[0]) if n else 0.0
    gap = default_min_gap(span) if min_gap is None else float(min_gap)
    if n < 4:
        return RateBreaks((), (), (), gap, n)
    first = np.ceil(tb[0] / candidate_step) * candidate_step + gap
    cand = np.arange(first, tb[-1] - gap, candidate_step)
    chosen: list[float] = []
    order: list[float] = []
    bic = [_bic(_rss(chosen, tb, yb), n, 0)]
    while len(chosen) < max_breaks and n > 2 + 2 * (len(chosen) + 1):
        best: tuple[float, float] | None = None
        for b in cand:
            if any(abs(b - c) < gap for c in chosen):
                continue
            r = _rss([*chosen, float(b)], tb, yb)
            if best is None or r < best[0]:
                best = (r, float(b))
        if best is None:
            break
        trial, rss = _refine([*chosen, best[1]], cand, gap, tb, yb)
        new = _bic(rss, n, len(chosen) + 1)
        if new > bic[-1] - min_bic_gain:
            break
        chosen = trial
        order.append(best[1])
        bic.append(new)
    return RateBreaks(tuple(sorted(chosen)), tuple(bic), tuple(order), gap, n)


def _refine(
    breaks: list[float], cand: FloatArray, gap: float, t: FloatArray, y: FloatArray
) -> tuple[list[float], float]:
    """Move each break to its best candidate with the others held, until none
    moves (coordinate descent on RSS; never increases it). Greedy selection
    alone keeps its first pick, which can sit between two true breaks."""
    current = list(breaks)
    rss = _rss(current, t, y)
    for _ in range(20):
        moved = False
        for i in range(len(current)):
            others = current[:i] + current[i + 1 :]
            for b in cand:
                if any(abs(b - c) < gap for c in others):
                    continue
                r = _rss([*others, float(b)], t, y)
                if r < rss - 1e-12 * max(rss, 1.0):
                    rss, current[i], moved = r, float(b), True
        if not moved:
            break
    return current, rss


def segment_edges(
    start: float, end: float, breaks: tuple[float, ...] | list[float]
) -> list[tuple[float, float]]:
    """The spans ``[(start, b1), (b1, b2), …, (bk, end)]`` between breaks.

    Breaks outside ``(start, end)`` are ignored.

    Raises:
        ValueError: ``end`` not after ``start``.
    """
    if not end > start:
        raise ValueError(f"end {end} must be after start {start}")
    inner = sorted(b for b in breaks if start < b < end)
    edges = [start, *inner, end]
    return list(zip(edges[:-1], edges[1:], strict=True))
