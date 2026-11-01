"""Is a station moving *now*? A step statistic for short windows.

A straight line through one week of daily solutions has a rate σ of a few
hundred mm/yr (7 points, ~2 mm daily noise, times the empirical k(T) of
:mod:`gps_analysis.rate_quality`): a 10 mm slip in a week never stands out
of it. The question a landslide map asks of its last week is simpler —
*has the station moved since the weeks before?* — and a step between two
medians answers it with far more of the data:

    before = the last ``before_days`` daily epochs before the window
             (looked for up to ``lookback_days`` back, so a step across a
             data gap is still measured)
    after  = the epochs in the window (its last ``window_days`` days)

    Δ   = median(after) − median(before)                     per component
    σ_Δ = 1.2533 · σ_day · √(1/n_before + 1/n_after)
    v   = Δ / (t̃_after − t̃_before)                          [L/yr]

``σ_day`` is the robust day-to-day noise of the series
(:func:`daily_noise`: 1.4826·MAD of the first differences / √2, which a
slow trend or a seasonal term hardly touches); 1.2533 = √(π/2) is the
efficiency loss of a median against a mean under white noise. ``t̃`` are
the median epochs of the two sets, so ``v`` is the mean rate between
them, in the units of a velocity map.

Written for the Seyðisfjörður slope stations (SEY?-SEYD baselines, BGÓ
2026-11-01): the step finds the slips of Dec 2022, Mar–Apr 2023, Jan 2024,
Jan 2025 and Dec 2025, and stays near zero on quiet weeks, where a 7-day
line could find neither.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: √(π/2): σ of a median of n white samples is this times σ/√n
MEDIAN_EFFICIENCY = 1.2533141373155003
#: days per year of the epochs (fractional years, ``yearf``)
YEAR_DAYS = 365.25


@dataclasses.dataclass(frozen=True)
class RecentStep:
    """The step of a window against the epochs before it, per component.

    Attributes:
        step: Δ = median(after) − median(before), shape (C,) [L].
        sigma: 1-σ of ``step`` [L].
        rate: ``step`` over the time between the two sets' median epochs,
            shape (C,) [L/yr].
        rate_sigma: 1-σ of ``rate`` [L/yr].
        n_before: Epochs in the before set.
        n_after: Epochs in the window.
        span_years: Time between the two sets' median epochs [yr].
    """

    step: FloatArray
    sigma: FloatArray
    rate: FloatArray
    rate_sigma: FloatArray
    n_before: int
    n_after: int
    span_years: float


def daily_noise(y: ArrayLike) -> FloatArray:
    """Robust day-to-day noise per component: 1.4826·MAD of the first
    differences, over √2 (each difference holds two days' noise).

    Args:
        y: Observations, shape (C, N) or (N,) [L], in time order; NaN
            epochs are left out per component.

    Returns:
        σ_day, shape (C,) [L] (NaN for a component with < 3 epochs).
    """
    a = np.atleast_2d(np.asarray(y, dtype=np.float64))
    out = np.full(a.shape[0], np.nan)
    for c, row in enumerate(a):
        d = np.diff(row[np.isfinite(row)])
        if d.size >= 2:
            out[c] = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    return out


def recent_step(
    t: ArrayLike,
    y: ArrayLike,
    end: float,
    *,
    window_days: float = 7.0,
    before_days: int = 14,
    lookback_days: float = 120.0,
    min_before: int = 5,
    min_after: int = 4,
    noise: ArrayLike | None = None,
) -> RecentStep | None:
    """The step of the window ``(end − window_days, end]`` against the
    ``before_days`` epochs before it (module docstring).

    Args:
        t: Epochs, shape (N,) [yr] (fractional years), any order.
        y: Observations, shape (C, N) or (N,) [L]; an epoch with any
            component NaN is left out.
        end: The window's end [yr], included.
        window_days: The window's length [days].
        before_days: How many epochs before the window make the before set.
        lookback_days: How far back from the window's start the before set
            may reach [days].
        min_before: Fewest epochs of the before set.
        min_after: Fewest epochs in the window.
        noise: σ_day per component, shape (C,) [L]; default
            :func:`daily_noise` of the whole input (pass a longer series
            than the two sets for a steadier σ).

    Returns:
        The :class:`RecentStep`, or None when a set has too few epochs.
    """
    tv = np.asarray(t, dtype=np.float64)
    a = np.atleast_2d(np.asarray(y, dtype=np.float64))
    order = np.argsort(tv)
    tv, a = tv[order], a[:, order]
    ok = np.isfinite(tv) & np.isfinite(a).all(axis=0)
    tv, a = tv[ok], a[:, ok]
    start = end - window_days / YEAR_DAYS
    tol = 1e-6
    after = (tv > start + tol) & (tv <= end + tol)
    before_idx = np.flatnonzero(
        (tv <= start + tol) & (tv > start - lookback_days / YEAR_DAYS)
    )[-before_days:]
    n_a, n_b = int(after.sum()), int(before_idx.size)
    if n_a < min_after or n_b < min_before:
        return None
    sig_day = daily_noise(a) if noise is None else np.asarray(noise, np.float64)
    step = np.median(a[:, after], axis=1) - np.median(a[:, before_idx], axis=1)
    sigma = MEDIAN_EFFICIENCY * sig_day * np.sqrt(1.0 / n_b + 1.0 / n_a)
    span = float(np.median(tv[after]) - np.median(tv[before_idx]))
    return RecentStep(
        step=step,
        sigma=sigma,
        rate=step / span,
        rate_sigma=sigma / span,
        n_before=n_b,
        n_after=n_a,
        span_years=span,
    )
