"""Displacement across a period: end levels, their σ, and rate ↔ displacement.

The displacement of a station between two epochs is the difference of its
LEVEL at each end, a level being the weighted mean of the solutions in a
short window (``--ends N`` days) at that end. One day per end is allowed
(the most intense unrest, where the motion dwarfs the daily noise), and
then a single blunder decides the number — :func:`neighbour_check` tests
such an end against a line through the adjacent days.

σ of a level is the largest of three independent estimates, never their
product (each already describes the same daily noise; multiplying them
counts it twice):

1. ``k_day · σ_apr`` — the formal σ scaled by the station's measured daily
   factor (:func:`~gps_analysis.empirical_sigma.sigma_scale_factor`);
2. ``σ_apr · √(χ²/ν)`` — the scatter inside the window (a blunder or real
   motion within the window inflates it);
3. ``s_day / √n`` — the station's robust day-to-day scatter
   (:func:`~gps_analysis.empirical_sigma.daily_scatter`) averaged down.

:func:`displacement_from_rate` and :func:`rate_from_displacement` are the
exact inverse conversions (rate × span, displacement / span) with σ carried
the same way. :func:`line_value_at` evaluates a
:class:`~gps_analysis.rate_quality.LineFit` at an epoch with the level σ
scaled like an end level and the rate σ scaled by the caller's k(T).

Pure: windows, k_day, s_day and k(T) are the caller's (geo_dataread
measures them per station); units are the caller's (L, L/yr).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from numpy.typing import ArrayLike

from .models import FloatArray
from .rate_quality import LineFit, line_fit

#: the N ≤ 2 end check: a level further than this many σ from the line
#: through its neighbours is disputed
NEIGHBOUR_SIGMA = 4.0
#: fewest neighbour epochs behind that line (fewer: the end is unchecked)
NEIGHBOUR_MIN_OBS = 3


@dataclasses.dataclass(frozen=True)
class EndLevel:
    """The level of one component in one end window.

    Attributes:
        level: Weighted mean ``L = Σwy/Σw`` [L].
        sigma_apriori: ``1/√Σw`` from the data σ alone [L].
        sigma: The σ used, the max of the three estimates (module doc) [L].
        n: Epochs used.
        t_mean: Weighted mean epoch [yr].
        span: (first, last) epoch used [yr].
        chi2nu: ``Σw(y−L)²/(n−1)``; NaN below 3 epochs.
    """

    level: float
    sigma_apriori: float
    sigma: float
    n: int
    t_mean: float
    span: tuple[float, float]
    chi2nu: float


def _window(
    t: ArrayLike, y: ArrayLike, sigma: ArrayLike, window: tuple[float, float]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ss = np.asarray(sigma, dtype=np.float64)
    if not tt.shape == yy.shape == ss.shape or tt.ndim != 1:
        raise ValueError("t, y, sigma must be 1-D arrays of one length")
    start, end = window
    keep = (tt >= start) & (tt < end) & np.isfinite(yy) & np.isfinite(ss) & (ss > 0)
    return tt[keep], yy[keep], ss[keep]


def level_sigma(
    sigma_apriori: float,
    chi2nu: float,
    n: int,
    *,
    k_day: float = 1.0,
    s_day: float = 0.0,
) -> float:
    """σ of a weighted mean of ``n`` epochs: the max of three estimates.

    Equation:
        ``σ = max(k_day·σ_apr, σ_apr·√(χ²/ν), s_day/√n)``

    Symbols → args:
        - ``σ_apr`` → ``sigma_apriori``: ``1/√Σw`` [L]
        - ``χ²/ν`` → ``chi2nu`` (NaN: term skipped) [–]
        - ``k_day`` → the station's daily σ factor (NaN: 1) [–]
        - ``s_day`` → the station's daily scatter (NaN: 0) [L]
    """
    k = k_day if math.isfinite(k_day) and k_day > 0 else 1.0
    out = k * sigma_apriori
    if math.isfinite(chi2nu):
        out = max(out, sigma_apriori * math.sqrt(chi2nu))
    if math.isfinite(s_day) and s_day > 0 and n > 0:
        out = max(out, s_day / math.sqrt(n))
    return out


def end_level(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    window: tuple[float, float],
    k_day: float = 1.0,
    s_day: float = 0.0,
) -> EndLevel:
    """The inverse-variance weighted level of one component in a window.

    Equation (w = 1/σᵢ², epochs ``start ≤ t < end`` with finite y, σ > 0):
        ``L = Σwy/Σw``, ``σ_apr = 1/√Σw``, ``χ²/ν = Σw(y−L)²/(n−1)``,
        σ by :func:`level_sigma`.

    Symbols → args:
        - ``t``, ``y``, ``sigma`` → one component, (N,) [yr, L, L]
        - ``window`` → half-open ``[start, end)`` [yr]
        - ``k_day``, ``s_day`` → :func:`level_sigma`

    Raises:
        ValueError: no usable epoch in the window.

    Numerical notes:
        Inverse-VARIANCE weights (minimum variance) — unlike the legacy
        1/σ weights of :func:`~gps_analysis.baseline.estimate_offset`,
        which stays as it is for its golden masters.
    """
    tt, yy, ss = _window(t, y, sigma, window)
    n = tt.size
    if n == 0:
        raise ValueError(f"no epochs in [{window[0]:.4f}, {window[1]:.4f})")
    w = 1.0 / ss**2
    sw = float(np.sum(w))
    level = float(np.sum(w * yy) / sw)
    chi2nu = float(np.sum(w * (yy - level) ** 2)) / (n - 1) if n >= 3 else math.nan
    s_apr = 1.0 / math.sqrt(sw)
    return EndLevel(
        level=level,
        sigma_apriori=s_apr,
        sigma=level_sigma(s_apr, chi2nu, n, k_day=k_day, s_day=s_day),
        n=n,
        t_mean=float(np.sum(w * tt) / sw),
        span=(float(tt[0]), float(tt[-1])),
        chi2nu=chi2nu,
    )


def end_displacement(start: EndLevel, end: EndLevel) -> tuple[float, float]:
    """``(Δ, σ_Δ)``: ``L_end − L_start``, ``√(σ_start² + σ_end²)``.

    The two ends are independent (disjoint windows).
    """
    return end.level - start.level, math.hypot(start.sigma, end.sigma)


def line_value_at(
    fit: LineFit,
    t_star: float,
    *,
    k_day: float = 1.0,
    k_rate: float = 1.0,
) -> tuple[float, float]:
    """The value of a fitted line at an epoch, with its σ.

    Equation:
        ``ŷ = a + v·(t* − t̄)``,
        ``σ_ŷ² = σ_a'² + (k_rate·σ_v)²·(t* − t̄)²``,
        ``σ_a' = max(k_day, √(χ²/ν))·σ_a``

    Symbols → args:
        - ``a``, ``σ_a``, ``t̄``, ``v``, ``σ_v``, ``χ²/ν`` → ``fit``
          (``level``, ``sigma_level_apriori``, ``t_mean``, ``rate``,
          ``sigma_short``, ``chi2nu``)
        - ``k_rate`` → the caller's k(T) of the fit window [–]

    The level σ is scaled like an end level (formal σ is too small by
    k_day); the rate σ by k(T), which is calibrated against the χ²-scaled
    formal σ of a rate.
    """
    if not math.isfinite(fit.level):
        raise ValueError(
            "LineFit has no level (made before gps_analysis 0.1 level fields)"
        )
    k = k_day if math.isfinite(k_day) and k_day > 0 else 1.0
    if math.isfinite(fit.chi2nu):
        k = max(k, math.sqrt(fit.chi2nu))
    dt = t_star - fit.t_mean
    sigma = math.hypot(k * fit.sigma_level_apriori, k_rate * fit.sigma_short * dt)
    return fit.level + fit.rate * dt, sigma


@dataclasses.dataclass(frozen=True)
class NeighbourCheck:
    """Whether a one- or two-day end level agrees with its neighbours.

    Attributes:
        disputed: True when it does not; None when it could not be checked
            (too few neighbour epochs).
        residual: ``L − ŷ`` [L] (NaN when unchecked).
        sigma: ``√(σ_L² + σ_ŷ²)`` [L] (NaN when unchecked).
        n_neighbour: Neighbour epochs used.
        reason: Why unchecked, or "".
    """

    disputed: bool | None
    residual: float
    sigma: float
    n_neighbour: int
    reason: str = ""


def neighbour_check(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    end: EndLevel,
    neighbour: tuple[float, float],
    k_day: float = 1.0,
    n_sigma: float = NEIGHBOUR_SIGMA,
    min_obs: int = NEIGHBOUR_MIN_OBS,
) -> NeighbourCheck:
    """Test an end level against the line through the adjacent days.

    Equation:
        ŷ, σ_ŷ by :func:`line_value_at` of the line through ``neighbour``
        at ``end.t_mean``; disputed when ``|L − ŷ| > n_sigma·√(σ_L² + σ_ŷ²)``.

    Symbols → args:
        - ``neighbour`` → ``[start, end)`` of the adjacent days (the days
          after a start window, the days before an end window) [yr]
        - ``end`` → the level under test (:func:`end_level`)

    The line, not a mean, so steady fast motion (tens of mm/day in a dike
    intrusion) is not mistaken for a blunder.
    """
    tt, yy, ss = _window(t, y, sigma, neighbour)
    n = tt.size
    if n < min_obs or float(np.ptp(tt)) <= 0:
        return NeighbourCheck(None, math.nan, math.nan, n, f"neighbour n={n}<{min_obs}")
    fit = line_fit(tt, yy, ss)
    pred, s_pred = line_value_at(fit, end.t_mean, k_day=k_day)
    resid = end.level - pred
    s = math.hypot(end.sigma, s_pred)
    return NeighbourCheck(bool(abs(resid) > n_sigma * s), resid, s, n)


def displacement_from_rate(
    rate: ArrayLike, sigma_rate: ArrayLike, span_years: float
) -> tuple[FloatArray, FloatArray]:
    """``(v·T, σ_v·T)``: a rate projected through a period of T years."""
    if not span_years > 0:
        raise ValueError(f"span must be > 0, got {span_years}")
    v = np.asarray(rate, dtype=np.float64)
    s = np.asarray(sigma_rate, dtype=np.float64)
    return v * span_years, s * span_years


def rate_from_displacement(
    displacement: ArrayLike, sigma: ArrayLike, span_years: float
) -> tuple[FloatArray, FloatArray]:
    """``(d/T, σ_d/T)``: the average rate of a displacement over T years."""
    if not span_years > 0:
        raise ValueError(f"span must be > 0, got {span_years}")
    d = np.asarray(displacement, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    return d / span_years, s / span_years


@dataclasses.dataclass(frozen=True)
class StepEstimate:
    """A displacement across an event from lines fitted on either side.

    Attributes:
        step: ``f_after(t_after) − f_before(t_before)`` [L].
        sigma: ``√(σ_before(t_before)² + σ_after(t_after)²)`` [L].
        at: (t_before, t_after), where each line is evaluated [yr].
        before, after: The two line fits.
    """

    step: float
    sigma: float
    at: tuple[float, float]
    before: LineFit
    after: LineFit


def step_displacement(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    before: tuple[float, float],
    after: tuple[float, float],
    at: tuple[float, float] | None = None,
    k_day: float = 1.0,
    k_rate: tuple[float, float] = (1.0, 1.0),
) -> StepEstimate:
    """The jump across an event: two lines, each evaluated at the event.

    Equation:
        ``Δ = f_after(t_a) − f_before(t_b)``,
        ``σ_Δ² = σ_before(t_b)² + σ_after(t_a)²`` (:func:`line_value_at`)

    Symbols → args:
        - ``before``, ``after`` → half-open windows ``[start, end)`` on
          either side of the event [yr]
        - ``t_b``, ``t_a`` → ``at``; default the event's edges, the end of
          ``before`` and the start of ``after`` (the event's own days are
          in neither line) [yr]
        - ``k_day`` → daily σ factor of the levels; ``k_rate`` → k(T) of
          each window's rate [–]

    The sign is AFTER − BEFORE (as
    :func:`~gps_analysis.baseline.estimate_step_offset`); the legacy
    ``find_line_offsets`` of gps_data_analyses reported before − after,
    evaluated at the midpoint of the gap — pass ``at=(m, m)`` for that.

    Raises:
        ValueError: fewer than 2 epochs (or one epoch time) on a side.
    """
    tb, yb, sb = _window(t, y, sigma, before)
    ta, ya, sa = _window(t, y, sigma, after)
    fb = line_fit(tb, yb, sb)
    fa = line_fit(ta, ya, sa)
    t_b, t_a = at if at is not None else (before[1], after[0])
    vb, s_b = line_value_at(fb, t_b, k_day=k_day, k_rate=k_rate[0])
    va, s_a = line_value_at(fa, t_a, k_day=k_day, k_rate=k_rate[1])
    return StepEstimate(va - vb, math.hypot(s_b, s_a), (t_b, t_a), fb, fa)
