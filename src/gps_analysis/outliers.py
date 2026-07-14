"""Model-aware outlier detection with signal protection for GNSS series.

Implements ``docs/DESIGN_outlier_detection.md`` §3–§4: outliers are
identified against a robustly fitted, **step-augmented** trajectory model
— never against the raw series — and every candidate must survive an
explicit signal-protection stage before it may be flagged. The result is
always a boolean mask plus reason/protection bitmasks and diagnostics;
observations are never filtered, replaced, or mutated.

Derivation chain
----------------
Per component c of one station, with epochs t ∈ ℝᴺ (fractional years),
observations y ∈ ℝᴺ [L] and optional formal 1-σ uncertainties σ ∈ ℝᴺ [L]:

0. **Stage-0 gross-blunder despike** (§3.0 addendum; **off by default**,
   ``despike=True`` opts in). Before any model is fitted, isolated
   single-epoch spikes that are extreme against BOTH temporal neighbors
   *and return to baseline* are masked out of every subsequent fit: with
   gap-aware first differences δ⁻/δ⁺ (:func:`neighbor_differences`) and
   the robust difference scale ``ŝ_Δ = mad_scale(δ⁻)``, epoch i is a
   gross blunder (:func:`spike_mask`) iff
   ``min(|δ⁻_i|, |δ⁺_i|) > k_d·ŝ_Δ`` ∧ ``δ⁻_i·δ⁺_i < 0`` ∧
   ``|δ⁻_i + δ⁺_i| ≤ c_r·ŝ_Δ``. A persistent offset (a real step: the
   level shifts and STAYS) fails the first two conditions and is never
   despiked. Despiked epochs are masked — never deleted — tagged
   ``REASON_GROSS``, excluded from the robust fit and the identifier
   candidates, force-flagged, and counted in ``n_despiked``; caller
   ``protect_windows`` disable despiking outright.
1. **Step-augmented robust fit** (§3.1). The trajectory
   ``f(t; p, a) = f_traj(t; p) + Σ_k a_k·H(t − t_k)``
   (:func:`gps_analysis.fitting.with_steps`,
   :func:`gps_analysis.models.heaviside_steps`; epochs t_k fixed,
   amplitudes a_k estimated) is fitted by an M-estimator (Huber loss,
   ``scipy.optimize.least_squares`` via
   :func:`gps_analysis.fitting._robust_params`; Huber 1964) so gross
   outliers cannot drag the model. Residuals ``r_i = y_i − f(t_i; p̂, â)``
   are evaluated on **all** epochs each sweep.
2. **Whitening** (§3.1). ``w_i = r_i / σ_i`` (:func:`whiten`; σ_i ≡ 1 when
   absent). Formal σ contributes the epoch-to-epoch quality *ratio* only —
   the absolute detection scale is always estimated from the residuals.
3. **Robust scale + global identifier** (§3.2–§3.3a).
   ``ẑ_i = (w_i − med(w)) / ŝ`` (:func:`standardize_robust`) with ŝ the
   normalized MAD (:func:`mad_scale`, default) or Qn (:func:`qn_scale`,
   optional); global candidates ``G_i : |ẑ_i| > k_g`` (Davies & Gather
   1993 identifier framing).
4. **Windowed Hampel identifier** (§3.3b). Local center ``m_i`` and
   scale ``s_i`` of selectable polynomial order (``window_order``):
   order 0 (default) is the local median
   ``m_i = med{w_j : |t_j − t_i| ≤ h}`` (:func:`rolling_median`) with
   the local MAD about it (:func:`rolling_mad`); orders 1/2 are a
   **robust local polynomial** (LOWESS — tricube distance weights +
   bisquare robustness iterations, Cleveland 1979;
   :func:`rolling_polyfit`) with ``m_i = f̂_i(t_i)`` and ``s_i`` the MAD
   of the local fit residuals — during fast transients a local constant
   lags the motion and over-flags on-trend epochs, a local line/parabola
   tracks the slope/curvature. Decision (either order):
   ``L_i : |w_i − m_i| > k_w·max(s_i, s_floor)`` (:func:`hampel_mask`;
   Hampel 1974; Pearson et al. 2016). Windows are defined **in time**;
   thin windows (< n_min samples) fall back to the global center/scale —
   a documented degradation, never a silent zero.
5. **Signal protection** (§3.4). Candidates ``G ∪ L`` are grouped into
   clusters (:func:`candidate_clusters`) and protected — recorded in the
   ``PROTECT_*`` bitmask, surfaced as :class:`SuspectedEvent` hints, and
   **not** flagged — by: the physical magnitude floor, the run-length /
   same-sign guard, the step-evidence statistic
   ``D = |med(r_post) − med(r_pre)| / ŝ`` (:func:`step_evidence`;
   Gazeaux et al. 2013 motivates the conservatism), and caller-supplied
   protected windows.
6. **Conservative iteration** (§3.5). :func:`detect_outliers` sweeps
   1 → 5 until the flags reach a fixed point or ``max_iterations``; if
   the per-component **candidate** fraction ever exceeds
   ``max_flag_fraction`` the masking is aborted loudly (all-False flags,
   ``excess_flag_abort=True``, diagnostics fully populated) — a series
   where > f_max of epochs look like outliers almost certainly contains
   unmodeled signal, and masking there deletes signal. Final reported
   fits are plain WLS on the inliers (Gauss–Markov covariance), the
   :func:`gps_analysis.fitting.reject_outliers` convention.

What this module is NOT (§3.6): not unknown-break *detection* (that is
GBIS4TS, :mod:`gps_analysis.transient`) — it only protects breaks and
hints at them; not formal-σ screening
(:func:`gps_analysis.preprocess.screen_uncertainty` runs upstream); not
imputation — flagged epochs are never filled.

All functions are pure and unit-agnostic ([L] = caller's unit): float64,
no I/O, no config reads, no logging, inputs never mutated. Every
threshold is a passed-in parameter (:class:`OutlierParams`) with a
documented default — the caller (gps_api precompute) owns config.
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from .fitting import (
    _MAD_TO_SIGMA,
    ModelFunc,
    _components_2d,
    _n_model_params,
    _per_component_p0,
    _per_component_sigma,
    _robust_params,
    fit_components,
    with_steps,
)
from .models import FloatArray, TrajectoryParams

__all__ = [
    "PROTECT_FLOOR",
    "PROTECT_RUN",
    "PROTECT_STEP",
    "PROTECT_WINDOW",
    "REASON_GLOBAL",
    "REASON_GROSS",
    "REASON_LOCAL",
    "OutlierDetection",
    "OutlierParams",
    "SuspectedEvent",
    "candidate_clusters",
    "detect_outliers",
    "hampel_mask",
    "mad_scale",
    "neighbor_differences",
    "qn_scale",
    "rolling_median",
    "rolling_mad",
    "rolling_polyfit",
    "spike_mask",
    "standardize_robust",
    "step_evidence",
    "whiten",
]

_DAYS_PER_YEAR = 365.25
"""Julian-year day count — the ``yearf`` day↔year conversion used for the
day-denominated thresholds of :class:`OutlierParams`."""

REASON_GLOBAL = 1
"""Reason bit: global identifier ``|ẑ_i| > k_g`` fired (§3.3a)."""

REASON_LOCAL = 2
"""Reason bit: windowed Hampel identifier fired (§3.3b)."""

REASON_GROSS = 4
"""Reason bit: Stage-0 gross-blunder despike fired (§3.0 addendum) —
an isolated single-epoch spike, extreme against both temporal neighbors
with the series returning to baseline. Set instead of (never together
with) ``REASON_GLOBAL``/``REASON_LOCAL``: despiked epochs are decided
before the model fit and removed from the identifier candidates."""

PROTECT_FLOOR = 1
"""Protection bit: candidate below the physical magnitude floor a_min
(§3.4.1)."""

PROTECT_RUN = 2
"""Protection bit: member of a sustained same-sign candidate run —
suspected transient/step signal (§3.4.2, run-length rule)."""

PROTECT_STEP = 4
"""Protection bit: member of a cluster whose step-evidence statistic D
fired (or was indeterminate) — the series stays offset after the cluster
(§3.4.2, step-evidence rule)."""

PROTECT_WINDOW = 8
"""Protection bit: candidate inside a caller-supplied protected event
window (§3.4.3)."""

_QN_CONSISTENCY = float(1.0 / (math.sqrt(2.0) * float(ndtri(0.625))))
"""Gaussian-consistency constant of Qn: ``d = 1/(√2·Φ⁻¹(5/8)) ≈ 2.21914``
(Rousseeuw & Croux 1993, JASA 88 §3; the spec's 2.2191)."""

_QN_SMALL_N_FACTORS = {
    2: 0.399,
    3: 0.994,
    4: 0.512,
    5: 0.844,
    6: 0.611,
    7: 0.857,
    8: 0.669,
    9: 0.872,
}
"""Finite-sample correction factors c_N of Qn for N ≤ 9 (Croux &
Rousseeuw 1992, Computational Statistics 1, Table 2)."""


def _validate_series(name: str, x: ArrayLike) -> FloatArray:
    """Coerce a 1-D finite float64 series, raising with the arg name."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {xx.shape}")
    if not np.all(np.isfinite(xx)):
        raise ValueError(f"{name} must be finite (no NaN/inf)")
    return xx


def _validate_sorted_time(t: ArrayLike) -> FloatArray:
    """Coerce epochs to 1-D finite float64, sorted ascending."""
    tt = _validate_series("t", t)
    if tt.size > 1 and np.any(np.diff(tt) < 0.0):
        raise ValueError("t must be sorted ascending")
    return tt


def mad_scale(x: ArrayLike, *, center: float | None = None) -> float:
    """Compute the normalized median-absolute-deviation scale ŝ.

    Equation:
        ``ŝ = 1.4826 · med_i |x_i − c|``,  ``c = med(x)`` when ``center``
        is None  (1.4826 = 1/Φ⁻¹(3/4), the Gaussian-consistency factor)

    Symbols → args:
        - ``x_i`` → ``x``: samples, shape (N,), N ≥ 3 [units of x]
        - ``c``   → ``center``: location to deviate about [units of x];
          ``None`` ⇒ the sample median

    Returns:
        Robust scale ŝ [units of x], float64 scalar. ``0.0`` on
        degenerate input (≥ 50 % of samples equal to the center) — the
        caller guards with a scale floor, e.g. ``s_floor`` in
        :func:`hampel_mask`.

    Raises:
        ValueError: If ``x`` is not 1-D, holds fewer than 3 samples, or
            contains non-finite values (NaN-free input required).

    Reference:
        Rousseeuw & Croux 1993, JASA 88(424) (MAD normalization;
        breakdown 50 %, Gaussian efficiency 37 %).

    Numerical notes:
        Two exact medians — no accumulation concerns. The N ≥ 3 guard
        rejects inputs where the MAD is meaningless (N ≤ 2 gives 0 or a
        half-range). MAD assumes symmetric spread; for one-sided
        contamination consider :func:`qn_scale`.
    """
    xx = _validate_series("x", x)
    if xx.size < 3:
        raise ValueError(f"mad_scale requires N >= 3 samples, got {xx.size}")
    c = float(np.median(xx)) if center is None else float(center)
    return float(_MAD_TO_SIGMA * float(np.median(np.abs(xx - c))))


def _qn_finite_sample_factor(n: int) -> float:
    """Finite-sample consistency factor c_N of Qn (Croux & Rousseeuw 1992)."""
    if n <= 9:
        return _QN_SMALL_N_FACTORS[n]
    if n % 2:
        return n / (n + 1.4)
    return n / (n + 3.8)


def qn_scale(x: ArrayLike) -> float:
    """Compute the Rousseeuw–Croux Qn robust scale estimate.

    Equation:
        ``Qn = d · c_N · {|x_i − x_j| : i < j}_(k)``,
        ``k = C(h, 2)``, ``h = ⌊N/2⌋ + 1``,
        ``d = 1/(√2·Φ⁻¹(5/8)) ≈ 2.21914``

    — the k-th order statistic of the N(N−1)/2 pairwise absolute
    differences, with the finite-sample correction factors c_N (table
    for N ≤ 9, ``N/(N+1.4)`` odd / ``N/(N+3.8)`` even beyond).

    Symbols → args:
        - ``x_i`` → ``x``: samples, shape (N,), N ≥ 2 [units of x]

    Returns:
        Robust scale Qn [units of x], float64 scalar. ``0.0`` on
        degenerate input (enough tied samples that the k-th smallest
        pairwise difference is zero).

    Raises:
        ValueError: If ``x`` is not 1-D, has fewer than 2 samples, or is
            non-finite.

    Reference:
        Rousseeuw & Croux 1993, JASA 88(424) (definition; breakdown
        50 %, Gaussian efficiency 82 %, **no symmetry assumption** —
        unlike the MAD, which matters for residuals during one-sided
        unmodeled signal); Croux & Rousseeuw 1992, Computational
        Statistics 1, 411–428 (finite-sample factors; the O(N log N)
        algorithm).

    Numerical notes:
        **O(N²) reference implementation** (BGÓ decision, spec §9-Q6:
        Qn is Phase 1.5 and non-default; the Croux–Rousseeuw
        O(N log N) selection algorithm replaces this body later without
        an interface change). All pairwise differences are materialized:
        ~N²/2 float64 values (≈ 213 MB at N = 7300, a 20-yr daily
        series) — acceptable for the non-default estimator, documented
        here. Selection via ``np.partition`` (introselect), exact.
    """
    xx = _validate_series("x", x)
    n = int(xx.size)
    if n < 2:
        raise ValueError(f"qn_scale requires N >= 2 samples, got {n}")
    h = n // 2 + 1
    k = h * (h - 1) // 2
    i_upper, j_upper = np.triu_indices(n, k=1)
    diffs = np.abs(xx[i_upper] - xx[j_upper])
    kth = float(np.partition(diffs, k - 1)[k - 1])
    return float(_QN_CONSISTENCY * _qn_finite_sample_factor(n) * kth)


def neighbor_differences(
    t: ArrayLike, x: ArrayLike, *, max_gap: float
) -> tuple[FloatArray, FloatArray]:
    """Compute the gap-aware first differences δ⁻, δ⁺ to both neighbors.

    Equation:
        ``δ⁻_i = x_i − x_{i−1}``  if ``t_i − t_{i−1} ≤ g``  else NaN,
        ``δ⁺_i = x_{i+1} − x_i``  if ``t_{i+1} − t_i ≤ g``  else NaN

    — the deviation of epoch i from its previous/next **temporally
    adjacent** sample. Adjacency is decided on the time axis, not on the
    array index: across a data gap wider than g the difference mixes
    real signal accumulated over the gap with the blunder amplitude, so
    it is reported NaN (= "no usable neighbor"). The series boundaries
    (i = 0 for δ⁻, i = N−1 for δ⁺) are NaN by construction.

    Symbols → args:
        - ``t_i`` → ``t``: epochs, shape (N,), sorted ascending [yr]
        - ``x_i`` → ``x``: samples, shape (N,) [units of x]
        - ``g``   → ``max_gap``: maximum neighbor spacing [units of t]
          (fractional years — pass e.g. ``1.5 / 365.25`` for 1.5 d)

    Returns:
        ``(δ⁻, δ⁺)`` — two float64 arrays, shape (N,); NaN marks a
        missing/too-distant neighbor.

    Raises:
        ValueError: If ``t`` is unsorted/non-finite, ``x`` is non-finite,
            shapes mismatch, or ``max_gap ≤ 0``.

    Reference:
        First-difference despiking statistics: Goring & Nikora 2002,
        J. Hydraul. Eng. 128(1) (difference-based spike detection);
        successive differences as a trend-immune noise probe:
        von Neumann et al. 1941, Ann. Math. Statist. 12.

    Numerical notes:
        One ``np.diff`` pass, no accumulation. A locally linear signal
        of rate v contributes v·Δt to both differences with the SAME
        sign, so the downstream two-sided rule (:func:`spike_mask`) is
        trend-immune by construction.
    """
    tt = _validate_sorted_time(t)
    xx = _validate_series("x", x)
    if xx.shape != tt.shape:
        raise ValueError(f"x shape {xx.shape} does not match t shape {tt.shape}")
    if max_gap <= 0.0:
        raise ValueError(f"max_gap must be > 0, got {max_gap}")
    n = int(tt.size)
    delta_prev = np.full(n, np.nan, dtype=np.float64)
    delta_next = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return delta_prev, delta_next
    d = np.diff(xx)
    adjacent = np.diff(tt) <= max_gap
    delta_prev[1:][adjacent] = d[adjacent]
    delta_next[:-1][adjacent] = d[adjacent]
    return delta_prev, delta_next


def spike_mask(
    delta_prev: ArrayLike,
    delta_next: ArrayLike,
    *,
    n_sigma: float,
    scale: float,
    return_n_sigma: float,
) -> NDArray[np.bool_]:
    """Apply the two-sided isolated-spike decision rule elementwise.

    Equation:
        ``S_i : min(|δ⁻_i|, |δ⁺_i|) > k_d·ŝ_Δ`` ∧ ``δ⁻_i·δ⁺_i < 0`` ∧
        ``|δ⁻_i + δ⁺_i| ≤ c_r·ŝ_Δ``

    — epoch i deviates strongly from BOTH temporal neighbors (first
    condition), in opposite difference directions, i.e. up-then-down or
    down-then-up (second condition), and the two neighbors agree with
    each other, ``δ⁻ + δ⁺ = x_{i+1} − x_{i−1}``, so the series RETURNS
    to baseline (third condition). A persistent offset (real step) has
    ``δ⁺ ≈ 0`` at the jump epoch and fails the first two conditions; a
    linear trend gives same-sign differences and fails the second — so
    only isolated single-epoch spikes fire.

    Symbols → args:
        - ``δ⁻_i`` → ``delta_prev``: backward differences, shape (N,)
          [units of x]; NaN = no usable neighbor
          (:func:`neighbor_differences`)
        - ``δ⁺_i`` → ``delta_next``: forward differences, shape (N,)
          [units of x]
        - ``k_d``  → ``n_sigma``: spike threshold [dimensionless]
        - ``ŝ_Δ``  → ``scale``: robust first-difference scale
          [units of x] — e.g. ``mad_scale`` of the finite δ⁻ (median-
          centered, so a constant rate does not inflate it); under white
          noise σ, ŝ_Δ estimates √2·σ, and a spike of amplitude A scores
          ``A/(√2·σ)`` on both sides
        - ``c_r``  → ``return_n_sigma``: neighbor-agreement (return-to-
          baseline) tolerance [dimensionless]; under white noise
          ``δ⁻ + δ⁺`` has the same √2·σ scale, so c_r ≈ 4 accepts
          essentially every genuine spike while a residual same-sign
          offset > c_r·ŝ_Δ (spike-plus-step) is left to the main,
          protection-aware stage

    Returns:
        Boolean mask S, shape (N,) — True where an isolated gross spike
        is identified. Any NaN δ (gap, boundary) yields **False**
        (conservative: no usable neighbor ⇒ never despike).

    Raises:
        ValueError: On shape mismatch, or non-positive ``n_sigma``,
            ``scale`` or ``return_n_sigma``.

    Reference:
        Spike-vs-step framing (a spike returns immediately, a level
        shift persists): Pearson, Neuvo, Astola & Gabbouj 2016, EURASIP
        J. Adv. Signal Process. 2016:87, §1; difference-based despiking:
        Goring & Nikora 2002, J. Hydraul. Eng. 128(1); why steps must
        never be auto-clipped: Gazeaux et al. 2013, JGR 118 (DOGEx).

    Numerical notes:
        Pure elementwise comparisons; NaN comparisons are IEEE-False so
        gap/boundary epochs drop out without special-casing. The rule is
        deliberately blind to multi-epoch clusters (two adjacent
        blunders give a small middle difference) — those remain the main
        detector's job.
    """
    dp = np.asarray(delta_prev, dtype=np.float64)
    dn = np.asarray(delta_next, dtype=np.float64)
    if dp.shape != dn.shape:
        raise ValueError(
            f"delta_prev shape {dp.shape} does not match delta_next shape {dn.shape}"
        )
    if n_sigma <= 0.0:
        raise ValueError(f"n_sigma must be > 0, got {n_sigma}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if return_n_sigma <= 0.0:
        raise ValueError(f"return_n_sigma must be > 0, got {return_n_sigma}")
    with np.errstate(invalid="ignore"):
        two_sided = np.minimum(np.abs(dp), np.abs(dn)) > n_sigma * scale
        opposite = dp * dn < 0.0
        returns = np.abs(dp + dn) <= return_n_sigma * scale
    return np.asarray(two_sided & opposite & returns, dtype=np.bool_)


def whiten(r: ArrayLike, sigma: ArrayLike | None) -> FloatArray:
    """Whiten residuals by their formal per-epoch uncertainties.

    Equation:
        ``w_i = r_i / σ_i``  (elementwise); ``σ ≡ 1`` when ``sigma`` is
        None (w is then a copy of r).

    Symbols → args:
        - ``r_i`` → ``r``: residuals, shape (N,) [L]
        - ``σ_i`` → ``sigma``: formal 1-σ uncertainties, shape (N,) [L],
          strictly positive; or ``None``

    Returns:
        Whitened residuals w, shape (N,), float64 [dimensionless when σ
        is given, [L] otherwise] — always a **new** array.

    Raises:
        ValueError: On shape mismatch, non-finite ``r``, or any
            ``σ_i ≤ 0`` (a NaN σ also fails the positivity check).

    Reference:
        Design spec §3.1: formal σ is used for **relative weighting
        only** — GAMIT/GipsyX formal errors are optimistic by a variable
        factor, so the absolute detection scale is always re-estimated
        from w itself (:func:`standardize_robust`).

    Numerical notes:
        Elementwise division, no accumulation. The strict σ > 0 guard
        prevents silent inf/NaN propagation into the identifiers.
    """
    rr = _validate_series("r", r)
    if sigma is None:
        return rr.copy()
    ss = np.asarray(sigma, dtype=np.float64)
    if ss.shape != rr.shape:
        raise ValueError(f"sigma shape {ss.shape} does not match r shape {rr.shape}")
    if not np.all(ss > 0.0):
        raise ValueError("sigma must be strictly positive (and finite)")
    return np.asarray(rr / ss, dtype=np.float64)


def standardize_robust(
    x: ArrayLike, *, scale: str = "mad"
) -> tuple[FloatArray, float, float]:
    """Robustly standardize a series about its median.

    Equation:
        ``ẑ_i = (x_i − c) / ŝ``,  ``c = med(x)``,
        ``ŝ = mad_scale(x)`` or ``qn_scale(x)``

    Symbols → args:
        - ``x_i`` → ``x``: samples, shape (N,), N ≥ 3 [units of x]
        - ŝ estimator → ``scale``: ``"mad"`` (default) | ``"qn"``

    Returns:
        ``(ẑ, c, ŝ)`` — the standardized series (float64, shape of x),
        the center c and the scale ŝ [units of x]. On degenerate input
        (ŝ = 0) ẑ is **all zeros** — no exceedances, the conservative
        never-flag convention (mirrors
        :func:`gps_analysis.fitting.reject_outliers`, which stops
        rejecting at ŝ = 0); callers see ŝ = 0.0 and may guard.

    Raises:
        ValueError: On invalid ``x`` (see :func:`mad_scale`) or an
            unknown ``scale`` name.

    Reference:
        Rousseeuw & Croux 1993, JASA 88(424); identifier framing:
        Davies & Gather 1993, JASA 88(423).

    Numerical notes:
        The MAD path reuses the median as the MAD center (one median
        computed twice at most); Qn is scale-only, the center stays the
        median in both paths.
    """
    xx = _validate_series("x", x)
    if xx.size < 3:
        raise ValueError(f"standardize_robust requires N >= 3, got {xx.size}")
    center = float(np.median(xx))
    if scale == "mad":
        s = mad_scale(xx, center=center)
    elif scale == "qn":
        s = qn_scale(xx)
    else:
        raise ValueError(f"scale must be 'mad' or 'qn', got {scale!r}")
    if s <= 0.0:
        return np.zeros_like(xx), center, s
    return np.asarray((xx - center) / s, dtype=np.float64), center, s


def rolling_median(
    t: ArrayLike, x: ArrayLike, *, half_window: float, min_count: int
) -> FloatArray:
    """Compute the time-windowed rolling median m of a series.

    Equation:
        ``m_i = med{ x_j : |t_j − t_i| ≤ h }``

    Windows are defined **in time, not samples** (daily GNSS series have
    gaps): the window at t_i is ``{j : |t_j − t_i| ≤ h}`` and always
    contains sample i itself.

    Symbols → args:
        - ``t_j`` → ``t``: epochs, shape (N,), sorted ascending [yr]
        - ``x_j`` → ``x``: samples, shape (N,) [units of x]
        - ``h``   → ``half_window``: half-window, **in the units of t**
          (fractional years — pass e.g. ``15.5 / 365.25`` for ±15.5 d)
        - ``min_count``: minimum in-window sample count for a valid
          median

    Returns:
        Rolling median m, shape (N,), float64 [units of x]. ``NaN`` at i
        when the window holds fewer than ``min_count`` samples — the
        caller substitutes its global fallback (documented degradation,
        §3.3b), never a silent zero.

    Raises:
        ValueError: If ``t`` is unsorted or non-finite, shapes mismatch,
            ``half_window ≤ 0``, or ``min_count < 1``.

    Reference:
        Pearson, Neuvo, Astola & Gabbouj 2016, EURASIP J. Adv. Signal
        Process. 2016:87 (generalized Hampel filter — the median window
        of eq. 4, here time-indexed).

    Numerical notes:
        Two-pointer sweep over the sorted epochs: window bounds are
        monotone in i, so bound maintenance is O(N) total and the cost
        is O(N·w̄ log w̄) for mean window size w̄ (exact medians per
        window, no incremental approximation).
    """
    tt = _validate_sorted_time(t)
    xx = _validate_series("x", x)
    if xx.shape != tt.shape:
        raise ValueError(f"x shape {xx.shape} does not match t shape {tt.shape}")
    if half_window <= 0.0:
        raise ValueError(f"half_window must be > 0, got {half_window}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    n = int(tt.size)
    out = np.full(n, np.nan, dtype=np.float64)
    lo = 0
    hi = 0
    for i in range(n):
        t_i = tt[i]
        while tt[lo] < t_i - half_window:
            lo += 1
        while hi < n and tt[hi] <= t_i + half_window:
            hi += 1
        if hi - lo >= min_count:
            out[i] = np.median(xx[lo:hi])
    return out


def rolling_mad(
    t: ArrayLike,
    x: ArrayLike,
    center: ArrayLike,
    *,
    half_window: float,
    min_count: int,
) -> FloatArray:
    """Compute the time-windowed rolling normalized MAD s about a center.

    Equation:
        ``s_i = 1.4826 · med{ |x_j − c_i| : |t_j − t_i| ≤ h }``

    — deviations are taken about the **per-epoch** center c_i (normally
    the :func:`rolling_median` of the same window), so s tracks local
    spread even where the low-frequency noise wanders.

    Symbols → args:
        - ``t_j`` → ``t``: epochs, shape (N,), sorted ascending [yr]
        - ``x_j`` → ``x``: samples, shape (N,) [units of x]
        - ``c_i`` → ``center``: per-epoch centers, shape (N,)
          [units of x]; NaN centers yield NaN s_i
        - ``h``   → ``half_window``: half-window [units of t]
        - ``min_count``: minimum in-window sample count

    Returns:
        Rolling scale s, shape (N,), float64 [units of x]. ``NaN``
        where the window is thin (same ``min_count`` rule as
        :func:`rolling_median`) or where c_i is NaN.

    Raises:
        ValueError: Same conditions as :func:`rolling_median`, plus a
            ``center`` shape mismatch (NaN center values are allowed —
            they mark thin windows upstream).

    Reference:
        Hampel 1974, JASA 69; Pearson et al. 2016, eq. 4 (the MAD window
        of the generalized Hampel filter); MAD normalization: Rousseeuw
        & Croux 1993.

    Numerical notes:
        Same two-pointer sweep as :func:`rolling_median`. The known
        Hampel degeneracy — ≥ 50 % of a window identical ⇒ s_i = 0 ⇒
        everything flagged — is the caller's to guard via ``s_floor``
        (:func:`hampel_mask`); this primitive reports the honest zero.
    """
    tt = _validate_sorted_time(t)
    xx = _validate_series("x", x)
    cc = np.asarray(center, dtype=np.float64)
    if xx.shape != tt.shape:
        raise ValueError(f"x shape {xx.shape} does not match t shape {tt.shape}")
    if cc.shape != tt.shape:
        raise ValueError(f"center shape {cc.shape} does not match t shape {tt.shape}")
    if half_window <= 0.0:
        raise ValueError(f"half_window must be > 0, got {half_window}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    n = int(tt.size)
    out = np.full(n, np.nan, dtype=np.float64)
    lo = 0
    hi = 0
    for i in range(n):
        t_i = tt[i]
        while tt[lo] < t_i - half_window:
            lo += 1
        while hi < n and tt[hi] <= t_i + half_window:
            hi += 1
        if hi - lo >= min_count and not math.isnan(cc[i]):
            out[i] = _MAD_TO_SIGMA * np.median(np.abs(xx[lo:hi] - cc[i]))
    return out


def _lowess_window_fit(
    dt: FloatArray,
    xw: FloatArray,
    order: int,
    robust_iterations: int,
) -> tuple[float, float]:
    """Robust local polynomial fit of one window (LOWESS, Cleveland 1979).

    Equation (window {j}, centered times dt_j = t_j − t_i, h = max|dt|):
        ``β̂ = argmin_β Σ_j u_j·δ_j·(x_j − Σ_{k=0}^{q} β_k·dt_j^k)²``
    with tricube distance weights ``u_j = (1 − (|dt_j|/h)³)³``
    (h = max_j|dt_j|; h = 0 — all epochs identical — is degenerate ⇒
    NaN/NaN) and bisquare robustness weights, re-estimated
    ``robust_iterations`` times from the residuals e_j = x_j − A·β̂:
        ``δ_j = (1 − (e_j / 6·med|e|)²)²`` for |e_j| < 6·med|e|, else 0
    (Cleveland 1979, JASA 74:829, eqs. 1–4 and the robustness step §3).
    Returns ``(m, s) = (β̂_0, 1.4826·med_j|e_j|)`` — the fit value at
    dt = 0 and the normalized MAD of the local residuals about the fit.
    Shared per-window core of :func:`rolling_polyfit`; inputs are
    pre-validated there. NaN/NaN on a rank-deficient weighted design.
    """
    h = float(np.max(np.abs(dt)))
    if h <= 0.0:
        return float("nan"), float("nan")
    u = (1.0 - (np.abs(dt) / h) ** 3) ** 3
    a = np.vander(dt, order + 1, increasing=True)
    weights = u
    beta = np.zeros(order + 1, dtype=np.float64)
    for _ in range(robust_iterations + 1):
        sw = np.sqrt(weights)
        beta_new, _, rank, _ = np.linalg.lstsq(
            a * sw[:, np.newaxis], xw * sw, rcond=None
        )
        if rank < order + 1:
            return float("nan"), float("nan")
        beta = np.asarray(beta_new, dtype=np.float64)
        e = xw - a @ beta
        s6 = 6.0 * float(np.median(np.abs(e)))
        if s6 <= 0.0:
            break  # exact fit on ≥ half the window — already robust
        delta = np.clip(1.0 - (e / s6) ** 2, 0.0, None) ** 2
        weights = u * delta
    resid = xw - a @ beta
    return float(beta[0]), float(_MAD_TO_SIGMA * float(np.median(np.abs(resid))))


def rolling_polyfit(
    t: ArrayLike,
    x: ArrayLike,
    *,
    half_window: float,
    min_count: int,
    order: int,
    robust_iterations: int = 2,
) -> tuple[FloatArray, FloatArray]:
    """Compute the time-windowed robust local polynomial center and scale.

    Equation (per epoch i, window ``W_i = {j : |t_j − t_i| ≤ h}``):
        ``m_i = f̂_i(t_i) = β̂_0``  from the robust local fit
        ``f̂_i(t) = Σ_{k=0}^{q} β̂_k·(t − t_i)^k``  (LOWESS: tricube
        distance weights × bisquare robustness iterations —
        :func:`_lowess_window_fit`), and
        ``s_i = 1.4826 · med{ |x_j − f̂_i(t_j)| : j ∈ W_i }``

    — the order-q generalization of the (order-0)
    :func:`rolling_median`/:func:`rolling_mad` pair: during fast
    transients a local constant lags the motion, so genuine on-trend
    samples acquire large ``|x_i − m_i|`` and over-flag; a local line
    (q = 1) or parabola (q = 2) tracks the local slope/curvature and
    leaves on-trend residuals small. Robustness iterations keep the fit
    from being pulled by the very outlier under test (masking).

    Symbols → args:
        - ``t_j`` → ``t``: epochs, shape (N,), sorted ascending [yr]
        - ``x_j`` → ``x``: samples, shape (N,) [units of x]
        - ``h``   → ``half_window``: half-window [units of t]
        - ``min_count``: minimum in-window sample count; must be
          ≥ q + 2 so the residual scale is not identically zero
        - ``q``   → ``order``: local polynomial order, 1 or 2 (order 0
          is exactly :func:`rolling_median` — the L1-optimal local
          constant — and stays on that path)
        - ``robust_iterations``: bisquare robustness re-weightings
          (Cleveland 1979 recommends 2; 0 = plain weighted LSQ,
          NOT robust — documented, not recommended)

    Returns:
        ``(m, s)`` — local fit value and local residual MAD, each shape
        (N,), float64 [units of x]. ``NaN`` at i when the window holds
        fewer than ``min_count`` samples or the weighted design is
        rank-deficient (e.g. duplicate epochs) — the caller substitutes
        its global fallback (same degradation convention as
        :func:`rolling_median`). ``s_i = 0`` is possible (≥ 50 % of a
        window fitted exactly) — the caller's ``scale_floor`` guards it,
        exactly the :func:`rolling_mad` Hampel degeneracy.

    Raises:
        ValueError: If ``t`` is unsorted/non-finite, shapes mismatch,
            ``half_window ≤ 0``, ``order ∉ {1, 2}``,
            ``min_count < order + 2``, or ``robust_iterations < 0``.

    Reference:
        Cleveland 1979, *Robust locally weighted regression and
        smoothing scatterplots*, JASA 74(368), 829–836 (LOWESS: tricube
        kernel, bisquare robustness iterations); windowed-identifier
        framing: Pearson et al. 2016, EURASIP J. Adv. Signal Process.
        2016:87 (generalized Hampel filter — here with a polynomial
        center function).

    Numerical notes:
        Same two-pointer window sweep as :func:`rolling_median`; per
        window one (q+1)-column Vandermonde LSQ per robustness pass —
        O(N·w̄·(q+1)²·(robust_iterations+1)) total. The design is built
        on **centered** times dt = t_j − t_i (|dt| ≤ h ≈ 0.04 yr), so
        conditioning is excellent regardless of the absolute ``yearf``
        epoch. ``np.linalg.lstsq`` (SVD) with an explicit rank check —
        rank-deficient windows degrade to NaN, never to a silent
        minimum-norm fit.
    """
    tt = _validate_sorted_time(t)
    xx = _validate_series("x", x)
    if xx.shape != tt.shape:
        raise ValueError(f"x shape {xx.shape} does not match t shape {tt.shape}")
    if half_window <= 0.0:
        raise ValueError(f"half_window must be > 0, got {half_window}")
    if order not in (1, 2):
        raise ValueError(
            f"order must be 1 or 2 (order 0 is rolling_median), got {order}"
        )
    if min_count < order + 2:
        raise ValueError(
            f"min_count must be >= order + 2 = {order + 2}, got {min_count}"
        )
    if robust_iterations < 0:
        raise ValueError(f"robust_iterations must be >= 0, got {robust_iterations}")
    n = int(tt.size)
    center = np.full(n, np.nan, dtype=np.float64)
    scale = np.full(n, np.nan, dtype=np.float64)
    lo = 0
    hi = 0
    for i in range(n):
        t_i = tt[i]
        while tt[lo] < t_i - half_window:
            lo += 1
        while hi < n and tt[hi] <= t_i + half_window:
            hi += 1
        if hi - lo >= min_count:
            center[i], scale[i] = _lowess_window_fit(
                tt[lo:hi] - t_i, xx[lo:hi], order, robust_iterations
            )
    return center, scale


def hampel_mask(
    x: ArrayLike,
    center: ArrayLike,
    scale: ArrayLike,
    *,
    n_sigma: float,
    scale_floor: float = 0.0,
) -> NDArray[np.bool_]:
    """Apply the Hampel identifier decision rule elementwise.

    Equation:
        ``L_i : |x_i − c_i| > k · max(s_i, s_floor)``

    Symbols → args:
        - ``x_i``     → ``x``: samples, shape (N,) [units of x]
        - ``c_i``     → ``center``: per-epoch centers, shape (N,)
          [units of x]
        - ``s_i``     → ``scale``: per-epoch scales, shape (N,)
          [units of x]
        - ``k``       → ``n_sigma``: decision threshold [dimensionless]
        - ``s_floor`` → ``scale_floor``: scale floor [units of x] —
          guards the Hampel degeneracy where ≥ 50 % of a window is
          identical (s_i = 0 would flag everything)

    Returns:
        Boolean mask L, shape (N,) — True where the identifier fires. A
        NaN ``s_i`` or ``c_i`` (thin window upstream) yields **False**
        (conservative: substituting a fallback scale is the caller's
        explicit business, never an implicit decision here).

    Raises:
        ValueError: On shape mismatches, non-finite ``x``,
            ``n_sigma ≤ 0`` or ``scale_floor < 0``.

    Reference:
        Hampel 1974, JASA 69 (the identifier); Pearson, Neuvo, Astola &
        Gabbouj 2016, EURASIP J. Adv. Signal Process. 2016:87, eq. 4
        (windowed decision rule + the MAD-collapse degeneracy).

    Numerical notes:
        Pure elementwise comparison; NaN comparisons are IEEE-False by
        construction (no warnings, no masking of the NaN itself).
    """
    xx = _validate_series("x", x)
    cc = np.asarray(center, dtype=np.float64)
    ss = np.asarray(scale, dtype=np.float64)
    if cc.shape != xx.shape or ss.shape != xx.shape:
        raise ValueError(
            f"center {cc.shape} and scale {ss.shape} must match x shape {xx.shape}"
        )
    if n_sigma <= 0.0:
        raise ValueError(f"n_sigma must be > 0, got {n_sigma}")
    if scale_floor < 0.0:
        raise ValueError(f"scale_floor must be >= 0, got {scale_floor}")
    deviation = np.abs(xx - cc)
    threshold = n_sigma * np.maximum(ss, scale_floor)
    return np.asarray(deviation > threshold, dtype=np.bool_)


def candidate_clusters(
    t: ArrayLike, candidates: NDArray[np.bool_], *, max_gap: float
) -> list[tuple[int, int]]:
    """Group candidate epochs into time-gap-limited index clusters.

    Definition (pure indexing, no arithmetic on values):
        a cluster is a maximal run of candidate epochs in which
        consecutive candidates are ``≤ max_gap`` apart in t; it is
        reported as the inclusive index pair ``(i_start, i_end)`` of its
        first and last candidate.

    Symbols → args:
        - ``t``        → ``t``: epochs, shape (N,), sorted ascending [yr]
        - candidates   → ``candidates``: boolean candidate mask, shape (N,)
        - ``max_gap``  → ``max_gap``: maximum intra-cluster gap
          [units of t]

    Returns:
        List of ``(i_start, i_end)`` inclusive index pairs, in time
        order; empty when there are no candidates. Non-candidate epochs
        may lie inside a cluster's index span (gaps ≤ max_gap).

    Raises:
        ValueError: If ``t`` is unsorted/non-finite, shapes mismatch, or
            ``max_gap ≤ 0``.

    Reference:
        Design spec §3.4.2 — clusters are the unit of the run-length /
        step-evidence signal-protection rules (Gazeaux et al. 2013:
        steps/transients produce *sustained* candidate runs; blunders
        are isolated or very short).

    Numerical notes:
        Single O(N) pass over the candidate indices; exact float
        comparison of time gaps (no tolerance — supply ``max_gap`` with
        headroom over the sampling interval, e.g. 1.5 d for daily data).
    """
    tt = _validate_sorted_time(t)
    cand = np.asarray(candidates, dtype=np.bool_)
    if cand.shape != tt.shape:
        raise ValueError(
            f"candidates shape {cand.shape} does not match t shape {tt.shape}"
        )
    if max_gap <= 0.0:
        raise ValueError(f"max_gap must be > 0, got {max_gap}")
    indices = np.flatnonzero(cand)
    clusters: list[tuple[int, int]] = []
    if indices.size == 0:
        return clusters
    start = prev = int(indices[0])
    for raw in indices[1:]:
        i = int(raw)
        if tt[i] - tt[prev] <= max_gap:
            prev = i
        else:
            clusters.append((start, prev))
            start = prev = i
    clusters.append((start, prev))
    return clusters


def _flank_medians(
    t: FloatArray,
    r: FloatArray,
    i_start: int,
    i_end: int,
    *,
    window: float,
    exclude: NDArray[np.bool_] | None,
) -> tuple[float, float]:
    """Median residuals of the two flank windows of a cluster.

    Equation:
        ``r̄_pre  = med{ r_j : t_j ∈ [t_start − W, t_start) }``,
        ``r̄_post = med{ r_j : t_j ∈ (t_end, t_end + W] }``

    with ``exclude``-masked epochs dropped from both flanks. Either
    median is NaN when its flank holds fewer than 3 usable samples.
    Shared numerator machinery of :func:`step_evidence` (D) and the
    elevated-background protection arm (§3.4.2 implementation note in
    :func:`_protect_component`). Inputs are pre-validated by callers.
    """
    pre = (t >= t[i_start] - window) & (t < t[i_start])
    post = (t > t[i_end]) & (t <= t[i_end] + window)
    if exclude is not None:
        ex = np.asarray(exclude, dtype=np.bool_)
        if ex.shape != t.shape:
            raise ValueError(
                f"exclude shape {ex.shape} does not match t shape {t.shape}"
            )
        pre &= ~ex
        post &= ~ex
    med_pre = (
        float(np.median(r[pre])) if int(np.count_nonzero(pre)) >= 3 else float("nan")
    )
    med_post = (
        float(np.median(r[post])) if int(np.count_nonzero(post)) >= 3 else float("nan")
    )
    return med_pre, med_post


def step_evidence(
    t: ArrayLike,
    r: ArrayLike,
    i_start: int,
    i_end: int,
    *,
    window: float,
    scale: float,
    exclude: NDArray[np.bool_] | None = None,
) -> float:
    """Compute the step-evidence statistic D of a candidate cluster.

    Equation:
        ``D = |med(r_post) − med(r_pre)| / ŝ`` with
        ``r_pre  = { r_j : t_j ∈ [t_start − W, t_start) }``,
        ``r_post = { r_j : t_j ∈ (t_end, t_end + W] }``

    — the cluster's flanking medians: a blunder cluster has D ≈ 0 (the
    series returns to the model), a step leaves the series *offset*
    (D large).

    Symbols → args:
        - ``t_j``     → ``t``: epochs, shape (N,), sorted ascending [yr]
        - ``r_j``     → ``r``: residuals, shape (N,) [units of r]
        - ``t_start``/``t_end`` → ``t[i_start]``/``t[i_end]``: cluster
          bounds (inclusive indices)
        - ``W``       → ``window``: flank window length [units of t]
        - ``ŝ``       → ``scale``: robust residual scale [units of r]
        - ``exclude`` → ``exclude``: boolean mask of epochs to drop from
          both flank medians (normally the full candidate mask, so
          neighboring outliers cannot bias the flanks)

    Returns:
        D [dimensionless], float64. ``NaN`` when either flank holds
        fewer than 3 usable samples — the caller treats NaN as "cannot
        rule out a step" and protects (Gazeaux et al. 2013 motivates the
        conservatism).

    Raises:
        ValueError: On invalid indices (``0 ≤ i_start ≤ i_end < N``),
            shape mismatches, ``window ≤ 0`` or ``scale ≤ 0``.

    Reference:
        Gazeaux et al. 2013, JGR 118 (DOGEx — offsets are hard to
        separate from outliers/noise ⇒ conservative protection); design
        spec §3.4.2.

    Numerical notes:
        Flank medians are exact; the 3-sample minimum per flank keeps
        the median meaningful. D is scale-normalized by the caller's ŝ
        so its threshold k_step is unit-free.
    """
    tt = _validate_sorted_time(t)
    rr = _validate_series("r", r)
    if rr.shape != tt.shape:
        raise ValueError(f"r shape {rr.shape} does not match t shape {tt.shape}")
    n = int(tt.size)
    if not (0 <= i_start <= i_end < n):
        raise ValueError(f"need 0 <= i_start <= i_end < {n}, got ({i_start}, {i_end})")
    if window <= 0.0:
        raise ValueError(f"window must be > 0, got {window}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    med_pre, med_post = _flank_medians(
        tt, rr, i_start, i_end, window=window, exclude=exclude
    )
    if math.isnan(med_pre) or math.isnan(med_post):
        return float("nan")
    return float(abs(med_post - med_pre) / scale)


@dataclasses.dataclass(frozen=True)
class OutlierParams:
    """Detection thresholds — constructed by the CALLER (config → here).

    Every threshold of §3 as a passed-in parameter with the spec default
    (design §4.3/§6 — defaults justified there); the leaf stays
    unit-agnostic and config-free. Day-denominated windows refer to the
    ``yearf`` time axis via 365.25 d/yr.

    Attributes:
        scale_estimator: Robust scale ŝ — ``"mad"`` (default; BGÓ
            decision §9-Q6: MAD ships first) | ``"qn"`` (Phase 1.5,
            selectable but non-default).
        global_n_sigma: Global identifier threshold k_g (§3.3a);
            deliberately high — colored noise fattens the global tails.
        window_days: FULL Hampel window length [d] (half-window
            h = window_days/2); ≳ 2× a typical outlier-cluster span, ≪
            seasonal period and typical transient τ.
        window_n_sigma: Hampel threshold k_w (§3.3b) — sits below k_g
            because the local center absorbs the noise wander.
        window_min_count: Minimum in-window samples n_min; thinner
            windows fall back to the global center/scale. With
            ``window_order`` q ≥ 1 it must be ≥ q + 2.
        window_order: Local polynomial order q of the windowed
            identifier — 0 (default; the current
            :func:`rolling_median`/:func:`rolling_mad` path, taken
            identically) | 1 | 2 (robust local line/parabola,
            :func:`rolling_polyfit`; Cleveland 1979). Orders ≥ 1 track
            fast transients (e.g. an unrest ramp) so on-trend epochs are
            not over-flagged by a lagging local constant.
        window_robust_iterations: Bisquare robustness re-weightings of
            the local polynomial fit (orders ≥ 1 only; Cleveland 1979
            recommends 2). 0 = plain local WLS — susceptible to masking
            by the very outlier under test; not recommended.
        despike: Enable the Stage-0 gross-blunder despike (§3.0
            addendum; default **False** — existing behavior is
            unchanged). When True, isolated single-epoch spikes that
            are extreme against both temporal neighbors AND return to
            baseline are masked out before the model fit, so they can
            neither drag the robust fit nor inflate the detection
            scale.
        despike_n_sigma: Stage-0 spike threshold k_d on
            ``min(|δ⁻|, |δ⁺|)/ŝ_Δ`` [dimensionless]; deliberately high
            (10) — Stage 0 removes only OBVIOUS extremes (a spike of
            amplitude A scores ≈ A/(√2σ) under white noise σ, so
            k_d = 10 ⇒ A ≳ 14σ).
        despike_return_sigma: Stage-0 return-to-baseline tolerance c_r
            on ``|δ⁻ + δ⁺|/ŝ_Δ`` [dimensionless] (:func:`spike_mask`).
        despike_gap_days: Maximum neighbor spacing of the Stage-0
            differences [d] — across wider gaps an epoch has no usable
            neighbor and is never despiked
            (:func:`neighbor_differences`).
        scale_floor: Hampel scale floor s_floor [whitened-residual
            units] — guards the MAD-collapse degeneracy.
        min_outlier: Physical magnitude floor a_min [L], applied per
            component (§3.4.1); the per-component vector form is
            :func:`detect_outliers`'s ``min_outlier`` argument.
        max_run_days: Run-length protection span L_max [d] — multi-day
            one-sided runs are the signature of signal (Gazeaux 2013).
        cluster_gap_days: Maximum intra-cluster candidate gap g [d].
        run_sign_fraction: Same-sign fraction q of the run rule.
        step_evidence_sigma: Step-evidence threshold k_step (§3.4.2).
        step_window_days: Step-evidence flank window W [d].
        max_flag_fraction: Abort threshold f_max on the per-component
            **candidate** fraction (§3.5 — "> f_max of epochs *look
            like* outliers ⇒ unmodeled signal, do nothing, loudly").
        max_iterations: Sweep cap of the conservative iteration.
        loss: Robust-fit loss (``scipy.optimize.least_squares``);
            ``"huber"`` per §3.1.
        f_scale: Robust-loss inlier margin, whitened-residual units.
        epoch_policy: Cross-component flag policy (§3.4.4) —
            ``"per_component"`` (default; N/E/U noise levels differ ~3×)
            | ``"union"`` (an epoch flagged in any component is flagged
            in all; overrides per-component protection of the other
            components by construction).
    """

    scale_estimator: str = "mad"
    global_n_sigma: float = 5.0
    window_days: float = 31.0
    window_n_sigma: float = 4.0
    window_min_count: int = 11
    window_order: int = 0
    window_robust_iterations: int = 2
    despike: bool = False
    despike_n_sigma: float = 10.0
    despike_return_sigma: float = 4.0
    despike_gap_days: float = 1.5
    scale_floor: float = 0.0
    min_outlier: float = 0.0
    max_run_days: float = 2.0
    cluster_gap_days: float = 1.5
    run_sign_fraction: float = 0.8
    step_evidence_sigma: float = 3.0
    step_window_days: float = 10.0
    max_flag_fraction: float = 0.05
    max_iterations: int = 3
    loss: str = "huber"
    f_scale: float = 1.0
    epoch_policy: str = "per_component"

    def __post_init__(self) -> None:
        if self.scale_estimator not in ("mad", "qn"):
            raise ValueError(
                f"scale_estimator must be 'mad' or 'qn', got {self.scale_estimator!r}"
            )
        if self.epoch_policy not in ("per_component", "union"):
            raise ValueError(
                "epoch_policy must be 'per_component' or 'union', "
                f"got {self.epoch_policy!r}"
            )
        positive = (
            "global_n_sigma",
            "window_days",
            "window_n_sigma",
            "cluster_gap_days",
            "run_sign_fraction",
            "step_evidence_sigma",
            "step_window_days",
            "f_scale",
            "despike_n_sigma",
            "despike_return_sigma",
            "despike_gap_days",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        non_negative = ("scale_floor", "min_outlier", "max_run_days")
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.run_sign_fraction > 1.0:
            raise ValueError("run_sign_fraction must be in (0, 1]")
        if not 0.0 < self.max_flag_fraction <= 1.0:
            raise ValueError("max_flag_fraction must be in (0, 1]")
        if self.window_min_count < 1:
            raise ValueError("window_min_count must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.window_order not in (0, 1, 2):
            raise ValueError(f"window_order must be 0, 1 or 2, got {self.window_order}")
        if self.window_order >= 1 and self.window_min_count < self.window_order + 2:
            raise ValueError(
                "window_min_count must be >= window_order + 2 = "
                f"{self.window_order + 2}, got {self.window_min_count}"
            )
        if self.window_robust_iterations < 0:
            raise ValueError("window_robust_iterations must be >= 0")


@dataclasses.dataclass(frozen=True)
class SuspectedEvent:
    """A protected candidate cluster — suspected real signal, not noise.

    Operator-facing hint for the ``steps.csv`` review flow and the
    GBIS4TS break-detection lane (§3.4/§9-Q5): sustained one-sided runs
    and persisting offsets survive detection *and* are surfaced here —
    never silently.

    Attributes:
        component: Row index into the input ``y`` (0 for 1-D input).
        i_start: Index of the first candidate epoch of the cluster.
        i_end: Index of the last candidate epoch (inclusive).
        t_start: Epoch of ``i_start`` [yr].
        t_end: Epoch of ``i_end`` [yr].
        sign: Dominant residual sign of the cluster members (−1/+1;
            ties report +1).
        step_evidence: The statistic D (:func:`step_evidence`);
            ``NaN`` = indeterminate (thin flank ⇒ protected as a
            possible step).
        kind: ``"step"`` when the step-evidence rule fired (D > k_step
            or indeterminate), else ``"transient_run"`` (run-length
            rule only).
    """

    component: int
    i_start: int
    i_end: int
    t_start: float
    t_end: float
    sign: int
    step_evidence: float
    kind: str


@dataclasses.dataclass(frozen=True)
class OutlierDetection:
    """Result of :func:`detect_outliers` — mask + diagnostics, never data.

    All per-epoch arrays have the shape of the input ``y``; the raw
    observations are NOT part of the result (requirement 3: detection
    flags, it never filters).

    Attributes:
        flags: Final outlier mask — True = OUTLIER (protection-surviving
            candidates plus Stage-0 despiked epochs). All-False when
            ``excess_flag_abort`` is True (§3.5 abort rule).
        candidates: Pre-protection candidate set of the final sweep —
            the identifier union G ∪ L **plus** the Stage-0 despiked
            epochs (so the §8.4 invariants ``flags ⊆ candidates`` and
            ``reasons == 0 off candidates`` hold with despike on).
            Despiked epochs carry ``REASON_GROSS`` (never
            ``REASON_GLOBAL``/``REASON_LOCAL``), are flagged directly,
            and are excluded from clustering, protection and the abort
            fraction; distinguish them by the reason bit, not by
            candidate membership.
        reasons: Per-epoch ``REASON_*`` bitmask (uint8; 0 exactly on
            non-candidates), final sweep.
        protected: Per-epoch ``PROTECT_*`` bitmask (uint8), final sweep.
        z: Final-sweep global detection statistic ẑ (shape of y).
        scale_global: Global robust scale ŝ per component, shape (C,)
            [whitened-residual units].
        scale_local: Local Hampel scale s_i, shape of y (NaN where the
            thin-window global fallback applied).
        fits: Final per-component inlier WLS fits of the step-augmented
            model (:class:`~gps_analysis.models.TrajectoryParams`,
            P base + K step-amplitude parameters).
        step_amplitudes: Fitted step amplitudes â, shape (C, K) in
            ``step_epochs`` input order [L]; ``None`` without steps.
        suspected_events: Protected clusters of the final sweep
            (:class:`SuspectedEvent`) — feed the operator review /
            ``steps.csv`` lane; the leaf only surfaces them.
        n_despiked: Stage-0 gross-blunder count per component, shape
            (C,) int64 — all zeros unless ``params.despike`` is True.
        n_iterations: Detection sweeps actually performed.
        converged: True when the flag mask reached a fixed point within
            ``max_iterations`` (False on abort).
        excess_flag_abort: True ⇒ the candidate fraction exceeded
            ``max_flag_fraction`` and ``flags`` is all-False by rule
            §3.5 — loud, diagnostics fully populated, never silent.
        params: Echo of the thresholds used — provenance building block
            (MATH_STANDARDS §6).
    """

    flags: NDArray[np.bool_]
    candidates: NDArray[np.bool_]
    reasons: NDArray[np.uint8]
    protected: NDArray[np.uint8]
    z: FloatArray
    scale_global: FloatArray
    scale_local: FloatArray
    fits: list[TrajectoryParams]
    step_amplitudes: FloatArray | None
    suspected_events: list[SuspectedEvent]
    n_despiked: NDArray[np.int64]
    n_iterations: int
    converged: bool
    excess_flag_abort: bool
    params: OutlierParams


def _resolve_floors(
    min_outlier: ArrayLike | None, params: OutlierParams, n_components: int
) -> FloatArray:
    """Resolve the per-component magnitude floors a_min (§3.4.1)."""
    if min_outlier is None:
        return np.full(n_components, float(params.min_outlier), dtype=np.float64)
    arr = np.atleast_1d(np.asarray(min_outlier, dtype=np.float64))
    if arr.size == 1:
        arr = np.full(n_components, float(arr[0]), dtype=np.float64)
    if arr.shape != (n_components,):
        raise ValueError(
            f"min_outlier must be scalar or shape ({n_components},), "
            f"got shape {arr.shape}"
        )
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("min_outlier must be finite and >= 0")
    return arr


def _despike_component(
    tt: FloatArray,
    y_c: FloatArray,
    protect_windows: Sequence[tuple[float, float]],
    params: OutlierParams,
    max_gap: float,
) -> NDArray[np.bool_]:
    """One component's Stage-0 gross-blunder despike (§3.0 addendum).

    Thin orchestration (no new math): gap-aware first differences on the
    RAW observations (:func:`neighbor_differences` — Stage 0 runs before
    any fit, so there are no residuals yet; formal σ is not consulted,
    which is acceptable for the k_d ≈ 10 gross-blunder regime) → robust
    difference scale ``ŝ_Δ = mad_scale(δ⁻)`` over the finite differences
    (median-centered, so a constant secular rate does not inflate it) →
    two-sided isolated-spike rule (:func:`spike_mask`) → caller
    ``protect_windows`` cleared (never despike inside a protected
    interval). Degenerate cases (< 3 usable differences, ŝ_Δ ≤ 0) despike
    nothing — the module's conservative never-flag convention.
    """
    mask = np.zeros(tt.size, dtype=np.bool_)
    delta_prev, delta_next = neighbor_differences(tt, y_c, max_gap=max_gap)
    finite = np.isfinite(delta_prev)
    if int(np.count_nonzero(finite)) < 3:
        return mask
    s_delta = mad_scale(delta_prev[finite])
    if s_delta <= 0.0:
        return mask
    mask = spike_mask(
        delta_prev,
        delta_next,
        n_sigma=params.despike_n_sigma,
        scale=s_delta,
        return_n_sigma=params.despike_return_sigma,
    )
    for t_a, t_b in protect_windows:
        mask &= ~((tt >= t_a) & (tt <= t_b))
    return mask


def _component_candidates(
    fit_model: ModelFunc,
    tt: FloatArray,
    y_c: FloatArray,
    sigma_c: FloatArray | None,
    guess: FloatArray | None,
    inliers: NDArray[np.bool_],
    params: OutlierParams,
    half_window: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    FloatArray,
    NDArray[np.bool_],
    NDArray[np.uint8],
]:
    """One component's §3.1–§3.3 sweep: robust fit → identifiers.

    Thin orchestration (no new math): WLS seed
    (:func:`~gps_analysis.fitting.fit_components`) → Huber M-fit
    (:func:`~gps_analysis.fitting._robust_params`) → residuals on ALL
    epochs → :func:`whiten` → :func:`standardize_robust` (global) +
    the windowed identifier — :func:`rolling_median`/:func:`rolling_mad`
    at ``window_order = 0`` (default), :func:`rolling_polyfit` at
    orders 1/2 — through :func:`hampel_mask` (thin windows falling back
    to the global center/scale either way).

    Returns:
        ``(r, w, z, s_global, s_local_raw, candidates, reasons)`` — raw
        residuals [L], whitened residuals, global statistic ẑ, global
        scale ŝ, raw local scale (NaN where fallback), the candidate
        mask G ∪ L and its ``REASON_*`` bitmask. Degenerate ŝ ≤ 0 ⇒ no
        candidates (documented never-flag convention,
        :func:`standardize_robust`).
    """
    sigma_in = None if sigma_c is None else sigma_c[inliers]
    seed = fit_components(
        fit_model, tt[inliers], y_c[inliers], sigma=sigma_in, p0=guess
    )[0].params
    p_hat = _robust_params(
        fit_model,
        tt[inliers],
        y_c[inliers],
        sigma_in,
        seed,
        params.loss,
        params.f_scale,
    )
    r = np.asarray(y_c - np.asarray(fit_model(tt, *p_hat), dtype=np.float64))
    w = whiten(r, sigma_c)
    z, center, s_global = standardize_robust(w, scale=params.scale_estimator)
    n = int(tt.size)
    s_local = np.full(n, np.nan, dtype=np.float64)
    candidates = np.zeros(n, dtype=np.bool_)
    reasons = np.zeros(n, dtype=np.uint8)
    if s_global <= 0.0:
        return r, w, z, s_global, s_local, candidates, reasons
    global_mask = np.abs(z) > params.global_n_sigma
    if params.window_order == 0:
        m = rolling_median(
            tt, w, half_window=half_window, min_count=params.window_min_count
        )
        s_local = rolling_mad(
            tt, w, m, half_window=half_window, min_count=params.window_min_count
        )
    else:
        m, s_local = rolling_polyfit(
            tt,
            w,
            half_window=half_window,
            min_count=params.window_min_count,
            order=params.window_order,
            robust_iterations=params.window_robust_iterations,
        )
    thin = np.isnan(s_local) | np.isnan(m)
    center_eff = np.where(thin, center, m)
    scale_eff = np.where(thin, s_global, s_local)
    local_mask = hampel_mask(
        w,
        center_eff,
        scale_eff,
        n_sigma=params.window_n_sigma,
        scale_floor=params.scale_floor,
    )
    candidates = global_mask | local_mask
    reasons[global_mask] |= np.uint8(REASON_GLOBAL)
    reasons[local_mask] |= np.uint8(REASON_LOCAL)
    return r, w, z, s_global, s_local, candidates, reasons


def _protect_component(
    tt: FloatArray,
    r: FloatArray,
    w: FloatArray,
    candidates: NDArray[np.bool_],
    s_global: float,
    floor: float,
    protect_windows: Sequence[tuple[float, float]],
    params: OutlierParams,
    max_gap: float,
    max_run: float,
    step_window: float,
    component: int,
    despiked: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.uint8], list[SuspectedEvent]]:
    """One component's §3.4 protection stage over the candidate mask.

    Thin orchestration of the protection rules (no new math): magnitude
    floor (``PROTECT_FLOOR``), caller protect-windows
    (``PROTECT_WINDOW``), and per cluster (:func:`candidate_clusters`):

    - the run-length/same-sign rule (``PROTECT_RUN``): span > L_max and
      ≥ q of the members share the residual sign — **unless** the cluster is
      a conclusive blunder (both flanks present, ``D`` small AND background
      small, i.e. it returns to the model), in which case the run-rule is
      released and the cluster is flagged (§3.4.2);
    - the step-evidence rule (``PROTECT_STEP``):
      ``D = |r̄_post − r̄_pre|/ŝ > k_step`` or D indeterminate (NaN thin
      flank — "cannot rule out a step");
    - the **elevated-background arm** (``PROTECT_RUN``, kind
      ``"transient_run"``): ``B = max(|r̄_pre|, |r̄_post|)/ŝ > k_step``
      — §3.4.2's blunder criterion is "D ≈ 0, the series *returns to
      the model*"; when the flank medians themselves sit far from the
      model the candidate rides unmodeled signal (transient wander —
      the §3.3 global-identifier over-flagging case), so it is
      protected even though D ≈ 0.

    Protected clusters become :class:`SuspectedEvent` records.
    Stage-0 ``despiked`` epochs (already-decided gross blunders,
    ``None`` ⇒ none) are excluded from the flank medians alongside the
    candidates so they cannot bias the step-evidence statistic.

    Returns:
        ``(protected, events)`` — the per-epoch ``PROTECT_*`` bitmask
        and the suspected-event records of this component.
    """
    n = int(tt.size)
    protected = np.zeros(n, dtype=np.uint8)
    events: list[SuspectedEvent] = []
    if not bool(np.any(candidates)):
        return protected, events
    flank_exclude = candidates if despiked is None else candidates | despiked
    floor_hit = candidates & (np.abs(r) < floor)
    protected[floor_hit] |= np.uint8(PROTECT_FLOOR)
    for t_a, t_b in protect_windows:
        in_window = candidates & (tt >= t_a) & (tt <= t_b)
        protected[in_window] |= np.uint8(PROTECT_WINDOW)
    for i_start, i_end in candidate_clusters(tt, candidates, max_gap=max_gap):
        members = np.flatnonzero(candidates[i_start : i_end + 1]) + i_start
        signs = np.sign(w[members])
        n_pos = int(np.count_nonzero(signs > 0))
        n_neg = int(np.count_nonzero(signs < 0))
        dominant = 1 if n_pos >= n_neg else -1
        sign_fraction = max(n_pos, n_neg) / members.size
        span = float(tt[i_end] - tt[i_start])
        background_rule = False
        if s_global > 0.0:
            med_pre, med_post = _flank_medians(
                tt, w, i_start, i_end, window=step_window, exclude=flank_exclude
            )
            if math.isnan(med_pre) or math.isnan(med_post):
                d = float("nan")
            else:
                d = abs(med_post - med_pre) / s_global
            flanks = [f for f in (med_pre, med_post) if not math.isnan(f)]
            if flanks:
                background = max(abs(f) for f in flanks) / s_global
                background_rule = background > params.step_evidence_sigma
        else:
            d = float("nan")
        step_rule = math.isnan(d) or d > params.step_evidence_sigma
        # A multi-day same-sign run is protected as possible unmodeled signal
        # UNLESS the step-evidence conclusively marks it a blunder cluster: it
        # returns to baseline (D small AND determinate) and both flanks sit at
        # the model (background small). Such a cluster is a decided blunder
        # (§3.4.2 "D ≈ 0, the series returns to the model") and must be flagged,
        # not protected. (BGÓ 2026-07-14: a 4-day −450 mm SENG Up streak that
        # returns to 0 was run-protected purely on span+sign.) A NaN/thin flank
        # keeps d indeterminate → not a conclusive blunder → run-rule stands.
        conclusive_blunder = (
            not math.isnan(d)
            and d <= params.step_evidence_sigma
            and not background_rule
        )
        run_rule = (
            span > max_run
            and sign_fraction >= params.run_sign_fraction
            and not conclusive_blunder
        )
        if run_rule or step_rule or background_rule:
            if run_rule or background_rule:
                protected[members] |= np.uint8(PROTECT_RUN)
            if step_rule:
                protected[members] |= np.uint8(PROTECT_STEP)
            events.append(
                SuspectedEvent(
                    component=component,
                    i_start=i_start,
                    i_end=i_end,
                    t_start=float(tt[i_start]),
                    t_end=float(tt[i_end]),
                    sign=dominant,
                    step_evidence=d,
                    kind="step" if step_rule else "transient_run",
                )
            )
    return protected, events


def detect_outliers(
    model: ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    step_epochs: ArrayLike | None = None,
    protect_windows: Sequence[tuple[float, float]] = (),
    min_outlier: ArrayLike | None = None,
    p0: ArrayLike | None = None,
    params: OutlierParams | None = None,
    names: Sequence[str] | None = None,
) -> OutlierDetection:
    """Detect outliers against a robust step-augmented trajectory model.

    Thin orchestration (MATH_STANDARDS §1) of the module's derivation
    chain, §3.0 → §3.5: optional Stage-0 gross-blunder despike
    (:func:`neighbor_differences` / :func:`spike_mask`; ``params.despike``,
    off by default) → :func:`~gps_analysis.fitting.with_steps` →
    robust Huber fit → :func:`whiten` → :func:`standardize_robust`
    (global identifier) + the windowed Hampel identifier
    (:func:`rolling_median` / :func:`rolling_mad` at
    ``params.window_order = 0``, the default; :func:`rolling_polyfit`
    at orders 1/2) through :func:`hampel_mask` →
    :func:`candidate_clusters` / :func:`step_evidence` signal protection
    → conservative iteration with the excess-candidate abort → final
    plain-WLS refit on the inliers. Returns a MASK plus diagnostics —
    **never** a filtered array; no input is mutated.

    Symbols → args (equations in the referenced primitives):
        - ``t_i``  → ``t``: epochs, shape (N,), fractional years
          (``yearf``), sorted ascending [yr]
        - ``y_ci`` → ``y``: observations, shape (N,) or (C, N) [L]
        - ``σ_ci`` → ``sigma``: formal 1-σ uncertainties, shape of ``y``
          [L]; relative weighting only (§3.1); ``None`` ⇒ unwhitened
        - ``t_k``  → ``step_epochs``: known step epochs, shape (K,) [yr]
          (per-station table — TOS equipment changes, coseismic offsets;
          the leaf never reads config)
        - ``[t_a, t_b]`` → ``protect_windows``: closed intervals [yr]
          inside which flagging is disabled outright (§3.4.3)
        - ``a_min`` → ``min_outlier``: per-component magnitude floor,
          scalar or shape (C,) [L]; ``None`` ⇒ ``params.min_outlier``
        - thresholds → ``params``: :class:`OutlierParams` (``None`` ⇒
          spec defaults); day-denominated windows are converted to the
          ``yearf`` axis here via 365.25 d/yr

    Args:
        model: Base trajectory model ``f(t, *p)`` (e.g.
            :func:`~gps_analysis.models.lineperiodic`, the production
            default — Bevis & Brown 2014 eq. 1).
        t: Epochs [yr]; finite, sorted ascending.
        y: Observations [L]; finite; NEVER mutated.
        sigma: Formal 1-σ uncertainties [L]; strictly positive.
        step_epochs: Known step epochs [yr]; amplitudes are estimated,
            epochs fixed. ``None``/empty ⇒ plain trajectory.
        protect_windows: Caller protect intervals ``(t_a, t_b)`` [yr].
        min_outlier: Magnitude floor(s) a_min [L] (§3.4.1).
        p0: Initial guess for the BASE model parameters, shape (P,) or
            (C, P); step amplitudes are seeded at 0.
        params: Detection thresholds; ``None`` ⇒ ``OutlierParams()``.
        names: Optional per-component labels for the returned fits.

    Returns:
        :class:`OutlierDetection` — final flags (True = outlier;
        Stage-0 despiked epochs included when ``params.despike``),
        pre-protection candidates, ``REASON_*``/``PROTECT_*`` bitmasks,
        the detection statistic ẑ, global/local scales, the final
        step-augmented inlier WLS fits, fitted step amplitudes,
        suspected-event hints, the per-component Stage-0 count
        ``n_despiked``, iteration/convergence/abort state, and
        the parameter echo. Under ``epoch_policy="union"`` an epoch
        flagged in any component is flagged in all (diagnostics stay
        per-component; union deliberately overrides per-component
        protection of the *other* components — §3.4.4).

    Raises:
        ValueError: On shape mismatches, unsorted ``t``, non-finite
            ``t``/``y``, non-positive ``sigma``, invalid
            ``protect_windows`` (t_b < t_a), or invalid
            ``min_outlier``.

    Reference:
        Design spec ``docs/DESIGN_outlier_detection.md`` §3–§4 and the
        references therein (Hampel 1974; Davies & Gather 1993; Rousseeuw
        & Croux 1993; Pearson et al. 2016; Gazeaux et al. 2013; Huber
        1964; Bevis & Brown 2014; Williams 2003 for why the two-tier
        identifier exists at all).

    Numerical notes:
        - Stage 0 (when enabled) runs ONCE, on the raw per-component
          observations, before the sweep loop: despiked epochs are
          excluded from every robust fit and from the identifier
          candidates (they carry ``REASON_GROSS`` and are flagged
          directly), so a gross blunder can neither drag the trajectory
          fit nor inflate the detection scales. Despiked epochs do NOT
          count toward the abort fraction — they are decided blunders,
          not "epochs that look like outliers".
        - The abort guard is evaluated on the per-component **candidate**
          fraction, pre-protection (§3.5's "epochs that *look like*
          outliers", §8.3 test contract) — strictly more conservative
          than a post-protection count, and immune to the protection
          stage rescuing a pathological series into silence. On abort
          the flags are all-False (Stage-0 despikes included — the §3.5
          rule is unconditional), ``excess_flag_abort=True``,
          diagnostics populated from the aborting sweep — loud, never a
          silent cap.
        - Flags are recomputed on ALL epochs each sweep (no ratchet — a
          previously flagged epoch may re-enter).
        - The Huber fit is seeded from the closed-form WLS optimum of
          the current inliers, so the M-estimation starts near the
          solution regardless of the absolute-``yearf`` conditioning.
        - Idempotence (§3.5): detection re-run on the inlier subset of a
          converged result reproduces an all-False mask (test-pinned).
    """
    detection_params = params if params is not None else OutlierParams()
    tt = _validate_sorted_time(t)
    yy, was_1d = _components_2d(y, "y")
    if yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    if not np.all(np.isfinite(yy)):
        raise ValueError("y must be finite (no NaN/inf)")
    sigmas = _per_component_sigma(sigma, yy, was_1d)
    n_components, n = yy.shape
    in_protect = np.zeros(n, dtype=np.bool_)
    for t_a, t_b in protect_windows:
        if t_b < t_a:
            raise ValueError(f"protect window ({t_a}, {t_b}) has t_b < t_a")
        in_protect |= (tt >= t_a) & (tt <= t_b)
    if names is not None and len(names) != n_components:
        raise ValueError(
            f"names has {len(names)} entries for {n_components} components"
        )
    floors = _resolve_floors(min_outlier, detection_params, n_components)

    half_window = detection_params.window_days / 2.0 / _DAYS_PER_YEAR
    max_gap = detection_params.cluster_gap_days / _DAYS_PER_YEAR
    max_run = detection_params.max_run_days / _DAYS_PER_YEAR
    step_window = detection_params.step_window_days / _DAYS_PER_YEAR

    if step_epochs is not None and np.asarray(step_epochs).size > 0:
        epochs = np.asarray(step_epochs, dtype=np.float64)
        fit_model = with_steps(model, epochs)
        n_steps = int(epochs.size)
    else:
        fit_model = model
        n_steps = 0
    n_base = _n_model_params(model)
    base_guesses = _per_component_p0(p0, n_components, was_1d)
    guesses: list[FloatArray | None] = [
        None if g is None else np.concatenate([g, np.zeros(n_steps, dtype=np.float64)])
        for g in base_guesses
    ]

    gross = np.zeros((n_components, n), dtype=np.bool_)
    if detection_params.despike:
        despike_gap = detection_params.despike_gap_days / _DAYS_PER_YEAR
        for c in range(n_components):
            gross[c] = _despike_component(
                tt, yy[c], protect_windows, detection_params, despike_gap
            )

    flags = np.zeros((n_components, n), dtype=np.bool_)
    candidates = np.zeros((n_components, n), dtype=np.bool_)
    reasons = np.zeros((n_components, n), dtype=np.uint8)
    protected = np.zeros((n_components, n), dtype=np.uint8)
    z_stat = np.zeros((n_components, n), dtype=np.float64)
    scale_global = np.zeros(n_components, dtype=np.float64)
    scale_local = np.full((n_components, n), np.nan, dtype=np.float64)
    events: list[SuspectedEvent] = []
    converged = False
    aborted = False
    n_iterations = 0

    for _sweep in range(detection_params.max_iterations):
        n_iterations += 1
        events = []
        aborted = False
        new_flags = np.zeros_like(flags)
        for c in range(n_components):
            r, w, z_c, s_g, s_loc, cand_c, reasons_c = _component_candidates(
                fit_model,
                tt,
                yy[c],
                sigmas[c],
                guesses[c],
                ~(flags[c] | gross[c] | in_protect),
                detection_params,
                half_window,
            )
            # Gross (Stage-0) epochs are decided BEFORE the identifiers:
            # remove them from the identifier candidate set used for
            # protection and the abort fraction (they are not "epochs that
            # look like outliers" — they are decided blunders), and stamp
            # their reason as REASON_GROSS only.
            cand_c &= ~gross[c]
            reasons_c = np.where(gross[c], np.uint8(REASON_GROSS), reasons_c).astype(
                np.uint8
            )
            prot_c, events_c = _protect_component(
                tt,
                r,
                w,
                cand_c,
                s_g,
                float(floors[c]),
                protect_windows,
                detection_params,
                max_gap,
                max_run,
                step_window,
                c,
                despiked=gross[c],
            )
            # Returned candidate mask INCLUDES gross so the documented §8.4
            # invariants hold with despike on: flags ⊆ candidates and
            # reasons == 0 exactly off candidates. Gross epochs carry
            # REASON_GROSS (never REASON_GLOBAL/LOCAL) and are never
            # clustered/protected/abort-counted.
            candidates[c] = cand_c | gross[c]
            reasons[c] = reasons_c
            protected[c] = prot_c
            z_stat[c] = z_c
            scale_global[c] = s_g
            scale_local[c] = s_loc
            events.extend(events_c)
            new_flags[c] = (cand_c & (prot_c == 0)) | gross[c]
            # Operator-declared protect windows are explicit "this is signal
            # here" intervals; their candidates are EXPECTED (by the operator's
            # own account the model is unrepresentative there) and must NOT trip
            # the pathological-series abort. Auto-protections (STEP/RUN/FLOOR)
            # are NOT excluded — a series the leaf must auto-protect wholesale
            # still signals a wrong model and should still abort. Numerator-only
            # (denominator stays n); protected epochs are never flagged anyway
            # (§3.5).
            abort_candidates = cand_c & ~in_protect
            n_abort = float(np.count_nonzero(abort_candidates))
            if n_abort / n > detection_params.max_flag_fraction:
                aborted = True
        if detection_params.epoch_policy == "union":
            union = np.any(new_flags, axis=0)
            new_flags = np.repeat(union[np.newaxis, :], n_components, axis=0)
        if aborted:
            flags = np.zeros_like(flags)
            converged = False
            break
        if np.array_equal(new_flags, flags):
            converged = True
            break
        flags = new_flags

    fits: list[TrajectoryParams] = []
    step_amplitudes = (
        np.zeros((n_components, n_steps), dtype=np.float64) if n_steps else None
    )
    for c in range(n_components):
        keep = ~flags[c]
        ss = sigmas[c]
        fit = fit_components(
            fit_model,
            tt[keep],
            yy[c][keep],
            sigma=None if ss is None else ss[keep],
            p0=guesses[c],
            names=None if names is None else [names[c]],
        )[0]
        fits.append(fit)
        if step_amplitudes is not None:
            step_amplitudes[c] = fit.params[n_base:]

    n_despiked = np.asarray(np.sum(gross, axis=1), dtype=np.int64)
    if was_1d:
        return OutlierDetection(
            flags=flags[0],
            candidates=candidates[0],
            reasons=reasons[0],
            protected=protected[0],
            z=z_stat[0],
            scale_global=scale_global,
            scale_local=scale_local[0],
            fits=fits,
            step_amplitudes=step_amplitudes,
            suspected_events=events,
            n_despiked=n_despiked,
            n_iterations=n_iterations,
            converged=converged,
            excess_flag_abort=aborted,
            params=detection_params,
        )
    return OutlierDetection(
        flags=flags,
        candidates=candidates,
        reasons=reasons,
        protected=protected,
        z=z_stat,
        scale_global=scale_global,
        scale_local=scale_local,
        fits=fits,
        step_amplitudes=step_amplitudes,
        suspected_events=events,
        n_despiked=n_despiked,
        n_iterations=n_iterations,
        converged=converged,
        excess_flag_abort=aborted,
        params=detection_params,
    )
