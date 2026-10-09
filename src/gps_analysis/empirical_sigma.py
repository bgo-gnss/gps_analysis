"""Empirical daily uncertainties: scatter, σ scaling, and distance-dependent common mode.

Why: GAMIT/GLOBK formal σ's do not describe the day-to-day scatter of IMO
series. Per station they are about right horizontally and ~1.7× too large in
Up, and a BASELINE's scatter is far below the quadrature sum √(σ_A²+σ_B²),
because neighbouring stations share most of their daily error (orbit,
atmosphere and the daily frame realisation) and the formal covariance
does not contain it. Measured 2026-10-09 on production TOT 2023–2026,
166 Icelandic stations, 1506 pairs < 50 km: shared fraction 0.92–0.96 at
< 1 km, levelling at ≈ 0.5 network-wide (curation
``~/gps-data/curation/sinex/common_mode.py``).

Derivation chain
----------------
1. :func:`consecutive_differences` — first differences ``Δy_i = y_{i+1} − y_i``
   over epoch pairs exactly one sampling step apart. Differencing removes
   trends, offsets, steps (one outlier each) and most slow (coloured) signal,
   leaving ``Var(Δy) = 2σ²`` for the white/daily part (von Neumann 1941).
2. :func:`daily_scatter` — ``σ̂ = MAD(Δy)/√2`` (robust: ``mad_scale``).
3. :func:`sigma_scale_factor` — ``k = MAD(Δy_i / √(σ_i² + σ_{i+1}²))``:
   the factor that makes formal σ match the scatter, robust to σ varying
   in time.
4. :func:`shared_fraction` — from the scatter of A, B and A − B on common
   epochs: ``ρ = 1 − s_AB² / (s_A² + s_B²)``.
5. :func:`shared_fraction_model` / :func:`fit_common_mode` — the network
   law ``ρ(d) = ρ∞ + (ρ₀ − ρ∞)·exp(−d/L)`` fitted over many pairs.
6. :func:`baseline_sigma` — ``σ_AB = √(σ_A² + σ_B² − 2ρ·σ_A·σ_B)`` with
   σ_A, σ_B already scaled (step 3) and ρ from step 5 (or step 4 for a
   pair with its own long record).

Scope: daily (white) σ only. Rate uncertainty needs the time-correlated
model in :mod:`gps_analysis.noise`; nothing here feeds a rate σ.

Conventions: time in fractional years, sampling ``step`` in the same unit;
displacements/σ in the caller's unit [L]; distances in the caller's unit
[D] (the fitted L carries it). Pure numpy/scipy, float64, inputs never
mutated.

References
----------
- von Neumann 1941, Ann. Math. Stat. 12(4), 367–395 (mean square successive
  difference: E[(Δy)²] = 2σ² for independent errors).
- Rousseeuw & Croux 1993, JASA 88(424), 1273–1283 (MAD scale, 50 % breakdown).
- Wdowinski et al. 1997, JGR 102(B8), 18057–18070 (common-mode error shared
  across a regional network; spatial filtering).
- Márquez-Azúa & DeMets 2003, JGR 108(B9), 2450 (correlation of daily
  residuals decreasing with station separation).
- Tian & Shen 2016, JGR Solid Earth 121, 1080–1096 (distance dependence of
  the regional common-mode component).
- The plateau + exponential form of ρ(d) is the empirical law chosen from the
  2026-10-09 IMO fit (a single exponential to zero misfit the ≈ 0.5 plateau).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .outliers import mad_scale

__all__ = [
    "CommonModeModel",
    "baseline_sigma",
    "consecutive_differences",
    "daily_scatter",
    "fit_common_mode",
    "shared_fraction",
    "shared_fraction_model",
    "sigma_scale_factor",
]

FloatArray = NDArray[np.float64]

#: GAMIT daily epochs in fractional years (one day; the yearf convention of
#: the .dat files uses 365 or 366 per year, inside the default tolerance).
DAY_YEARS = 1.0 / 365.25

#: Fewer usable differences than this → NaN (a MAD of a handful of
#: differences is too noisy to act on).
MIN_PAIRS_DEFAULT = 60


def _series(t: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if yy.ndim == 1:
        yy = yy[None, :]
    if yy.ndim != 2 or yy.shape[1] != tt.size:
        raise ValueError(f"y must be (N,) or (C, N) with N = {tt.size}, got {yy.shape}")
    if tt.size > 1 and np.any(np.diff(tt) <= 0):
        raise ValueError("t must be strictly increasing")
    return tt, yy


def consecutive_differences(
    t: ArrayLike,
    y: ArrayLike,
    *,
    step: float = DAY_YEARS,
    tol: float = 0.3,
) -> tuple[FloatArray, NDArray[np.intp]]:
    """First differences over consecutive epochs one sampling step apart.

    Equation:
        ``Δy_c,i = y_c(t_{i+1}) − y_c(t_i)`` for every i with
        ``|t_{i+1} − t_i − step| < tol·step``

    Symbols → args:
        - ``t`` → ``t``: epochs, (N,), strictly increasing [yr]
        - ``y`` → ``y``: values, (N,) or (C, N) [L]
        - ``step`` → sampling interval [yr] (default one day)
        - ``tol`` → accepted deviation as a fraction of ``step`` [–]

    Returns:
        ``(Δy (C, M), i (M,))`` — the differences and the index of each
        pair's first epoch. Pairs across a data gap are excluded, never
        bridged. Pairs where either value is non-finite yield NaN.

    Reference:
        von Neumann 1941 (successive differences).
    """
    tt, yy = _series(t, y)
    if step <= 0 or not 0 < tol < 0.5:
        raise ValueError("step must be > 0 and 0 < tol < 0.5")
    ok = np.flatnonzero(np.abs(np.diff(tt) - step) < tol * step).astype(np.intp)
    return yy[:, ok + 1] - yy[:, ok], ok


def _robust_or_nan(v: FloatArray, min_pairs: int) -> float:
    v = v[np.isfinite(v)]
    return mad_scale(v) if v.size >= max(min_pairs, 3) else float("nan")


def daily_scatter(
    t: ArrayLike,
    y: ArrayLike,
    *,
    step: float = DAY_YEARS,
    tol: float = 0.3,
    min_pairs: int = MIN_PAIRS_DEFAULT,
) -> FloatArray:
    """Robust day-to-day scatter σ̂ of a series, per component.

    Equation:
        ``σ̂_c = MAD(Δy_c) / √2``,  ``MAD = 1.4826·med|Δy − med Δy|``
        (``Var(Δy) = 2σ²`` for independent daily errors)

    Symbols → args:
        - ``Δy`` → :func:`consecutive_differences` of ``t``, ``y`` [L]
        - ``min_pairs`` → minimum finite differences, else NaN

    Returns:
        ``σ̂`` (C,) [L]. Under coloured noise this measures the
        high-frequency (daily) part only — the right σ for one epoch, NOT
        for a rate.

    Reference:
        von Neumann 1941; Rousseeuw & Croux 1993 (MAD).

    Numerical notes:
        The median of Δy is ~0 for a smooth signal; centring on it (not on
        0) keeps a strong constant drift between days out of the scale.
    """
    dy, _ = consecutive_differences(t, y, step=step, tol=tol)
    out: FloatArray = np.array(
        [_robust_or_nan(row, min_pairs) for row in dy]
    ) / np.sqrt(2.0)
    return out


def sigma_scale_factor(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    step: float = DAY_YEARS,
    tol: float = 0.3,
    min_pairs: int = MIN_PAIRS_DEFAULT,
) -> FloatArray:
    """Factor k by which formal σ must be multiplied to match the scatter.

    Equation:
        ``z_c,i = Δy_c,i / √(σ_c(t_i)² + σ_c(t_{i+1})²)``,  ``k_c = MAD(z_c)``

    Symbols → args:
        - ``Δy`` → :func:`consecutive_differences` [L]
        - ``σ`` → ``sigma``: formal σ, same shape as ``y`` [L]

    Returns:
        ``k`` (C,) [–]; ``k = 1`` means the formal σ is realistic, ``k < 1``
        pessimistic. Normalising each difference by its own σ pair makes k
        insensitive to σ changing over the record (receiver or network
        changes).

    Reference:
        von Neumann 1941; Rousseeuw & Croux 1993.
    """
    tt, yy = _series(t, y)
    ss = np.asarray(sigma, dtype=np.float64)
    if ss.ndim == 1:
        ss = ss[None, :]
    if ss.shape != yy.shape:
        raise ValueError(f"sigma shape {ss.shape} != y shape {yy.shape}")
    dy, i = consecutive_differences(tt, yy, step=step, tol=tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = dy / np.sqrt(ss[:, i] ** 2 + ss[:, i + 1] ** 2)
    return np.array([_robust_or_nan(row, min_pairs) for row in z])


def shared_fraction(s_a: ArrayLike, s_b: ArrayLike, s_ab: ArrayLike) -> FloatArray:
    """Fraction ρ of the daily error two stations share, from three scatters.

    Equation:
        ``ρ = 1 − s_AB² / (s_A² + s_B²)``
        (from ``s_AB² = s_A² + s_B² − 2·cov``: ``ρ = cov / ((s_A² + s_B²)/2)``,
        the correlation when ``s_A = s_B``)

    Symbols → args:
        - ``s_A``, ``s_B`` → :func:`daily_scatter` of each station [L]
        - ``s_AB`` → :func:`daily_scatter` of the baseline A − B [L]
        (all three on the SAME common epochs)

    Returns:
        ρ, broadcast shape [–]; may fall slightly outside [0, 1] from
        sampling noise — not clipped, so fits stay unbiased.

    Reference:
        Wdowinski et al. 1997 (common-mode error); Márquez-Azúa & DeMets 2003.
    """
    a, b, ab = (np.asarray(v, dtype=np.float64) for v in (s_a, s_b, s_ab))
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = 1.0 - ab**2 / (a**2 + b**2)
    return out


@dataclasses.dataclass(frozen=True)
class CommonModeModel:
    """Per-component ``ρ(d) = ρ∞ + (ρ₀ − ρ∞)·exp(−d/L)`` parameters.

    Attributes:
        rho0: ρ at zero separation, (C,) [–]
        rho_inf: network-wide plateau, (C,) [–]
        length: decay length L, (C,) [D, the caller's distance unit]
        components: component labels, e.g. ``("N", "E", "U")``
        distance_unit: label of [D], e.g. ``"km"`` (carried, not used)
    """

    rho0: tuple[float, ...]
    rho_inf: tuple[float, ...]
    length: tuple[float, ...]
    components: tuple[str, ...] = ("N", "E", "U")
    distance_unit: str = "km"

    def __post_init__(self) -> None:
        n = len(self.components)
        if not len(self.rho0) == len(self.rho_inf) == len(self.length) == n:
            raise ValueError(
                "rho0, rho_inf, length and components must have equal length"
            )
        for r0, ri, el in zip(self.rho0, self.rho_inf, self.length, strict=True):
            if not (0.0 <= ri <= r0 <= 1.0) or el <= 0:
                raise ValueError("need 0 <= rho_inf <= rho0 <= 1 and length > 0")

    def to_record(self) -> dict[str, Any]:
        """JSON-ready dict (round-trips through :meth:`from_record`)."""
        return {"model": "plateau_exponential", **dataclasses.asdict(self)}

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> CommonModeModel:
        if rec.get("model") != "plateau_exponential":
            raise ValueError(f"unknown common-mode model {rec.get('model')!r}")
        return cls(
            rho0=tuple(rec["rho0"]),
            rho_inf=tuple(rec["rho_inf"]),
            length=tuple(rec["length"]),
            components=tuple(rec["components"]),
            distance_unit=str(rec["distance_unit"]),
        )


def shared_fraction_model(distance: ArrayLike, model: CommonModeModel) -> FloatArray:
    """Shared daily-error fraction ρ(d) predicted by ``model``, per component.

    Equation:
        ``ρ_c(d) = ρ∞_c + (ρ₀_c − ρ∞_c)·exp(−d/L_c)``

    Symbols → args:
        - ``d`` → ``distance``: separation(s), any shape, ≥ 0 [D]
        - ``ρ₀, ρ∞, L`` → ``model`` fields

    Returns:
        ρ, shape ``(C,) + distance.shape`` [–].

    Reference:
        Empirical form (module docstring); cf. Tian & Shen 2016.
    """
    d = np.asarray(distance, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError("distance must be >= 0")
    r0, ri, el = (
        np.asarray(v, dtype=np.float64).reshape((-1,) + (1,) * d.ndim)
        for v in (model.rho0, model.rho_inf, model.length)
    )
    out: FloatArray = ri + (r0 - ri) * np.exp(-d[None] / el)
    return out


def fit_common_mode(
    distance: ArrayLike,
    rho: ArrayLike,
    *,
    components: tuple[str, ...] = ("N", "E", "U"),
    distance_unit: str = "km",
    f_scale: float = 0.05,
    length_bounds: tuple[float, float] = (0.05, 1000.0),
) -> CommonModeModel:
    """Fit ``ρ(d) = ρ∞ + (ρ₀ − ρ∞)·exp(−d/L)`` to per-pair shared fractions.

    Equation (per component, over pairs j):
        ``min Σ_j ϱ((ρ_j − ρ(d_j))/f)``, ``ϱ(z) = 2(√(1+z²) − 1)`` (soft-L1),
        subject to ``0 ≤ ρ∞ ≤ ρ₀ ≤ 1``, ``L ∈ length_bounds``

    Symbols → args:
        - ``d_j`` → ``distance`` (M,) [D]
        - ``ρ_j`` → ``rho`` (C, M) from :func:`shared_fraction` [–]
        - ``f`` → ``f_scale``: residual size where the loss turns linear [–]

    Returns:
        :class:`CommonModeModel`.

    Reference:
        scipy ``least_squares`` (trust-region reflective, soft-L1 loss);
        model form: module docstring.

    Numerical notes:
        ρ₀ ≥ ρ∞ is imposed by fitting ``Δ = ρ₀ − ρ∞ ∈ [0, 1]`` and clipping
        ρ₀ to 1. Non-finite ρ_j are dropped per component. Needs pairs
        spanning both short and long separations, else L is unidentified.
    """
    d = np.asarray(distance, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    if r.ndim == 1:
        r = r[None]
    if d.ndim != 1 or r.shape != (len(components), d.size):
        raise ValueError(
            f"need distance (M,) and rho ({len(components)}, M); got {d.shape}, {r.shape}"
        )
    r0s, ris, els = [], [], []
    for row in r:
        ok = np.isfinite(row) & np.isfinite(d)
        if ok.sum() < 5:
            raise ValueError("fit_common_mode needs >= 5 finite pairs per component")
        dd, rr = d[ok], row[ok]

        def resid(
            p: FloatArray, dd: FloatArray = dd, rr: FloatArray = rr
        ) -> FloatArray:
            ri, delta, el = p
            out: FloatArray = ri + delta * np.exp(-dd / el) - rr
            return out

        sol = optimize.least_squares(
            resid,
            x0=[0.5, 0.4, float(np.clip(np.median(dd), *length_bounds))],
            bounds=([0.0, 0.0, length_bounds[0]], [1.0, 1.0, length_bounds[1]]),
            loss="soft_l1",
            f_scale=f_scale,
        )
        ri, delta, el = (float(v) for v in sol.x)
        r0s.append(min(ri + delta, 1.0))
        ris.append(ri)
        els.append(el)
    return CommonModeModel(
        tuple(r0s), tuple(ris), tuple(els), components, distance_unit
    )


def baseline_sigma(
    sigma_a: ArrayLike, sigma_b: ArrayLike, rho: ArrayLike
) -> FloatArray:
    """σ of the difference A − B of two stations sharing a fraction ρ of their error.

    Equation:
        ``σ_AB = √(σ_A² + σ_B² − 2ρ·σ_A·σ_B)``  (ρ = 0: quadrature)

    Symbols → args:
        - ``σ_A``, ``σ_B`` → ``sigma_a``, ``sigma_b`` (realistic, e.g. scaled
          by :func:`sigma_scale_factor`) [L]
        - ``ρ`` → ``rho``: shared fraction, broadcastable, ``[−1, 1]`` [–]

    Returns:
        σ_AB [L], broadcast shape; the radicand is floored at 0 (ρ ≈ 1 with
        σ_A ≈ σ_B can round slightly negative).

    Reference:
        Variance of a difference of correlated variables; ρ from
        :func:`shared_fraction_model` (Wdowinski et al. 1997 for the
        common-mode interpretation).
    """
    a, b, p = (np.asarray(v, dtype=np.float64) for v in (sigma_a, sigma_b, rho))
    if np.any(np.abs(p) > 1):
        raise ValueError("rho must lie in [-1, 1]")
    out: FloatArray = np.sqrt(np.maximum(a**2 + b**2 - 2.0 * p * a * b, 0.0))
    return out
