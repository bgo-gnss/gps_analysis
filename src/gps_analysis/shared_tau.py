"""One transient time scale shared by many series, and the rate it implies.

What this estimates
-------------------
Many series (stations × components) after one event, each with its own
linear amplitudes but ONE shared e-folding time τ of the transient term:

    ``yᵢ = Φᵢ(τ)·cᵢ + εᵢ``,   ``i = 1..S``,  ``εᵢ ~ N(0, diag(σᵢ²))``.

The use is a recharging volcano (one source, one relaxation time seen by
every station) — e.g. Svartsengi after each eruption, where
``offset + rate·t + Φ·(1 − e^(−Δt/τ))`` fits the curved recharge that no
straight line over a year does. The curve's derivative is then a velocity
on any chosen date (:func:`transient_rate`), which is what tracking an
ongoing event needs.

Derivation chain
----------------
1. **Concentration, summed** — for fixed θ = ln τ each series has the exact
   WLS optimum, so the joint profile objective is the SUM of the series'
   concentrated objectives (the series share no amplitude):

       ``χ²(θ) = Σᵢ ‖(I − UᵢUᵢᵀ)·y_w,i‖₂²``

   (Golub & Pereyra 1973; :mod:`gps_analysis.varpro` derivation step 1).
   The block-diagonal joint design never has to be formed.
2. **Search** — coarse grid in θ over ``tau_bounds`` + bounded Brent
   polish, keeping the better (the :mod:`gps_analysis.varpro` structure).
3. **σ_θ** — from the profile curvature, ``σ_θ² = 2·ŝ² / χ²''(θ̂)``,
   ``ŝ² = χ²(θ̂)/(N − P − 1)``. The second derivative of the concentrated
   objective is the Schur complement of the bordered information matrix,
   so this equals varpro's bordered ``C[-1, -1]`` (O'Leary & Rust 2013
   §2.5) to finite-difference accuracy — test-pinned against
   :func:`gps_analysis.varpro.estimate_varpro` on a single series.
4. **Interval** — the scaled Δχ² = ŝ² profile interval
   (Venzon & Moolgavkar 1988), as varpro step 5; an edge that does not
   close inside the box is reported open.
5. **Per-series rate** (:func:`transient_rate`) — with τ fixed at τ̂ the
   model is linear; ``dx/dt`` at an epoch is ``gᵀc`` with
   ``g = ∂Φ/∂t``, its variance ``gᵀCg`` plus ``(∂r/∂θ)²·σ_θ²`` with the
   TOTAL derivative ∂r/∂θ (amplitudes refitted at θ ± h). The τ part
   treats θ̂ as independent of the one series — exact when θ̂ comes from
   other data or a fixed number, conservative-to-neutral when the series
   is one of many that set it.

Each series' result depends on that series and on (τ̂, σ_θ) only: a shared
τ is a separate, named input — the k(T)-table pattern of the velocity
product, so a station's rate does not silently depend on which other
stations were in a run.

Conventions (``docs/MATH_STANDARDS.md``): t in fractional years, [L]
unit-agnostic (mm in IMO production), parameter vectors returned
absolute-t. :class:`~gps_analysis.terms.ExpTransient`'s amplitude ``Φ``
is the accumulated form (``models.exp_linear``'s ``A = −Φ``,
``k = 1/τ``).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize_scalar

from .models import FloatArray
from .terms import _HOUSE_TRANSIENTS, TrajectoryModel

_N_GRID_DEFAULT = 33
_D_THETA = 1e-3  # finite-difference step in θ = ln τ (0.1 % of τ)
_D_T = 1e-4  # finite-difference step in t [yr] for the design derivative

Series = tuple[ArrayLike, ArrayLike, ArrayLike]


@dataclasses.dataclass(frozen=True)
class SharedTauFit:
    """Result of :func:`profile_shared_tau`.

    Attributes:
        tau: τ̂ [yr]; ``1/tau`` is ``models.exp_linear``'s decay rate k.
        tau_sigma: 1-σ of τ̂ by the delta method ``τ̂·σ_θ`` [yr].
        theta, theta_sigma: θ̂ = ln τ̂ and its σ [ln yr] — the pair
            :func:`transient_rate` takes.
        tau_interval: Δχ² = ŝ² profile interval in τ [yr].
        interval_open_lower / interval_open_upper: that side did not close
            inside ``tau_bounds`` — a bound, not an estimate.
        chisq: Σ whitened residual² at τ̂.
        scale_sq: ŝ² = chisq/(N − P − 1).
        n_obs, n_params, n_series: N, P (linear amplitudes, all series), S.
        series_chisq: each series' χ² at τ̂, in input order.
    """

    tau: float
    tau_sigma: float
    theta: float
    theta_sigma: float
    tau_interval: tuple[float, float]
    interval_open_lower: bool
    interval_open_upper: bool
    chisq: float
    scale_sq: float
    n_obs: int
    n_params: int
    n_series: int
    series_chisq: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class TransientRate:
    """Result of :func:`transient_rate` for one series.

    Attributes:
        rate: dx/dt at each requested epoch [L/yr], shape (M,).
        sigma: its 1-σ including the τ uncertainty [L/yr], shape (M,).
        sigma_fixed_tau: the 1-σ with τ held exact [L/yr], shape (M,).
        params: absolute-t amplitudes at τ, ``model.param_names`` order.
        covariance: their covariance (formal × ŝ²), shape (P, P).
        chi2nu: χ²/(N − P) of the series at τ.
        n_obs: N used (finite y).
        model: the model with τ substituted.
    """

    rate: FloatArray
    sigma: FloatArray
    sigma_fixed_tau: FloatArray
    params: FloatArray
    covariance: FloatArray
    chi2nu: float
    n_obs: int
    model: TrajectoryModel


def _transient_index(model: TrajectoryModel, term_index: int | None) -> int:
    found = [i for i, tm in enumerate(model.terms) if tm.group == "transient"]
    if term_index is None:
        if len(found) != 1:
            raise ValueError(f"model has {len(found)} transient terms; pass term_index")
        term_index = found[0]
    if not 0 <= term_index < len(model.terms):
        raise ValueError(f"term_index {term_index} out of range")
    if not isinstance(model.terms[term_index], _HOUSE_TRANSIENTS):
        raise ValueError(
            f"term {term_index} is {model.terms[term_index].kind!r}; a house "
            "transient (exp_transient / log_transient) with a tau is needed"
        )
    return term_index


def _with_tau(model: TrajectoryModel, index: int, tau: float) -> TrajectoryModel:
    terms = list(model.terms)
    terms[index] = dataclasses.replace(terms[index], tau=tau)  # type: ignore[type-var]
    return TrajectoryModel(terms)


def _clean(t: ArrayLike, y: ArrayLike, s: ArrayLike) -> tuple[FloatArray, ...]:
    tt, yy, ss = (np.asarray(a, dtype=np.float64) for a in (t, y, s))
    if not (tt.ndim == yy.ndim == ss.ndim == 1 and tt.shape == yy.shape == ss.shape):
        raise ValueError("each series needs 1-D t, y, sigma of equal length")
    ok = np.isfinite(tt) & np.isfinite(yy) & np.isfinite(ss)
    if np.any(ss[ok] <= 0):
        raise ValueError("sigma must be > 0")
    return tt[ok], yy[ok], ss[ok]


def _wls(
    model: TrajectoryModel, t: FloatArray, y: FloatArray, s: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Centered WLS: (absolute-t params, unscaled cov, χ²)."""
    t_ref = float(np.mean(t))
    a = model.design(t, t_ref=t_ref) / s[:, None]
    b = y / s
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    if sv[-1] <= sv[0] * max(a.shape) * np.finfo(float).eps:
        raise ValueError("design is rank deficient (series too short?)")
    c = vt.T @ ((u.T @ b) / sv)
    cov = (vt.T / sv**2) @ vt
    chisq = float(np.sum((b - a @ c) ** 2))
    m = model.uncentering(t_ref)
    return m @ c, m @ cov @ m.T, chisq


def profile_shared_tau(
    model: TrajectoryModel,
    series: Sequence[Series],
    *,
    term_index: int | None = None,
    tau_bounds: tuple[float, float] = (0.01, 10.0),
    n_grid: int = _N_GRID_DEFAULT,
) -> SharedTauFit:
    """Estimate one τ shared by many series (summed VARPRO profile).

    Equation:
        ``θ̂ = argmin_θ Σᵢ ‖(I − Uᵢ(θ)Uᵢ(θ)ᵀ)·y_w,i‖²``,  ``τ̂ = e^θ̂``
        (module derivation steps 1–4).

    Symbols → args:
        - Φᵢ(τ) → ``model``: the trajectory model every series is fitted
          with (typically ``Polynomial(1)`` + ``ExpTransient(epoch, τ₀)``;
          the τ₀ in it is ignored)
        - transient term → ``term_index`` (None ⇒ the single one)
        - (tᵢ, yᵢ, σᵢ) → ``series``: [yr], [L], [L]; non-finite rows dropped
        - τ search box → ``tau_bounds`` [yr], 0 < lo < hi
        - grid density → ``n_grid`` (≥ 2)

    Returns:
        :class:`SharedTauFit`.

    Raises:
        ValueError: no series, bad bounds, a rank-deficient series, or
            N ≤ P + 1.

    Warns:
        UserWarning: the profile interval is open on a side (a bound), or
            the data after the event span less than 3·τ̂.

    Reference:
        Golub & Pereyra 1973 (concentration); O'Leary & Rust 2013, §2.5
        (covariance — the profile curvature is its Schur complement);
        Venzon & Moolgavkar 1988 (profile interval).
    """
    if not series:
        raise ValueError("no series")
    lo, hi = float(tau_bounds[0]), float(tau_bounds[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError(f"tau_bounds must be 0 < lo < hi, got {tau_bounds}")
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    idx = _transient_index(model, term_index)
    data = [_clean(*s) for s in series]
    n_obs = sum(len(d[0]) for d in data)
    n_params = model.n_params * len(data)
    if n_obs <= n_params + 1:
        raise ValueError(f"N = {n_obs} observations for P + 1 = {n_params + 1}")

    def chi(theta: float) -> float:
        m = _with_tau(model, idx, math.exp(theta))
        return float(sum(_wls(m, *d)[2] for d in data))

    th_lo, th_hi = math.log(lo), math.log(hi)
    grid = np.linspace(th_lo, th_hi, n_grid)
    vals = np.array([chi(g) for g in grid])
    j = int(np.argmin(vals))
    best_th, best = float(grid[j]), float(vals[j])
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, n_grid - 1)]
    pol = minimize_scalar(chi, bounds=(a, b), method="bounded",
                          options={"xatol": 1e-6})  # fmt: skip
    if pol.success and float(pol.fun) < best:
        best_th, best = float(pol.x), float(pol.fun)
    scale_sq = best / (n_obs - n_params - 1)

    h = 1e-3
    lo_th, hi_th = max(best_th - h, th_lo), min(best_th + h, th_hi)
    mid = (lo_th + hi_th) / 2
    hh = (hi_th - lo_th) / 2
    curv = (chi(mid + hh) - 2 * chi(mid) + chi(mid - hh)) / hh**2
    theta_sigma = math.sqrt(2 * scale_sq / curv) if curv > 0 else math.inf

    def crossing(edge: float) -> tuple[float, bool]:
        def f(th: float) -> float:
            return chi(th) - best - scale_sq

        if f(edge) < 0:
            return edge, True
        root = brentq(f, min(best_th, edge), max(best_th, edge), xtol=1e-6)
        return float(root), False

    (i_lo, open_lo), (i_hi, open_hi) = crossing(th_lo), crossing(th_hi)
    tau = math.exp(best_th)
    m_hat = _with_tau(model, idx, tau)
    series_chisq = tuple(_wls(m_hat, *d)[2] for d in data)
    if open_lo or open_hi:
        warnings.warn(
            f"shared tau profile interval is open ({'lower' if open_lo else ''}"
            f"{' ' if open_lo and open_hi else ''}{'upper' if open_hi else ''}): "
            "publish a bound on tau, not an estimate",
            UserWarning,
            stacklevel=2,
        )
    epoch = float(model.terms[idx].epoch)  # type: ignore[attr-defined]
    t_post = max(float(np.max(d[0])) for d in data) - epoch
    if t_post < 3 * tau:
        warnings.warn(
            f"data span {t_post:.2f} yr after the event is < 3·tau "
            f"({3 * tau:.2f} yr): tau and the long-term rate trade off",
            UserWarning,
            stacklevel=2,
        )
    return SharedTauFit(
        tau=tau,
        tau_sigma=tau * theta_sigma,
        theta=best_th,
        theta_sigma=theta_sigma,
        tau_interval=(math.exp(i_lo), math.exp(i_hi)),
        interval_open_lower=open_lo,
        interval_open_upper=open_hi,
        chisq=best,
        scale_sq=scale_sq,
        n_obs=n_obs,
        n_params=n_params,
        n_series=len(data),
        series_chisq=series_chisq,
    )


def model_rate(
    model: TrajectoryModel, params: ArrayLike, covariance: ArrayLike, at: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """dx/dt of a fitted linear-in-parameters model, with its 1-σ.

    Equation:
        ``r(t) = g(t)ᵀ·c``,  ``g = ∂Φ(t)/∂t``,  ``σ_r² = gᵀ·C·g``

    with g by a central difference in t (step 1e-4 yr; every house term is
    smooth away from its own onset epoch).

    Args:
        model: The model (τ fixed).
        params: Absolute-t amplitudes, ``model.param_names`` order.
        covariance: Their covariance, (P, P).
        at: Epochs [yr], scalar or (M,).

    Returns:
        (rate, sigma), each shape (M,) [L/yr].
    """
    tt = np.atleast_1d(np.asarray(at, dtype=np.float64))
    g = (model.design(tt + _D_T) - model.design(tt - _D_T)) / (2 * _D_T)
    c = np.asarray(params, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    return g @ c, np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))


def transient_rate(
    model: TrajectoryModel,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    at: ArrayLike,
    *,
    tau: float,
    tau_sigma: float = 0.0,
    term_index: int | None = None,
) -> TransientRate:
    """Fit one series with τ fixed and give dx/dt on chosen dates.

    Equation:
        ``r(t) = gᵀĉ(τ)``,
        ``σ_r² = gᵀ·C·g + (∂r/∂θ)²·σ_θ²``,  ``θ = ln τ``,  ``σ_θ = σ_τ/τ``

    For ``Polynomial(1) + ExpTransient``:
    ``r(t) = rate + (Φ/τ)·e^(−(t−t₀)/τ)`` (``models.exp_linear_rate`` with
    ``A = −Φ``, ``k = 1/τ``) — the long-term ``rate`` plus a decaying
    excess.

    Symbols → args:
        - model → ``model`` (its τ is replaced by ``tau``)
        - (t, y, σ) → ``t``, ``y``, ``sigma`` [yr], [L], [L]
        - epochs → ``at`` [yr]
        - τ̂, σ_τ → ``tau``, ``tau_sigma`` [yr] (e.g. from
          :func:`profile_shared_tau`; 0 for a τ taken as exact)

    Returns:
        :class:`TransientRate`; covariance and σ are formal × ŝ²
        (ŝ² = χ²/(N − P), the curve_fit convention).

    Raises:
        ValueError: bad τ, or a rank-deficient series.
    """
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be > 0, got {tau}")
    idx = _transient_index(model, term_index)
    tt, yy, ss = _clean(t, y, sigma)
    m = _with_tau(model, idx, tau)
    params, cov, chisq = _wls(m, tt, yy, ss)
    dof = len(tt) - m.n_params
    if dof <= 0:
        raise ValueError(f"{len(tt)} observations for {m.n_params} parameters")
    s2 = chisq / dof
    cov = cov * s2
    rate, sig = model_rate(m, params, cov, at)
    total = sig
    if tau_sigma > 0:
        th = math.log(tau)
        rs = []
        for d in (+_D_THETA, -_D_THETA):
            mm = _with_tau(model, idx, math.exp(th + d))
            p, c, _ = _wls(mm, tt, yy, ss)
            rs.append(model_rate(mm, p, c, at)[0])
        drdth = (rs[0] - rs[1]) / (2 * _D_THETA)
        total = np.sqrt(sig**2 + (drdth * tau_sigma / tau) ** 2)
    return TransientRate(
        rate=rate,
        sigma=total,
        sigma_fixed_tau=sig,
        params=params,
        covariance=cov,
        chi2nu=s2,
        n_obs=len(tt),
        model=m,
    )
