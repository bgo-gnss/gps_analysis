"""Colored-noise (power-law + white) maximum-likelihood estimation for GNSS series.

The honest-σ velocity engine (plan §9b): fit a **white + power-law** noise
model jointly with a linear-in-parameters trajectory (intercept, secular
rate, seasonal terms) by maximum likelihood, so the reported rate
uncertainty reflects the temporal correlation of real GNSS daily solutions
instead of the optimistic white-noise formal error (Zhang et al. 1997;
Williams 2003; Williams et al. 2004). This module supplies the noise/MLE
machinery; :func:`gps_analysis.velocity.estimate_velocity_mle` is the
velocity-product orchestrator (``method="mle"``) and
:func:`gps_analysis.velocity.detectability_floor` the derived alarm
threshold.

Noise model (shared with :mod:`gps_analysis.transient` — same family)
----------------------------------------------------------------------
For n uniformly sampled epochs the data covariance is (Williams 2003,
J. Geodesy 76, eq. 4; Yang et al. 2023 SI Text S1):

    ``C = σ_w²·I + β²·ΔT^(−κ/2)·(T Tᵀ)``,  ``ΔT = 1/365 yr``

with T the unit lower-triangular Toeplitz matrix of the fractional-
integration coefficients ψ of ``(1 − L)^{κ/2}``
(:func:`gps_analysis.transient._powerlaw_psi`; Hosking 1981) — σ_w the
white-noise amplitude [L], β the power-law amplitude [L·yr^(−κ/4)],
κ ≤ 0 the spectral index (0 white, −1 flicker, −2 random walk). C is
**not Toeplitz** (power-law noise with κ ≤ −1 is nonstationary), but it
has displacement rank 2, so the generalized Schur algorithm factorizes it
exactly in O(n²) — the machinery of
:func:`gps_analysis.transient._schur_logdet_quad` (task H3), generalized
here to whiten a *matrix* right-hand side (:func:`_schur_whiten`).

Derivation chain (the MLE)
--------------------------
Reparameterize the covariance by a total scale s and a mixing angle φ
(the CATS form — Williams 2008, GPS Solut. 12; also Bos et al. 2013 §2):

    ``C = s²·C₀(φ, κ)``, ``C₀ = cos²φ·I + sin²φ·ΔT^(−κ/2)·(T Tᵀ)``
    with ``σ_w = s·cos φ``, ``β = s·sin φ``.

For a trajectory that is linear in its parameters, ``y = A·p + ε``,
``ε ~ N(0, C)``, both p and s² have closed-form conditional maxima, so the
numeric search is only 2-D over (φ, κ):

1. **Whitening** — :func:`_schur_whiten` computes ``L₀⁻¹[A | y]`` and
   ``ln det C₀`` in one O(n²·(P+1)) generalized-Schur pass
   (``C₀ = L₀L₀ᵀ``), never forming C₀.
2. **GLS profile over p** — :func:`_profile_fit`:
   ``p̂(φ, κ) = argmin_p ‖ỹ − Ã·p‖₂²`` (``Ã = L₀⁻¹A``, ``ỹ = L₀⁻¹y``),
   solved by the SVD path of :func:`gps_analysis.fitting._wls_solve`,
   with unit-scale covariance ``(ÃᵀÃ)⁻¹ = (AᵀC₀⁻¹A)⁻¹``.
3. **Scale profile** — ``ŝ² = RSS/n``, ``RSS = ‖ỹ − Ã·p̂‖₂²`` (the ML
   optimum of s² given everything else), leaving the concentrated
   objective ``−2 ln L(φ, κ) = n·ln ŝ² + ln det C₀ + n·(1 + ln 2π)``.
4. **2-D maximization** — :func:`estimate_noise_mle`: coarse
   (φ, κ) grid then a bounded Nelder–Mead polish
   (``scipy.optimize.minimize``), returning the noise triple
   (σ̂_w, β̂, κ̂), the trajectory parameters p̂ and their
   **colored-noise** covariance ``Ĉ_p = ŝ²·(AᵀC₀⁻¹A)⁻¹`` — the honest
   σ_v lives in its rate slot.
5. **Rate-uncertainty primitive** — :func:`powerlaw_rate_sigma`: the
   exact finite-n GLS σ_v of a pure linear fit under a *given* noise
   triple; reproduces the Williams 2003 (eqs. 23–30) span scalings
   σ_v² ∝ T⁻³ (white), T⁻² (flicker), T⁻¹ (random walk) and feeds
   :func:`gps_analysis.velocity.detectability_floor`.

Conventions and caveats (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------------------
- Time numeric **fractional years**; displacement/amplitudes in the
  caller's unit [L] (mm in IMO production); rates [L/yr]; κ dimensionless;
  β in [L·yr^(−κ/4)] (Williams 2003 normalization — the ΔT^(−κ/4)
  scaling makes β sampling-interval independent and decorrelates β from κ).
- The covariance recursion is **index-lagged**: uniform *daily* sampling
  is assumed (ΔT = 1/365 yr, the hard-coded convention shared with
  :mod:`gps_analysis.transient` / ``UniVarMatrix.m``); data gaps are
  treated as if absent — same fidelity flag as the transient module.
  Bos et al. 2013 (J. Geodesy 87) treat missing data properly; out of
  scope here.
- ML, not REML: ŝ² = RSS/n biases σ estimates low by O(P/n) — negligible
  for the multi-hundred-epoch windows this is built for; documented, not
  corrected.
- Pure leaf: numpy/scipy only, no I/O, float64 throughout, inputs never
  mutated.

References
----------
- Zhang et al. 1997, JGR 102(B8), 18035–18055 (first white+flicker MLE
  for GPS coordinate series).
- Williams 2003, *The effect of coloured noise on the uncertainties of
  rates estimated from geodetic time series*, J. Geodesy 76, 483–494,
  doi:10.1007/s00190-002-0283-4 (covariance eq. 4; rate-uncertainty
  eqs. 23–30; amplitude normalization).
- Williams et al. 2004, JGR 109, B03412, doi:10.1029/2003JB002741
  (global MLE noise analysis; typical flicker-driven σ_v inflation).
- Langbein 2004, JGR 109, B04406, doi:10.1029/2003JB002819 (noise-model
  choice and rate-error consequences).
- Williams 2008, *CATS: GPS coordinate time series analysis software*,
  GPS Solut. 12, 147–153 (the (s, φ) parameterization).
- Bos et al. 2013, *Fast error analysis of continuous GNSS observations
  (Hector)*, J. Geodesy 87, 351–360, doi:10.1007/s00190-012-0605-0
  (fast MLE; the performance reference for the O(n²) path).
- Kailath & Sayed 1995, SIAM Review 37(3) (generalized Schur algorithm);
  Hosking 1981, Biometrika 68(1) (ψ coefficients, nonstationarity).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .fitting import _wls_solve
from .models import FloatArray
from .transient import _DELTA_T_YR, _powerlaw_psi

__all__ = [
    "NoiseMLEFit",
    "NoiseModel",
    "estimate_noise_mle",
    "powerlaw_rate_sigma",
]

_LOG_2PI = float(np.log(2.0 * np.pi))

#: Default κ search bounds: 0 (white) down to −2.5 — GNSS daily solutions
#: live in the flicker/random-walk band κ ∈ (−2, 0) (Zhang et al. 1997;
#: Williams et al. 2004); the −2.5 headroom keeps random walk (κ = −2)
#: off the boundary so a boundary optimum is a real diagnostic.
_KAPPA_BOUNDS_DEFAULT = (-2.5, 0.0)

#: Hard κ domain: below −3 the rate variance integral diverges and the
#: ψ recursion grows super-linearly (Williams 2003 §3 treats κ > −3).
_KAPPA_DOMAIN = (-3.0, 0.0)

#: Power-law variance fractions sin²φ of the coarse MLE start grid —
#: spread over near-white … near-pure-power-law so the Nelder–Mead
#: polish starts in the basin of the global optimum.
_PHI_GRID_FRACTIONS = (0.02, 0.15, 0.35, 0.60, 0.85, 0.98)

#: κ points of the coarse grid (relative position inside kappa_bounds).
_KAPPA_GRID_REL = (0.08, 0.28, 0.50, 0.72, 0.92)

#: Mixing-angle domain [rad]: φ = 0 pure white, φ = π/2 pure power-law.
_PHI_MAX = math.pi / 2.0


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    """MLE white + power-law noise parameters of one component series.

    The (σ_w, β, κ) triple of ``C = σ_w²·I + β²·ΔT^(−κ/2)·(T Tᵀ)``
    (Williams 2003, J. Geodesy 76, eq. 4) at the likelihood optimum,
    with the attained log-likelihood — the provenance record behind an
    ``method="mle"`` velocity (MATH_STANDARDS §6).

    Attributes:
        sigma_white: White-noise amplitude σ̂_w [L].
        amplitude_powerlaw: Power-law amplitude β̂ [L·yr^(−κ/4)]
            (Williams 2003 normalization, ΔT = 1/365 yr daily sampling).
        spectral_index: Spectral index κ̂ (0 white, −1 flicker, −2 random
            walk) [dimensionless]. A value at a ``kappa_bounds`` edge
            means the optimum saturated the search bound — widen it.
        log_likelihood: Attained Gaussian log-likelihood ln L̂ (all
            constants included) [dimensionless].
        n_obs: Number of observations n the model was fitted to.

    Numerical notes:
        Frozen value object; no arrays. σ_w and β are reported ≥ 0
        (s ≥ 0, φ ∈ [0, π/2] by construction).
    """

    sigma_white: float
    amplitude_powerlaw: float
    spectral_index: float
    log_likelihood: float
    n_obs: int


@dataclasses.dataclass(frozen=True)
class NoiseMLEFit:
    """Joint trajectory + noise MLE result for one component series.

    Result of :func:`estimate_noise_mle`: the GLS trajectory parameters
    at the noise-model optimum with their **colored-noise** covariance
    ``Ĉ_p = ŝ²·(AᵀC₀⁻¹A)⁻¹`` — the honest replacement for the
    white-noise WLS covariance — plus the :class:`NoiseModel` itself.

    Attributes:
        params: Trajectory parameters p̂, shape (P,), float64 — same
            positional order as the design-matrix columns (house model
            convention: ``params[1]`` is the secular rate [L/yr]).
        covariance: Colored-noise parameter covariance Ĉ_p, shape (P, P),
            float64. ``√Ĉ_p[1,1]`` is the honest σ_v.
        noise: The fitted :class:`NoiseModel` (σ̂_w, β̂, κ̂, ln L̂, n).

    Numerical notes:
        Arrays coerced to float64 and shape-validated at construction;
        frozen dataclass, ndarray contents read-only by convention.
    """

    params: FloatArray
    covariance: FloatArray
    noise: NoiseModel

    def __post_init__(self) -> None:
        params = np.asarray(self.params, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D, got shape {params.shape}")
        if covariance.shape != (params.size, params.size):
            raise ValueError(
                f"covariance shape {covariance.shape} does not match "
                f"{params.size} parameters"
            )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "covariance", covariance)


def _schur_whiten(
    mat: NDArray[np.float64], wn_amp: float, kappa: float, pln_amp_scaled: float
) -> tuple[float, NDArray[np.float64]]:
    """Whiten a matrix by the exact Cholesky factor of C, in O(n²·m).

    Computes ``(ln det C, L⁻¹M)`` for the white + power-law covariance

        ``C = σ_w²·I + β̃²·(T Tᵀ) = L·Lᵀ``

    (T the unit lower-triangular Toeplitz matrix of ψ =
    :func:`gps_analysis.transient._powerlaw_psi`; β̃ the **already
    ΔT-scaled** power-law amplitude) via the same generalized-Schur
    recursion as :func:`gps_analysis.transient._schur_logdet_quad`
    (displacement rank 2, generators ``[σ_w·e₀, β̃·ψ]``, orthogonal
    Givens rotations only — task H3), generalized from one
    forward-substituted vector to an (n, m) right-hand-side block:

        ``Z[k, :] = (M[k, :] − Σ_{j<k} L_{kj}·Z[j, :]) / L_{kk}``,
        ``ln det C = 2·Σ_k ln L_{kk}``.

    Symbols → args:
        - ``M``  → ``mat``: right-hand-side block, shape (n, m) — e.g.
          ``[A | y]`` (design columns and observations) [column units]
        - ``σ_w`` → ``wn_amp``: white-noise amplitude [L], ≥ 0
        - ``κ``  → ``kappa``: power-law spectral index [dimensionless]
        - ``β̃`` → ``pln_amp_scaled``: power-law amplitude *including*
          the sampling-interval factor, ``β̃ = β·ΔT^(−κ/4)`` [L], ≥ 0.
          (:func:`~gps_analysis.transient._schur_logdet_quad` applies the
          ΔT factor internally; here the caller does, so non-daily
          sampling intervals are expressible.)

    Returns:
        ``(ln det C, L⁻¹M)`` — float and (n, m) float64 array (new).

    Raises:
        ValueError: If ``mat`` is not 2-D or is empty, or both
            amplitudes are zero (C singular).
        numpy.linalg.LinAlgError: If a Schur pivot is not positive
            (cannot happen for ``wn_amp > 0``; for ``wn_amp = 0`` the
            power-law part alone is positive definite since T is unit
            lower triangular, so ``pln_amp_scaled > 0`` also suffices).

    Reference:
        Kailath, Kung & Morf 1979, J. Math. Anal. Appl. 68 (displacement
        rank); Kailath & Sayed 1995, SIAM Review 37(3), §1–4 (generalized
        Schur algorithm); Chandrasekaran & Sayed 1996, SIAM J. Matrix
        Anal. Appl. 17(4) (backward stability, positive-definite case);
        Williams 2003, J. Geodesy 76, eq. 4 (the covariance).

    Numerical notes:
        Identical generator bookkeeping to
        :func:`~gps_analysis.transient._schur_logdet_quad` (u shortened
        from the end, v advanced from the front, rotation skipped when
        ``v[k] = 0``); parity with it on ``m = 1`` is test-pinned at
        rtol 1e-11 (both are exact factorizations of the same matrix,
        differing only in rounding order). The forward substitution
        updates the whole trailing block with one rank-1 ``outer`` per
        step — O(n²·m) flops, O(n·m) memory, no C or L ever stored.
        Inputs are not mutated (``mat`` copied once).
    """
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"mat must be 2-D, got shape {m.shape}")
    n = m.shape[0]
    if n == 0 or m.shape[1] == 0:
        raise ValueError(f"mat must be non-empty, got shape {m.shape}")
    if wn_amp == 0.0 and pln_amp_scaled == 0.0:
        raise ValueError("wn_amp and pln_amp_scaled cannot both be zero")
    # Generators: u = σ_w·e0 (column 1), v = β̃·ψ (column 2).
    v = pln_amp_scaled * _powerlaw_psi(n, kappa)
    u = np.zeros(n, dtype=np.float64)
    u[0] = wn_amp
    w = m.copy()  # M → L⁻¹M, in place row by row
    log_diag_sum = 0.0
    for k in range(n):
        rows = n - k
        b = float(v[k])
        if b != 0.0:
            a = float(u[0])
            h = math.hypot(a, b)
            c, s = a / h, b / h
            u_act = u[:rows]
            v_act = v[k:]
            rotated = c * u_act + s * v_act
            v[k:] = c * v_act - s * u_act
            u[:rows] = rotated
        d = float(u[0])
        if d <= 0.0:
            raise np.linalg.LinAlgError(
                f"covariance not positive definite: Schur pivot {d} at step {k}"
            )
        log_diag_sum += math.log(d)
        z_row = w[k] / d
        w[k] = z_row
        if rows > 1:
            # w[k+1:, :] -= L[k+1:, k] ⊗ z_row
            w[k + 1 :] -= np.outer(u[1:rows], z_row)
    return 2.0 * log_diag_sum, w


@dataclasses.dataclass(frozen=True)
class _ProfileFit:
    """Concentrated-likelihood evaluation at one (φ, κ) point.

    Attributes:
        neg2_loglike: ``−2 ln L(φ, κ)`` after profiling out p and s²
            (all constants included, so values are comparable across
            (φ, κ) and convertible to ln L).
        params: GLS trajectory parameters p̂(φ, κ), shape (P,).
        cov_unit: Unit-scale parameter covariance ``(AᵀC₀⁻¹A)⁻¹``,
            shape (P, P) — multiply by ŝ² for the physical covariance.
        scale_sq: Profiled ML variance scale ŝ² = RSS/n [L²].
    """

    neg2_loglike: float
    params: FloatArray
    cov_unit: FloatArray
    scale_sq: float


def _profile_fit(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    phi: float,
    kappa: float,
) -> _ProfileFit:
    """Evaluate the (p, s²)-profiled negative log-likelihood at (φ, κ).

    Equation (profile/concentrated likelihood of the linear-Gaussian
    model ``y = A·p + ε``, ``ε ~ N(0, s²·C₀(φ, κ))``):

        ``p̂ = argmin_p ‖L₀⁻¹(y − A·p)‖₂²``  (GLS via whitening),
        ``ŝ² = RSS/n``,  ``RSS = ‖L₀⁻¹(y − A·p̂)‖₂²``,
        ``−2 ln L(φ, κ) = n·ln ŝ² + ln det C₀ + n·(1 + ln 2π)``

    with ``C₀ = cos²φ·I + sin²φ·ΔT^(−κ/2)·(T Tᵀ) = L₀L₀ᵀ`` — both inner
    maximizations are exact, so the outer search is 2-D only.

    Symbols → args:
        - ``A`` → ``design``: trajectory design matrix, shape (n, P)
          [basis units]; must have full column rank
        - ``y`` → ``y``: observations, shape (n,) [L]
        - ``φ`` → ``phi``: noise mixing angle [rad], 0 = white,
          π/2 = pure power-law
        - ``κ`` → ``kappa``: power-law spectral index [dimensionless]

    Returns:
        :class:`_ProfileFit` — the objective value with the p̂,
        ``(AᵀC₀⁻¹A)⁻¹`` and ŝ² byproducts the optimum needs.

    Raises:
        ValueError: If the whitened design is rank deficient (covariance
            not estimable) or the residuals are identically zero (ŝ² = 0
            — a noise-free series has no likelihood optimum).

    Reference:
        Profile likelihood of the linear-Gaussian model: Williams 2008,
        GPS Solut. 12 (CATS, the (s, φ) split); Bos et al. 2013,
        J. Geodesy 87, §2 (same concentration in Hector). Whitening:
        :func:`_schur_whiten`; GLS solve:
        :func:`gps_analysis.fitting._wls_solve`.

    Numerical notes:
        A and y are whitened together in ONE Schur pass over ``[A | y]``
        (the factorization cost is shared); the GLS solve reuses the
        SVD-based :func:`~gps_analysis.fitting._wls_solve`
        (``absolute_sigma=True`` ⇒ covariance exactly ``(ÃᵀÃ)⁻¹``,
        no χ² rescale — the scale is profiled analytically instead).
        ``cos φ, sin φ ≥ 0`` on the domain, matching the amplitude-sign
        convention. All terms of −2 ln L are kept so likelihoods are
        absolute, not merely comparable.
    """
    n = y.size
    beta_scaled = math.sin(phi) * float(_DELTA_T_YR ** (-kappa / 4.0))
    logdet0, whitened = _schur_whiten(
        np.column_stack((design, y)), math.cos(phi), kappa, beta_scaled
    )
    a_w, y_w = whitened[:, :-1], whitened[:, -1]
    params, cov_unit = _wls_solve(a_w, y_w, None, absolute_sigma=True)
    if not bool(np.all(np.isfinite(cov_unit))):
        raise ValueError(
            "design matrix is rank deficient under the noise model - "
            "colored-noise MLE requires a full-rank trajectory design"
        )
    resid = y_w - a_w @ params
    rss = float(resid @ resid)
    if rss <= 0.0:
        raise ValueError(
            "residual sum of squares is zero - a noise-free series has "
            "no maximum-likelihood noise model"
        )
    scale_sq = rss / n
    neg2 = n * math.log(scale_sq) + logdet0 + n * (1.0 + _LOG_2PI)
    return _ProfileFit(
        neg2_loglike=neg2, params=params, cov_unit=cov_unit, scale_sq=scale_sq
    )


def estimate_noise_mle(
    design: ArrayLike,
    y: ArrayLike,
    *,
    kappa_bounds: tuple[float, float] = _KAPPA_BOUNDS_DEFAULT,
) -> NoiseMLEFit:
    """Jointly estimate trajectory parameters and white+power-law noise by MLE.

    Equation (the full estimator; see the module derivation chain):

        ``(φ̂, κ̂) = argmax_{φ,κ} ln L(φ, κ)``  (profile likelihood,
        :func:`_profile_fit`), then
        ``σ̂_w = ŝ·cos φ̂``, ``β̂ = ŝ·sin φ̂``, ``ŝ = √(RSS/n)``,
        ``p̂ = p̂(φ̂, κ̂)``, ``Ĉ_p = ŝ²·(AᵀC₀(φ̂, κ̂)⁻¹A)⁻¹``

    — the maximum-likelihood joint estimate of (intercept, rate,
    seasonal, …) and (σ_w, β, κ) for a linear-in-parameters trajectory
    under the Williams 2003 covariance. ``√Ĉ_p[1,1]`` is the honest,
    colored-noise-inflated σ_v (typically several × the white-noise WLS
    formal error for flicker-dominated GNSS series — Zhang et al. 1997
    §5; Williams et al. 2004 §4).

    Symbols → args:
        - ``A`` → ``design``: design matrix, shape (n, P) [basis units];
          rows must be **time-ordered, uniformly (daily) sampled** epochs
          — the covariance lag is the row index (module caveat)
        - ``y`` → ``design`` companion → ``y``: observations, shape (n,)
          [L], finite
        - ``κ`` search range → ``kappa_bounds``: (κ_min, κ_max) within
          [−3, 0] [dimensionless]

    Returns:
        :class:`NoiseMLEFit` — p̂ with colored-noise covariance Ĉ_p and
        the :class:`NoiseModel` (σ̂_w [L], β̂ [L·yr^(−κ/4)], κ̂, ln L̂, n).

    Raises:
        ValueError: On shape/finiteness violations, n ≤ P + 2 (no room
            for the two noise parameters and the scale), invalid
            ``kappa_bounds``, a rank-deficient design, or identically
            zero residuals.

    Reference:
        Zhang et al. 1997, JGR 102(B8) (white+flicker GPS MLE);
        Williams 2003, J. Geodesy 76 (covariance + rate errors);
        Williams et al. 2004, JGR 109, B03412 (MLE practice, inflation);
        Williams 2008, GPS Solut. 12 (CATS (s, φ) parameterization);
        Bos et al. 2013, J. Geodesy 87 (fast MLE reference);
        Langbein 2004, JGR 109, B04406 (model-choice consequences).

    Numerical notes:
        The concentrated surface is smooth in (φ, κ) but can be flat
        along κ when sin φ → 0 (no power-law noise ⇒ κ unidentified) and
        shows the well-known κ–amplitude correlation; the ΔT^(−κ/4)
        normalization (Williams 2003) reduces the latter. A 6×5 coarse
        grid over (sin²φ, κ) followed by a bounded Nelder–Mead polish
        (derivative-free — the objective is exact but its gradient is
        not available analytically) makes the search robust to that
        geometry; each evaluation is one exact O(n²·P) Schur pass. The
        best of grid and polish is returned, so a failed polish can
        never worsen the estimate. κ̂ landing on a ``kappa_bounds`` edge
        is reported as-is — widen the bounds rather than trusting it.
        Bias note: ML (not REML) scale ⇒ amplitudes low by O(P/n).
    """
    a = np.asarray(design, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {a.shape}")
    if yy.ndim != 1 or yy.size != a.shape[0]:
        raise ValueError(
            f"y must be 1-D with design.shape[0] = {a.shape[0]} samples, "
            f"got shape {yy.shape}"
        )
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(yy))):
        raise ValueError("design and y must be finite")
    n, p = a.shape
    if n < p + 3:
        raise ValueError(
            f"need at least P + 3 = {p + 3} samples to estimate {p} trajectory "
            f"parameters plus (kappa, phi, scale), got {n}"
        )
    k_lo, k_hi = float(kappa_bounds[0]), float(kappa_bounds[1])
    if not (_KAPPA_DOMAIN[0] <= k_lo < k_hi <= _KAPPA_DOMAIN[1]):
        raise ValueError(
            f"kappa_bounds must satisfy {_KAPPA_DOMAIN[0]} <= lo < hi <= "
            f"{_KAPPA_DOMAIN[1]}, got {kappa_bounds}"
        )

    def objective(x: NDArray[np.float64]) -> float:
        phi = min(max(float(x[0]), 0.0), _PHI_MAX)
        kap = min(max(float(x[1]), k_lo), k_hi)
        return _profile_fit(a, yy, phi, kap).neg2_loglike

    best_x = np.array([0.0, 0.5 * (k_lo + k_hi)], dtype=np.float64)
    best_val = math.inf
    for frac in _PHI_GRID_FRACTIONS:
        phi = math.asin(math.sqrt(frac))
        for rel in _KAPPA_GRID_REL:
            kap = k_lo + rel * (k_hi - k_lo)
            val = objective(np.array([phi, kap], dtype=np.float64))
            if val < best_val:
                best_val = val
                best_x = np.array([phi, kap], dtype=np.float64)

    result = optimize.minimize(
        objective,
        best_x,
        method="Nelder-Mead",
        bounds=[(0.0, _PHI_MAX), (k_lo, k_hi)],
        options={"xatol": 1e-4, "fatol": 1e-8},
    )
    if float(result.fun) < best_val:
        best_x = np.asarray(result.x, dtype=np.float64)

    phi_hat = min(max(float(best_x[0]), 0.0), _PHI_MAX)
    kappa_hat = min(max(float(best_x[1]), k_lo), k_hi)
    fit = _profile_fit(a, yy, phi_hat, kappa_hat)
    s_hat = math.sqrt(fit.scale_sq)
    noise = NoiseModel(
        sigma_white=s_hat * math.cos(phi_hat),
        amplitude_powerlaw=s_hat * math.sin(phi_hat),
        spectral_index=kappa_hat,
        log_likelihood=-0.5 * fit.neg2_loglike,
        n_obs=n,
    )
    covariance = np.asarray(fit.scale_sq * fit.cov_unit, dtype=np.float64)
    return NoiseMLEFit(params=fit.params, covariance=covariance, noise=noise)


def powerlaw_rate_sigma(
    sigma_white: float,
    amplitude_powerlaw: float,
    spectral_index: float,
    n_epochs: int,
    *,
    dt_years: float = _DELTA_T_YR,
) -> float:
    """Exact GLS rate uncertainty σ_v of a linear fit under colored noise.

    Equation (finite-n generalized-least-squares rate variance):

        ``σ_v² = [(AᵀC⁻¹A)⁻¹]₁₁``,  ``A = [1  t]``,
        ``tᵢ = ΔT·(i − (n−1)/2)``,  i = 0 … n−1,
        ``C = σ_w²·I + β²·ΔT^(−κ/2)·(T Tᵀ)``

    — the quantity whose large-n behaviour Williams 2003 (J. Geodesy 76,
    eqs. 23–30) derives for power-law noise: over a span T = (n−1)·ΔT at
    fixed ΔT, ``σ_v² ∝ T⁻³`` for white noise (κ = 0; the exact finite-n
    form is ``σ_v² = 12σ_w²/(ΔT²·n·(n²−1))``), ``∝ T⁻²`` for flicker
    (κ = −1) and ``∝ T⁻¹`` for random walk (κ = −2) — i.e.
    ``σ_v² ∝ T^(−3−κ)``. This function evaluates the exact finite-n
    variance rather than those asymptotics (they are its validated
    limits — test-pinned), so short windows are handled honestly.

    Symbols → args:
        - ``σ_w`` → ``sigma_white``: white-noise amplitude [L], ≥ 0
        - ``β``  → ``amplitude_powerlaw``: power-law amplitude
          [L·yr^(−κ/4)] (Williams 2003 normalization), ≥ 0; not both 0
        - ``κ``  → ``spectral_index``: spectral index ∈ [−3, 0]
          [dimensionless]
        - ``n``  → ``n_epochs``: number of uniformly spaced epochs, ≥ 3
        - ``ΔT`` → ``dt_years``: sampling interval [yr], > 0 (default
          1/365 — the daily convention shared with
          :mod:`gps_analysis.transient`)

    Returns:
        σ_v [L/yr] — the 1-σ rate uncertainty of a GLS straight-line fit
        under the specified noise (float, > 0).

    Raises:
        ValueError: On negative amplitudes, both amplitudes zero, κ
            outside [−3, 0], ``n_epochs < 3`` or ``dt_years ≤ 0``.

    Reference:
        Williams 2003, J. Geodesy 76, eqs. 23–30 (power-law rate
        uncertainty and its span scalings); Bos et al. 2013, J. Geodesy
        87 (the same GLS variance inside Hector). Assumes a pure
        two-parameter linear design — co-estimated seasonal terms
        (Blewitt & Lavallée 2002) inflate σ_v further on sub-annual to
        few-year windows.

    Numerical notes:
        Computed by whitening the centered design with the exact O(n²)
        Schur pass (:func:`_schur_whiten`) and reading the rate variance
        off the SVD covariance of :func:`gps_analysis.fitting._wls_solve`
        (``(ÃᵀÃ)⁻¹``; called with a zero observation vector — only the
        covariance is used). Centering t makes the two whitened columns
        far from collinear; cross-checked against the dense
        ``(Aᵀ C⁻¹ A)⁻¹`` built from
        :func:`gps_analysis.transient.noise_covariance` (test-pinned,
        rtol 1e-8).
    """
    if sigma_white < 0.0 or amplitude_powerlaw < 0.0:
        raise ValueError(
            f"amplitudes must be >= 0, got sigma_white={sigma_white}, "
            f"amplitude_powerlaw={amplitude_powerlaw}"
        )
    if sigma_white == 0.0 and amplitude_powerlaw == 0.0:
        raise ValueError("sigma_white and amplitude_powerlaw cannot both be zero")
    if not _KAPPA_DOMAIN[0] <= spectral_index <= _KAPPA_DOMAIN[1]:
        raise ValueError(
            f"spectral_index must be in [{_KAPPA_DOMAIN[0]}, {_KAPPA_DOMAIN[1]}], "
            f"got {spectral_index}"
        )
    if n_epochs < 3:
        raise ValueError(f"n_epochs must be >= 3, got {n_epochs}")
    if dt_years <= 0.0:
        raise ValueError(f"dt_years must be > 0, got {dt_years}")
    t = dt_years * (np.arange(n_epochs, dtype=np.float64) - (n_epochs - 1) / 2.0)
    a = np.column_stack((np.ones_like(t), t))
    beta_scaled = amplitude_powerlaw * float(dt_years ** (-spectral_index / 4.0))
    _, a_w = _schur_whiten(a, sigma_white, spectral_index, beta_scaled)
    _, cov = _wls_solve(
        a_w, np.zeros(n_epochs, dtype=np.float64), None, absolute_sigma=True
    )
    return float(math.sqrt(cov[1, 1]))
