"""Term-agnostic variable projection (VARPRO) for separable nonlinear least squares.

What this estimates
-------------------
A model that is linear in P amplitudes ``c`` and nonlinear in ONE scalar
parameter θ,

    ``y = Φ(θ)·c + ε``,  ``ε ~ N(0, diag(σᵢ²))``,

with the caller supplying Φ(θ) as a callable that builds the full (N, P)
design. In the intended transient application θ is ``ln τ`` (the log
e-folding time), but nothing here knows that: the module is deliberately
**term-agnostic**, so the transient terms (blocked on the Bevis & Brown
2014 amplitude convention) can adopt it later without this solver
changing. VARPRO **is** the concentration of the linear parameters, not a
special optimizer (Golub & Pereyra 1973).

Derivation chain (MATH_STANDARDS §2, module contract)
-----------------------------------------------------
1. **Concentration** — for fixed θ the amplitudes have the exact WLS
   optimum ``ĉ(θ) = Φ_w(θ)⁺·y_w`` (σ-whitened pseudo-inverse), so the
   profile objective

       ``χ²(θ) = ‖(I − U·Uᵀ)·y_w‖₂²``,  ``Φ_w(θ) = U·S·Vᵀ``

   depends on θ alone (Golub & Pereyra 1973; O'Leary & Rust 2013, §2).
   The rank-revealing SVD is the same factorization
   :func:`gps_analysis.fitting._wls_solve` documents (identical cutoff);
   at the optimum it is computed **once** and reused by the amplitude
   solve (:func:`_svd_solve`), the Jacobian (:func:`_varpro_jacobian`)
   and the bordered covariance (:func:`_bordered_covariance`) — never
   recomputed per consumer.
2. **Search** (:func:`estimate_varpro`) — coarse grid over
   ``theta_bounds`` followed by a bounded derivative-free polish, keeping
   the best of both, mirroring the structure of
   :func:`gps_analysis.noise.estimate_noise_mle` (grid + polish +
   best-of-both + edge clamping). The objective is smooth and 1-D, so a
   hand-rolled Gauss–Newton gains nothing; being 1-D, scipy's bounded
   Brent search stands in for the Nelder–Mead that the 2-D noise search
   uses. A failed polish can never worsen the grid best (the
   ``noise.py`` guarantee, reproduced verbatim: the polish result is
   taken only if its objective value is strictly lower).
3. **Jacobian** — O'Leary & Rust 2013, eq. (8), p. 585 (verified
   against the primary source): ``J = −(A + B)`` with columns

       ``a = P·D·c = D·c − U·(Uᵀ·(D·c))``,  ``P = I − U·Uᵀ``
       ``b = (Φ_w†)ᵀ·Dᵀ·r_w = U·S⁻¹·Vᵀ·(Dᵀ·r_w)``

   and ``D = ∂Φ_w/∂θ``. ``B`` is the term Kaufman 1975 drops; it is
   kept here. Dropping it leaves the gradient ``rᵀJ`` unchanged
   (``Uᵀr_w = 0`` exactly), which is why the approximation is tempting,
   but O'Leary & Rust §3.1 measure iterations rising 3→7 (function
   evaluations 8→12, three stalls instead of one) with ``‖B‖/‖J‖``
   never above 2 % — and the term costs one back-substitution. The
   paper's grouping is deliberate — columns from matrix-vector products
   only, never matrix-matrix (p. 585) — and is preserved verbatim here.
   Computed once, at θ̂, for the covariance and diagnostics (the search
   itself is derivative-free).
4. **Covariance** — from the **bordered** matrix ``H = [Φ_w, J]``
   (O'Leary & Rust 2013, §2.5, p. 587: ``H = W[Φ, J]``, linear
   parameters first, nonlinear last): ``C = ŝ²·(HᵀH)⁻¹`` — never
   ``ŝ²/(JᵀJ)``, which conditions on the amplitudes and understates
   σ_θ (Schur complement: ``σ_θ,bordered² = ŝ²/‖(I−UUᵀ)·J‖₂²
   ≥ ŝ²/‖J‖₂²``, strict whenever the B term is nonzero, since
   ``(I−UUᵀ)·J = −A`` and ``U·Uᵀ·J = −B``).
5. **Interval** — the Δχ²=1 profile-likelihood interval in θ
   (Venzon & Moolgavkar 1988), root-found on the concentrated χ²(θ).
   With θ = ln τ the interval is near-symmetric essentially always in
   the well-identified regime (which independently ratifies searching in
   ln τ); it degenerates to a one-sided bound when identification is
   marginal. A side that fails to close inside ``theta_bounds`` is
   reported as open (``interval_open_*``) with a :class:`UserWarning` —
   the ``noise.py`` κ-edge convention (report, don't hide).

Design contract (why one callable, not blocks + mask)
-----------------------------------------------------
The caller passes a single ``design(theta) -> (N, P)`` building the FULL
design, plus optionally ``d_design(theta) -> (N, P)`` = ∂Φ/∂θ. A
fixed/nonlinear block split (or a column mask) was rejected: eq. (8)
consumes the full ``D`` anyway — θ-independent columns simply carry zero
derivative columns — so a split contract forces every caller to
re-implement column concatenation and bookkeeping, and makes the fixed
block's zero derivative an invariant this module must trust rather than
one it can see. With one callable, column semantics live entirely in the
caller's closure and the module stays term-agnostic. When ``d_design``
is absent, ``D`` is a documented central finite difference of ``design``
(:func:`_finite_difference_columns`); fixed columns difference to
exactly zero, so correctness is unaffected.

Identification (operator guidance, surfaced as diagnostics)
-----------------------------------------------------------
For a transient with e-folding time τ observed over a post-onset span
``T_post``: τ fixed needs ``T_post ≥ τ``; τ profiled needs
``T_post ≳ 5τ`` **and** amplitude SNR ``b/σ_b ≳ 5`` for a bounded
two-sided interval. At ``T_post ≈ 3τ`` the profile interval spans
decades or is one-sided regardless of SNR — publish a τ **bound**, not
an estimate, and carry that caveat into the amplitude. This module
cannot evaluate ``T_post/τ`` (it is term-agnostic); what it surfaces is
the *symptom*: a one-sided or bound-hitting profile interval
(``interval_open_lower`` / ``interval_open_upper`` + warning). The
term-aware caller owns the span/SNR checks.

Conventions and caveats (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------------------
- The covariance is exact under **diagonal** Σ with ``W = Σ⁻¹``; under
  temporally correlated (colored) noise the usual Williams 2003 caveat
  applies — the same caveat class as every WLS covariance in this
  package. Notation map: O'Leary & Rust's ``W`` is the σ-whitening
  ``diag(1/σᵢ)`` — this module's ``W^{1/2}`` — so their ``Φ_w = WΦ``
  and ``r_w = W(y − Φc)`` (p. 585) are exactly the whitened design and
  residual here.
- No centering or reparameterization happens here: the returned
  amplitudes mean exactly what the caller's Φ columns define (the
  package invariant that parameter vectors crossing a boundary are in
  the absolute-t parameterization; conditioning/centering lives only
  inside :func:`gps_analysis.fitting._fit_linear_design`, and callers
  must pre-condition polynomial-in-t columns exactly as for
  :func:`gps_analysis.fitting._wls_solve`).
- Precedent: :func:`gps_analysis.transient._preliminary_start` is the
  same grid-over-a-nonlinear-parameter pattern hand-written for break
  epochs; its MATLAB bit-parity is CI-gated, so it is cited here as
  precedent and deliberately NOT refactored onto this module.
- Pure leaf: numpy/scipy only, float64 throughout, inputs never mutated.

References
----------
- Golub & Pereyra 1973, *The differentiation of pseudo-inverses and
  nonlinear least squares problems whose variables separate*, SIAM J.
  Numer. Anal. 10(2), 413–432. (the variable-projection method.)
- O'Leary & Rust 2013, *Variable projection for nonlinear least squares
  problems*, Comput. Optim. Appl. 54(3), 579–593. (eq. (8) Jacobian,
  p. 585; §2.5 bordered-matrix covariance and residual-mean-square dof
  convention, pp. 586–587; §3.1 Kaufman-approximation cost, pp. 588–589
  — all verified against the primary source.)
- Kaufman 1975, BIT 15(1), 49–57. (the dropped-B-term approximation —
  cited as what this module deliberately does NOT do.)
- Venzon & Moolgavkar 1988, J. R. Stat. Soc. C 37(1), 87–94. (the
  Δχ²=1 profile-likelihood interval.)
- Williams 2003, J. Geodesy 76. (formal-σ caveat under colored noise.)
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from .models import FloatArray

__all__ = ["VarproFit", "estimate_varpro"]


_N_GRID_DEFAULT = 33
"""Coarse-grid density over ``theta_bounds`` (uniform, endpoints included).

33 points over a typical ±3 ln-τ search box is a spacing of ~0.19 in
ln τ — comfortably inside the basin of any profile minimum whose Δχ²=1
interval is worth reporting (such minima are O(1) wide in ln τ; see the
identification note in the module docstring)."""

_POLISH_XATOL = 1e-10
"""Absolute θ tolerance of the bounded polish — tight enough that the
noise-free exact-recovery test (~1e-10 in θ) is limited by the objective's
conditioning, not by the search."""

_FD_RELATIVE_STEP = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)
"""Central-difference step factor ``ε^{1/3} ≈ 6.1e-6`` — the standard
truncation/roundoff balance for second-order central differences
(Nocedal & Wright 2006, *Numerical Optimization* 2nd ed., §8.1)."""


@dataclasses.dataclass(frozen=True)
class _DesignSVD:
    """Rank-truncated SVD of one σ-whitened design, with the whitened data.

    Attributes:
        u: Left singular vectors of ``Φ_w``, shape (N, r) — kept columns
            only (singular values above the cutoff).
        s: Kept singular values, shape (r,), descending.
        vt: Right singular vectors (transposed), shape (r, P).
        y_w: σ-whitened observations ``W^{1/2}·y``, shape (N,)
            [dimensionless when σ is supplied, else [L]].
        n_params: P — the full column count of Φ (rank r ≤ P).
        rank: r — numerical rank under the shared cutoff.
    """

    u: FloatArray
    s: FloatArray
    vt: FloatArray
    y_w: FloatArray
    n_params: int
    rank: int


def _whiten(
    a: FloatArray, y: FloatArray, sigma: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """Apply the σ-whitening ``W^{1/2}`` to a design and its observations.

    Equation:
        ``A_w = W^{1/2}·A``, ``y_w = W^{1/2}·y``, ``W = diag(1/σᵢ²)``
        (identity when ``sigma`` is None).

    Symbols → args:
        - ``A``  → ``a``: design matrix, shape (N, P) [basis units]
        - ``yᵢ`` → ``y``: observations, shape (N,) [L]
        - ``σᵢ`` → ``sigma``: 1-σ observation uncertainties, shape (N,)
          [L]; ``None`` ⇒ unit weights

    Returns:
        ``(A_w, y_w)`` — float64; new arrays (inputs never mutated).

    Reference:
        Aitken WLS whitening, exactly as in
        :func:`gps_analysis.fitting._wls_solve` (Aitken 1936; Strang &
        Borre 1997, ch. 9).

    Numerical notes:
        Row-wise division only — no matrix products, so no conditioning
        change beyond the weights themselves.
    """
    if sigma is None:
        return np.array(a, dtype=np.float64), np.array(y, dtype=np.float64)
    return (
        np.asarray(a / sigma[:, np.newaxis], dtype=np.float64),
        np.asarray(y / sigma, dtype=np.float64),
    )


def _design_svd(a_w: FloatArray, y_w: FloatArray) -> _DesignSVD:
    """Compute the rank-revealing SVD of one whitened design.

    Equation:
        ``A_w = U·S·Vᵀ`` (thin SVD), truncated to the numerical rank
        ``r = #{sᵢ > ε·max(N, P)·s_max}``.

    Symbols → args:
        - ``A_w`` → ``a_w``: σ-whitened design, shape (N, P)
        - ``y_w`` → ``y_w``: σ-whitened observations, shape (N,)
          (carried along so downstream consumers share one container)

    Returns:
        :class:`_DesignSVD` with ``u`` (N, r), ``s`` (r,), ``vt`` (r, P).

    Reference:
        The factorization and cutoff of
        :func:`gps_analysis.fitting._wls_solve` (the ``numpy.linalg.lstsq``
        default cutoff), exposed so the amplitude solve, the O'Leary &
        Rust 2013 eq. (8) Jacobian and the §2.5 bordered covariance reuse
        ONE factorization (module derivation chain, step 1).

    Numerical notes:
        SVD rather than QR/normal equations: rank-safe and backward
        stable; forming ``A_wᵀA_w`` would square the condition number.
        Cutoff identical to ``_wls_solve`` so both agree on rank —
        test-pinned parity (``tests/test_varpro.py``).
    """
    n, p = a_w.shape
    u, s, vt = np.linalg.svd(a_w, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(n, p) * (float(s[0]) if s.size else 0.0)
    kept = s > cutoff
    rank = int(np.count_nonzero(kept))
    return _DesignSVD(
        u=np.asarray(u[:, kept], dtype=np.float64),
        s=np.asarray(s[kept], dtype=np.float64),
        vt=np.asarray(vt[kept, :], dtype=np.float64),
        y_w=y_w,
        n_params=p,
        rank=rank,
    )


def _svd_solve(svd: _DesignSVD) -> FloatArray:
    """Solve for the concentrated linear amplitudes ĉ(θ).

    Equation:
        ``ĉ = V·S⁻¹·Uᵀ·y_w``  (minimum-norm WLS solution)

    Symbols → args:
        - ``U, S, V, y_w`` → ``svd``: the shared whitened factorization
          (:class:`_DesignSVD`); amplitudes in the caller's column units
          (typically mm per unit basis function).

    Returns:
        ĉ, shape (P,), float64.

    Reference:
        Golub & Pereyra 1973 (the inner linear solve of the variable
        projection); O'Leary & Rust 2013, p. 585 (``Φ_w† = V·Σ⁻¹·Uᵀ``,
        ``c = Φ_w†·y``; their footnote 6 endorses the minimum-norm
        solution under rank deficiency); identical formula and
        conventions to :func:`gps_analysis.fitting._wls_solve` — parity
        is test-pinned.

    Numerical notes:
        Uses only kept singular values (rank-truncated), so a
        rank-deficient design yields the minimum-norm ĉ; the estimator
        raises on rank deficiency *before* trusting such a solution.
    """
    return np.asarray(
        svd.vt.T @ ((svd.u.T @ svd.y_w) / svd.s),
        dtype=np.float64,
    )


def _projection_residual(svd: _DesignSVD) -> FloatArray:
    """Compute the variable-projection residual r_w(θ).

    Equation:
        ``r_w = (I − U·Uᵀ)·y_w``  — the whitened residual with the
        linear amplitudes at their exact conditional optimum;
        ``χ²(θ) = ‖r_w‖₂²`` is the VARPRO profile objective.

    Symbols → args:
        - ``U, y_w`` → ``svd``: shared whitened factorization
          (:class:`_DesignSVD`) [r_w dimensionless when σ supplied].

    Returns:
        r_w, shape (N,), float64.

    Reference:
        Golub & Pereyra 1973 (the variable projection functional);
        O'Leary & Rust 2013, p. 585 (``P = I − U·Uᵀ``, ``Pᵀ = P``,
        ``r_w = W(y − Φc) = P·y``).

    Numerical notes:
        ``Uᵀ·r_w = 0`` to roundoff by construction — the identity behind
        the Kaufman-gradient equivalence (see :func:`_varpro_jacobian`).
    """
    return np.asarray(svd.y_w - svd.u @ (svd.u.T @ svd.y_w), dtype=np.float64)


def _finite_difference_columns(
    design: Callable[[float], ArrayLike],
    theta: float,
    n_obs: int,
    n_params: int,
) -> FloatArray:
    """Approximate the design derivative D = ∂Φ/∂θ by central differences.

    Equation:
        ``D ≈ [Φ(θ+h) − Φ(θ−h)] / (2h)``,
        ``h = ε^{1/3}·max(1, |θ|)``,  ε = float64 machine epsilon

    Symbols → args:
        - ``Φ`` → ``design``: full-design builder, θ → (N, P)
          [basis units]
        - ``θ`` → ``theta``: nonlinear parameter [caller's units,
          typically ln yr]
        - N, P → ``n_obs``, ``n_params``: expected shape (validated)

    Returns:
        D, shape (N, P), float64 — exactly zero in θ-independent columns.

    Reference:
        Second-order central difference, truncation O(h²), step balance
        h ∝ ε^{1/3}: Nocedal & Wright 2006, *Numerical Optimization*
        (2nd ed.), §8.1. Used only when the caller supplies no analytic
        ``d_design`` (eq. (8) of O'Leary & Rust 2013 needs D either way).

    Numerical notes:
        Error ~ ε^{2/3} ≈ 4e-11 relative for smooth Φ — negligible
        against the ~1e-6 Jacobian test tolerance. Two extra design
        builds per call; the estimator calls this once, at θ̂.
    """
    h = _FD_RELATIVE_STEP * max(1.0, abs(theta))
    hi = _eval_design(design, theta + h, n_obs, n_params)
    lo = _eval_design(design, theta - h, n_obs, n_params)
    return np.asarray((hi - lo) / (2.0 * h), dtype=np.float64)


def _varpro_jacobian(
    svd: _DesignSVD,
    d_w: FloatArray,
    coeffs: FloatArray,
    residual: FloatArray,
    *,
    kaufman: bool = False,
) -> FloatArray:
    """Compute the VARPRO residual Jacobian J = ∂r_w/∂θ (scalar θ).

    Equation (O'Leary & Rust 2013, eq. (8), p. 585 — verified against
    the primary source): ``J = −(A + B)``, single columns here (q = 1)

        ``a = P·D·c = D·c − U·(Uᵀ·(D·c))``,  ``P = I − U·Uᵀ``
        ``b = (P·D·Φ_w†)ᵀ·y = (Φ_w†)ᵀ·Dᵀ·Pᵀ·y = (Φ_w†)ᵀ·Dᵀ·r_w
             = U·S⁻¹·Vᵀ·(Dᵀ·r_w)``   (using ``Pᵀ = P``, ``P·y = r_w``)

    ``A`` is the projected direct term, ``B`` (from the θ-dependence of
    ĉ) the term Kaufman 1975 drops.

    Symbols → args:
        - ``U, S, V`` → ``svd``: shared whitened factorization
          (:class:`_DesignSVD`)
        - ``D`` → ``d_w``: σ-whitened design derivative ``∂Φ_w/∂θ``,
          shape (N, P) [basis units / θ-unit]
        - ``c`` → ``coeffs``: concentrated amplitudes ĉ(θ), shape (P,)
        - ``r_w`` → ``residual``: projection residual, shape (N,)
        - ``kaufman``: drop the B term — **diagnostics/tests only**,
          never used by the estimator

    Returns:
        J, shape (N,), float64 [per θ-unit].

    Reference:
        O'Leary & Rust 2013, Comput. Optim. Appl. 54(3) 579–593,
        eq. (8), p. 585; Kaufman 1975, BIT 15(1) (the approximation
        deliberately NOT used). The gradient ``rᵀJ`` is identical under
        both (``Uᵀr = 0`` exactly — test-pinned to ~4e-13), but O'Leary
        & Rust §3.1 (pp. 588–589) measure iterations rising 3→7 and
        function evaluations 8→12, with three stalls instead of one,
        while ``‖B‖/‖J‖`` never exceeds 2 %.

    Numerical notes:
        Reuses the single SVD of the design — no new factorization; the
        B term costs one (P,) back-substitution. The grouping into
        matrix-vector products only (never matrix-matrix) is the paper's
        stated design of eq. (8) (p. 585) and is preserved verbatim.
        ``S⁻¹`` uses kept singular values only (rank-truncated
        pseudo-inverse).
    """
    dc = d_w @ coeffs
    projected = dc - svd.u @ (svd.u.T @ dc)
    if kaufman:
        return np.asarray(-projected, dtype=np.float64)
    b_term = svd.u @ ((svd.vt @ (d_w.T @ residual)) / svd.s)
    return np.asarray(-(projected + b_term), dtype=np.float64)


def _bordered_covariance(
    a_w: FloatArray,
    jacobian: FloatArray,
    scale_sq: float,
) -> FloatArray:
    """Compute the joint (ĉ, θ̂) covariance from the bordered matrix.

    Equation (O'Leary & Rust 2013, §2.5, p. 587 — verified against the
    primary source):
        ``H = W[Φ, J] = [Φ_w, J]``,  ``C_v = ŝ²·(HᵀH)⁻¹``
    with the linear parameters ordered first and θ LAST, exactly the
    paper's convention — never ``ŝ²/(JᵀJ)`` for σ_θ², which conditions
    on the amplitudes: by the Schur complement
    ``σ_θ² = ŝ²/‖(I−UUᵀ)·J‖₂² ≥ ŝ²/‖J‖₂²``, strictly larger whenever
    the B term is nonzero (``(I−UUᵀ)·J = −A``, ``U·Uᵀ·J = −B``).

    Symbols → args:
        - ``Φ_w`` → ``a_w``: σ-whitened design at θ̂, shape (N, P)
        - ``J`` → ``jacobian``: the eq.-(8) VARPRO Jacobian
          ``∂r_w/∂θ`` (:func:`_varpro_jacobian`, B term included),
          shape (N,) — the paper borders with J itself; all variances
          are invariant to the sign of this column (similarity by
          ``diag(1, …, 1, −1)``), the amplitude–θ cross-covariances
          follow the paper's sign convention
        - ``ŝ²`` → ``scale_sq``: residual variance scale
          [dimensionless when σ supplied]; 1.0 under ``absolute_sigma``

    Returns:
        C, shape (P+1, P+1), float64, symmetrized exactly
        (``C ← (C + Cᵀ)/2``); amplitudes first, θ last.

    Raises:
        ValueError: If H is numerically rank deficient — θ is then not
            identifiable at θ̂ (its derivative column lies in the span
            of the design), and no finite covariance exists.

    Reference:
        O'Leary & Rust 2013, §2.5, p. 587 (``H = W[Φ, J]``,
        ``C_v = σ²(HᵀH)⁻¹``, linear first / nonlinear last); scale
        convention as :func:`gps_analysis.fitting._wls_solve`
        (curve_fit-compatible). Exact under diagonal Σ; Williams 2003
        caveat under colored noise (module docstring).

    Numerical notes:
        SVD of H (not normal equations): ``C = ŝ²·V_h·S_h⁻²·V_hᵀ`` —
        rank-safe, condition number not squared; same cutoff convention
        as :func:`_design_svd`. The paper uses a pivoted-QR R factor
        here; the SVD gives the identical C in exact arithmetic, is at
        least as stable, and matches the package-wide factorization
        convention — the one documented divergence from §2.5.
    """
    h = np.column_stack((a_w, jacobian))
    n, p_full = h.shape
    _, s_h, vt_h = np.linalg.svd(h, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(n, p_full) * float(s_h[0])
    if int(np.count_nonzero(s_h > cutoff)) < p_full:
        raise ValueError(
            "bordered matrix [design, d(model)/d(theta)] is rank deficient - "
            "theta is not identifiable at the optimum (its derivative column "
            "lies in the span of the design columns)"
        )
    cov = np.asarray(scale_sq * ((vt_h.T / s_h**2) @ vt_h), dtype=np.float64)
    return np.asarray(0.5 * (cov + cov.T), dtype=np.float64)


def _profile_interval(
    objective: Callable[[float], float],
    theta_hat: float,
    chisq_min: float,
    delta: float,
    lo: float,
    hi: float,
) -> tuple[float, float, bool, bool]:
    """Compute the Δχ²=1 profile-likelihood interval for θ.

    Equation (Venzon & Moolgavkar 1988):
        ``{θ : χ²(θ) ≤ χ²(θ̂) + Δ}``,  ``Δ = ŝ²·Δχ²``,  ``Δχ² = 1``
    with endpoints root-found on ``g(θ) = χ²(θ) − (χ²(θ̂) + Δ)`` on each
    side of θ̂.

    Symbols → args:
        - ``χ²(θ)`` → ``objective``: concentrated profile objective
          [dimensionless when σ supplied]
        - ``θ̂`` → ``theta_hat``: profile optimum [caller's θ units]
        - ``χ²(θ̂)`` → ``chisq_min``; ``Δ`` → ``delta`` (> 0)
        - search box → ``lo``, ``hi`` (= ``theta_bounds``)

    Returns:
        ``(lower, upper, open_lower, open_upper)`` — a side is **open**
        (endpoint pinned to the bound, flag True) when θ̂ sits on that
        bound or g never crosses zero before it: the data establish only
        a one-sided bound there. Mirrors the
        :func:`gps_analysis.noise.estimate_noise_mle` κ-edge convention
        (report the edge, don't hide it).

    Reference:
        Venzon & Moolgavkar 1988, J. R. Stat. Soc. C 37(1), 87–94.
        The ŝ² plug-in for unknown scale is the same convention as the
        curve_fit-compatible covariance rescale in
        :func:`gps_analysis.fitting._wls_solve`.

    Numerical notes:
        Brent root-finding (``brentq``, xtol 1e-10) — the bracket is
        guaranteed by ``g(θ̂) = −Δ < 0`` against ``g(bound) > 0``.
        Assumes χ² is quasi-convex per side within the box (true near
        any minimum worth reporting); a multi-modal profile would return
        the innermost crossing.
    """

    def g(theta: float) -> float:
        return objective(theta) - (chisq_min + delta)

    if theta_hat <= lo:
        lower, open_lower = lo, True
    elif g(lo) <= 0.0:
        lower, open_lower = lo, True
    else:
        lower = float(optimize.brentq(g, lo, theta_hat, xtol=1e-10))
        open_lower = False

    if theta_hat >= hi:
        upper, open_upper = hi, True
    elif g(hi) <= 0.0:
        upper, open_upper = hi, True
    else:
        upper = float(optimize.brentq(g, theta_hat, hi, xtol=1e-10))
        open_upper = False

    return lower, upper, open_lower, open_upper


def _eval_design(
    design: Callable[[float], ArrayLike],
    theta: float,
    n_obs: int,
    n_params: int | None,
) -> FloatArray:
    """Build and validate one design matrix Φ(θ) from the caller's callable.

    Pure validation shim (no math): coerces to float64 and checks
    2-D shape, row count = ``n_obs``, column count = ``n_params`` when
    known, and finiteness — raising ``ValueError`` naming θ so a bad
    builder fails loudly at the offending evaluation, not downstream.
    """
    a = np.asarray(design(theta), dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != n_obs:
        raise ValueError(
            f"design(theta={theta!r}) must return shape ({n_obs}, P), " f"got {a.shape}"
        )
    if n_params is not None and a.shape[1] != n_params:
        raise ValueError(
            f"design(theta={theta!r}) returned {a.shape[1]} columns, "
            f"expected {n_params} (column count must not depend on theta)"
        )
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"design(theta={theta!r}) returned non-finite values")
    return a


@dataclasses.dataclass(frozen=True)
class VarproFit:
    """Result of one separable (VARPRO) nonlinear least-squares estimate.

    Attributes:
        theta: Profile optimum θ̂ [caller's θ units, typically ln yr for
            θ = ln τ], clamped into ``theta_bounds``.
        theta_sigma: 1-σ uncertainty of θ̂ from the bordered covariance
            (``sqrt(covariance[-1, -1])``) — NOT the JᵀJ value.
        theta_interval: ``(lower, upper)`` Δχ²=1 profile-likelihood
            interval in θ (Venzon & Moolgavkar 1988).
        interval_open_lower: The lower endpoint is pinned to
            ``theta_bounds[0]`` without a Δχ²=1 crossing — only an upper
            bound on θ is established.
        interval_open_upper: Mirror image — only a lower bound on θ is
            established. Either flag ⇒ publish a θ bound, not an
            estimate, and carry the caveat into the amplitudes.
        params: Concentrated linear amplitudes ĉ(θ̂), shape (P,)
            [caller's column units].
        covariance: Joint (ĉ, θ̂) covariance from the bordered matrix,
            shape (P+1, P+1), amplitudes first, **θ last**.
        chisq: Whitened residual sum of squares χ²(θ̂)
            [dimensionless when σ supplied].
        scale_sq: ŝ² applied to the covariance and the interval —
            ``chisq/(N−P−1)`` under ``absolute_sigma=False``, else 1.0.
            The dof convention is O'Leary & Rust 2013 §2.5, p. 586
            (``σ² = ‖r_w‖²/(m−n−q)``, q = 1 nonlinear parameter), which
            coincides with the curve_fit convention of
            :func:`gps_analysis.fitting._wls_solve` once θ is counted.
        n_obs: Number of observations N.
    """

    theta: float
    theta_sigma: float
    theta_interval: tuple[float, float]
    interval_open_lower: bool
    interval_open_upper: bool
    params: FloatArray
    covariance: FloatArray
    chisq: float
    scale_sq: float
    n_obs: int


def estimate_varpro(
    design: Callable[[float], ArrayLike],
    y: ArrayLike,
    *,
    theta_bounds: tuple[float, float],
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    d_design: Callable[[float], ArrayLike] | None = None,
    n_grid: int = _N_GRID_DEFAULT,
) -> VarproFit:
    """Estimate a separable model ``y = Φ(θ)·c + ε`` by variable projection.

    Equation (the full estimator; see the module derivation chain):

        ``θ̂ = argmin_θ χ²(θ)``,  ``χ²(θ) = ‖(I − UUᵀ)·y_w‖₂²``
        (concentrated profile — Golub & Pereyra 1973), then
        ``ĉ = V·S⁻¹·Uᵀ·y_w``, the eq.-(8) Jacobian J (O'Leary & Rust
        2013, p. 585, B term included — no Kaufman 1975),
        ``C = ŝ²·([Φ_w, J]ᵀ[Φ_w, J])⁻¹`` (§2.5, p. 587, bordered
        covariance) and the Δχ²=1 profile interval in θ
        (Venzon & Moolgavkar 1988).

    Symbols → args:
        - ``Φ(θ)`` → ``design``: θ → full design, shape (N, P)
          [basis units]; column count must not depend on θ. Callers
          must pre-condition polynomial-in-t columns (no centering
          happens here — module docstring).
        - ``y`` → ``y``: observations, shape (N,) [L], finite
        - ``θ`` search box → ``theta_bounds``: (lo, hi), lo < hi, finite
          [caller's θ units — pass **ln τ** for transient time scales;
          the profile is then near-symmetric when well identified]
        - ``σᵢ`` → ``sigma``: 1-σ uncertainties, shape (N,) [L], > 0;
          ``None`` ⇒ unit weights
        - ``absolute_sigma`` → ``absolute_sigma``: as in
          :func:`gps_analysis.fitting._wls_solve` — True trusts σ
          (ŝ² = 1); False rescales by ``ŝ² = χ²/(N−P−1)`` (the −1 is θ)
        - ``∂Φ/∂θ`` → ``d_design``: optional analytic derivative
          builder, θ → (N, P); absent ⇒ central finite difference
          (:func:`_finite_difference_columns`)
        - grid density → ``n_grid``: coarse-grid points over the box
          (≥ 2; default ``_N_GRID_DEFAULT``)

    Returns:
        :class:`VarproFit` — θ̂ with bordered σ_θ and profile interval,
        amplitudes ĉ, the joint (P+1, P+1) covariance (θ last), χ², ŝ²
        and N.

    Raises:
        ValueError: On shape/finiteness violations, non-positive σ,
            inverted or non-finite ``theta_bounds``, ``n_grid < 2``,
            N < P + 2 (no room for θ plus one degree of freedom), a
            rank-deficient linear design at θ̂ (amplitudes not
            identifiable — the :func:`gps_analysis.noise` precedent for
            propagating :func:`~gps_analysis.fitting._wls_solve`'s
            failure convention in an estimator), zero residual under
            ``absolute_sigma=False`` (no scale estimable), or an
            unidentifiable θ (:func:`_bordered_covariance`).

    Warns:
        UserWarning: When the profile interval fails to close on a side
            within ``theta_bounds`` — the estimate is then a one-sided
            **bound**; see :class:`VarproFit.interval_open_lower` /
            ``interval_open_upper``.

    Reference:
        Golub & Pereyra 1973, SIAM J. Numer. Anal. 10(2) (variable
        projection); O'Leary & Rust 2013, Comput. Optim. Appl. 54(3),
        eq. (8) p. 585 + §2.5 p. 587; Venzon & Moolgavkar 1988 (profile
        interval);
        Kaufman 1975 (rejected approximation); search structure mirrors
        :func:`gps_analysis.noise.estimate_noise_mle`.

    Numerical notes:
        One SVD per objective evaluation (O(N·P²)); the SVD at θ̂ is
        computed once and shared by solve, Jacobian and covariance. The
        polish (bounded Brent, xatol ``_POLISH_XATOL``) is accepted only
        if it strictly improves on the grid best — a failed polish never
        worsens the estimate (the ``noise.py:534`` guarantee). θ̂ is
        clamped into ``theta_bounds`` (edge-clamping convention of
        :func:`~gps_analysis.noise.estimate_noise_mle`); an edge optimum
        surfaces as an open interval side, not as a hidden failure.
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {yy.shape}")
    if not bool(np.all(np.isfinite(yy))):
        raise ValueError("y must be finite")
    n = int(yy.size)

    sig: FloatArray | None
    if sigma is None:
        sig = None
    else:
        sig = np.asarray(sigma, dtype=np.float64)
        if sig.shape != (n,):
            raise ValueError(
                f"sigma must have shape ({n},) matching y, got {sig.shape}"
            )
        if not bool(np.all(np.isfinite(sig)) and np.all(sig > 0.0)):
            raise ValueError("sigma must be finite and strictly positive")

    lo, hi = float(theta_bounds[0]), float(theta_bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(
            f"theta_bounds must be finite with lo < hi, got {theta_bounds}"
        )
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    p = int(_eval_design(design, 0.5 * (lo + hi), n, None).shape[1])
    if n < p + 2:
        raise ValueError(
            f"need at least P + 2 = {p + 2} observations for {p} linear "
            f"amplitudes plus theta and one degree of freedom, got {n}"
        )

    def concentrated(theta: float) -> tuple[_DesignSVD, FloatArray]:
        theta_c = min(max(theta, lo), hi)
        a = _eval_design(design, theta_c, n, p)
        a_w, y_w = _whiten(a, yy, sig)
        svd = _design_svd(a_w, y_w)
        return svd, a_w

    def objective(theta: float) -> float:
        svd, _ = concentrated(theta)
        r = _projection_residual(svd)
        return float(r @ r)

    best_theta = lo
    best_val = np.inf
    for theta in np.linspace(lo, hi, n_grid):
        val = objective(float(theta))
        if val < best_val:
            best_val = val
            best_theta = float(theta)

    result = optimize.minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": _POLISH_XATOL},
    )
    if float(result.fun) < best_val:
        best_theta = float(result.x)

    theta_hat = min(max(best_theta, lo), hi)
    svd, a_w = concentrated(theta_hat)
    if svd.rank < p:
        raise ValueError(
            f"linear design is rank deficient at the optimum (rank {svd.rank} "
            f"< P = {p}) - the amplitudes are not identifiable; remove or "
            "recombine collinear columns"
        )
    coeffs = _svd_solve(svd)
    residual = _projection_residual(svd)
    chisq = float(residual @ residual)

    if absolute_sigma:
        scale_sq = 1.0
    else:
        if chisq <= 0.0:
            raise ValueError(
                "residual sum of squares is zero - no chi-square scale is "
                "estimable; pass absolute_sigma=True with trusted sigma"
            )
        scale_sq = chisq / float(n - p - 1)

    if d_design is None:
        d = _finite_difference_columns(design, theta_hat, n, p)
    else:
        d = np.asarray(d_design(theta_hat), dtype=np.float64)
        if d.shape != (n, p):
            raise ValueError(
                f"d_design(theta={theta_hat!r}) must return shape "
                f"({n}, {p}) matching the design, got {d.shape}"
            )
        if not bool(np.all(np.isfinite(d))):
            raise ValueError(
                f"d_design(theta={theta_hat!r}) returned non-finite values"
            )
    d_w = d if sig is None else np.asarray(d / sig[:, np.newaxis], dtype=np.float64)

    jac = _varpro_jacobian(svd, d_w, coeffs, residual)
    covariance = _bordered_covariance(a_w, jac, scale_sq)

    lower, upper, open_lower, open_upper = _profile_interval(
        objective, theta_hat, chisq, scale_sq, lo, hi
    )
    if open_lower or open_upper:
        side = (
            "both sides"
            if (open_lower and open_upper)
            else ("the lower side" if open_lower else "the upper side")
        )
        warnings.warn(
            f"Delta-chi-square = 1 profile interval for theta did not close on "
            f"{side} within theta_bounds = ({lo}, {hi}) - report a one-sided "
            "bound on theta, not a two-sided estimate, and carry the caveat "
            "into the linear amplitudes",
            UserWarning,
            stacklevel=2,
        )

    return VarproFit(
        theta=theta_hat,
        theta_sigma=float(np.sqrt(covariance[-1, -1])),
        theta_interval=(lower, upper),
        interval_open_lower=open_lower,
        interval_open_upper=open_upper,
        params=coeffs,
        covariance=covariance,
        chisq=chisq,
        scale_sq=scale_sq,
        n_obs=n,
    )
