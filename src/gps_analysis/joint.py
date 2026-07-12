"""Joint GPS + InSAR deformation-source inversion (LOS projection + VCE).

GNSS-only source inversions suffer the classic **depth–ΔV trade-off**: a
deeper Mogi source with a larger volume change produces almost the same
sparse-network displacement field as a shallower, smaller one (the pattern
*width* carries the depth information, and a handful of stations barely
samples it). InSAR breaks that trade-off twice over — dense spatial coverage
pins the pattern width, and the oblique line-of-sight (LOS) geometry mixes
the components differently than three-component GNSS (Fialko, Simons & Agnew
2001; Wright, Parsons & Lu 2004). This module supplies the leaf machinery:
LOS projection, the downsampled-InSAR data contract, and a joint
GPS + InSAR Mogi inversion with Variance Component Estimation (VCE) for the
relative dataset weighting (Sudhaus & Jónsson 2009) and per-track
offset/ramp nuisance parameters (Bagnardi & Hooper 2018).

Derivation chain
----------------
1. Geometry — :func:`los_unit_vector` builds the ground→satellite unit
   vector (E, N, U) from radar incidence and heading;
   :func:`los_project` contracts a 3-component ENU displacement onto it,
   ``d_LOS = u·d`` (Hanssen 2001 ch. 2; Fialko et al. 2001, eq. 1).
2. Data contract — :class:`InsarLos` holds ONE track of already-downsampled
   InSAR observations: local (e, n) points, LOS displacements, per-point LOS
   unit vectors, and a data covariance (dense or per-point σ). Quadtree
   downsampling and variogram covariance estimation from rasters are the
   *reader's* job (Lohman & Simons 2005; the KITE/geo_dataread lane) — the
   leaf consumes arrays.
3. Nuisance — :func:`ramp_design` builds the per-track design matrix of a
   constant offset and optional linear ramp, co-estimated with the source
   (orbital/reference-frame residuals; Bagnardi & Hooper 2018 §2).
4. Joint estimator — :func:`mogi_invert_joint` minimizes the concatenated
   covariance-whitened misfit of the GPS ENU block and every InSAR LOS block
   over ``[x, y, depth, ΔV]`` + nuisance (scipy ``least_squares``, analytic
   Jacobian, formal covariance), with the relative GPS/InSAR variance levels
   estimated by iterated Helmert VCE — :func:`variance_components`
   (Sudhaus & Jónsson 2009; Koch 1999 ch. 3) — instead of being fixed a
   priori.
5. Diagnostics — :func:`param_correlation` reads the trade-off off the
   posterior covariance (the depth–ΔV correlation is *the* number the joint
   data set must shrink; :class:`JointFit` exposes it directly).

Sign convention (binding)
-------------------------
``los_unit`` is the unit vector **from the ground point to the satellite**
(E, N, U components), and consequently **positive LOS displacement means
motion toward the satellite, i.e. a range decrease** (uplift toward a
right-looking SAR gives d_LOS > 0). Products delivered in the opposite
convention (positive = range increase / motion away) must be negated by the
reader before entering :class:`InsarLos`.

Conventions (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------
- Local Cartesian frame as :mod:`gps_analysis.deformation`: x = east,
  y = north, z = up; depths positive down [L]; ENU displacement blocks are
  ``(3, N)`` float64, rows **(east, north, up)**.
- Angles in **degrees** at the API surface: incidence measured from the
  vertical, heading = azimuth of the satellite ground velocity, clockwise
  from north (Sentinel-1: ≈ 348° ascending, ≈ 192° descending).
- One length unit [L] throughout (IMO production: meters); GPS and InSAR
  observations and σ share it.
- All functions are pure: no I/O, inputs never mutated.

Reference:
    Hanssen 2001, *Radar Interferometry — Data Interpretation and Error
    Analysis* (Kluwer), ch. 2 (imaging geometry). Fialko, Simons & Agnew
    2001, GRL 28(16), 3063–3066, eq. (1) (LOS decomposition). Wright,
    Parsons & Lu 2004, GRL 31, L01607 (multi-LOS 3-D retrieval). Lohman &
    Simons 2005, G³ 6, Q01007 (InSAR downsampling + noise covariance).
    Sudhaus & Jónsson 2009, GJI 176, 389–404 (joint InSAR+GPS inversion,
    VCE weighting). Bagnardi & Hooper 2018, G³ 19 (GBIS joint framework,
    per-track nuisance). Koch 1999, *Parameter Estimation and Hypothesis
    Testing in Linear Models* (2nd ed., Springer), ch. 3 (Helmert VCE).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import least_squares

from gps_analysis.deformation import (
    DEFAULT_NU,
    MogiSource,
    _as_obs_arrays,
    _lsq_covariance,
    _mogi_jacobian,
    _mogi_start,
    mogi_forward,
)
from gps_analysis.models import FloatArray

__all__ = [
    "InsarLos",
    "JointFit",
    "los_project",
    "los_unit_vector",
    "mogi_invert_joint",
    "param_correlation",
    "ramp_design",
    "variance_components",
]

#: Nuisance-parameter count per ramp model (see :func:`ramp_design`).
_RAMP_COLUMNS: dict[str, int] = {"none": 0, "offset": 1, "linear": 3}

#: Tolerance on ‖los_unit‖ = 1 (float64 build noise ≪ 1e-6 ≪ any real
#: geometry error).
_UNIT_NORM_TOL = 1.0e-6

#: VCE variance-component floor: a group whose estimated component falls
#: below this is residual-free at float precision (noiseless synthetics) —
#: iterating further would divide by ~0, so the VCE loop stops (documented
#: in :func:`mogi_invert_joint`).
_VCE_DEGENERATE = 1.0e-8

_RampName = Literal["none", "offset", "linear"]


# =====================================================================
# LOS geometry
# =====================================================================


def los_unit_vector(
    incidence: ArrayLike,
    heading: ArrayLike,
    look: Literal["right", "left"] = "right",
) -> FloatArray:
    """Ground→satellite LOS unit vector (E, N, U) from incidence and heading.

    Equation (right-looking SAR; θ incidence from vertical, α heading of the
    satellite ground velocity, clockwise from north — the horizontal part
    points from the ground toward the satellite, azimuth α − 90°):
        ``u = (−sin θ·cos α,  sin θ·sin α,  cos θ)``
    and for a left-looking system the horizontal components change sign,
        ``u = (sin θ·cos α, −sin θ·sin α, cos θ)``.

    Symbols → args:
        - ``θ`` → ``incidence``: radar incidence angle [deg], 0 ≤ θ < 90,
          scalar or per-point array
        - ``α`` → ``heading``: satellite heading (azimuth of the ground
          velocity) [deg], scalar or per-point array (broadcast with θ)
        - ``look``: antenna look direction, ``"right"`` (Sentinel-1, ERS,
          Envisat) or ``"left"``

    Returns:
        ``(3,)`` float64 for scalar inputs, ``(3, N)`` for array inputs —
        rows (east, north, up); unit norm by construction. **Positive
        projections onto this vector mean motion toward the satellite
        (range decrease)** — the module's binding sign convention.

    Reference:
        Hanssen 2001, *Radar Interferometry* (Kluwer), ch. 2 (SAR imaging
        geometry); Fialko, Simons & Agnew 2001, GRL 28(16), 3063–3066,
        eq. (1) (the LOS decomposition of an ENU displacement). Checked
        against the Sentinel-1 frame conventions: descending
        (α ≈ 192°) looks WNW so the satellite sits ESE of the target
        (u_E > 0); ascending (α ≈ 348°) gives u_E < 0.

    Numerical notes:
        Exact trigonometry, float64. θ = 0 degenerates to pure vertical
        (heading irrelevant); θ ≥ 90 (satellite at/below the horizon) is
        rejected.
    """
    if look not in ("right", "left"):
        raise ValueError(f'look must be "right" or "left", got {look!r}')
    theta = np.deg2rad(np.asarray(incidence, dtype=np.float64))
    alpha = np.deg2rad(np.asarray(heading, dtype=np.float64))
    if bool(np.any(theta < 0.0)) or bool(np.any(theta >= math.pi / 2.0)):
        raise ValueError("incidence must satisfy 0 <= incidence < 90 degrees")
    theta, alpha = np.broadcast_arrays(theta, alpha)
    sign = 1.0 if look == "right" else -1.0
    st = np.sin(theta)
    return np.stack(
        (
            -sign * st * np.cos(alpha),
            sign * st * np.sin(alpha),
            np.cos(theta),
        )
    )


def los_project(enu_disp: ArrayLike, los_unit: ArrayLike) -> FloatArray:
    """Project ENU displacements onto the satellite line of sight.

    Equation (inner product with the ground→satellite unit vector):
        ``d_LOS = u_E·d_E + u_N·d_N + u_U·d_U``

    Symbols → args:
        - ``d = (d_E, d_N, d_U)`` → ``enu_disp``: ``(3, N)`` (or ``(3,)``)
          displacements [L], rows **(east, north, up)**
        - ``u = (u_E, u_N, u_U)`` → ``los_unit``: ``(3,)`` (shared by all
          points) or ``(3, N)`` (per-point) ground→satellite unit vector,
          e.g. from :func:`los_unit_vector`

    Returns:
        ``(N,)`` float64 LOS displacements [L] (``()`` for ``(3,)`` input).
        **Positive = motion toward the satellite = range decrease** (module
        sign convention; negate for range-increase products).

    Reference:
        Fialko, Simons & Agnew 2001, GRL 28(16), 3063–3066, eq. (1);
        Hanssen 2001, ch. 2. This scalar projection is what makes a single
        interferogram a 1-component measurement — full 3-D recovery needs
        multiple viewing geometries (Wright, Parsons & Lu 2004, GRL 31,
        L01607).

    Numerical notes:
        Exact contraction (one multiply-add per component), float64;
        broadcasting of a shared ``(3,)`` unit vector is done by numpy.
        Unit-norm of ``los_unit`` is the caller's contract (validated when
        entering through :class:`InsarLos`).
    """
    d = np.asarray(enu_disp, dtype=np.float64)
    u = np.asarray(los_unit, dtype=np.float64)
    if d.shape[0] != 3 or u.shape[0] != 3:
        raise ValueError(
            f"enu_disp and los_unit must have 3 (E, N, U) rows, "
            f"got shapes {d.shape} and {u.shape}"
        )
    if u.ndim == 1 and d.ndim == 2:
        u = u[:, None]
    elif d.ndim == 1 and u.ndim == 2:
        d = d[:, None]
    out: FloatArray = (u * d).sum(axis=0)
    return out


def ramp_design(e: ArrayLike, n: ArrayLike, ramp: str) -> FloatArray:
    """Design matrix of the per-track InSAR nuisance model (offset + ramp).

    Equation (nuisance surface evaluated at the track's own points):
        ``r(e, n) = c₀``                                    (``"offset"``)
        ``r(e, n) = c₀ + c₁·(e − ē) + c₂·(n − n̄)``          (``"linear"``)
    assembled as ``A`` with ``r = A·c``; ``ē, n̄`` are the track's mean
    coordinates, and ``"none"`` yields a ``(N, 0)`` matrix.

    Symbols → args:
        - ``e, n``: track point coordinates [L]
        - ``ramp``: ``"none" | "offset" | "linear"`` — nuisance model
        - ``c = (c₀[, c₁, c₂])``: the parameters the caller estimates —
          offset [L] and ramp slopes [L/L = dimensionless]

    Returns:
        ``(N, p)`` float64 with p ∈ {0, 1, 3}; the linear columns are
        mean-centered.

    Reference:
        Bagnardi & Hooper 2018, G³ 19, §2 (constant offset and linear ramp
        as free nuisance parameters per InSAR data set — residual orbital /
        reference-frame signals); same practice in Sudhaus & Jónsson 2009,
        GJI 176, §4.

    Numerical notes:
        Mean-centering the ramp columns decorrelates them from the offset
        and keeps the least-squares system well-scaled; c₀ is then the
        nuisance value at the track centroid, not at (0, 0).
    """
    if ramp not in _RAMP_COLUMNS:
        raise ValueError(f"ramp must be one of {sorted(_RAMP_COLUMNS)}, got {ramp!r}")
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if ee.ndim != 1 or ee.shape != nn.shape:
        raise ValueError(f"e/n must be equal-length 1-D, got {ee.shape}/{nn.shape}")
    n_cols = _RAMP_COLUMNS[ramp]
    if n_cols == 0:
        return np.empty((ee.size, 0), dtype=np.float64)
    if n_cols == 1:
        return np.ones((ee.size, 1), dtype=np.float64)
    return np.column_stack(
        (np.ones(ee.size, dtype=np.float64), ee - ee.mean(), nn - nn.mean())
    )


# =====================================================================
# InSAR data contract (leaf input: already-downsampled points)
# =====================================================================


@dataclass(frozen=True)
class InsarLos:
    """One downsampled InSAR LOS dataset (one track/geometry) — leaf contract.

    The leaf consumes arrays: quadtree downsampling of the raster and the
    variogram-based data covariance are the reader's job (Lohman & Simons
    2005, G³ 6, Q01007; the KITE/geo_dataread lane). Exactly one of
    ``sigma``/``cov`` may be given; neither means unit weights.

    Attributes:
        e: ``(N,)`` point east coordinates [L] (same local frame as the GPS,
            e.g. :func:`gps_analysis.deformation.local_coordinates`).
        n: ``(N,)`` point north coordinates [L].
        d_los: ``(N,)`` LOS displacements [L], **positive toward the
            satellite (range decrease)** — module sign convention.
        los_unit: ``(3, N)`` (or ``(3,)`` shared) ground→satellite unit
            vectors, rows (E, N, U) — :func:`los_unit_vector`. Unit norm
            enforced to 1e-6.
        sigma: ``(N,)`` per-point 1-σ [L] (diagonal covariance), or None.
        cov: ``(N, N)`` dense data covariance [L²] (symmetric positive
            definite; e.g. from a variogram/covariogram fit — Lohman &
            Simons 2005 §3; Sudhaus & Jónsson 2009 §3), or None.
        ramp: Nuisance model co-estimated for this track —
            ``"none" | "offset" | "linear"`` (see :func:`ramp_design`;
            default ``"offset"``, the GBIS baseline).
    """

    e: FloatArray
    n: FloatArray
    d_los: FloatArray
    los_unit: FloatArray
    sigma: FloatArray | None = None
    cov: FloatArray | None = None
    ramp: _RampName = "offset"

    def __post_init__(self) -> None:
        ee = np.atleast_1d(np.asarray(self.e, dtype=np.float64))
        nn = np.atleast_1d(np.asarray(self.n, dtype=np.float64))
        dd = np.atleast_1d(np.asarray(self.d_los, dtype=np.float64))
        if ee.ndim != 1 or ee.shape != nn.shape or ee.shape != dd.shape:
            raise ValueError(
                f"e, n, d_los must be equal-length 1-D, got "
                f"{ee.shape}/{nn.shape}/{dd.shape}"
            )
        uu = np.asarray(self.los_unit, dtype=np.float64)
        if uu.shape == (3,):
            uu = np.repeat(uu[:, None], ee.size, axis=1)
        if uu.shape != (3, ee.size):
            raise ValueError(
                f"los_unit must have shape (3,) or (3, {ee.size}), got {uu.shape}"
            )
        norms = np.sqrt((uu * uu).sum(axis=0))
        if not bool(np.all(np.abs(norms - 1.0) <= _UNIT_NORM_TOL)):
            raise ValueError("los_unit columns must be unit vectors (|‖u‖−1| ≤ 1e-6)")
        if self.sigma is not None and self.cov is not None:
            raise ValueError("give sigma or cov, not both")
        ss: FloatArray | None = None
        if self.sigma is not None:
            ss = np.atleast_1d(np.asarray(self.sigma, dtype=np.float64))
            if ss.shape != ee.shape:
                raise ValueError(f"sigma must have shape {ee.shape}, got {ss.shape}")
            if not bool(np.all(ss > 0.0)):
                raise ValueError("sigma must be strictly positive")
        cc: FloatArray | None = None
        if self.cov is not None:
            cc = np.asarray(self.cov, dtype=np.float64)
            if cc.shape != (ee.size, ee.size):
                raise ValueError(
                    f"cov must have shape ({ee.size}, {ee.size}), got {cc.shape}"
                )
            if not np.allclose(cc, cc.T, rtol=1.0e-10, atol=0.0):
                raise ValueError("cov must be symmetric")
        if self.ramp not in _RAMP_COLUMNS:
            raise ValueError(
                f"ramp must be one of {sorted(_RAMP_COLUMNS)}, got {self.ramp!r}"
            )
        finite = (
            bool(np.all(np.isfinite(ee)))
            and bool(np.all(np.isfinite(nn)))
            and bool(np.all(np.isfinite(dd)))
            and bool(np.all(np.isfinite(uu)))
        )
        if not finite:
            raise ValueError("e, n, d_los, los_unit must be finite (mask upstream)")
        object.__setattr__(self, "e", ee)
        object.__setattr__(self, "n", nn)
        object.__setattr__(self, "d_los", dd)
        object.__setattr__(self, "los_unit", uu)
        object.__setattr__(self, "sigma", ss)
        object.__setattr__(self, "cov", cc)

    @property
    def n_points(self) -> int:
        """Number of LOS observations N."""
        return int(self.e.size)

    @property
    def n_nuisance(self) -> int:
        """Number of nuisance parameters of this track's ramp model."""
        return _RAMP_COLUMNS[self.ramp]


# =====================================================================
# Covariance diagnostics + variance component estimation
# =====================================================================


def param_correlation(covariance: ArrayLike, i: int, j: int) -> float:
    """Correlation coefficient of two parameters from a covariance matrix.

    Equation:
        ``ρ_ij = C_ij / √(C_ii · C_jj)``

    Symbols → args:
        - ``C`` → ``covariance``: ``(p, p)`` parameter covariance
        - ``i, j``: parameter indices (e.g. depth = 2, ΔV = 3 in the Mogi
          ``[x, y, depth, dv]`` order)

    Returns:
        ρ_ij ∈ [−1, 1] (dimensionless). |ρ| → 1 is the algebraic signature
        of a parameter trade-off: the data constrain only a combination of
        the two — for GPS-only Mogi fits ρ(depth, ΔV) ≈ +1 (deeper source,
        larger volume, same field).

    Reference:
        Standard covariance algebra (e.g. Aster, Borchers & Thurber 2018,
        §2.3); the depth–ΔV trade-off diagnosed this way follows the joint
        GPS+InSAR literature (Sudhaus & Jónsson 2009, table 3 reports the
        analogous posterior correlations).

    Numerical notes:
        Raises for non-positive diagonal entries (degenerate covariance);
        exact algebra otherwise.
    """
    c = np.asarray(covariance, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"covariance must be square, got shape {c.shape}")
    cii = float(c[i, i])
    cjj = float(c[j, j])
    if cii <= 0.0 or cjj <= 0.0:
        raise ValueError("covariance diagonal must be positive at i and j")
    return float(c[i, j]) / math.sqrt(cii * cjj)


def variance_components(
    residuals: Sequence[ArrayLike],
    jacobians: Sequence[ArrayLike],
) -> FloatArray:
    """Helmert variance components of whitened observation groups.

    Equation (iterated almost-unbiased VCE; one update step): for group k
    with whitened residual vector ``r_k`` and whitened Jacobian block
    ``J_k`` at the solution,
        ``σ̂²_k = (r_kᵀ·r_k) / red_k``,
        ``red_k = n_k − tr(N⁻¹·N_k)``,  ``N_k = J_kᵀ·J_k``,  ``N = Σ_k N_k``
    — ``red_k`` is the group's partial redundancy (its share of the total
    degrees of freedom ``Σ red_k = n − p``). σ̂²_k = 1 means the group's
    covariance is consistent with its residuals; the caller rescales
    ``C_k ← σ̂²_k·C_k`` and iterates to convergence (σ̂²_k → 1 ∀k).

    Symbols → args:
        - ``r_k`` → ``residuals[k]``: ``(n_k,)`` residuals, whitened by the
          group's CURRENT covariance (dimensionless)
        - ``J_k`` → ``jacobians[k]``: ``(n_k, p)`` Jacobian of the whitened
          residuals w.r.t. all p parameters (sign-free: only JᵀJ enters)

    Returns:
        ``(K,)`` float64 — the variance factor σ̂²_k per group, relative to
        the covariance each group was whitened with.

    Reference:
        Koch 1999, *Parameter Estimation and Hypothesis Testing in Linear
        Models* (2nd ed., Springer), ch. 3 (Helmert-type variance component
        estimation, partial redundancies); applied to joint InSAR + GPS
        source inversion by Sudhaus & Jónsson 2009, GJI 176, 389–404
        (relative dataset weighting instead of a fixed weight ratio).

    Numerical notes:
        ``N`` is solved by Cholesky (`cho_solve`), never inverted
        explicitly; ``tr(N⁻¹N_k)`` is evaluated as ``Σ_ij (N⁻¹J_kᵀ)∘J_kᵀ``
        without forming N_k. Nonlinear models: this is the linearized update
        at the current solution (standard practice — Sudhaus & Jónsson 2009
        iterate it with the model fit). Raises when N is singular (rank
        deficiency: parameters unconstrained by ALL groups combined) or a
        partial redundancy is ≤ 0 (a group contributes no redundant
        information — its variance component is not estimable).
    """
    if len(residuals) != len(jacobians) or len(residuals) == 0:
        raise ValueError("need >= 1 group with matching residual/jacobian entries")
    r_blocks = [np.atleast_1d(np.asarray(r, dtype=np.float64)) for r in residuals]
    j_blocks = [np.asarray(j, dtype=np.float64) for j in jacobians]
    p = j_blocks[0].shape[1]
    for r_k, j_k in zip(r_blocks, j_blocks, strict=True):
        if r_k.ndim != 1 or j_k.ndim != 2 or j_k.shape != (r_k.size, p):
            raise ValueError(
                f"group blocks must be (n_k,) and (n_k, {p}), got "
                f"{r_k.shape} and {j_k.shape}"
            )
    normal = np.zeros((p, p), dtype=np.float64)
    for j_k in j_blocks:
        normal += j_k.T @ j_k
    try:
        chol = cho_factor(normal)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - scipy raises its own
        raise ValueError("normal matrix N is singular") from exc
    out = np.empty(len(r_blocks), dtype=np.float64)
    for k, (r_k, j_k) in enumerate(zip(r_blocks, j_blocks, strict=True)):
        trace = float(np.sum(np.asarray(cho_solve(chol, j_k.T)) * j_k.T))
        red = r_k.size - trace
        if red <= 0.0:
            raise ValueError(
                f"group {k} has non-positive partial redundancy ({red:.3g}) — "
                "its variance component is not estimable"
            )
        out[k] = float(r_k @ r_k) / red
    return out


# =====================================================================
# Joint GPS + InSAR Mogi inversion
# =====================================================================


@dataclass(frozen=True)
class JointFit:
    """Joint GPS + InSAR inversion result (:func:`mogi_invert_joint`).

    Attributes:
        source: Best-fit :class:`~gps_analysis.deformation.MogiSource`.
        param_names: Names of ALL estimated parameters, covariance order:
            ``("x", "y", "depth", "dv", "los0_offset", …)``.
        covariance: ``(p, p)`` formal covariance of all parameters (source
            block first — already marginal over the nuisance parameters).
        sigma: ``(p,)`` formal 1-σ, ``√diag(covariance)``.
        nuisance: Per-track nuisance vectors (offset [L]; ramp slopes
            [L/L]), ordering as :func:`ramp_design`; empty arrays for
            ``ramp="none"``.
        variance_components: ``(1 + K,)`` final VCE variance factors
            relative to the INPUT covariances — GPS first, then each track.
            ≈ 1 means the stated covariance was already honest; 4 means the
            stated σ were understated by 2×. Diagnostic only (weights
            unchanged) when ``vce=False``.
        vce_iterations: Number of fit+VCE-update cycles performed.
        chi2_reduced: Reduced chi-square of the final whitened fit.
        rms_gps: Unweighted RMS of the GPS ENU residuals [L].
        rms_los: Per-track unweighted RMS of the LOS residuals (nuisance
            removed) [L].
        n_obs: Total scalar observations (3·N_gps + Σ N_k).
    """

    source: MogiSource
    param_names: tuple[str, ...]
    covariance: FloatArray
    sigma: FloatArray
    nuisance: tuple[FloatArray, ...]
    variance_components: FloatArray
    vce_iterations: int
    chi2_reduced: float
    rms_gps: float
    rms_los: tuple[float, ...]
    n_obs: int

    @property
    def depth_dv_correlation(self) -> float:
        """Posterior depth–ΔV correlation ρ(d, ΔV) = C₂₃/√(C₂₂C₃₃).

        The trade-off diagnostic (see :func:`param_correlation`): GPS-only
        Mogi fits sit near +1; a joint GPS+InSAR fit must pull it toward 0.
        """
        return param_correlation(self.covariance, 2, 3)


def _track_whitener(
    track: InsarLos, factor: float
) -> Callable[[FloatArray], FloatArray]:
    """Whitening operator x ↦ C_k^(−1/2)·x / √s_k for one InSAR track.

    Equation:
        dense ``cov``:  ``x_w = L⁻¹·x / √s`` with ``C = L·Lᵀ`` (Cholesky);
        diagonal:       ``x_w = x / (σ·√s)``;
        with ``s`` the track's current VCE variance factor.

    Symbols → args:
        - ``C, σ`` → ``track.cov`` / ``track.sigma`` (None ⇒ unit weights)
        - ``s`` → ``factor``: VCE variance factor (> 0)

    Returns:
        Callable applying the whitening to ``(n_k,)`` vectors or ``(n_k, p)``
        matrices (columns whitened independently).

    Reference:
        Standard generalized least squares whitening (Aster, Borchers &
        Thurber 2018, §2.2 — weighting by the inverse square root of the
        data covariance); dense-C InSAR weighting per Sudhaus & Jónsson
        2009, §3; Lohman & Simons 2005, §3.

    Numerical notes:
        Cholesky + triangular solve (never an explicit C⁻¹): O(n²) per
        apply after one O(n³) factorization, and backward stable. Raises
        ``scipy.linalg.LinAlgError`` for a non-positive-definite ``cov``.
    """
    if not factor > 0.0:
        raise ValueError(f"variance factor must be > 0, got {factor}")
    root = math.sqrt(factor)
    if track.cov is not None:
        lo = np.asarray(cholesky(track.cov, lower=True), dtype=np.float64)

        def dense(x: FloatArray) -> FloatArray:
            out: FloatArray = (
                np.asarray(solve_triangular(lo, x, lower=True), dtype=np.float64) / root
            )
            return out

        return dense
    sig = track.sigma if track.sigma is not None else np.ones(track.n_points)

    def diagonal(x: FloatArray) -> FloatArray:
        div = sig * root
        out: FloatArray = x / (div if x.ndim == 1 else div[:, None])
        return out

    return diagonal


def mogi_invert_joint(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    insar: Sequence[InsarLos],
    x0: MogiSource | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    nu: float = DEFAULT_NU,
    vce: bool = True,
    vce_tol: float = 0.02,
    vce_max_iter: int = 20,
) -> JointFit:
    """Joint GPS ENU + InSAR LOS Mogi inversion with VCE dataset weighting.

    Estimator (generalized least squares over source + nuisance):
        ``m* = argmin_m  Σ_k ‖ C_k^(−1/2)·(d_k − G_k(m)) ‖²`` over
        ``m = [x_s, y_s, d, ΔV, c₁, …, c_K]`` where group k = 0 is the GPS
        block (``G₀`` = :func:`~gps_analysis.deformation.mogi_forward`,
        diagonal C₀ from ``sigma``) and each InSAR track k ≥ 1 predicts
        ``G_k(m) = u_k·u_mogi(e_k, n_k; m) + A_k·c_k``
    — the LOS projection (:func:`los_project`) of the Mogi field plus the
    track's nuisance surface (:func:`ramp_design`), whitened by the track
    covariance. The **relative weighting** of GPS vs each InSAR track is not
    fixed: after each fit the Helmert variance components
    (:func:`variance_components`) rescale every group's covariance,
    ``C_k ← σ̂²_k·C_k``, and the fit repeats until all σ̂²_k ≈ 1 (Sudhaus &
    Jónsson 2009). Solved by scipy ``least_squares`` (TRF, fully analytic
    Jacobian); formal covariance from
    :func:`~gps_analysis.deformation._lsq_covariance`.

    Why joint: GPS alone leaves the depth–ΔV direction of the Mogi problem
    ill-constrained (ρ(d, ΔV) → 1); the dense LOS field pins the anomaly's
    spatial wavelength — and a second viewing geometry approaches full 3-D
    (Wright, Parsons & Lu 2004) — collapsing both σ_d and the correlation.

    Symbols → args:
        - ``d₀`` → ``obs``: ``(3, N)`` GPS displacements [L], rows (east,
          north, up); ``σ₀`` → ``sigma``: ``(3, N)`` 1-σ [L] or None
        - ``e, n``: GPS station coordinates [L] (same local frame as every
          track's points — one origin for the whole problem)
        - ``d_k, C_k, u_k, A_k`` → ``insar``: ≥ 1 :class:`InsarLos` tracks
        - ``x0``: optional starting source; default
          :func:`~gps_analysis.deformation._mogi_start` on the GPS block
        - ``bounds``: optional ``(lower, upper)`` 4-vectors for the SOURCE
          parameters ``[x, y, depth, dv]``; nuisance parameters are always
          unbounded. Default: x, y within the joint data footprint ± one
          span, depth ∈ [10⁻³·span, 20·span], ΔV free.
        - ``ν`` → ``nu``: Poisson's ratio [-]
        - ``vce``: iterate the VCE reweighting (True) or fit once with the
          stated covariances and report the components as diagnostics only
        - ``vce_tol``: convergence: max_k |σ̂²_k − 1| < vce_tol
        - ``vce_max_iter``: cap on fit+update cycles

    Returns:
        :class:`JointFit` — source, full covariance (+ per-parameter names),
        per-track nuisance, VCE variance components, χ²_red, per-dataset
        RMS. The trade-off diagnostics: ``sigma[2]`` (σ_depth) and
        ``depth_dv_correlation``.

    Reference:
        Sudhaus & Jónsson 2009, GJI 176, 389–404 (joint InSAR+GPS NLLS with
        VCE weighting — the estimator this implements deterministically);
        Bagnardi & Hooper 2018, G³ 19 (joint GBIS: per-track offset/ramp
        nuisance; the Bayesian counterpart on :mod:`gps_analysis._mcmc` is
        the planned next slice); Fialko et al. 2001 (LOS geometry); Wright
        et al. 2004 (multi-LOS 3-D); Lohman & Simons 2005 (downsampled-point
        + covariance input contract); forward model Mogi 1958 / Segall 2010
        §7.1.

    Numerical notes:
        - Whitening by Cholesky solves, never explicit inverses; the
          analytic Mogi Jacobian is LOS-projected per track, so no
          finite differences anywhere.
        - Parameter scaling: network span for x/y/depth, |ΔV₀| for volume,
          each track's LOS scatter for its offset (slopes: scatter/span) —
          keeps the trust region even across ~10⁷ m³ vs ~10⁻³ m parameters.
        - VCE with noise-free (synthetic) groups: a variance component
          collapsing below 1e-8 stops the iteration (weights would diverge);
          use ``vce=False`` for exact-recovery tests.
        - The formal covariance is scaled by the reduced χ² (as
          :func:`~gps_analysis.deformation.mogi_invert`, so GPS-only and
          joint uncertainties are directly comparable); after converged VCE
          that factor is ≈ 1 by construction.
        - Degrees of freedom must be positive: 3·N_gps + Σ n_k > 4 + Σ p_k.
    """
    ee, nn, dd, ssg = _as_obs_arrays(e, n, obs, sigma)
    tracks = tuple(insar)
    if not tracks:
        raise ValueError(
            "insar must contain >= 1 InsarLos dataset "
            "(GPS-only inversion: use mogi_invert)"
        )
    if not 0.0 < vce_tol < 1.0:
        raise ValueError(f"vce_tol must be in (0, 1), got {vce_tol}")
    if vce_max_iter < 1:
        raise ValueError(f"vce_max_iter must be >= 1, got {vce_max_iter}")

    # ---- parameter layout: [x, y, depth, dv] + per-track nuisance ---------
    names: list[str] = ["x", "y", "depth", "dv"]
    nuis_slices: list[slice] = []
    offset = 4
    for k, trk in enumerate(tracks):
        nuis_slices.append(slice(offset, offset + trk.n_nuisance))
        names += [
            f"los{k}_{c}" for c in ("offset", "ramp_e", "ramp_n")[: trk.n_nuisance]
        ]
        offset += trk.n_nuisance
    n_params = offset
    block_sizes = [dd.size] + [trk.n_points for trk in tracks]
    n_total = sum(block_sizes)
    if n_total <= n_params:
        raise ValueError(
            f"underdetermined: {n_total} observations for {n_params} parameters"
        )

    # ---- start, bounds, scaling -------------------------------------------
    start_src = x0.as_array() if x0 is not None else _mogi_start(ee, nn, dd, nu)
    all_e = np.concatenate([ee] + [trk.e for trk in tracks])
    all_n = np.concatenate([nn] + [trk.n for trk in tracks])
    span = max(float(np.ptp(all_e)), float(np.ptp(all_n)), 1.0)
    if bounds is not None:
        lo_src = np.asarray(bounds[0], dtype=np.float64)
        up_src = np.asarray(bounds[1], dtype=np.float64)
        if lo_src.shape != (4,) or up_src.shape != (4,):
            raise ValueError("bounds must be a pair of length-4 source vectors")
    else:
        lo_src = np.array(
            [all_e.min() - span, all_n.min() - span, 1.0e-3 * span, -np.inf]
        )
        up_src = np.array([all_e.max() + span, all_n.max() + span, 20.0 * span, np.inf])
    lower = np.concatenate((lo_src, np.full(n_params - 4, -np.inf)))
    upper = np.concatenate((up_src, np.full(n_params - 4, np.inf)))
    start = np.concatenate((np.clip(start_src, lo_src, up_src), np.zeros(n_params - 4)))
    x_scale = np.empty(n_params, dtype=np.float64)
    x_scale[:4] = (span, span, span, max(abs(float(start[3])), 1.0))
    for trk, sl in zip(tracks, nuis_slices, strict=True):
        if trk.n_nuisance:
            amp = max(float(np.std(trk.d_los)), 1.0e-6)
            x_scale[sl] = [amp, amp / span, amp / span][: trk.n_nuisance]

    designs = [ramp_design(trk.e, trk.n, trk.ramp) for trk in tracks]
    dflat = dd.ravel()
    sgflat = ssg.ravel()

    # ---- one whitened fit for the current variance factors ----------------
    def fit_once(s: FloatArray) -> Any:
        g_root = math.sqrt(float(s[0]))
        whits = [_track_whitener(trk, float(s[1 + k])) for k, trk in enumerate(tracks)]

        def residual(m: FloatArray) -> FloatArray:
            src = MogiSource.from_array(m[:4])
            parts = [
                (dflat - mogi_forward(ee, nn, src, nu).ravel()) / (sgflat * g_root)
            ]
            for trk, a_mat, sl, wh in zip(
                tracks, designs, nuis_slices, whits, strict=True
            ):
                model = los_project(mogi_forward(trk.e, trk.n, src, nu), trk.los_unit)
                if trk.n_nuisance:
                    model = model + a_mat @ m[sl]
                parts.append(wh(trk.d_los - model))
            return np.concatenate(parts)

        def jacobian(m: FloatArray) -> FloatArray:
            jac = np.zeros((n_total, n_params), dtype=np.float64)
            jac[: dflat.size, :4] = -_mogi_jacobian(m[:4], ee, nn, nu) / (
                sgflat[:, None] * g_root
            )
            row = dflat.size
            for trk, a_mat, sl, wh in zip(
                tracks, designs, nuis_slices, whits, strict=True
            ):
                n_k = trk.n_points
                j_enu = _mogi_jacobian(m[:4], trk.e, trk.n, nu).reshape(3, n_k, 4)
                j_los = np.einsum("cn,cnp->np", trk.los_unit, j_enu)
                jac[row : row + n_k, :4] = wh(-j_los)
                if trk.n_nuisance:
                    jac[row : row + n_k, sl] = wh(-a_mat)
                row += n_k
            return jac

        res: Any = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale=x_scale,
        )
        if not res.success:
            raise RuntimeError(f"mogi_invert_joint did not converge: {res.message}")
        return res

    # ---- VCE loop (Sudhaus & Jónsson 2009: rescale C_k, refit) ------------
    s = np.ones(1 + len(tracks), dtype=np.float64)
    n_iter = 0
    while True:
        res = fit_once(s)
        n_iter += 1
        edges = np.cumsum([0] + block_sizes)
        r_blocks = [res.fun[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)]
        j_blocks = [
            np.asarray(res.jac[a:b], dtype=np.float64)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        factors = variance_components(r_blocks, j_blocks)
        s_est = s * factors
        if not vce:
            s = s_est  # report-only estimate; weights were never rescaled
            break
        s = np.maximum(s_est, _VCE_DEGENERATE)
        if (
            float(np.max(np.abs(factors - 1.0))) < vce_tol
            or bool(np.any(s_est < _VCE_DEGENERATE))
            or n_iter >= vce_max_iter
        ):
            break

    # ---- results ------------------------------------------------------------
    cov = _lsq_covariance(
        np.asarray(res.jac, dtype=np.float64), float(res.cost), n_total, n_params
    )
    m_opt = np.asarray(res.x, dtype=np.float64)
    source = MogiSource.from_array(m_opt[:4])
    model_gps = mogi_forward(ee, nn, source, nu)
    rms_gps = float(np.sqrt(np.mean((dd - model_gps) ** 2)))
    rms_los: list[float] = []
    for trk, a_mat, sl in zip(tracks, designs, nuis_slices, strict=True):
        model = los_project(mogi_forward(trk.e, trk.n, source, nu), trk.los_unit)
        if trk.n_nuisance:
            model = model + a_mat @ m_opt[sl]
        rms_los.append(float(np.sqrt(np.mean((trk.d_los - model) ** 2))))
    return JointFit(
        source=source,
        param_names=tuple(names),
        covariance=cov,
        sigma=np.sqrt(np.diag(cov)),
        nuisance=tuple(m_opt[sl].copy() for sl in nuis_slices),
        variance_components=s,
        vce_iterations=n_iter,
        chi2_reduced=2.0 * float(res.cost) / (n_total - n_params),
        rms_gps=rms_gps,
        rms_los=tuple(rms_los),
        n_obs=n_total,
    )
