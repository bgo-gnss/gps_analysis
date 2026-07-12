"""Model fitting, detrending and outlier rejection for GNSS time series.

Consolidated from ``~/work/projects/gps_data_analyses`` (see
``docs/CONSOLIDATION_MAP.md``): the legacy per-component fit loop
``fittimes`` (``detrend-reykjanes/detrend_rnes.py``, duplicated across the
``detrend-*`` family) and its half-refactored descendants
``svartsengi_model.fitting.fit_curve`` / ``fit_three_components``, plus the
legacy ``detrend(...)`` (rewritten without its in-place mutation,
function-body imports and I/O side effects). The robust-loss path of
:func:`reject_outliers` is net-new (the legacy running-median pre-filter
``RunningMedian`` is superseded by it).

Derivation chain
----------------
Given epochs t ∈ ℝᴺ (fractional years, ``yearf``), per-component
observations y ∈ ℝᴺ with 1-σ uncertainties σ ∈ ℝᴺ, and a trajectory model
f(t; p) from :mod:`gps_analysis.models` with parameters p ∈ ℝᴾ:

1. :func:`fit_components` solves the weighted least-squares problem, per
   component,

       ``p̂ = argmin_p Σᵢ [ (yᵢ − f(tᵢ; p)) / σᵢ ]²``

   and packs (p̂, C_p̂) into :class:`~gps_analysis.models.TrajectoryParams`.
   Models that are **linear in their parameters**
   (:func:`~gps_analysis.models.linear`,
   :func:`~gps_analysis.models.periodic`,
   :func:`~gps_analysis.models.lineperiodic` — registered in
   ``_LINEAR_DESIGNS``) are solved in closed form: the design matrix A is
   built once and the σ-whitened system is solved by SVD
   (:func:`_wls_solve`), giving the exact optimum
   ``p̂ = (AᵀWA)⁻¹AᵀWy`` and covariance ``C_p̂ = (AᵀWA)⁻¹`` with
   ``W = diag(1/σᵢ²)`` — no iteration. Genuinely nonlinear models
   (:func:`~gps_analysis.models.exp_linear`,
   :func:`~gps_analysis.models.poly2`, custom callables) go through
   ``scipy.optimize.curve_fit`` (Levenberg–Marquardt / TRF; Moré 1978) as
   before. Both paths share the covariance semantics (reduced-χ²
   rescaling unless ``absolute_sigma=True``).
2. :func:`remove_trend` evaluates the fitted model and subtracts it,
   ``r = y − f(t; p̂)`` — the detrended series / residuals.
3. :func:`detrend_fit` composes 1 → 2 (the legacy ``detrend`` behavior,
   made pure).
4. :func:`reject_outliers` iterates a **robust** fit — the M-estimator

       ``p̂ = argmin_p Σᵢ ρ( (yᵢ − f(tᵢ; p)) / (σᵢ·f_scale) )``

   with ρ = ``soft_l1``/``huber``/… (``scipy.optimize.least_squares``;
   Huber 1964) — with rejection of points whose whitened residuals exceed
   ``n_sigma`` × the normalized-MAD scale (Rousseeuw & Croux 1993), then
   refits inliers with plain WLS (step 1) so the reported covariance is
   the standard Gauss–Markov one.

Everything is array-first (decision 1, ``docs/CONSOLIDATION_MAP.md``):
``y`` is 1-D (one component) or (C, N) (component-major, e.g. N/E/U rows);
units are the caller's business ([L] below); inputs are never mutated.
Formal WLS errors assume temporally white noise — for honest GNSS rate
uncertainties see Williams 2003 (J. Geodesy 76) and the GBIS4TS lane
(:mod:`gps_analysis.transient`).
"""

import dataclasses
import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from . import models
from .models import FloatArray, TrajectoryParams

__all__ = [
    "ModelFunc",
    "OutlierRejection",
    "detrend_fit",
    "fit_components",
    "reject_outliers",
    "remove_trend",
]

ModelFunc = Callable[..., FloatArray]
"""Trajectory-model callable ``f(t, *params) -> ndarray`` (see
:mod:`gps_analysis.models`)."""

_MAD_TO_SIGMA = 1.4826
"""Normalized-MAD factor: 1/Φ⁻¹(3/4), Gaussian-consistent scale
(Rousseeuw & Croux 1993, JASA 88)."""


def _n_model_params(model: ModelFunc) -> int:
    """Number of fit parameters P of ``model(t, *params)`` by signature."""
    n = len(inspect.signature(model).parameters) - 1
    if n < 1:
        raise ValueError("model must accept at least one parameter after t")
    return n


def _components_2d(y: ArrayLike, name: str) -> tuple[FloatArray, bool]:
    """Coerce y to (C, N) float64; report whether input was 1-D."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"{name} must be 1-D or 2-D (C, N), got shape {arr.shape}")


def _per_component_p0(
    p0: ArrayLike | None, n_components: int, was_1d: bool
) -> list[FloatArray | None]:
    """Split an initial-guess spec into one optional vector per component."""
    if p0 is None:
        return [None] * n_components
    arr = np.asarray(p0, dtype=np.float64)
    if was_1d:
        if arr.ndim != 1:
            raise ValueError("p0 must be 1-D for a 1-D y")
        return [arr]
    if arr.ndim != 2 or arr.shape[0] != n_components:
        raise ValueError(
            f"p0 must have shape (C, P) with C = {n_components}, got shape {arr.shape}"
        )
    return [arr[i] for i in range(n_components)]


def _per_component_sigma(
    sigma: ArrayLike | None, yy: FloatArray, was_1d: bool
) -> list[FloatArray | None]:
    """Split uncertainties into one optional (N,) vector per component."""
    if sigma is None:
        return [None] * yy.shape[0]
    ss, s_was_1d = _components_2d(sigma, "sigma")
    if s_was_1d != was_1d or ss.shape != yy.shape:
        raise ValueError(
            f"sigma shape {np.asarray(sigma).shape} does not match y shape "
            f"{yy.shape if not was_1d else (yy.shape[1],)}"
        )
    return [ss[i] for i in range(ss.shape[0])]


@dataclasses.dataclass(frozen=True)
class _LinearDesign:
    """Design-matrix recipe for a linear-in-parameters trajectory model.

    Pairs the model with a builder for its design matrix A ∈ ℝᴺˣᴾ (one
    column per model parameter, in the model's positional parameter
    order), so f(t; p) = A(t)·p exactly. Registered per model in
    ``_LINEAR_DESIGNS``; :func:`fit_components` dispatches on it.

    Attributes:
        build: Callable ``A = build(t)`` evaluating the basis columns at
            the **absolute** epochs t [yr] — for the seasonal models the
            trig columns use absolute ``yearf`` so the fitted phase
            convention matches :func:`gps_analysis.models.periodic`
            exactly.
        trend_column: Column index holding the raw epoch t (the secular
            rate column), or ``None`` if the design has no polynomial-in-t
            column. This column is re-centered (t → t − t_ref) before the
            solve for conditioning — see :func:`_fit_linear_design`.
        intercept_column: Column index of the constant (all-ones)
            intercept column that absorbs the centering shift, or
            ``None``.
    """

    build: Callable[[FloatArray], FloatArray]
    trend_column: int | None = None
    intercept_column: int | None = None


def _design_linear(t: FloatArray) -> FloatArray:
    """Build the design matrix of :func:`gps_analysis.models.linear`.

    Equation:
        ``A = [1  t]``  (N×2), so ``linear(t; x₀, v) = A·[x₀, v]ᵀ``

    Symbols → args:
        - ``t`` → ``t``: epochs, fractional years (``yearf``) [yr]

    Returns:
        Design matrix A, shape (N, 2), float64 — columns [1, t] matching
        the ``[offset, rate]`` parameter order.

    Reference:
        Strang & Borre 1997, *Linear Algebra, Geodesy and GPS*, ch. 9
        (the straight-line observation model); Bevis & Brown 2014, eq. (1)
        with m = 1.

    Numerical notes:
        Columns are built at absolute t — near-collinear for
        t ≈ 2×10³ yr; :func:`_fit_linear_design` re-centers the t column
        before solving.
    """
    return np.column_stack((np.ones_like(t), t))


def _design_periodic(t: FloatArray) -> FloatArray:
    """Build the design matrix of :func:`gps_analysis.models.periodic`.

    Equation:
        ``A = [cos 2πt  sin 2πt  cos 4πt  sin 4πt]``  (N×4), so
        ``periodic(t; a, b, c, d) = A·[a, b, c, d]ᵀ``

    Symbols → args:
        - ``t`` → ``t``: epochs, fractional years (``yearf``) [yr]

    Returns:
        Design matrix A, shape (N, 4), float64 — columns in the
        ``[cos_annual, sin_annual, cos_semiannual, sin_semiannual]``
        parameter order.

    Reference:
        Blewitt & Lavallée 2002, JGR 107(B7), eq. (2) (annual +
        semiannual truncation); Bevis & Brown 2014, eq. (1), n_F = 2.

    Numerical notes:
        The trig arguments replicate :func:`gps_analysis.models.periodic`
        verbatim (``2πt`` and ``2·2πt`` on absolute ``yearf``) so the
        basis — and therefore the fitted phase convention — is identical
        to the model evaluator's. Columns are bounded in [−1, 1]: no
        centering needed.
    """
    two_pi_t = 2.0 * np.pi * t
    return np.column_stack(
        (
            np.cos(two_pi_t),
            np.sin(two_pi_t),
            np.cos(2.0 * two_pi_t),
            np.sin(2.0 * two_pi_t),
        )
    )


def _design_lineperiodic(t: FloatArray) -> FloatArray:
    """Build the design matrix of :func:`gps_analysis.models.lineperiodic`.

    Equation:
        ``A = [1  t  cos 2πt  sin 2πt  cos 4πt  sin 4πt]``  (N×6) —
        the column-wise concatenation of :func:`_design_linear` and
        :func:`_design_periodic`, matching the ``[offset, rate,
        cos_annual, sin_annual, cos_semiannual, sin_semiannual]``
        parameter order.

    Symbols → args:
        - ``t`` → ``t``: epochs, fractional years (``yearf``) [yr]

    Returns:
        Design matrix A, shape (N, 6), float64.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (1) (m = 1, n_F = 2);
        Blewitt & Lavallée 2002, JGR 107(B7).

    Numerical notes:
        See :func:`_design_linear` (t column re-centered before the
        solve) and :func:`_design_periodic` (absolute-t trig phase).
    """
    return np.column_stack((_design_linear(t), _design_periodic(t)))


_LINEAR_DESIGNS: dict[ModelFunc, _LinearDesign] = {
    models.linear: _LinearDesign(_design_linear, trend_column=1, intercept_column=0),
    models.periodic: _LinearDesign(_design_periodic),
    models.lineperiodic: _LinearDesign(
        _design_lineperiodic, trend_column=1, intercept_column=0
    ),
}
"""Registry of linear-in-parameters models → design recipes.

Dispatch is by callable identity — only the house models listed here take
the closed-form path; every other callable (``exp_linear``, ``poly2``,
custom models) keeps the iterative ``curve_fit`` path."""

_COV_WARNING = "Covariance of the parameters could not be estimated"
"""``curve_fit``-compatible OptimizeWarning message (singular design or
zero degrees of freedom) — kept identical so callers filtering on the
scipy warning keep working."""


def _wls_solve(
    a: FloatArray,
    y: FloatArray,
    sigma: FloatArray | None,
    absolute_sigma: bool,
) -> tuple[FloatArray, FloatArray]:
    """Solve one weighted linear least-squares system with covariance.

    Equation:
        ``p̂ = argmin_p ‖W^{1/2}(y − A·p)‖₂²  =  (AᵀWA)⁻¹AᵀWy``,
        ``C_p̂ = (AᵀWA)⁻¹``,  ``W = diag(1/σᵢ²)``

    computed from the SVD of the σ-whitened design ``A_w = W^{1/2}A =
    U·S·Vᵀ`` as ``p̂ = V·S⁻¹·Uᵀ(W^{1/2}y)`` and ``C_p̂ = V·S⁻²·Vᵀ``
    (Gauss–Markov / Aitken estimator). Unless ``absolute_sigma`` is True,
    C_p̂ is rescaled by the reduced chi-square
    ``s² = χ²/(N−P)``, ``χ² = ‖W^{1/2}(y − A·p̂)‖₂²`` — exactly the
    ``scipy.optimize.curve_fit`` covariance convention, preserved so the
    closed-form path is a drop-in replacement for the iterative one.

    Symbols → args:
        - ``A``  → ``a``: design matrix, shape (N, P) [basis units]
        - ``yᵢ`` → ``y``: observations, shape (N,) [L]
        - ``σᵢ`` → ``sigma``: 1-σ observation uncertainties, shape (N,)
          [L]; ``None`` ⇒ unit weights
        - ``absolute_sigma`` → ``absolute_sigma``: skip the reduced-χ²
          rescaling of C_p̂ (σ trusted as absolute 1-σ errors)

    Returns:
        ``(p̂, C_p̂)`` — parameters (P,) and covariance (P, P), float64.
        C_p̂ is filled with ``inf`` (and an
        ``scipy.optimize.OptimizeWarning`` is emitted, message identical
        to ``curve_fit``'s) when it cannot be estimated: rank(A_w) < P
        (p̂ is then the minimum-norm solution) or N ≤ P with
        ``absolute_sigma=False`` (no degrees of freedom for s²).

    Reference:
        WLS / Gauss–Markov: Aitken 1936, Proc. R. Soc. Edinb. 55;
        Strang & Borre 1997, *Linear Algebra, Geodesy and GPS*, ch. 9.
        Covariance propagation: JCGM 100:2008 (GUM) §5.1.2. Formal-σ
        caveat for temporally correlated GNSS noise: Williams 2003,
        J. Geodesy 76.

    Numerical notes:
        SVD (rank-safe, backward stable) rather than normal-equations
        Cholesky: forming AᵀWA squares the condition number, and the SVD
        gives rank + covariance from one factorization. Singular values
        below ``eps·max(N, P)·s_max`` are treated as zero (the
        ``numpy.linalg.lstsq`` default cutoff). Callers must pre-condition
        polynomial-in-t columns (see :func:`_fit_linear_design`).
    """
    if sigma is None:
        aw, yw = a, y
    else:
        aw = a / sigma[:, np.newaxis]
        yw = y / sigma
    n, p = aw.shape
    u, s, vt = np.linalg.svd(aw, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(n, p) * (float(s[0]) if s.size else 0.0)
    kept = s > cutoff
    rank = int(np.count_nonzero(kept))
    s_inv = np.zeros_like(s)
    s_inv[kept] = 1.0 / s[kept]
    params = np.asarray(vt.T @ (s_inv * (u.T @ yw)), dtype=np.float64)
    cov = np.asarray((vt.T * s_inv**2) @ vt, dtype=np.float64)
    if rank < p:
        cov = np.full((p, p), np.inf, dtype=np.float64)
        warnings.warn(_COV_WARNING, optimize.OptimizeWarning, stacklevel=2)
    elif not absolute_sigma:
        if n > p:
            residual = yw - aw @ params
            chisq = float(residual @ residual)
            cov = cov * (chisq / (n - p))
        else:
            cov = np.full((p, p), np.inf, dtype=np.float64)
            warnings.warn(_COV_WARNING, optimize.OptimizeWarning, stacklevel=2)
    return params, cov


def _fit_linear_design(
    design: _LinearDesign,
    t: FloatArray,
    y: FloatArray,
    sigma: FloatArray | None,
    absolute_sigma: bool,
) -> tuple[FloatArray, FloatArray]:
    """Fit one component of a linear-in-parameters model in closed form.

    Equation (conditioning + exact back-substitution):
        solve :func:`_wls_solve` on the centered design — the trend
        column t replaced by ``t′ = t − t_ref``, ``t_ref = mean(t)`` —
        giving p̂′ with ``x₀′ = x₀ + v·t_ref``; then map back

        ``p̂ = M·p̂′``, ``C_p̂ = M·C_p̂′·Mᵀ``,  M = I except
        ``M[i₀, i₁] = −t_ref``

        (i₀/i₁ the intercept/trend columns), so the returned parameters
        refer to **absolute t** exactly as the model callable defines
        them. The map is exact (linear reparameterization), so the
        centering changes conditioning only, not the estimate or its
        first-order (GUM) uncertainty.

    Symbols → args:
        - ``t``  → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``y``  → ``y``: observations, one component, shape (N,) [L]
        - ``σ``  → ``sigma``: 1-σ uncertainties, shape (N,) [L] or None
        - design → ``design``: the model's :class:`_LinearDesign`

    Returns:
        ``(p̂, C_p̂)`` in the model's absolute-t parameterization —
        drop-in for ``curve_fit``'s ``(popt, pcov)``.

    Reference:
        Strang & Borre 1997, ch. 9 (WLS); JCGM 100:2008 (GUM) §5.1.2
        (linear covariance propagation through M).

    Numerical notes:
        For absolute ``yearf`` (t ≈ 2×10³) the raw [1, t] columns are
        nearly collinear (condition number ~10⁷); centering makes them
        orthogonal-ish so the SVD solve is fully accurate. Seasonal trig
        columns stay on absolute t (bounded, well-conditioned; phase
        convention of :func:`gps_analysis.models.periodic` preserved —
        no coefficient rotation needed). The covariance transform is
        skipped when C_p̂′ is non-finite (singular design ⇒ ``inf`` fill,
        ``curve_fit`` convention).
    """
    a = design.build(t)
    t_ref = 0.0
    if design.trend_column is not None:
        t_ref = float(np.mean(t))
        a[:, design.trend_column] = t - t_ref
    params, cov = _wls_solve(a, y, sigma, absolute_sigma)
    if (
        design.trend_column is not None
        and design.intercept_column is not None
        and t_ref != 0.0
    ):
        i0, i1 = design.intercept_column, design.trend_column
        params = params.copy()
        params[i0] -= t_ref * params[i1]
        if np.all(np.isfinite(cov)):
            m = np.eye(params.size, dtype=np.float64)
            m[i0, i1] = -t_ref
            cov = m @ cov @ m.T
    return params, cov


def fit_components(
    model: ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    p0: ArrayLike | None = None,
    *,
    absolute_sigma: bool = False,
    maxfev: int | None = None,
    names: Sequence[str] | None = None,
) -> list[TrajectoryParams]:
    """Fit a trajectory model to each coordinate component by WLS.

    Equation (per component c):
        ``p̂_c = argmin_p Σᵢ [ (y_cᵢ − f(tᵢ; p)) / σ_cᵢ ]²``

    For the linear-in-parameters house models
    (:func:`~gps_analysis.models.linear`,
    :func:`~gps_analysis.models.periodic`,
    :func:`~gps_analysis.models.lineperiodic`) this is solved **in closed
    form** — ``p̂ = (AᵀWA)⁻¹AᵀWy`` via SVD of the σ-whitened design with a
    centered trend column (:func:`_fit_linear_design` / :func:`_wls_solve`);
    all other models go through ``scipy.optimize.curve_fit``. Both paths
    return the same parameter covariance ``C_p̂ = (JᵀWJ)⁻¹``
    (W = diag σ⁻²; J ≡ A for the linear models), rescaled by the reduced
    chi-square unless ``absolute_sigma=True``.

    Symbols → args:
        - ``tᵢ``   → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``y_cᵢ`` → ``y``: observations, component-major [L]
        - ``σ_cᵢ`` → ``sigma``: 1-σ observation uncertainties [L]
        - ``f``    → ``model``: trajectory model ``f(t, *p)`` from
          :mod:`gps_analysis.models`
        - ``p``    → ``p0``: initial parameter guess (units per model)

    Args:
        model: Model callable ``f(t, *params) -> ndarray``.
        t: Epochs, shape (N,) [yr]. Must be finite (filter NaNs first).
        y: Observations, shape (N,) or (C, N) [L]. Must be finite.
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; ``None`` for an
            unweighted fit.
        p0: Initial guess, shape (P,) for 1-D ``y`` or (C, P) for 2-D
            ``y``; ``None`` starts every parameter at 1.0 (``curve_fit``
            convention — supply a guess for nonlinear models such as
            :func:`~gps_analysis.models.exp_linear`). The closed-form
            linear path ignores the values (the WLS optimum is unique)
            but still validates the shape.
        absolute_sigma: If True, treat ``sigma`` as absolute 1-σ errors
            (no chi-square rescaling of the covariance). Legacy scripts
            used False.
        maxfev: Optional cap on function evaluations (legacy
            ``fit_curve`` used 100000); ``None`` keeps the scipy default.
            Irrelevant (ignored) on the closed-form linear path — the
            solve is non-iterative.
        names: Optional per-component labels stored on the results
            (e.g. ``("north", "east", "up")``).

    Returns:
        One :class:`~gps_analysis.models.TrajectoryParams` per component,
        in row order of ``y`` (a single-element list for 1-D ``y``).

    Raises:
        ValueError: On shape mismatches between ``t``/``y``/``sigma``/
            ``p0``/``names``, or non-finite ``t``/``y``.
        RuntimeError: Propagated from ``curve_fit`` when a nonlinear fit
            does not converge (cannot occur on the closed-form linear
            path).

    Reference:
        Levenberg–Marquardt: Moré 1978 (in *Numerical Analysis*, LNM 630);
        WLS estimator and covariance: Aitken 1936, Proc. R. Soc. Edinb.
        55; Strang & Borre 1997, *Linear Algebra, Geodesy and GPS*, ch. 9.
        Legacy sources: ``fittimes`` (``detrend_rnes.py``) and
        ``svartsengi_model.fitting.fit_curve``/``fit_three_components``.

    Numerical notes:
        Formal errors are white-noise-optimistic for GNSS daily solutions
        (Williams 2003). Both paths fill the covariance with ``inf`` (and
        emit ``scipy.optimize.OptimizeWarning``) when it cannot be
        estimated — singular design/Jacobian, or N ≤ P without
        ``absolute_sigma``. For absolute ``yearf`` epochs the
        intercept/rate columns are nearly collinear: the linear path
        re-centers the trend column internally (t → t − mean t) and maps
        the intercept and covariance back exactly
        (:func:`_fit_linear_design`), so callers get full accuracy at
        absolute t; for nonlinear models re-reference ``t`` yourself when
        conditioning matters.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy, was_1d = _components_2d(y, "y")
    if tt.ndim != 1 or yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    sigmas = _per_component_sigma(sigma, yy, was_1d)
    guesses = _per_component_p0(p0, yy.shape[0], was_1d)
    if names is not None and len(names) != yy.shape[0]:
        raise ValueError(f"names has {len(names)} entries for {yy.shape[0]} components")

    design = _LINEAR_DESIGNS.get(model)
    if design is not None:
        # Closed-form WLS fast path — mirror curve_fit's check_finite
        # guard (lstsq would silently propagate NaN otherwise).
        n_params = _n_model_params(model)
        if not np.all(np.isfinite(tt)) or not np.all(np.isfinite(yy)):
            raise ValueError("array must not contain infs or NaNs")
        for guess in guesses:
            if guess is not None and guess.size != n_params:
                raise ValueError(
                    f"p0 has {guess.size} parameters for a {n_params}-parameter model"
                )

    kwargs: dict[str, Any] = {"absolute_sigma": absolute_sigma}
    if maxfev is not None:
        kwargs["maxfev"] = maxfev

    fits: list[TrajectoryParams] = []
    for i in range(yy.shape[0]):
        if design is not None:
            popt, pcov = _fit_linear_design(
                design, tt, yy[i], sigmas[i], absolute_sigma
            )
        else:
            popt, pcov = optimize.curve_fit(
                model, tt, yy[i], p0=guesses[i], sigma=sigmas[i], **kwargs
            )
        fits.append(
            TrajectoryParams(
                params=np.asarray(popt, dtype=np.float64),
                covariance=np.asarray(pcov, dtype=np.float64),
                component=None if names is None else names[i],
            )
        )
    return fits


def remove_trend(
    model: ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    fits: TrajectoryParams | Sequence[TrajectoryParams],
) -> FloatArray:
    """Subtract a fitted trajectory model from the observations.

    Equation (per component c):
        ``r_cᵢ = y_cᵢ − f(tᵢ; p̂_c)``

    Symbols → args:
        - ``tᵢ``   → ``t``: epochs [yr]
        - ``y_cᵢ`` → ``y``: observations [L]
        - ``p̂_c`` → ``fits``: fitted parameters per component
        - ``f``    → ``model``: the model the parameters were fitted with

    Args:
        model: Model callable ``f(t, *params) -> ndarray`` — must be the
            same model used to produce ``fits``.
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        fits: One :class:`~gps_analysis.models.TrajectoryParams` (1-D
            ``y``) or a sequence of C of them in row order of ``y`` — as
            returned by :func:`fit_components`.

    Returns:
        Detrended series r [L], float64, a new array with the shape of
        ``y`` (input is not mutated — unlike the legacy ``detrend``,
        which overwrote ``y`` in place).

    Raises:
        ValueError: If the number of parameter sets does not match the
            number of components.

    Reference:
        Residual of the trajectory-model fit (Bevis & Brown 2014). Legacy
        source: the subtraction loop of ``detrend`` in
        ``detrend_rnes.py``.

    Numerical notes:
        Pure evaluation + subtraction; the residuals retain the
        observation noise and any unmodeled signal (steps, transients).
    """
    tt = np.asarray(t, dtype=np.float64)
    yy, was_1d = _components_2d(y, "y")
    fit_list = [fits] if isinstance(fits, TrajectoryParams) else list(fits)
    if len(fit_list) != yy.shape[0]:
        raise ValueError(
            f"got {len(fit_list)} parameter sets for {yy.shape[0]} components"
        )
    detrended = np.empty_like(yy)
    for i, fit in enumerate(fit_list):
        detrended[i] = yy[i] - np.asarray(model(tt, *fit.params), dtype=np.float64)
    return detrended[0] if was_1d else detrended


def detrend_fit(
    model: ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    p0: ArrayLike | None = None,
    *,
    absolute_sigma: bool = False,
    maxfev: int | None = None,
    names: Sequence[str] | None = None,
) -> tuple[FloatArray, list[TrajectoryParams]]:
    """Fit a trajectory model and return the detrended series with the fit.

    Composition (no new math):
        ``fits = fit_components(model, t, y, σ, p₀)`` then
        ``r = remove_trend(model, t, y, fits)`` — see those functions for
        the equations, symbol/unit mapping and references.

    Args:
        model: Model callable ``f(t, *params) -> ndarray``; the legacy
            default was :func:`~gps_analysis.models.lineperiodic`.
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; optional.
        p0: Initial guess, (P,) or (C, P); optional.
        absolute_sigma: Passed to :func:`fit_components`.
        maxfev: Passed to :func:`fit_components`.
        names: Passed to :func:`fit_components`.

    Returns:
        ``(detrended, fits)`` — the residual series r [L] (new array,
        shape of ``y``) and the per-component
        :class:`~gps_analysis.models.TrajectoryParams`.

    Reference:
        Legacy source: ``detrend`` in ``detrend_rnes.py``, minus its
        in-place mutation of ``y``, its function-body imports, and its
        ``getDetrFit``/``vshift`` I/O side paths (seeding ``p0`` from a
        prior fit file is the caller's business; re-referencing is
        :func:`gps_analysis.baseline.remove_offset`).

    Numerical notes:
        See :func:`fit_components`; the returned fit carries the trend
        that was removed, so the operation is exactly invertible via
        ``r + model(t, *fit.params)``.
    """
    fits = fit_components(
        model,
        t,
        y,
        sigma=sigma,
        p0=p0,
        absolute_sigma=absolute_sigma,
        maxfev=maxfev,
        names=names,
    )
    detrended = remove_trend(model, t, y, fits)
    return detrended, fits


@dataclasses.dataclass(frozen=True)
class OutlierRejection:
    """Result of :func:`reject_outliers` — inlier mask plus final WLS fits.

    Attributes:
        inliers: Boolean mask, shape of the input ``y`` (per-component) —
            True where the sample was kept.
        fits: Final per-component WLS fits on the inliers
            (:class:`~gps_analysis.models.TrajectoryParams`), row order
            of ``y``.
        n_iterations: Robust-fit/rejection sweeps actually performed per
            component (max over components).
    """

    inliers: NDArray[np.bool_]
    fits: list[TrajectoryParams]
    n_iterations: int


def _robust_params(
    model: ModelFunc,
    t: FloatArray,
    y: FloatArray,
    sigma: FloatArray | None,
    x0: FloatArray,
    loss: str,
    f_scale: float,
) -> FloatArray:
    """Solve the per-component M-estimator (robust loss) for p̂."""

    def residuals(p: FloatArray) -> FloatArray:
        r = y - np.asarray(model(t, *p), dtype=np.float64)
        return r if sigma is None else r / sigma

    result = optimize.least_squares(
        residuals, x0, loss=loss, f_scale=f_scale, method="trf"
    )
    return np.asarray(result.x, dtype=np.float64)


def reject_outliers(
    model: ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    p0: ArrayLike | None = None,
    *,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    n_sigma: float = 3.0,
    max_iterations: int = 5,
    absolute_sigma: bool = False,
    names: Sequence[str] | None = None,
) -> OutlierRejection:
    """Iteratively flag outliers against a robustly fitted trajectory model.

    Per component and per sweep:

    1. Robust M-estimate (``scipy.optimize.least_squares``):
       ``p̂ = argmin_p Σᵢ ρ( zᵢ(p) )``, whitened residuals
       ``zᵢ = (yᵢ − f(tᵢ; p)) / σᵢ`` (σᵢ ≡ 1 if ``sigma`` is None) with
       loss ρ — ``soft_l1``: ρ(z) = 2·f²·(√(1+(z/f)²) − 1); ``huber``:
       quadratic for |z| ≤ f, linear beyond (f = ``f_scale``).
    2. Robust scale: ``ŝ = 1.4826 · median(|z − median(z)|)`` over the
       current inliers (normalized MAD).
    3. Rejection: sample i is an outlier iff
       ``|zᵢ − median(z)| > n_sigma · ŝ`` — evaluated over *all* samples,
       so previously rejected points may re-enter.
    4. Repeat from 1 on the new inlier set until the mask is unchanged or
       ``max_iterations`` sweeps; finally refit the inliers by plain WLS
       (:func:`fit_components`) so the reported covariance is the
       standard Gauss–Markov one, not a robust-loss approximation.

    Symbols → args:
        - ``tᵢ``, ``yᵢ``, ``σᵢ`` → ``t``, ``y``, ``sigma`` ([yr], [L], [L])
        - ``ρ``, ``f`` → ``loss``, ``f_scale`` (f in whitened-residual
          units: σ-units when ``sigma`` is given, [L] otherwise)
        - ``n_sigma`` → ``n_sigma``: rejection threshold in robust-scale
          units [dimensionless]

    Args:
        model: Model callable ``f(t, *params) -> ndarray``.
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; optional.
        p0: Initial guess, (P,) or (C, P); ``None`` starts at 1.0 per
            parameter (nonlinear models need a real guess).
        loss: ``least_squares`` loss — ``"soft_l1"`` (default) or
            ``"huber"`` per the module plan; ``"cauchy"``/``"arctan"``
            are accepted too.
        f_scale: Soft inlier/outlier margin f of the loss.
        n_sigma: Rejection threshold in units of the normalized-MAD scale.
        max_iterations: Sweep cap (≥ 1); guards against mask oscillation.
        absolute_sigma: Passed to the final :func:`fit_components`.
        names: Optional component labels for the final fits.

    Returns:
        :class:`OutlierRejection` with the inlier mask (shape of ``y``),
        the final per-component WLS fits, and the sweep count.

    Raises:
        ValueError: On shape mismatches or ``max_iterations < 1``.

    Reference:
        M-estimation / huber loss: Huber 1964, Ann. Math. Statist. 35;
        soft_l1: Triggs et al. 2000 (*Bundle Adjustment — A Modern
        Synthesis*), as implemented in ``scipy.optimize.least_squares``;
        normalized MAD: Rousseeuw & Croux 1993, JASA 88. Net-new per
        ``docs/CONSOLIDATION_MAP.md`` (supersedes the legacy
        ``RunningMedian`` pre-filter of ``detrend_rnes.py``).

    Numerical notes:
        If ŝ = 0 (degenerate: ≥ 50 % of inlier residuals identical) the
        component stops rejecting rather than flagging everything.
        Rejection uses residuals about their median, so a constant model
        bias does not inflate the outlier set. The three-point minimum
        for a meaningful MAD is not enforced — very short series are the
        caller's responsibility.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    tt = np.asarray(t, dtype=np.float64)
    yy, was_1d = _components_2d(y, "y")
    if tt.ndim != 1 or yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    sigmas = _per_component_sigma(sigma, yy, was_1d)
    guesses = _per_component_p0(p0, yy.shape[0], was_1d)
    n_params = _n_model_params(model)

    inliers = np.ones(yy.shape, dtype=np.bool_)
    sweeps_used = 0
    for i in range(yy.shape[0]):
        params = (
            guesses[i]
            if guesses[i] is not None
            else np.ones(n_params, dtype=np.float64)
        )
        mask = inliers[i]
        for sweep in range(1, max_iterations + 1):
            ss = sigmas[i]
            params = _robust_params(
                model,
                tt[mask],
                yy[i][mask],
                None if ss is None else ss[mask],
                np.asarray(params, dtype=np.float64),
                loss,
                f_scale,
            )
            z = yy[i] - np.asarray(model(tt, *params), dtype=np.float64)
            if ss is not None:
                z = z / ss
            center = np.median(z[mask])
            scale = _MAD_TO_SIGMA * np.median(np.abs(z[mask] - center))
            sweeps_used = max(sweeps_used, sweep)
            if scale <= 0.0:
                break
            new_mask = np.abs(z - center) <= n_sigma * scale
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        inliers[i] = mask

    fits: list[TrajectoryParams] = []
    for i in range(yy.shape[0]):
        ss = sigmas[i]
        fit = fit_components(
            model,
            tt[inliers[i]],
            yy[i][inliers[i]],
            sigma=None if ss is None else ss[inliers[i]],
            p0=guesses[i],
            absolute_sigma=absolute_sigma,
            names=None if names is None else [names[i]],
        )
        fits.append(fit[0])

    return OutlierRejection(
        inliers=inliers[0] if was_1d else inliers,
        fits=fits,
        n_iterations=sweeps_used,
    )
