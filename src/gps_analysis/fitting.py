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

1. :func:`fit_components` solves the weighted nonlinear least-squares
   problem, per component,

       ``p̂ = argmin_p Σᵢ [ (yᵢ − f(tᵢ; p)) / σᵢ ]²``

   via ``scipy.optimize.curve_fit`` (Levenberg–Marquardt / TRF; Moré 1978)
   and packs (p̂, C_p̂) into :class:`~gps_analysis.models.TrajectoryParams`.
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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

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
            f"p0 must have shape (C, P) with C = {n_components}, "
            f"got shape {arr.shape}"
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

    solved by ``scipy.optimize.curve_fit``; the parameter covariance is
    ``C_p̂ = (JᵀWJ)⁻¹`` (W = diag σ⁻²), rescaled by the reduced chi-square
    unless ``absolute_sigma=True``.

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
            :func:`~gps_analysis.models.exp_linear`).
        absolute_sigma: If True, treat ``sigma`` as absolute 1-σ errors
            (no chi-square rescaling of the covariance). Legacy scripts
            used False.
        maxfev: Optional cap on function evaluations (legacy
            ``fit_curve`` used 100000); ``None`` keeps the scipy default.
        names: Optional per-component labels stored on the results
            (e.g. ``("north", "east", "up")``).

    Returns:
        One :class:`~gps_analysis.models.TrajectoryParams` per component,
        in row order of ``y`` (a single-element list for 1-D ``y``).

    Raises:
        ValueError: On shape mismatches between ``t``/``y``/``sigma``/
            ``p0``/``names``.
        RuntimeError: Propagated from ``curve_fit`` when the fit does not
            converge.

    Reference:
        Levenberg–Marquardt: Moré 1978 (in *Numerical Analysis*, LNM 630);
        WLS covariance: Strang & Borre 1997, *Linear Algebra, Geodesy and
        GPS*, ch. 9. Legacy sources: ``fittimes`` (``detrend_rnes.py``)
        and ``svartsengi_model.fitting.fit_curve``/``fit_three_components``.

    Numerical notes:
        Formal errors are white-noise-optimistic for GNSS daily solutions
        (Williams 2003). ``curve_fit`` fills the covariance with ``inf``
        when the Jacobian is singular at the solution. For absolute
        ``yearf`` epochs the intercept/rate columns of J are nearly
        collinear — re-reference ``t`` when conditioning matters.
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

    kwargs: dict[str, Any] = {"absolute_sigma": absolute_sigma}
    if maxfev is not None:
        kwargs["maxfev"] = maxfev

    fits: list[TrajectoryParams] = []
    for i in range(yy.shape[0]):
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
