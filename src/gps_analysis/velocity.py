"""Secular-velocity estimation for GNSS coordinate time series.

Fixed-window and sliding-window **weighted least-squares (WLS)** velocities
with formal 1-σ uncertainties, plus the horizontal magnitude/azimuth
products served by the API (plan §10.5: velocity vectors as GeoJSON
features with per-component rate/σ, magnitude, azimuth and ``method``
props). The estimator is the one every legacy ``detrend-*`` script
re-implements: the rate term of a :func:`gps_analysis.models.lineperiodic`
trajectory fit.

Derivation chain
----------------
Given epochs t ∈ ℝᴺ (fractional years, ``yearf``), per-component
observations y ([L], caller's unit — mm in IMO production) with 1-σ
uncertainties σ ([L]), and a trajectory model f(t; p) whose **second
parameter p₁ is the secular rate** (:func:`~gps_analysis.models.linear`,
:func:`~gps_analysis.models.lineperiodic`, or any compatible callable):

1. **Windowing** — :func:`gps_analysis.baseline.slice_window` masks the
   samples inside [t_start, t_end] (``±tol`` boundary tolerance).
2. **Conditioning** — epochs are re-referenced, t′ = t − t_ref with t_ref
   the mean windowed epoch, so the intercept and rate columns of the
   design matrix are (near-)orthogonal instead of collinear at
   t ≈ 2×10³ yr. A time translation leaves the rate p₁ and its variance
   invariant; only the intercept and the seasonal phase parameters change.
3. **WLS trajectory fit** — :func:`gps_analysis.fitting.fit_components`
   solves, per component,

       ``p̂ = argmin_p Σᵢ [ (yᵢ − f(t′ᵢ; p)) / σᵢ ]²``

   with parameter covariance ``Ĉ = (JᵀWJ)⁻¹``, ``W = diag(1/σᵢ²)``
   (Aitken 1936; Strang & Borre 1997 ch. 9), J the model Jacobian —
   equal to the design matrix A for the linear-in-parameters models here.
4. **Rate extraction** — the secular velocity is ``v̂ = p̂₁`` [L/yr] and
   its formal 1-σ uncertainty ``σ_v = √Ĉ₁₁`` (the ``params[1]`` /
   ``uncertainties[1]`` slots of
   :class:`~gps_analysis.models.TrajectoryParams`).
5. **Horizontal products** — from east/north rates:
   ``|v_h| = √(v_E² + v_N²)`` and ``α = atan2(v_E, v_N)`` degrees
   clockwise from geographic north, with first-order (delta-method)
   σ propagation (:func:`horizontal_magnitude`, :func:`horizontal_azimuth`
   and their ``*_sigma`` companions).
6. **Sliding windows** — :func:`sliding_velocity` repeats 1–4 on windows
   of fixed length stepped along the series, yielding a dynamic-velocity
   series (rate + formal σ per window centre).

Method provenance (binding, ``docs/MATH_STANDARDS.md`` §6)
----------------------------------------------------------
Every result carries ``method="wls"``. The white-noise formal σ_v is
**optimistic** for real GNSS daily solutions — temporally correlated
(flicker/random-walk) noise inflates true rate uncertainty by factors of
several (Williams 2003, J. Geodesy 76, eqs. 23–30). Honest colored-noise
uncertainties arrive with the GBIS4TS lane (:mod:`gps_analysis.transient`,
plan §10.7) and will carry ``method="gbis"``; the API contract
distinguishes the two so consumers never mistake a WLS σ for an honest one.
The per-station :func:`detectability_floor` (velocity-change alarm
threshold) depends on that noise model and is a documented stub here.

All functions are pure: float64 arithmetic, no I/O, inputs never mutated,
units the caller's business ([L] below; velocity [L/yr], azimuth degrees,
time ``yearf``).
"""

import dataclasses
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from . import models
from .baseline import slice_window
from .fitting import (
    ModelFunc,
    _components_2d,
    _n_model_params,
    _per_component_p0,
    _per_component_sigma,
    fit_components,
)
from .models import FloatArray, TrajectoryParams

__all__ = [
    "SlidingVelocity",
    "VelocityEstimate",
    "detectability_floor",
    "estimate_velocity",
    "horizontal_azimuth",
    "horizontal_azimuth_sigma",
    "horizontal_magnitude",
    "horizontal_magnitude_sigma",
    "sliding_velocity",
]

#: Window boundary tolerance [yr] — the ``slice_window`` legacy default
#: (0.001 yr ≈ 8.77 h, keeps a daily solution stamped exactly at a bound).
_DEFAULT_TOL = 1e-3

_RATE_INDEX = 1
"""Parameter slot of the secular rate: ``params[1]`` by the house model
convention (``models.linear`` / ``models.lineperiodic`` positional order)."""

_METHOD_WLS = "wls"
"""Method tag of this estimator (API contract, plan §10.5 /
MATH_STANDARDS §6); the GBIS4TS upgrade will tag ``"gbis"``."""

_NAMED_MODELS: dict[str, ModelFunc] = {
    "linear": models.linear,
    "lineperiodic": models.lineperiodic,
}


def _resolve_model(model: str | ModelFunc) -> ModelFunc:
    """Resolve a model name or callable to a trajectory-model function."""
    if callable(model):
        return model
    try:
        return _NAMED_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; named models: {sorted(_NAMED_MODELS)}"
        ) from None


def _rate_param_count(model_func: ModelFunc) -> int:
    """Parameter count P of the model, requiring the rate slot to exist."""
    n_params = _n_model_params(model_func)
    if n_params < _RATE_INDEX + 1:
        raise ValueError(
            "model must have at least 2 parameters - params[1] is the secular rate"
        )
    return n_params


def horizontal_magnitude(v_east: ArrayLike, v_north: ArrayLike) -> FloatArray:
    """Compute the horizontal velocity magnitude |v_h|.

    Equation:
        ``|v_h| = √(v_E² + v_N²)``

    Symbols → args:
        - ``v_E`` → ``v_east``: east velocity component [L/yr]
        - ``v_N`` → ``v_north``: north velocity component [L/yr]

    Args:
        v_east: East velocity component(s) [L/yr].
        v_north: North velocity component(s) [L/yr], broadcastable
            against ``v_east``.

    Returns:
        Horizontal magnitude |v_h| [L/yr], float64, broadcast shape
        (0-d for scalar inputs).

    Reference:
        Euclidean norm of the horizontal velocity vector — the
        ``magnitude`` property of the API velocity feature (plan §10.5).

    Numerical notes:
        Implemented with ``np.hypot`` — no intermediate overflow/underflow
        for extreme components (unlike a naive ``√(e²+n²)``).
    """
    ve = np.asarray(v_east, dtype=np.float64)
    vn = np.asarray(v_north, dtype=np.float64)
    return np.asarray(np.hypot(ve, vn), dtype=np.float64)


def horizontal_azimuth(v_east: ArrayLike, v_north: ArrayLike) -> FloatArray:
    """Compute the horizontal velocity azimuth α, degrees CW from north.

    Equation:
        ``α = atan2(v_E, v_N) · 180/π  (mod 360)``   →  α ∈ [0, 360)

    Symbols → args:
        - ``v_E`` → ``v_east``: east velocity component [L/yr]
        - ``v_N`` → ``v_north``: north velocity component [L/yr]

    Args:
        v_east: East velocity component(s) [L/yr].
        v_north: North velocity component(s) [L/yr], broadcastable.

    Returns:
        Azimuth α [degrees], clockwise from geographic north, wrapped to
        [0, 360): north 0°, east 90°, south 180°, west 270°. Float64,
        broadcast shape (0-d for scalars).

    Reference:
        Geodetic azimuth convention (clockwise from geographic north) —
        note the swapped ``atan2`` argument order versus the mathematical
        counterclockwise-from-x convention. The ``azimuth`` property of
        the API velocity feature (plan §10.5).

    Numerical notes:
        ``atan2`` handles all quadrants and axis cases exactly; the
        modulo wrap maps the (−180, 180] range onto [0, 360). Azimuth is
        undefined for a zero vector — ``atan2(0, 0) = 0`` is returned by
        IEEE convention; check :func:`horizontal_magnitude` (or the NaN
        from :func:`horizontal_azimuth_sigma`) before trusting it.
    """
    ve = np.asarray(v_east, dtype=np.float64)
    vn = np.asarray(v_north, dtype=np.float64)
    az = np.degrees(np.arctan2(ve, vn))
    return np.asarray(np.mod(az, 360.0), dtype=np.float64)


def horizontal_magnitude_sigma(
    v_east: ArrayLike,
    v_north: ArrayLike,
    sigma_east: ArrayLike,
    sigma_north: ArrayLike,
) -> FloatArray:
    """Propagate component σ to the horizontal magnitude, first order.

    Equation:
        ``σ_|v| = √(v_E²·σ_E² + v_N²·σ_N²) / |v_h|``,
        ``|v_h| = √(v_E² + v_N²)``

    (delta method on |v_h|, assuming **zero east–north covariance** — the
    components are fitted independently by
    :func:`gps_analysis.fitting.fit_components`, and any inter-component
    correlation from the GAMIT processing is not available at this level.)

    Symbols → args:
        - ``v_E``, ``v_N`` → ``v_east``, ``v_north``: velocity
          components [L/yr]
        - ``σ_E``, ``σ_N`` → ``sigma_east``, ``sigma_north``: their 1-σ
          uncertainties [L/yr]

    Args:
        v_east: East velocity component(s) [L/yr].
        v_north: North velocity component(s) [L/yr].
        sigma_east: 1-σ uncertainty of ``v_east`` [L/yr].
        sigma_north: 1-σ uncertainty of ``v_north`` [L/yr].

    Returns:
        1-σ magnitude uncertainty σ_|v| [L/yr], float64, broadcast shape;
        **NaN where |v_h| = 0** (see notes).

    Reference:
        First-order uncertainty propagation: JCGM 100:2008 (GUM), §5.1.2.

    Numerical notes:
        The linearization breaks down as |v_h| → 0 (|v_h| is then
        Rayleigh-distributed, not Gaussian) — NaN is returned there
        rather than a misleading number. The vector-component σ are the
        primary uncertainties; σ_|v| is a derived convenience. For
        σ_E = σ_N = σ the expression collapses to σ_|v| = σ exactly.
    """
    ve = np.asarray(v_east, dtype=np.float64)
    vn = np.asarray(v_north, dtype=np.float64)
    se = np.asarray(sigma_east, dtype=np.float64)
    sn = np.asarray(sigma_north, dtype=np.float64)
    mag = np.hypot(ve, vn)
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.sqrt(ve**2 * se**2 + vn**2 * sn**2) / mag
    return np.asarray(np.where(mag > 0.0, prop, np.nan), dtype=np.float64)


def horizontal_azimuth_sigma(
    v_east: ArrayLike,
    v_north: ArrayLike,
    sigma_east: ArrayLike,
    sigma_north: ArrayLike,
) -> FloatArray:
    """Propagate component σ to the horizontal azimuth, first order.

    Equation:
        ``σ_α = √(v_N²·σ_E² + v_E²·σ_N²) / |v_h|² · 180/π``,
        ``|v_h|² = v_E² + v_N²``

    (delta method on α = atan2(v_E, v_N): ∂α/∂v_E = v_N/|v_h|²,
    ∂α/∂v_N = −v_E/|v_h|²; zero east–north covariance assumed as in
    :func:`horizontal_magnitude_sigma`.)

    Symbols → args:
        - ``v_E``, ``v_N`` → ``v_east``, ``v_north``: velocity
          components [L/yr]
        - ``σ_E``, ``σ_N`` → ``sigma_east``, ``sigma_north``: their 1-σ
          uncertainties [L/yr]

    Args:
        v_east: East velocity component(s) [L/yr].
        v_north: North velocity component(s) [L/yr].
        sigma_east: 1-σ uncertainty of ``v_east`` [L/yr].
        sigma_north: 1-σ uncertainty of ``v_north`` [L/yr].

    Returns:
        1-σ azimuth uncertainty σ_α [degrees], float64, broadcast shape;
        **NaN where |v_h| = 0** (azimuth undefined).

    Reference:
        First-order uncertainty propagation: JCGM 100:2008 (GUM), §5.1.2.

    Numerical notes:
        Valid only while σ_α is small (≲ tens of degrees); as
        |v_h| → σ the azimuth distribution wraps and the delta method
        fails — for near-zero velocities report the component σ instead.
    """
    ve = np.asarray(v_east, dtype=np.float64)
    vn = np.asarray(v_north, dtype=np.float64)
    se = np.asarray(sigma_east, dtype=np.float64)
    sn = np.asarray(sigma_north, dtype=np.float64)
    mag_sq = ve**2 + vn**2
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.degrees(np.sqrt(vn**2 * se**2 + ve**2 * sn**2) / mag_sq)
    return np.asarray(np.where(mag_sq > 0.0, prop, np.nan), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class VelocityEstimate:
    """Fixed-window WLS secular velocity for one or more components.

    Result of :func:`estimate_velocity` — the per-component rates v̂ = p̂₁
    with their formal 1-σ errors σ_v = √Ĉ₁₁, the full trajectory fits they
    came from, and (when north/east components are labelled) the derived
    horizontal magnitude/azimuth.

    Maps directly onto the API velocity GeoJSON feature (plan §10.5):
    per-component rate/σ props (``east``/``north``/``up`` selected by the
    ``components`` labels), ``magnitude``, ``azimuth``, and ``method``
    provenance (MATH_STANDARDS §6).

    Attributes:
        rates: Secular rates v̂, shape (C,), float64 [L/yr] — one entry
            per component row of the input ``y`` (C = 1 for 1-D input,
            mirroring :func:`~gps_analysis.fitting.fit_components`).
        sigmas: Formal 1-σ rate uncertainties σ_v, shape (C,) [L/yr].
            White-noise formal errors — optimistic for correlated GNSS
            noise (Williams 2003); see ``method``.
        fits: Full per-component
            :class:`~gps_analysis.models.TrajectoryParams` in row order.
            **Fitted in re-referenced time** t′ = t − ``t_ref``: the
            intercept and seasonal-phase parameters refer to t′, the
            rate is translation-invariant.
        components: Component labels as passed via ``names`` (or None).
        n_obs: Number of observations inside the window.
        t_ref: Reference epoch subtracted before fitting [yr] (the mean
            windowed epoch).
        span: (first, last) epoch actually used [yr] — fit provenance.
        method: Estimator tag — ``"wls"`` here; ``"gbis"`` when the
            colored-noise estimator supersedes it (API contract §10.5).
        magnitude: Horizontal magnitude |v_h| [L/yr]
            (:func:`horizontal_magnitude`), or None when no unique
            "north"+"east" labels were given.
        azimuth: Horizontal azimuth α [degrees CW from north, 0–360)
            (:func:`horizontal_azimuth`), or None as above.
        magnitude_sigma: Delta-method σ_|v| [L/yr]
            (:func:`horizontal_magnitude_sigma`; NaN at |v_h| = 0), or None.
        azimuth_sigma: Delta-method σ_α [degrees]
            (:func:`horizontal_azimuth_sigma`; NaN at |v_h| = 0), or None.

    Numerical notes:
        Arrays are coerced to float64 and shape-validated at construction;
        the dataclass is frozen but ndarrays are not immutable — treat the
        contents as read-only.
    """

    rates: FloatArray
    sigmas: FloatArray
    fits: tuple[TrajectoryParams, ...]
    components: tuple[str, ...] | None
    n_obs: int
    t_ref: float
    span: tuple[float, float]
    method: str = _METHOD_WLS
    magnitude: float | None = None
    azimuth: float | None = None
    magnitude_sigma: float | None = None
    azimuth_sigma: float | None = None

    def __post_init__(self) -> None:
        rates = np.asarray(self.rates, dtype=np.float64)
        sigmas = np.asarray(self.sigmas, dtype=np.float64)
        if rates.ndim != 1:
            raise ValueError(f"rates must be 1-D, got shape {rates.shape}")
        if sigmas.shape != rates.shape:
            raise ValueError(
                f"sigmas shape {sigmas.shape} does not match rates {rates.shape}"
            )
        if len(self.fits) != rates.size:
            raise ValueError(f"got {len(self.fits)} fits for {rates.size} components")
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "sigmas", sigmas)


@dataclasses.dataclass(frozen=True)
class SlidingVelocity:
    """Sliding-window dynamic velocity series (rate + formal σ per centre).

    Result of :func:`sliding_velocity`. Windows with too few observations
    or a failed/degenerate fit carry NaN rate and σ — the epoch grid stays
    regular so gaps are visible, not silently dropped.

    Attributes:
        centers: Window centre epochs, shape (K,), float64 [yr].
        rates: Secular rates v̂ per component and window, shape (C, K),
            float64 [L/yr] (C = 1 for 1-D input ``y``); NaN for skipped
            windows.
        sigmas: Formal 1-σ rate uncertainties, shape (C, K) [L/yr]; NaN
            for skipped windows (white-noise formal errors — Williams 2003
            caveat applies as in :class:`VelocityEstimate`).
        counts: Observations inside each window, shape (K,), int64 —
            recorded even for skipped windows.
        window_years: Window length T [yr].
        step_years: Centre-to-centre step [yr].
        components: Component labels as passed via ``names`` (or None).
        method: Estimator tag — ``"wls"`` (see :class:`VelocityEstimate`).

    Numerical notes:
        Arrays coerced to float64/int64 and shape-validated at
        construction; frozen dataclass, contents read-only by convention.
    """

    centers: FloatArray
    rates: FloatArray
    sigmas: FloatArray
    counts: NDArray[np.int64]
    window_years: float
    step_years: float
    components: tuple[str, ...] | None
    method: str = _METHOD_WLS

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        rates = np.asarray(self.rates, dtype=np.float64)
        sigmas = np.asarray(self.sigmas, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.int64)
        if centers.ndim != 1:
            raise ValueError(f"centers must be 1-D, got shape {centers.shape}")
        if rates.ndim != 2 or rates.shape[1] != centers.size:
            raise ValueError(
                f"rates must have shape (C, {centers.size}), got {rates.shape}"
            )
        if sigmas.shape != rates.shape:
            raise ValueError(
                f"sigmas shape {sigmas.shape} does not match rates {rates.shape}"
            )
        if counts.shape != centers.shape:
            raise ValueError(
                f"counts shape {counts.shape} does not match centers {centers.shape}"
            )
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "counts", counts)


def estimate_velocity(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    model: str | ModelFunc = "lineperiodic",
    window: tuple[float | None, float | None] | None = None,
    tol: float = _DEFAULT_TOL,
    p0: ArrayLike | None = None,
    names: Sequence[str] | None = None,
    absolute_sigma: bool = False,
) -> VelocityEstimate:
    """Estimate the fixed-window WLS secular velocity with its formal σ.

    Equation (per component, over the windowed samples):
        ``p̂ = argmin_p Σᵢ [ (yᵢ − f(tᵢ − t_ref; p)) / σᵢ ]²``,
        ``Ĉ = (JᵀWJ)⁻¹``, ``W = diag(1/σᵢ²)``  →
        ``v̂ = p̂₁`` [L/yr], ``σ_v = √Ĉ₁₁``

    solved via :func:`gps_analysis.fitting.fit_components`
    (``scipy.optimize.curve_fit``); the rate v̂ and its 1-σ error come
    straight from the returned
    :class:`~gps_analysis.models.TrajectoryParams` — ``params[1]`` /
    ``uncertainties[1]``. When ``names`` contains exactly one ``"north"``
    and one ``"east"`` (case-insensitive), the horizontal magnitude,
    azimuth and their delta-method σ are computed from those two rates.

    Symbols → args:
        - ``tᵢ``    → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``yᵢ``    → ``y``: observations, component-major [L]
        - ``σᵢ``    → ``sigma``: 1-σ observation uncertainties [L]
        - ``f``     → ``model``: trajectory model with ``params[1]`` =
          secular rate — ``"lineperiodic"`` (default), ``"linear"``, or a
          compatible callable ``f(t, *p)``
        - window    → ``window``/``tol``: [t_start, t_end] ± δ [yr]
          (see :func:`gps_analysis.baseline.slice_window`)
        - ``t_ref`` → internal: mean windowed epoch (returned on the
          result) [yr]

    Args:
        t: Epochs, shape (N,) [yr]. Must be finite (filter NaNs first).
        y: Observations, shape (N,) or (C, N) [L]. Must be finite.
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; ``None`` for an
            unweighted fit.
        model: Named model (``"lineperiodic"``/``"linear"``) or a callable
            ``f(t, *params)`` whose second parameter is the secular rate.
        window: Optional (start, end) window [yr]; either bound may be
            ``None`` (open). ``None`` uses the whole series.
        tol: Window boundary tolerance δ [yr] (legacy 0.001 yr default).
        p0: Initial guess, shape (P,) or (C, P); ``None`` starts every
            parameter at 1.0 (``curve_fit`` convention, as in
            :func:`~gps_analysis.fitting.fit_components`) — always
            sufficient for the linear-in-parameters named models;
            nonlinear custom models need a real guess. Avoid seeding
            parameters at denormal-tiny nonzero values: MINPACK's
            relative finite-difference step ``h ∝ |p₀ⱼ|`` collapses
            there and stalls the fit.
        names: Optional per-component labels (e.g. ``("north", "east",
            "up")``) — stored on the result, forwarded to the fits, and
            used to locate the horizontal components.
        absolute_sigma: If True, treat ``sigma`` as absolute 1-σ errors
            (no reduced-chi-square rescaling of Ĉ — pass True when the
            input σ are trusted). Default False matches
            :func:`~gps_analysis.fitting.fit_components` and the legacy
            scripts; with ``sigma=None`` the χ²-rescaled covariance is
            the only meaningful one.

    Returns:
        :class:`VelocityEstimate` — rates/σ per component (shape (C,);
        C = 1 for 1-D ``y``), the full fits, window provenance, the
        ``method="wls"`` tag, and horizontal products when derivable.

    Raises:
        ValueError: On non-finite ``t``, shape mismatches, an unknown
            model name, a model without a rate parameter, or a window
            with fewer than P + 1 samples (no degrees of freedom).
        RuntimeError: Propagated from ``curve_fit`` when the fit does not
            converge.

    Reference:
        WLS / Gauss–Markov covariance: Aitken 1936, Proc. R. Soc. Edinb.
        55; Strang & Borre 1997, *Linear Algebra, Geodesy and GPS*, ch. 9.
        Seasonal co-estimation requirement: Blewitt & Lavallée 2002, JGR
        107(B7) — for windows ≲ 4.5 yr fit ``lineperiodic``, not
        ``linear``, or the annual signal biases v̂. Formal-σ caveat:
        Williams 2003, J. Geodesy 76 (see module docstring). Legacy
        sources: the rate extraction of the ``detrend-*`` family
        (``detrend_rnes.py``).

    Numerical notes:
        Epochs are re-referenced to t_ref (mean windowed epoch) before
        fitting, decorrelating intercept and rate (absolute ``yearf``
        makes those Jacobian columns nearly collinear); the rate and σ_v
        are invariant under this translation, the returned fit parameters
        refer to t − t_ref. Ĉ comes from ``curve_fit``'s SVD-based
        pseudo-inverse of JᵀWJ — no explicit matrix inverse is formed
        here. σ_v is ``inf`` if the Jacobian is singular at the solution
        (``curve_fit`` convention).
    """
    model_func = _resolve_model(model)
    n_params = _rate_param_count(model_func)

    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must be finite")
    yy, was_1d = _components_2d(y, "y")
    if yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    sigma_rows = _per_component_sigma(sigma, yy, was_1d)
    p0_rows = _per_component_p0(p0, yy.shape[0], was_1d)
    if names is not None and len(names) != yy.shape[0]:
        raise ValueError(f"names has {len(names)} entries for {yy.shape[0]} components")

    if window is None:
        mask = np.ones(tt.shape, dtype=np.bool_)
    else:
        mask = slice_window(tt, window[0], window[1], tol=tol)
    n_obs = int(np.count_nonzero(mask))
    if n_obs < n_params + 1:
        raise ValueError(
            f"window has {n_obs} samples for {n_params} parameters - "
            f"need at least {n_params + 1}"
        )

    t_win = tt[mask]
    t_ref = float(np.mean(t_win))
    t_local = t_win - t_ref

    fits: list[TrajectoryParams] = []
    for i in range(yy.shape[0]):
        y_i = yy[i][mask]
        s_i = sigma_rows[i]
        guess = p0_rows[i]
        (fit,) = fit_components(
            model_func,
            t_local,
            y_i,
            sigma=None if s_i is None else s_i[mask],
            p0=guess,
            absolute_sigma=absolute_sigma,
            names=None if names is None else [names[i]],
        )
        fits.append(fit)

    rates = np.asarray([f.params[_RATE_INDEX] for f in fits], dtype=np.float64)
    sigmas = np.asarray([f.uncertainties[_RATE_INDEX] for f in fits], dtype=np.float64)

    magnitude: float | None = None
    azimuth: float | None = None
    magnitude_sigma: float | None = None
    azimuth_sigma: float | None = None
    if names is not None:
        lowered = [name.lower() for name in names]
        if lowered.count("north") == 1 and lowered.count("east") == 1:
            i_n, i_e = lowered.index("north"), lowered.index("east")
            v_e, v_n = rates[i_e], rates[i_n]
            s_e, s_n = sigmas[i_e], sigmas[i_n]
            magnitude = float(horizontal_magnitude(v_e, v_n))
            azimuth = float(horizontal_azimuth(v_e, v_n))
            magnitude_sigma = float(horizontal_magnitude_sigma(v_e, v_n, s_e, s_n))
            azimuth_sigma = float(horizontal_azimuth_sigma(v_e, v_n, s_e, s_n))

    return VelocityEstimate(
        rates=rates,
        sigmas=sigmas,
        fits=tuple(fits),
        components=None if names is None else tuple(names),
        n_obs=n_obs,
        t_ref=t_ref,
        span=(float(np.min(t_win)), float(np.max(t_win))),
        method=_METHOD_WLS,
        magnitude=magnitude,
        azimuth=azimuth,
        magnitude_sigma=magnitude_sigma,
        azimuth_sigma=azimuth_sigma,
    )


def sliding_velocity(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    window_years: float,
    step_years: float,
    model: str | ModelFunc = "lineperiodic",
    min_obs: int | None = None,
    tol: float = _DEFAULT_TOL,
    p0: ArrayLike | None = None,
    names: Sequence[str] | None = None,
    absolute_sigma: bool = False,
) -> SlidingVelocity:
    """Estimate a sliding-window dynamic velocity series (WLS per window).

    Equation:
        window centres ``c_k = t_min + T/2 + k·Δ``,
        ``k = 0 … K−1``, ``K = 1 + ⌊(t_max − t_min − T)/Δ⌋``;
        per centre, over the samples with |tᵢ − c_k| ≤ T/2 (± ``tol``):

        ``v̂_k = p̂₁``, ``σ_v,k = √Ĉ₁₁``

    — the fixed-window WLS estimator of :func:`estimate_velocity` applied
    to each window (:func:`gps_analysis.baseline.slice_window` masks,
    epochs re-referenced to the window-mean epoch, fit via
    :func:`gps_analysis.fitting.fit_components`).

    Symbols → args:
        - ``tᵢ``, ``yᵢ``, ``σᵢ`` → ``t``, ``y``, ``sigma``
          ([yr], [L], [L]) — as in :func:`estimate_velocity`
        - ``T`` → ``window_years``: window length [yr]
        - ``Δ`` → ``step_years``: centre-to-centre step [yr]
        - ``f`` → ``model``: trajectory model, ``params[1]`` = rate

    Gap / degeneracy policy (documented behavior, not an error):
        a window is **skipped** — NaN rate and σ, count still recorded —
        when it holds fewer than ``min_obs`` samples, when the fit does
        not converge (``curve_fit`` ``RuntimeError``), or when its
        covariance is not estimable (singular Jacobian,
        ``OptimizeWarning``). The centre grid stays regular so data gaps
        appear as NaN runs rather than silently shifting epochs.

    Args:
        t: Epochs, shape (N,) [yr]; finite, need not be sorted.
        y: Observations, shape (N,) or (C, N) [L]. Must be finite.
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; optional.
        window_years: Window length T [yr]. **Trade-off** (Blewitt &
            Lavallée 2002, JGR 107(B7)): short windows resolve rate
            changes but alias the annual signal into v̂ — with the default
            ``lineperiodic`` model keep T ≳ 2.5 yr so the seasonal terms
            are separable from the rate; with ``model="linear"`` use
            integer-year T so the annual signal averages out.
        step_years: Centre step Δ [yr], > 0. Δ < T gives overlapping
            windows — successive estimates are then strongly correlated
            (they share samples); treat the series as a smoothed rate
            history, not K independent measurements.
        model: Named model or callable, as in :func:`estimate_velocity`.
        min_obs: Minimum samples per window; default ``2·P`` (P model
            parameters) so the fit keeps ≥ P degrees of freedom for the
            χ²-rescaled covariance. Must be ≥ P + 1.
        tol: Window boundary tolerance δ [yr].
        p0: Initial guess, (P,) or (C, P), used for every window;
            ``None`` starts every parameter at 1.0 (``curve_fit``
            convention — see :func:`estimate_velocity`).
        names: Optional per-component labels, stored on the result.
        absolute_sigma: Passed through to the per-window fits (see
            :func:`estimate_velocity`).

    Returns:
        :class:`SlidingVelocity` — centres (K,), rates/σ (C, K) with NaN
        at skipped windows, per-window counts, the window geometry, and
        the ``method="wls"`` tag.

    Raises:
        ValueError: On non-finite ``t``, shape mismatches, non-positive
            ``window_years``/``step_years``, ``min_obs < P + 1``, an
            unknown model, or ``window_years`` exceeding the data span
            (no window fits).

    Reference:
        Window-length vs seasonal-aliasing trade-off: Blewitt & Lavallée
        2002, JGR 107(B7). WLS estimator and formal-σ caveat: see
        :func:`estimate_velocity` (Strang & Borre 1997 ch. 9;
        Williams 2003). Sliding-window dynamic velocity: plan §10.2 /
        analysis-lane task H5 (net-new — no ``gps_data_analyses``
        ancestor).

    Numerical notes:
        Each window is fitted independently in window-local time
        (t − mean windowed epoch) — the conditioning argument of
        :func:`estimate_velocity` applies per window. Components are
        fitted one at a time so a failure in one component NaNs only that
        (component, window) cell. The window count K uses a 10⁻⁹ yr guard
        against float truncation at exact multiples of ``step_years``.
        NaN σ (skipped) is distinct from ``inf`` σ — the latter cannot
        occur here because singular-covariance windows are demoted to NaN.
    """
    model_func = _resolve_model(model)
    n_params = _rate_param_count(model_func)
    if window_years <= 0.0:
        raise ValueError(f"window_years must be > 0, got {window_years}")
    if step_years <= 0.0:
        raise ValueError(f"step_years must be > 0, got {step_years}")
    if min_obs is None:
        min_obs = 2 * n_params
    if min_obs < n_params + 1:
        raise ValueError(
            f"min_obs must be >= {n_params + 1} (P + 1 for {n_params} "
            f"parameters), got {min_obs}"
        )

    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must be finite")
    yy, was_1d = _components_2d(y, "y")
    if yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    sigma_rows = _per_component_sigma(sigma, yy, was_1d)
    p0_rows = _per_component_p0(p0, yy.shape[0], was_1d)
    if names is not None and len(names) != yy.shape[0]:
        raise ValueError(f"names has {len(names)} entries for {yy.shape[0]} components")

    t_min, t_max = float(np.min(tt)), float(np.max(tt))
    span = t_max - t_min
    if span + tol < window_years:
        raise ValueError(
            f"window_years = {window_years} exceeds the data span {span:.6g} yr"
        )
    n_windows = 1 + int(np.floor(max(span - window_years, 0.0) / step_years + 1e-9))
    centers = (
        t_min + window_years / 2.0 + step_years * np.arange(n_windows, dtype=np.float64)
    )

    n_components = yy.shape[0]
    half = window_years / 2.0
    rate_series = np.full((n_components, n_windows), np.nan, dtype=np.float64)
    sigma_series = np.full((n_components, n_windows), np.nan, dtype=np.float64)
    counts = np.zeros(n_windows, dtype=np.int64)

    for k, center in enumerate(centers):
        mask = slice_window(tt, center - half, center + half, tol=tol)
        count = int(np.count_nonzero(mask))
        counts[k] = count
        if count < min_obs:
            continue
        t_local = tt[mask] - float(np.mean(tt[mask]))
        for i in range(n_components):
            y_i = yy[i][mask]
            s_i = sigma_rows[i]
            guess = p0_rows[i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", optimize.OptimizeWarning)
                    (fit,) = fit_components(
                        model_func,
                        t_local,
                        y_i,
                        sigma=None if s_i is None else s_i[mask],
                        p0=guess,
                        absolute_sigma=absolute_sigma,
                    )
            except (RuntimeError, optimize.OptimizeWarning):
                continue  # window stays NaN for this component
            rate_series[i, k] = fit.params[_RATE_INDEX]
            sigma_series[i, k] = fit.uncertainties[_RATE_INDEX]

    return SlidingVelocity(
        centers=centers,
        rates=rate_series,
        sigmas=sigma_series,
        counts=counts,
        window_years=float(window_years),
        step_years=float(step_years),
        components=None if names is None else tuple(names),
        method=_METHOD_WLS,
    )


def detectability_floor(
    sigma_white: float,
    amplitude_powerlaw: float,
    spectral_index: float,
    window_years: float,
    *,
    confidence: float = 0.95,
) -> float:
    """Per-station velocity-change alarm threshold — **stub** (Phase 2).

    Planned quantity: the smallest velocity change |Δv| between adjacent
    analysis windows of length T that is detectable at the given
    confidence against the station's noise,

        ``Δv_min = z_{1−α/2} · √2 · σ_v(T; σ_w, A, κ)``

    where σ_v(T) is the **colored-noise** rate uncertainty of a window of
    length T under a white (σ_w) + power-law (amplitude A, spectral index
    κ) noise model — Williams 2003 (J. Geodesy 76, eqs. 23–30: σ_v² ∝ T⁻³
    for white noise, ∝ T⁻² for flicker, ∝ T⁻¹ for random walk), and √2
    accounts for differencing two independent window estimates.

    The per-station noise parameters (σ_w, A, κ) come from the GBIS4TS
    MLE/MCMC noise model (Yang, Sigmundsson & Geirsson 2023, GRL
    2023GL103432) — analysis-lane task H1 / plan §10.7, currently
    backburnered. Implementing the floor with the optimistic WLS
    white-noise σ_v would defeat its purpose (alarms would fire on noise),
    so this function is deliberately **not** implemented until the noise
    model lands; results produced meanwhile carry ``method="wls"`` and
    the Williams-2003 caveat.

    Args:
        sigma_white: White-noise amplitude σ_w [L].
        amplitude_powerlaw: Power-law noise amplitude A [L·yr^(−κ/4)]
            (Williams 2003 normalization).
        spectral_index: Spectral index κ (0 white, −1 flicker, −2 random
            walk) [dimensionless].
        window_years: Analysis-window length T [yr].
        confidence: Two-sided detection confidence 1 − α [dimensionless].

    Returns:
        Detectable velocity change Δv_min [L/yr] — once implemented.

    Raises:
        NotImplementedError: Always, until the GBIS4TS colored-noise
            model (task H1) provides σ_v(T; σ_w, A, κ).

    Reference:
        Williams 2003, J. Geodesy 76, eqs. 23–30; Yang, Sigmundsson &
        Geirsson 2023, GRL 2023GL103432; plan §10.7 (detectability floor).
    """
    raise NotImplementedError(
        "detectability_floor requires the GBIS4TS colored-noise model "
        "(analysis-lane task H1, plan §10.7); WLS velocities carry "
        "method='wls' and Williams-2003-optimistic sigmas until then"
    )
