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
WLS results carry ``method="wls"``. The white-noise formal σ_v is
**optimistic** for real GNSS daily solutions — temporally correlated
(flicker/random-walk) noise inflates true rate uncertainty by factors of
several (Williams 2003, J. Geodesy 76, eqs. 23–30). The honest-σ upgrade
is :func:`estimate_velocity_mle` (``method="mle"``, plan §9b): the same
windowed trajectory fit under a **white + power-law colored-noise model**
estimated jointly by maximum likelihood (:mod:`gps_analysis.noise` —
Zhang et al. 1997; Williams 2003; Williams et al. 2004; Bos et al. 2013),
whose σ_v comes from the colored-noise GLS covariance. Posterior
(MCMC) noise estimates from the GBIS4TS lane
(:mod:`gps_analysis.transient`, plan §10.7) carry ``method="gbis"``; the
API contract distinguishes all three so consumers never mistake a WLS σ
for an honest one. The per-station :func:`detectability_floor`
(velocity-change alarm threshold) evaluates the exact colored-noise rate
uncertainty for any (σ_w, β, κ) triple — MLE- or GBIS-estimated.

All functions are pure: float64 arithmetic, no I/O, inputs never mutated,
units the caller's business ([L] below; velocity [L/yr], azimuth degrees,
time ``yearf``).
"""

import dataclasses
import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from . import models
from .baseline import slice_window
from .fitting import (
    _LINEAR_DESIGNS,
    ModelFunc,
    _components_2d,
    _n_model_params,
    _per_component_p0,
    _per_component_sigma,
    _wls_solve,
    fit_components,
)
from .models import FloatArray, TrajectoryParams
from .noise import NoiseModel, estimate_noise_mle, powerlaw_rate_sigma
from .transient import _DELTA_T_YR

__all__ = [
    "SlidingVelocity",
    "VelocityEstimate",
    "VelocityEstimateMLE",
    "detectability_floor",
    "estimate_velocity",
    "estimate_velocity_mle",
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
"""Method tag of the WLS estimator (API contract, plan §10.5 /
MATH_STANDARDS §6)."""

_METHOD_MLE = "mle"
"""Method tag of the colored-noise MLE estimator
(:func:`estimate_velocity_mle`; API contract, plan §10.5 / §9b). The
GBIS4TS posterior lane tags ``"gbis"``."""

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


def _horizontal_products(
    names: Sequence[str] | None,
    rates: FloatArray,
    sigmas: FloatArray,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Derive the horizontal magnitude/azimuth products from labelled rates.

    Equation (composition of the four atomic horizontal functions):
        ``|v_h| = √(v_E² + v_N²)``, ``α = atan2(v_E, v_N)·180/π (mod 360)``
        with their delta-method σ — evaluated only when ``names`` contains
        exactly one ``"north"`` and one ``"east"`` (case-insensitive).

    Symbols → args:
        - component labels → ``names``: per-component labels or None
        - ``v`` → ``rates``: per-component rates, shape (C,) [L/yr]
        - ``σ_v`` → ``sigmas``: their 1-σ uncertainties, shape (C,) [L/yr]

    Returns:
        ``(magnitude, azimuth, magnitude_sigma, azimuth_sigma)`` floats
        [L/yr, deg, L/yr, deg], or ``(None, None, None, None)`` when the
        horizontal pair is not identifiable from the labels.

    Reference:
        Thin orchestration over :func:`horizontal_magnitude`,
        :func:`horizontal_azimuth`, :func:`horizontal_magnitude_sigma`,
        :func:`horizontal_azimuth_sigma` (see those for the math) —
        shared by the WLS and MLE estimators so the API products are
        method-independent (plan §10.5).

    Numerical notes:
        No math of its own; the NaN conventions of the ``*_sigma``
        functions at |v_h| = 0 pass through.
    """
    if names is None:
        return None, None, None, None
    lowered = [name.lower() for name in names]
    if lowered.count("north") != 1 or lowered.count("east") != 1:
        return None, None, None, None
    i_n, i_e = lowered.index("north"), lowered.index("east")
    v_e, v_n = rates[i_e], rates[i_n]
    s_e, s_n = sigmas[i_e], sigmas[i_n]
    return (
        float(horizontal_magnitude(v_e, v_n)),
        float(horizontal_azimuth(v_e, v_n)),
        float(horizontal_magnitude_sigma(v_e, v_n, s_e, s_n)),
        float(horizontal_azimuth_sigma(v_e, v_n, s_e, s_n)),
    )


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
class VelocityEstimateMLE(VelocityEstimate):
    """Fixed-window colored-noise **MLE** secular velocity (honest σ_v).

    Result of :func:`estimate_velocity_mle` — shape-compatible with
    :class:`VelocityEstimate` (same rates/sigmas/fits/horizontal products,
    ``method="mle"``) plus the per-component white + power-law noise
    models the uncertainties are conditioned on. ``sigmas`` here are the
    **colored-noise GLS** 1-σ rate errors ``√(ŝ²·(AᵀC₀⁻¹A)⁻¹)₁₁`` —
    typically several × the WLS formal error for flicker-dominated GNSS
    series (Zhang et al. 1997; Williams et al. 2004).

    Attributes:
        noise: Per-component :class:`gps_analysis.noise.NoiseModel`
            (σ̂_w [L], β̂ [L·yr^(−κ/4)], κ̂, ln L̂, n), in the same row
            order as ``rates`` — the MATH_STANDARDS §6 provenance that
            makes the σ honest.

    Numerical notes:
        Inherits the coercion/validation of :class:`VelocityEstimate`;
        additionally requires one noise model per component.
    """

    noise: tuple[NoiseModel, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        n_components = np.asarray(self.rates).size
        if len(self.noise) != n_components:
            raise ValueError(
                f"got {len(self.noise)} noise models for {n_components} components"
            )


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

    solved via :func:`gps_analysis.fitting.fit_components` (closed-form
    weighted least squares for the linear-in-parameters named models;
    ``scipy.optimize.curve_fit`` for nonlinear custom callables); the
    rate v̂ and its 1-σ error come straight from the returned
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
        RuntimeError: Propagated from ``curve_fit`` when a nonlinear
            custom-model fit does not converge (cannot occur for the
            named linear-in-parameters models — closed-form solve).

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
        refer to t − t_ref. Ĉ comes from an SVD-based pseudo-inverse of
        JᵀWJ (``gps_analysis.fitting._wls_solve`` for the linear models,
        ``curve_fit`` internally for nonlinear ones) — no explicit matrix
        inverse is formed here. σ_v is ``inf`` if the design/Jacobian is
        singular at the solution (``curve_fit`` convention, mirrored by
        the closed-form path).
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

    magnitude, azimuth, magnitude_sigma, azimuth_sigma = _horizontal_products(
        names, rates, sigmas
    )

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


def estimate_velocity_mle(
    t: ArrayLike,
    y: ArrayLike,
    *,
    model: str | ModelFunc = "lineperiodic",
    window: tuple[float | None, float | None] | None = None,
    tol: float = _DEFAULT_TOL,
    names: Sequence[str] | None = None,
    kappa_bounds: tuple[float, float] = (-2.5, 0.0),
) -> VelocityEstimateMLE:
    """Estimate the secular velocity with an honest colored-noise MLE σ_v.

    Equation (per component, over the windowed samples):
        ``(p̂, σ̂_w, β̂, κ̂) = argmax  ln L(p, σ_w, β, κ)``,
        ``y = A(t−t_ref)·p + ε``,  ``ε ~ N(0, C(σ_w, β, κ))``,
        ``C = σ_w²·I + β²·ΔT^(−κ/2)·(T Tᵀ)``  →
        ``v̂ = p̂₁`` [L/yr],  ``σ_v = √(Ĉ_p)₁₁``,
        ``Ĉ_p = ŝ²·(AᵀC₀⁻¹A)⁻¹`` (colored-noise GLS covariance)

    solved by :func:`gps_analysis.noise.estimate_noise_mle` — the same
    white + power-law covariance family as :mod:`gps_analysis.transient`,
    factorized exactly in O(n²) by the generalized-Schur machinery. This
    is the honest-σ upgrade of :func:`estimate_velocity`: the rate v̂ is
    essentially the WLS/GLS estimate, but σ_v is inflated by the fitted
    temporal correlation (typically several × the white-noise formal
    error for flicker-dominated GNSS series). Result carries
    ``method="mle"``; horizontal magnitude/azimuth and their delta-method
    σ are derived exactly as in :func:`estimate_velocity` when
    ``names`` labels a unique north/east pair.

    Symbols → args:
        - ``tᵢ`` → ``t``: epochs, fractional years (``yearf``) [yr] —
          **time-ordered, uniformly (daily) sampled** (the covariance lag
          is the sample index; :mod:`gps_analysis.noise` caveat)
        - ``yᵢ`` → ``y``: observations, component-major [L]
        - ``A``  → ``model``: linear-in-parameters trajectory design with
          ``params[1]`` = secular rate — ``"lineperiodic"`` (default) or
          ``"linear"`` (or a callable registered in ``_LINEAR_DESIGNS``)
        - ``t_ref`` → internal: mean windowed epoch (returned) [yr]
        - ``κ`` search range → ``kappa_bounds``: (κ_min, κ_max) ⊂ [−3, 0]

    Args:
        t: Epochs, shape (N,) [yr]. Finite; sort ascending.
        y: Observations, shape (N,) or (C, N) [L]. Finite.
        model: A **linear-in-parameters** named model
            (``"lineperiodic"``/``"linear"``) or a callable registered in
            :data:`gps_analysis.fitting._LINEAR_DESIGNS`. Nonlinear models
            are rejected — the closed-form GLS profile of the MLE needs a
            fixed design matrix.
        window: Optional (start, end) window [yr]; either bound ``None``
            (open). ``None`` uses the whole series.
        tol: Window boundary tolerance δ [yr] (legacy 0.001 yr default).
        names: Optional per-component labels (e.g. ``("north", "east",
            "up")``) — stored, and used to locate the horizontal pair.
        kappa_bounds: Spectral-index search bounds forwarded to
            :func:`gps_analysis.noise.estimate_noise_mle`; default
            (−2.5, 0) spans white … beyond random walk.

    Returns:
        :class:`VelocityEstimateMLE` — per-component rate v̂ and honest
        colored-noise σ_v, the WLS trajectory fits (for reference — the
        ``fits`` carry the white-noise covariance and the same rate), the
        per-component :class:`gps_analysis.noise.NoiseModel`, window
        provenance, ``method="mle"``, and horizontal products.

    Raises:
        ValueError: On non-finite ``t``, shape mismatches, a nonlinear or
            unknown model, a model without a rate parameter, a window with
            too few samples for the trajectory + noise parameters, or a
            rank-deficient design / noise-free series (propagated from the
            MLE).

    Reference:
        Colored-noise rate uncertainty: Williams 2003, J. Geodesy 76;
        MLE practice and typical flicker-driven inflation: Zhang et al.
        1997, JGR 102(B8); Williams et al. 2004, JGR 109, B03412;
        Langbein 2004, JGR 109, B04406; fast MLE: Bos et al. 2013,
        J. Geodesy 87. Seasonal co-estimation on short windows: Blewitt &
        Lavallée 2002, JGR 107(B7). The estimator itself:
        :func:`gps_analysis.noise.estimate_noise_mle`.

    Numerical notes:
        Epochs are re-referenced to t_ref (mean windowed epoch) before
        building the design, exactly as :func:`estimate_velocity`; the
        rate and σ_v are translation-invariant so ``params[1]`` /
        ``√Ĉ_p[1,1]`` are read directly. Each component is fitted
        independently (a 2-D (φ, κ) search over an exact O(n²·P) profile
        likelihood — coarse grid + Nelder–Mead polish). For provenance,
        the reference white-noise WLS ``fits`` are also computed (cheap
        closed form) so callers can compare formal vs honest σ side by
        side. A κ̂ landing on a ``kappa_bounds`` edge is a diagnostic
        (widen the bounds) — surfaced via the returned ``noise``.
    """
    model_func = _resolve_model(model)
    n_params = _rate_param_count(model_func)
    design = _LINEAR_DESIGNS.get(model_func)
    if design is None:
        raise ValueError(
            "estimate_velocity_mle requires a linear-in-parameters model "
            "(a fixed design matrix); got a nonlinear/unregistered model - "
            "use 'lineperiodic', 'linear', or another _LINEAR_DESIGNS model"
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
    if names is not None and len(names) != yy.shape[0]:
        raise ValueError(f"names has {len(names)} entries for {yy.shape[0]} components")

    if window is None:
        mask = np.ones(tt.shape, dtype=np.bool_)
    else:
        mask = slice_window(tt, window[0], window[1], tol=tol)
    n_obs = int(np.count_nonzero(mask))
    # Need P trajectory params + (kappa, phi, scale) degrees of freedom.
    if n_obs < n_params + 3:
        raise ValueError(
            f"window has {n_obs} samples for {n_params} trajectory parameters "
            f"plus (kappa, phi, scale) - need at least {n_params + 3}"
        )

    t_win = tt[mask]
    t_ref = float(np.mean(t_win))
    a_full = design.build(t_win)
    if design.trend_column is not None:
        a_full[:, design.trend_column] = t_win - t_ref

    fits: list[TrajectoryParams] = []
    noise_models: list[NoiseModel] = []
    rates_list: list[float] = []
    sigmas_list: list[float] = []
    for i in range(yy.shape[0]):
        y_i = yy[i][mask]
        mle = estimate_noise_mle(a_full, y_i, kappa_bounds=kappa_bounds)
        noise_models.append(mle.noise)
        rates_list.append(float(mle.params[_RATE_INDEX]))
        sigmas_list.append(float(np.sqrt(mle.covariance[_RATE_INDEX, _RATE_INDEX])))
        # Reference white-noise WLS fit (provenance: formal vs honest sigma).
        wls_params, wls_cov = _wls_solve(a_full, y_i, None, absolute_sigma=False)
        fits.append(
            TrajectoryParams(
                params=wls_params,
                covariance=wls_cov,
                component=None if names is None else names[i],
            )
        )

    rates = np.asarray(rates_list, dtype=np.float64)
    sigmas = np.asarray(sigmas_list, dtype=np.float64)
    magnitude, azimuth, magnitude_sigma, azimuth_sigma = _horizontal_products(
        names, rates, sigmas
    )

    return VelocityEstimateMLE(
        rates=rates,
        sigmas=sigmas,
        fits=tuple(fits),
        components=None if names is None else tuple(names),
        n_obs=n_obs,
        t_ref=t_ref,
        span=(float(np.min(t_win)), float(np.max(t_win))),
        method=_METHOD_MLE,
        magnitude=magnitude,
        azimuth=azimuth,
        magnitude_sigma=magnitude_sigma,
        azimuth_sigma=azimuth_sigma,
        noise=tuple(noise_models),
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
        when it holds fewer than ``min_obs`` samples, when its covariance
        is not estimable (singular design/Jacobian, ``OptimizeWarning``
        from either fit path), or — nonlinear custom models only — when
        the fit does not converge (``curve_fit`` ``RuntimeError``). The
        centre grid stays regular so data gaps appear as NaN runs rather
        than silently shifting epochs.

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
        For the linear-in-parameters named models the full-series design
        matrix (incl. the seasonal trig columns, on **absolute** t) is
        built once and each window solves a row slice of it in closed
        form (:func:`gps_analysis.fitting._wls_solve`) — no per-window
        basis rebuild, no iteration. The absolute-t trig basis spans the
        same column space as the window-local one (a time translation is
        an exact rotation of the (a,b)/(c,d) seasonal pairs), so v̂ and
        σ_v are unchanged; only the raw-t trend column is re-centered per
        window (t − mean windowed epoch), which the rate is invariant
        under — the conditioning argument of :func:`estimate_velocity`
        applies per window. Nonlinear custom models keep the per-window
        iterative fit in window-local time. Components are fitted one at
        a time so a failure in one component NaNs only that
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

    # Linear-in-parameters models: build the full-series design once and
    # solve each window from row slices of it (finding #5) — the seasonal
    # trig columns are evaluated a single time, on absolute t (same fitted
    # rate/σ_v as the window-local-time basis: a time translation only
    # rotates the intercept/phase coefficients, spanning the identical
    # column space). Only the raw-t trend column is re-centered per window
    # for conditioning; the rate and its variance are invariant under that
    # centering.
    design = _LINEAR_DESIGNS.get(model_func)
    basis = None if design is None else design.build(tt)

    for k, center in enumerate(centers):
        mask = slice_window(tt, center - half, center + half, tol=tol)
        count = int(np.count_nonzero(mask))
        counts[k] = count
        if count < min_obs:
            continue
        if design is not None and basis is not None:
            a_win = basis[mask]
            if design.trend_column is not None:
                a_win = a_win.copy()
                trend = a_win[:, design.trend_column]
                a_win[:, design.trend_column] = trend - float(np.mean(trend))
            for i in range(n_components):
                s_i = sigma_rows[i]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", optimize.OptimizeWarning)
                        params, cov = _wls_solve(
                            a_win,
                            yy[i][mask],
                            None if s_i is None else s_i[mask],
                            absolute_sigma,
                        )
                except optimize.OptimizeWarning:
                    continue  # window stays NaN for this component
                rate_series[i, k] = params[_RATE_INDEX]
                sigma_series[i, k] = float(np.sqrt(cov[_RATE_INDEX, _RATE_INDEX]))
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
    dt_years: float = _DELTA_T_YR,
    single_window: bool = False,
) -> float:
    """Minimum detectable velocity change under a colored-noise model.

    Equation (Williams 2003, J. Geodesy 76, §5; two-sided z-test on the
    difference of two window rate estimates):

        ``Δv_min = z_{1−α/2} · √2 · σ_v(T; σ_w, β, κ)``

    — the smallest velocity change |Δv| between two adjacent, independent
    analysis windows of length T detectable at confidence 1 − α, where
    σ_v(T; σ_w, β, κ) = :func:`gps_analysis.noise.powerlaw_rate_sigma`
    is the exact finite-n **colored-noise** GLS rate uncertainty of a
    straight-line fit under white (σ_w) + power-law (amplitude β, spectral
    index κ) noise. The √2 propagates the two independent window rate
    errors of the difference Δv = v₂ − v₁ (σ_Δv = √2·σ_v for equal-length
    windows); set ``single_window=True`` to drop it and get the detection
    threshold on a *single* rate (Δv_min = z·σ_v), i.e. the smallest rate
    distinguishable from zero. Williams 2003 (eqs. 23–30) gives the span
    scalings σ_v² ∝ T^(−3−κ): T⁻³ white (κ=0), T⁻² flicker (κ=−1), T⁻¹
    random walk (κ=−2) — reproduced by the exact σ_v used here.

    Symbols → args:
        - ``σ_w`` → ``sigma_white``: white-noise amplitude [L], ≥ 0
        - ``β``  → ``amplitude_powerlaw``: power-law amplitude
          [L·yr^(−κ/4)] (Williams 2003 normalization), ≥ 0; not both 0
        - ``κ``  → ``spectral_index``: spectral index ∈ [−3, 0]
          [dimensionless]
        - ``T``  → ``window_years``: analysis-window length [yr], such
          that ``T/ΔT + 1 ≥ 3`` epochs
        - ``ΔT`` → ``dt_years``: sampling interval [yr], > 0 (default
          1/365, the daily convention of :mod:`gps_analysis.transient`)
        - ``z_{1−α/2}`` → from ``confidence`` = 1 − α (standard-normal
          two-sided quantile)

    Args:
        sigma_white: White-noise amplitude σ_w [L].
        amplitude_powerlaw: Power-law amplitude β [L·yr^(−κ/4)].
        spectral_index: Spectral index κ [dimensionless].
        window_years: Analysis-window length T [yr].
        confidence: Two-sided detection confidence 1 − α ∈ (0, 1)
            [dimensionless]; 0.95 ⇒ z ≈ 1.95996.
        dt_years: Sampling interval ΔT [yr]; default daily (1/365).
        single_window: Drop the √2 (single-rate detection vs zero) when
            True; default False (velocity *change* between two windows).

    Returns:
        Detectable velocity change Δv_min [L/yr] (float, > 0). The noise
        triple (σ_w, β, κ) comes from a colored-noise estimate —
        :func:`estimate_velocity_mle` (``method="mle"``) or the GBIS4TS
        posterior (``method="gbis"``); passing the optimistic WLS
        white-noise σ here would under-report the floor (alarms on noise).

    Raises:
        ValueError: On negative amplitudes / both zero, κ outside [−3, 0],
            a window shorter than 3 epochs, ``dt_years ≤ 0``, or
            ``confidence`` outside (0, 1).

    Reference:
        Williams 2003, J. Geodesy 76, §5 and eqs. 23–30 (rate uncertainty
        and its span scaling); Williams et al. 2004, JGR 109, B03412
        (typical colored-noise levels); Langbein 2004, JGR 109, B04406
        (detection implications of the noise model); Bos et al. 2013,
        J. Geodesy 87 (the GLS σ_v inside Hector). Noise parameters from
        the MLE (:func:`gps_analysis.noise.estimate_noise_mle`) or the
        GBIS4TS posterior (Yang, Sigmundsson & Geirsson 2023, GRL
        2023GL103432). Plan §9b / §10.7 (detectability floor).

    Numerical notes:
        σ_v is the exact finite-n GLS value (:func:`~gps_analysis.noise.
        powerlaw_rate_sigma`), not the large-n asymptotic, so short
        windows are handled honestly. The window is discretized to
        ``n = round(T/ΔT) + 1`` uniformly spaced epochs (the covariance
        lag is the sample index — uniform-sampling assumption of the
        noise model). z is ``scipy.stats.norm.ppf((1+confidence)/2)``.
        Assumes a pure two-parameter linear fit per window; co-estimated
        seasonal terms inflate σ_v (hence Δv_min) further on sub-annual
        to few-year windows (Blewitt & Lavallée 2002).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if window_years <= 0.0:
        raise ValueError(f"window_years must be > 0, got {window_years}")
    if dt_years <= 0.0:
        raise ValueError(f"dt_years must be > 0, got {dt_years}")
    n_epochs = int(round(window_years / dt_years)) + 1
    if n_epochs < 3:
        raise ValueError(
            f"window_years = {window_years} yr is fewer than 3 epochs at "
            f"dt_years = {dt_years} yr"
        )
    sigma_v = powerlaw_rate_sigma(
        sigma_white,
        amplitude_powerlaw,
        spectral_index,
        n_epochs,
        dt_years=dt_years,
    )
    z = float(stats.norm.ppf(0.5 * (1.0 + confidence)))
    factor = 1.0 if single_window else math.sqrt(2.0)
    return factor * z * sigma_v
