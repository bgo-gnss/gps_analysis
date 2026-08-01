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

MIDAS robust lane (``method="midas"``)
--------------------------------------
:func:`estimate_velocity_midas` — Blewitt, Kreemer, Hammond & Gazeaux
2016 (JGR Solid Earth 121, doi:10.1002/2015JB012552): a Theil–Sen
variant whose slope pairs are separated by 1 year (eq 3 — cancels the
seasonal signal exactly and minimizes step-spanning pairs), with the
§2.4 relaxed selection for gapped/campaign series, a 2σ trim of the
slope distribution (eqs 4–5 — the "Adjusted for Skewness" of the
acronym), and a scaled MAD-based uncertainty (eqs 6–8). It needs **no
step detection or declaration at all** — the honest estimator for a
station whose steps are uncatalogued. Derivation chain:
:func:`midas_pair_indices` (pair set) → slopes →
:func:`midas_trimmed_median` (v̂, via :func:`midas_mad_sigma`) →
:func:`midas_rate_uncertainty` (ŝ_v). Blind-test pedigree: RMS
±0.33 mm/yr horizontal, ±1.1 mm/yr up on the Gazeaux et al. 2013 DOGEx
synthetics, best 5th-percentile range of the 20 automatic estimators
tested (Blewitt et al. 2016, §3.2).

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
    ModelFunc,
    _components_2d,
    _n_model_params,
    _per_component_p0,
    _per_component_sigma,
    _resolve_linear_design,
    _wls_solve,
    fit_components,
)
from .models import FloatArray, TrajectoryParams
from .noise import NoiseModel, estimate_noise_mle, powerlaw_rate_sigma
from .transient import _DELTA_T_YR

__all__ = [
    "SlidingVelocity",
    "VelocityEstimate",
    "VelocityEstimateMIDAS",
    "VelocityEstimateMLE",
    "detectability_floor",
    "estimate_velocity",
    "estimate_velocity_midas",
    "estimate_velocity_mle",
    "horizontal_azimuth",
    "horizontal_azimuth_sigma",
    "horizontal_magnitude",
    "horizontal_magnitude_sigma",
    "midas_mad_sigma",
    "midas_pair_indices",
    "midas_rate_uncertainty",
    "midas_trimmed_median",
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

_METHOD_MIDAS = "midas"
"""Method tag of the MIDAS robust estimator
(:func:`estimate_velocity_midas`; MATH_STANDARDS §6 provenance)."""

_MIDAS_PAIR_TOL = 1e-3
"""Interannual pair tolerance δ [yr] of Blewitt et al. 2016 eq. (3):
``0.999 yr < tⱼ − tᵢ < 1.001 yr`` — δ = 0.001 yr forces pairs separated
by 365 days for daily data (``midas.f``: ``tol=0.001d+0``)."""

_MIDAS_TRIM_SIGMAS = 2.0
"""Trim threshold of eq. (5): slopes beyond ±2σ of the first-pass median
are removed — the "Adjusted for Skewness" step (step-spanning pairs
populate one tail; a symmetric percentage trim would not remove them)."""

_MIDAS_MIN_PAIRS = 10
"""Minimum usable slope pairs (both passes, pre-trim) — the reference
implementation's ``minn`` FATAL threshold (``midas.f``, MIDAS4)."""

_MAD_TO_SIGMA = 1.4826
"""σ/MAD for a Gaussian core: 1/Φ⁻¹(3/4) ≈ 1.4826 — the constant of
Blewitt et al. 2016 eq. (4) (after Wilcox 2005), also used verbatim by
``midas.f``."""

_MIDAS_PAIR_REUSE = 4.0
"""Effective-sample-size divisor of eq. (7): ``N = N_actual/4`` — each
coordinate datum is used a nominal 4 times to form pairs (once as first
and once as second member, in each of the forward and backward passes)."""

_MIDAS_ERROR_SCALE = 3.0
"""Uncertainty scale factor of eq. (8): ``ŝ = 3·σ̂`` — chosen so the
reported error matches the RMS accuracy on simulated data; needed
because autocorrelated (power-law/flicker) noise shrinks the effective
number of independent observations (Blewitt et al. 2016, §2.5)."""

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
class VelocityEstimateMIDAS(VelocityEstimate):
    """Fixed-window **MIDAS** robust secular velocity (no step detection).

    Result of :func:`estimate_velocity_midas` — shape-compatible with
    :class:`VelocityEstimate` (same rates/sigmas/fits/horizontal
    products, ``method="midas"``) plus the slope-distribution
    diagnostics of Blewitt et al. 2016. ``sigmas`` here are the scaled
    standard errors of the median, ``ŝ_v = 3·√(π/2)·σ/√(N_actual/4)``
    (eqs. 6–8) — calibrated against RMS accuracy on real (autocorrelated)
    GPS series, so on white-noise synthetics they overbound the true
    scatter by design.

    Attributes:
        n_pairs: Total slope samples n selected by the forward + backward
            passes (pre-trim; shared by all components — pair selection
            depends only on ``t``).
        n_used: Per-component N_actual — slopes surviving the 2σ trim of
            eq. (5), in ``rates`` row order (differs per component:
            "N will generally be different for each of the three
            coordinates", Blewitt et al. 2016 §2.5).
        scale_sigmas: Per-component robust σ of the trimmed slope
            distribution (eq. 4 on the second pass), shape (C,) [L/yr].
        fraction_removed: Per-component trimmed fraction
            (n − N_actual)/n, shape (C,) [dimensionless] — the ``midas.f``
            ``fe/fn/fu`` diagnostics; large values flag heavy
            step/outlier contamination of the pair set.

    Numerical notes:
        The ``fits`` entries hold the ``models.linear`` parameterization
        ``[intercept, rate]`` with the intercept the median of
        ``x − v̂·(t − t_ref)`` over the windowed epochs (``midas.f``
        parity, ``t_ref`` = first used epoch) and covariance
        ``diag(NaN, ŝ_v²)`` — MIDAS estimates no intercept uncertainty
        and no parameter cross-covariance.
    """

    n_pairs: int = 0
    n_used: tuple[int, ...] = ()
    scale_sigmas: FloatArray = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    fraction_removed: FloatArray = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        n_components = np.asarray(self.rates).size
        scale = np.asarray(self.scale_sigmas, dtype=np.float64)
        frac = np.asarray(self.fraction_removed, dtype=np.float64)
        if len(self.n_used) != n_components:
            raise ValueError(
                f"got {len(self.n_used)} n_used entries for "
                f"{n_components} components"
            )
        if scale.shape != (n_components,):
            raise ValueError(
                f"scale_sigmas shape {scale.shape} does not match "
                f"{n_components} components"
            )
        if frac.shape != (n_components,):
            raise ValueError(
                f"fraction_removed shape {frac.shape} does not match "
                f"{n_components} components"
            )
        object.__setattr__(self, "scale_sigmas", scale)
        object.__setattr__(self, "fraction_removed", frac)


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
    # _resolve_linear_design, NOT _LINEAR_DESIGNS.get: the registry is keyed by
    # callable identity and holds only the three house models, so a DERIVED model
    # -- anything from `with_steps`, and every composed model to come -- carries
    # its design on the _LINEAR_DESIGN_ATTR attribute instead and was rejected
    # here as "nonlinear" despite being perfectly linear in its parameters.
    design = _resolve_linear_design(model_func)
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


def _select_pairs_forward(t: FloatArray, pair_tol: float) -> list[tuple[int, int]]:
    """Single forward pass of the MIDAS relaxed pair selection.

    Equation (selection rule; Blewitt et al. 2016 eq. 3 + §2.4):
        for each epoch i (in time order), pair it with **one** later
        epoch: the first j with ``tⱼ − tᵢ ≥ 1 − δ``; if additionally
        ``tⱼ − tᵢ < 1 + δ`` the pair is the 1-year match, otherwise the
        pair is (i, k) with k the next not-yet-consumed partner pointer
        (k = max(k, j), advanced on use; on exhaustion at the series end
        the pointer resets so the next search re-catches the closest
        partner at least 1 year ahead). Epochs with no later point
        ≥ 1 − δ away (``tᵢ > t_last + δ − 1``) form no pair.

    Symbols → args:
        - ``tᵢ`` → ``t``: epochs, 1-D float64, sorted ascending [yr]
        - ``δ``  → ``pair_tol``: interannual tolerance [yr]

    Returns:
        List of (i, j) index pairs, i < j, one pair per eligible i.

    Reference:
        Blewitt et al. 2016, JGR 121, §2.4 (relaxed pair selection
        principles 1–4); exact-parity port of ``midas.f`` (MIDAS4)
        ``selectpair``, no-step-file case.

    Numerical notes:
        The scan for the first partner is ``np.searchsorted`` (t sorted
        ascending — caller-checked), matching the Fortran linear scan's
        exit condition ``tⱼ ≥ tᵢ + 1 − δ`` exactly; the unmatched-pointer
        bookkeeping (k = max(k, j); wrap at the last index) reproduces
        ``selectpair``'s ``k`` variable including its reset-to-closest
        behavior. Relaxed pairs always have ``tⱼ − tᵢ > 1 + δ`` (partner
        pointer never points before the 1-year horizon), so pair Δt is
        bounded below by 1 − δ and division by Δt is safe.
    """
    m = t.size
    if m < 2:
        return []
    limit = float(t[-1]) + pair_tol - 1.0
    first_partner = np.searchsorted(t, t + (1.0 - pair_tol), side="left")
    pairs: list[tuple[int, int]] = []
    k = -1  # next-unmatched partner pointer (relaxed branch only)
    for i in range(m):
        if t[i] > limit:
            break  # t sorted: no later i can find a partner >= 1 yr away
        j = int(first_partner[i])
        if k < j:
            k = j
        if t[j] - t[i] < 1.0 + pair_tol:
            pairs.append((i, j))  # 1-year match, eq. (3)
        else:
            pairs.append((i, k))  # relaxed: next unmatched partner, §2.4
            if k == m - 1:
                k = -1  # exhausted: next i re-catches its closest partner
            k += 1
    return pairs


def midas_pair_indices(
    t: ArrayLike, *, pair_tol: float = _MIDAS_PAIR_TOL
) -> NDArray[np.int64]:
    """Select the MIDAS interannual data-pair set (indices into ``t``).

    Equation (Blewitt et al. 2016 eq. 3, relaxed per §2.4):
        ``0.999 yr < tⱼ − tᵢ < 1.001 yr``  (δ = ``pair_tol`` = 0.001 yr)
    selected **twice** — once forward in time and once backward
    (on ``−t`` reversed), the backward pairs mapped to forward indices —
    so the estimator is time-symmetric; when no 1-year partner exists
    (gapped/campaign series) the §2.4 relaxation pairs the epoch with the
    next not-yet-matched point ≥ 1 year ahead instead
    (:func:`_select_pairs_forward`).

    The 1-year separation is the defining feature of MIDAS versus
    ordinary Theil–Sen: it cancels the seasonal signal exactly (any
    period-1 component differences away) and minimizes the fraction of
    pairs spanning a step discontinuity, while an integer-year-only rule
    would re-admit long step-spanning pairs. For continuous daily series
    the forward and backward passes select the same pairs, so each slope
    appears twice — the origin of the nominal pair-reuse factor 4 in
    eq. (7).

    Symbols → args:
        - ``tᵢ`` → ``t``: epochs, shape (M,), fractional years
          (``yearf``), sorted ascending [yr]
        - ``δ``  → ``pair_tol``: interannual tolerance [yr] > 0
          (0.001 yr default — 365-day pairs for daily data)

    Args:
        t: Epochs, 1-D, finite, sorted ascending [yr].
        pair_tol: Tolerance δ of eq. (3) [yr].

    Returns:
        Index pairs, shape (n, 2) int64, each row (i, j) with i < j and
        ``t[j] − t[i] ≥ 1 − δ``; forward-pass pairs first, then the
        backward pass. Duplicate rows are **intentional** (see above).
        Shape (0, 2) when no pair exists (span < 1 − δ).

    Raises:
        ValueError: If ``t`` is not 1-D, not finite, not sorted
            ascending, or ``pair_tol`` is not in (0, 0.5).

    Reference:
        Blewitt et al. 2016, JGR 121, doi:10.1002/2015JB012552, §2.2
        (eq. 3) and §2.4 (relaxed selection); ``midas.f`` (MIDAS4)
        ``selectpair`` + ``tback``.

    Numerical notes:
        Float64; O(M log M) via ``searchsorted``. The backward pass runs
        the same forward routine on ``−t`` reversed and maps a reversed
        pair (a, b) to (M−1−b, M−1−a) — the slope of a pair is invariant
        under the mapping since both numerator and denominator negate.
        Duplicate epochs (Δt = 0) are legal input: pair Δt is always
        ≥ 1 − δ, so no zero division downstream.
    """
    if not 0.0 < pair_tol < 0.5:
        raise ValueError(f"pair_tol must be in (0, 0.5) yr, got {pair_tol}")
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must be finite")
    if tt.size >= 2 and bool(np.any(np.diff(tt) < 0.0)):
        raise ValueError("t must be sorted ascending for MIDAS pair selection")
    forward = _select_pairs_forward(tt, pair_tol)
    reversed_pairs = _select_pairs_forward(-tt[::-1], pair_tol)
    m = tt.size
    backward = [(m - 1 - j, m - 1 - i) for (i, j) in reversed_pairs]
    return np.asarray(forward + backward, dtype=np.int64).reshape(-1, 2)


def midas_mad_sigma(values: ArrayLike, center: float) -> float:
    """Robust standard deviation σ via the scaled MAD about a center.

    Equation (Blewitt et al. 2016 eq. 4):
        ``MAD = median_p |v_p − v̂|``,  ``σ = 1.4826 · MAD``

    Symbols → args:
        - ``v_p`` → ``values``: sample (slopes), shape (n,) [L/yr]
        - ``v̂``  → ``center``: reference value, normally the sample
          median [L/yr]

    Args:
        values: Sample values, 1-D, finite [L/yr].
        center: Center v̂ the deviations are taken about [L/yr].

    Returns:
        Robust σ [L/yr], float64 ≥ 0.

    Raises:
        ValueError: On an empty or non-1-D sample, or non-finite input.

    Reference:
        Blewitt et al. 2016, JGR 121, eq. (4); scaling constant after
        Wilcox 2005 (σ = MAD/Φ⁻¹(3/4) ≈ 1.4826·MAD for a Gaussian core —
        insensitive to a minority of outliers).

    Numerical notes:
        Float64. The literal 1.4826 (not 1/Φ⁻¹(3/4) to machine
        precision) is used for parity with the paper and ``midas.f``;
        the difference is < 3·10⁻⁵ relative. ``np.median`` averages the
        two middle order statistics for even n (eq. 2 convention).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"values must be 1-D and non-empty, got shape {v.shape}")
    if not (np.all(np.isfinite(v)) and math.isfinite(center)):
        raise ValueError("values and center must be finite")
    return float(_MAD_TO_SIGMA * np.median(np.abs(v - center)))


def midas_trimmed_median(
    slopes: ArrayLike, *, trim_sigmas: float = _MIDAS_TRIM_SIGMAS
) -> tuple[float, float, int]:
    """Two-pass trimmed median of the slope distribution (the MIDAS v̂).

    Equation (Blewitt et al. 2016 eq. 5):
        Step 1: ``v̂ = median_p v_p``, ``σ = 1.4826·median_p |v_p − v̂|``
        Step 2: select ``q = p`` for all ``|v_p − v̂| < 2σ``;
        ``v̂ = median_q v_q``, ``σ = 1.4826·median_q |v_q − v̂|``

    This trim-and-recompute is the "Adjusted for Skewness" of the MIDAS
    acronym: pairs spanning a step discontinuity populate **one** tail of
    the slope distribution, skewing (or multimodalizing) it and biasing
    the plain median; removing slopes beyond 2σ of the first-pass median
    (a threshold in *deviation*, not a symmetric percentage — a
    percentage trim would leave the skew in place) and re-taking the
    median removes that bias while barely touching a Gaussian core.

    Symbols → args:
        - ``v_p`` → ``slopes``: slope sample, shape (n,) [L/yr]
        - trim threshold ``2σ`` → ``trim_sigmas``·σ (default 2, the
          paper's choice) [dimensionless multiplier]

    Args:
        slopes: Slope sample, 1-D, non-empty, finite [L/yr].
        trim_sigmas: Trim threshold in robust-σ units, > 0.

    Returns:
        ``(v_hat, sigma, n_used)`` — the second-pass median v̂ [L/yr],
        second-pass robust σ [L/yr] (both float64), and the number of
        slopes N_actual that survived the trim (the eq.-7 input).

    Raises:
        ValueError: On an empty/non-1-D/non-finite sample or
            ``trim_sigmas ≤ 0``.

    Reference:
        Blewitt et al. 2016, JGR 121, eqs. (2), (4), (5) and §2.3;
        ``midas.f`` (MIDAS4) estimation section.

    Numerical notes:
        Float64; medians via ``np.median`` (even-n average, eq. 2). The
        keep rule is the strict ``< 2σ`` of ``midas.f``; when σ = 0
        (≥ half the slopes identical — e.g. noise-free data, where the
        strict rule would empty the sample) the trim set degenerates to
        the exact-tie slopes ``|v_p − v̂| = 0``, which the majority
        satisfies by construction, so the result is the exact common
        slope with σ = 0. At least half the sample always survives
        (deviations ≤ MAD < 2σ hold for ≥ n/2 slopes), so the
        second-pass median is well-defined.
    """
    v = np.asarray(slopes, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"slopes must be 1-D and non-empty, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("slopes must be finite")
    if trim_sigmas <= 0.0:
        raise ValueError(f"trim_sigmas must be > 0, got {trim_sigmas}")
    v_hat = float(np.median(v))
    sigma = midas_mad_sigma(v, v_hat)
    deviations = np.abs(v - v_hat)
    if sigma > 0.0:
        keep = deviations < trim_sigmas * sigma
    else:
        keep = deviations == 0.0  # degenerate scale: keep the exact ties
    trimmed = v[keep]
    v_hat = float(np.median(trimmed))
    sigma = midas_mad_sigma(trimmed, v_hat)
    return v_hat, sigma, int(trimmed.size)


def midas_rate_uncertainty(
    sigma: float,
    n_actual: int,
    *,
    pair_reuse: float = _MIDAS_PAIR_REUSE,
    error_scale: float = _MIDAS_ERROR_SCALE,
) -> float:
    """Scaled standard error ŝ_v of the MIDAS median rate.

    Equation (Blewitt et al. 2016 eqs. 6–8, composed):
        ``σ̂ = √(π/2) · σ/√N ≈ 1.2533 σ/√N``   (eq. 6)
        ``N = N_actual/4``                      (eq. 7)
        ``ŝ_v = 3 · σ̂``                        (eq. 8)

    The two corrections are separate, with separate justifications:
    the **/4** of eq. (7) converts the trimmed slope count to an
    effective sample size, accounting for the nominal number of times
    each coordinate datum is used to form pairs (first and second member,
    forward and backward pass); the **×3** of eq. (8) is calibrated so
    the reported error matches the RMS accuracy on simulated data — it
    absorbs the further loss of independence from autocorrelated
    (power-law/flicker) noise, which eq. (6)'s i.i.d. assumption cannot
    see (Blewitt et al. 2016 §2.5, citing Zięba & Ramza 2011).

    Symbols → args:
        - ``σ`` → ``sigma``: second-pass robust σ of the trimmed slope
          distribution (:func:`midas_trimmed_median`) [L/yr]
        - ``N_actual`` → ``n_actual``: trimmed slope count
          [dimensionless]
        - divisor 4 → ``pair_reuse``; factor 3 → ``error_scale``

    Args:
        sigma: Robust slope-scatter σ ≥ 0 [L/yr].
        n_actual: Number of slopes after trimming, ≥ 1.
        pair_reuse: Eq.-7 divisor, > 0 (default 4).
        error_scale: Eq.-8 scale, > 0 (default 3).

    Returns:
        ŝ_v [L/yr], float64 ≥ 0 — a *realistic* (RMS-calibrated), not
        formal, 1-σ velocity uncertainty.

    Raises:
        ValueError: On ``sigma < 0`` / non-finite, ``n_actual < 1``, or
            non-positive ``pair_reuse``/``error_scale``.

    Reference:
        Blewitt et al. 2016, JGR 121, eqs. (6)–(8) and §2.5; standard
        error of a median: Kenney & Keeping 1954, p. 212.

    Numerical notes:
        Float64; ``√(π/2)`` evaluated exactly (``math``), not the
        4-digit 1.2533 of the paper/``midas.f`` (< 3·10⁻⁵ relative
        difference). Assumes the trimmed distribution is approximately
        normal — justified by the eq.-4 MAD basis (Gaussian majority).
    """
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be finite and >= 0, got {sigma}")
    if n_actual < 1:
        raise ValueError(f"n_actual must be >= 1, got {n_actual}")
    if pair_reuse <= 0.0 or error_scale <= 0.0:
        raise ValueError("pair_reuse and error_scale must be > 0")
    n_eff = n_actual / pair_reuse
    return float(error_scale * math.sqrt(math.pi / 2.0) * sigma / math.sqrt(n_eff))


def estimate_velocity_midas(
    t: ArrayLike,
    y: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    window: tuple[float | None, float | None] | None = None,
    tol: float = _DEFAULT_TOL,
    pair_tol: float = _MIDAS_PAIR_TOL,
    trim_sigmas: float = _MIDAS_TRIM_SIGMAS,
    min_pairs: int = _MIDAS_MIN_PAIRS,
) -> VelocityEstimateMIDAS:
    """Estimate the MIDAS robust secular velocity (no step detection).

    Equation (per component; Blewitt et al. 2016, composed):
        pairs (i, j): ``0.999 yr < tⱼ − tᵢ < 1.001 yr`` (eq. 3, relaxed
        per §2.4 for gaps) → slopes ``v_p = (x_j − x_i)/(t_j − t_i)``
        (eq. 1 restricted to the pair set) →
        2σ-trimmed re-median (eqs. 4–5): ``v̂ = median_q v_q``,
        ``σ = 1.4826·median_q |v_q − v̂|`` →
        ``ŝ_v = 3·√(π/2)·σ/√(N_actual/4)`` (eqs. 6–8).

    Thin orchestration of the four atomic pieces —
    :func:`midas_pair_indices`, the slope quotient,
    :func:`midas_trimmed_median` (with :func:`midas_mad_sigma`) and
    :func:`midas_rate_uncertainty` — see those docstrings for the math.
    Requires **no step detection, declaration, or outlier screening**:
    resistance to steps, outliers, seasonality, skewness and
    heteroscedasticity is built into the pair selection and the trimmed
    median. This is the honest estimator for a station whose steps are
    uncatalogued — the principled replacement for borrowing a rate from
    a neighbour. Components are treated independently (inter-component
    correlations ~0.1; steps hit components differently — §2.5).

    .. warning::
        **Local parameterization — never feed these fits to a staged
        ``held`` value.** Like everything in :mod:`~gps_analysis.velocity`
        (the deliberate exception to the package's absolute-t invariant),
        the returned ``fits`` are t-local: ``params = [intercept, rate]``
        with the intercept referred to ``t_ref`` (the first used epoch,
        ``midas.f`` parity). The rate and ŝ_v are translation-invariant;
        the intercept is NOT — re-referencing is the caller's job before
        any parameter vector crosses a boundary (returned/held/stored/
        borrowed), e.g. before wiring MIDAS into ``staged.py``.

    Symbols → args:
        - ``tᵢ`` → ``t``: epochs, fractional years (``yearf``), sorted
          ascending [yr]
        - ``xᵢ`` → ``y``: observations, component-major [L]
        - ``δ`` → ``pair_tol``: eq.-3 tolerance [yr]
        - trim ``2σ`` → ``trim_sigmas`` [σ units]
        - window → ``window``/``tol``: [t_start, t_end] ± δ_w [yr]
          (:func:`gps_analysis.baseline.slice_window`)
        - ``t_ref`` → internal: first windowed epoch (returned) [yr]

    Args:
        t: Epochs, shape (N,) [yr]. Finite, sorted ascending (the pair
            selection scans in time order — unlike the WLS/MLE
            estimators, order matters here).
        y: Observations, shape (N,) or (C, N) [L]. Must be finite —
            MIDAS pair selection is shared across components, so drop
            non-finite epochs before calling (an all-NaN component
            cannot be estimated and raises).
        names: Optional per-component labels (e.g. ``("north", "east",
            "up")``) — stored, and used to locate the horizontal pair.
        window: Optional (start, end) window [yr]; either bound may be
            ``None`` (open). ``None`` uses the whole series.
        tol: Window boundary tolerance δ_w [yr] (legacy 0.001 default).
        pair_tol: Interannual pair tolerance δ of eq. (3) [yr].
        trim_sigmas: Eq.-5 trim threshold in robust-σ units (default 2).
        min_pairs: Minimum usable slope pairs (pre-trim); default 10 —
            the reference implementation's ``minn``. Values < 2 are
            floored to 2 (a 1-pair "median" is not an estimate).

    Returns:
        :class:`VelocityEstimateMIDAS` — per-component robust rate v̂
        and RMS-calibrated ŝ_v (shape (C,); C = 1 for 1-D ``y``),
        ``method="midas"``, linear ``fits`` (median intercept at t_ref +
        rate; covariance ``diag(NaN, ŝ_v²)``), pair/trim diagnostics
        (``n_pairs``, ``n_used``, ``scale_sigmas``,
        ``fraction_removed``), window provenance, and horizontal
        magnitude/azimuth products when derivable from ``names``.

    Raises:
        ValueError: On non-finite ``t``/``y`` (incl. an all-NaN
            component), unsorted ``t``, shape mismatches, a window with
            < 2 samples, a used time span < 1 − δ yr (**no interannual
            pair can exist** — this includes all-identical epochs), or
            fewer than ``max(min_pairs, 2)`` usable pairs.

    Reference:
        Blewitt, Kreemer, Hammond & Gazeaux 2016, *MIDAS robust trend
        estimator for accurate GPS station velocities without step
        detection*, JGR Solid Earth 121, 2054–2068,
        doi:10.1002/2015JB012552 — eqs. (1)–(8), §2.2–§2.5; reference
        implementation ``midas.f`` (MIDAS4, G. Blewitt). Blind-test
        accuracy: ±0.33 mm/yr horizontal, ±1.1 mm/yr up RMS on the
        Gazeaux et al. 2013 DOGEx synthetics (§3.2). Seasonal-bias
        motivation: Blewitt & Lavallée 2002, JGR 107(B7).

    Numerical notes:
        Float64 throughout; inputs never mutated. No per-epoch σ input:
        MIDAS is deliberately unweighted (medians give resistance to
        heteroscedasticity; §1.2). ŝ_v is calibrated against RMS
        accuracy on real, autocorrelated GPS series — on white-noise
        synthetics it overbounds the empirical scatter (by roughly the
        eq.-8 factor 3), which is by design, not a bug. Sub-daily/
        duplicate epochs are legal; pair Δt ≥ 1 − δ always, so the
        slope quotient never divides by ≈ 0.
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must be finite")
    yy, _was_1d = _components_2d(y, "y")
    if yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    if not np.all(np.isfinite(yy)):
        all_nan = [
            i for i in range(yy.shape[0]) if not bool(np.any(np.isfinite(yy[i])))
        ]
        detail = (
            f" (component(s) {all_nan} contain no finite value at all)"
            if all_nan
            else ""
        )
        raise ValueError(
            "y must be finite - MIDAS pair selection is shared across "
            "components, so drop non-finite epochs before calling" + detail
        )
    if names is not None and len(names) != yy.shape[0]:
        raise ValueError(f"names has {len(names)} entries for {yy.shape[0]} components")

    if window is None:
        mask = np.ones(tt.shape, dtype=np.bool_)
    else:
        mask = slice_window(tt, window[0], window[1], tol=tol)
    n_obs = int(np.count_nonzero(mask))
    if n_obs < 2:
        raise ValueError(f"window has {n_obs} samples - need at least 2 epochs")

    t_win = tt[mask]
    if bool(np.any(np.diff(t_win) < 0.0)):
        raise ValueError(
            "t must be sorted ascending - MIDAS pair selection scans in time order"
        )
    span = float(t_win[-1] - t_win[0])
    if span < 1.0 - pair_tol:
        raise ValueError(
            f"used time span {span:.6g} yr < {1.0 - pair_tol:.6g} yr - no "
            "interannual pair (Blewitt et al. 2016 eq. 3) can exist; MIDAS "
            "needs more than 1 year of data"
        )

    pairs = midas_pair_indices(t_win, pair_tol=pair_tol)
    n_pairs = int(pairs.shape[0])
    min_pairs_eff = max(int(min_pairs), 2)
    if n_pairs < min_pairs_eff:
        raise ValueError(
            f"only {n_pairs} slope pairs selected - need at least "
            f"{min_pairs_eff} (min_pairs; the reference implementation "
            "aborts below 10)"
        )
    idx_i = pairs[:, 0]
    idx_j = pairs[:, 1]
    dt = t_win[idx_j] - t_win[idx_i]
    t_ref = float(t_win[0])

    fits: list[TrajectoryParams] = []
    rates_list: list[float] = []
    sigmas_list: list[float] = []
    n_used_list: list[int] = []
    scale_list: list[float] = []
    for c in range(yy.shape[0]):
        x = yy[c][mask]
        slopes = (x[idx_j] - x[idx_i]) / dt
        v_hat, scale_sigma, n_used = midas_trimmed_median(
            slopes, trim_sigmas=trim_sigmas
        )
        s_v = midas_rate_uncertainty(scale_sigma, n_used)
        intercept = float(np.median(x - v_hat * (t_win - t_ref)))
        fits.append(
            TrajectoryParams(
                params=np.array([intercept, v_hat], dtype=np.float64),
                covariance=np.array(
                    [[np.nan, np.nan], [np.nan, s_v**2]], dtype=np.float64
                ),
                component=None if names is None else names[c],
            )
        )
        rates_list.append(v_hat)
        sigmas_list.append(s_v)
        n_used_list.append(n_used)
        scale_list.append(scale_sigma)

    rates = np.asarray(rates_list, dtype=np.float64)
    sigmas = np.asarray(sigmas_list, dtype=np.float64)
    magnitude, azimuth, magnitude_sigma, azimuth_sigma = _horizontal_products(
        names, rates, sigmas
    )

    return VelocityEstimateMIDAS(
        rates=rates,
        sigmas=sigmas,
        fits=tuple(fits),
        components=None if names is None else tuple(names),
        n_obs=n_obs,
        t_ref=t_ref,
        span=(float(t_win[0]), float(t_win[-1])),
        method=_METHOD_MIDAS,
        magnitude=magnitude,
        azimuth=azimuth,
        magnitude_sigma=magnitude_sigma,
        azimuth_sigma=azimuth_sigma,
        n_pairs=n_pairs,
        n_used=tuple(n_used_list),
        scale_sigmas=np.asarray(scale_list, dtype=np.float64),
        fraction_removed=(n_pairs - np.asarray(n_used_list, dtype=np.float64))
        / n_pairs,
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
    # As in estimate_velocity_mle: resolve through the attribute fallback, so a
    # step-augmented model keeps the prebuilt-basis fast path instead of silently
    # dropping to per-window curve_fit (same result, ~10-100x slower, and a
    # different covariance path with no warning that it happened).
    design = _resolve_linear_design(model_func)
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
