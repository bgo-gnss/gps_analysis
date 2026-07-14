"""Trajectory models for GNSS coordinate time series.

Consolidated from ``~/work/projects/gps_data_analyses`` (see
``docs/CONSOLIDATION_MAP.md``): the ``line``/``periodic``/``lineperiodic``
trio of ``detrend-reykjanes/detrend_rnes.py`` (copy-pasted across the
``detrend-*`` family) and the transient models of
``svartsengi_model.fitting`` (``expf_long``/``expf_short``/``dexpf*``,
``polynomial_transient``/``dpolynomial_transient`` and the polynomial peak
helpers). New names only — the legacy ``expf``/``secondorder`` aliases stay
behind.

Derivation chain
----------------
The standard GNSS trajectory model (Bevis & Brown 2014, J. Geodesy 88,
283–311, eq. 1–7) decomposes a coordinate component x(t) into a polynomial
(secular) part, seasonal oscillations, step offsets, and transient
(post-event) terms. This module provides the atomic model evaluators:

1. :func:`linear` — degree-1 polynomial part, x(t) = x₀ + v·t. The secular
   rate v is the primary product of :mod:`gps_analysis.velocity`.
2. :func:`periodic` — annual (angular frequency 2π yr⁻¹) plus semiannual
   (4π yr⁻¹) cosine/sine pairs. Seasonal terms must be co-estimated with v
   for windows ≲ 4.5 yr or the rate is biased (Blewitt & Lavallée 2002).
3. :func:`lineperiodic` — the sum of 1 and 2; the default detrending model
   of the legacy ``detrend-*`` scripts and of
   :func:`gps_analysis.fitting.detrend_fit`.
4. :func:`exp_linear` (+ :func:`exp_linear_rate`) — linear part plus a
   decaying-exponential transient (postseismic / magmatic relaxation);
   generalizes the legacy ``expf_long``/``expf_short``.
5. :func:`poly2` (+ :func:`poly2_rate`, :func:`poly2_peak_time`,
   :func:`poly2_peak_value`) — degree-2 polynomial transient (empirical
   viscoelastic relaxation proxy used for the Sundhnúkur intrusions).
6. :func:`heaviside_steps` — the known-step (Heaviside jump) term
   Σₖ aₖ·H(t − tₖ), H(0) = 1; composed with any of the above by
   :func:`gps_analysis.fitting.with_steps` for the step-augmented
   trajectory used in outlier detection
   (:mod:`gps_analysis.outliers`).

Fitted parameters and their covariance travel in the typed
:class:`TrajectoryParams` container (descendant of the svartsengi-model
``FitParameters``/``LongTermParams``/``TransientParams`` dataclasses),
which :mod:`gps_analysis.fitting` produces and
:mod:`gps_analysis.velocity` / :mod:`gps_analysis.baseline` consume.

Conventions (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------
- Time ``t`` is numeric, in **fractional years** (``yearf``); the seasonal
  phase in :func:`periodic` is defined by cos/sin of 2πt, so an absolute
  ``yearf`` puts phase zero at the calendar new year. If the caller
  re-references time (t → t − t_ref) the cos/sin coefficients absorb the
  phase shift — fitted coefficients are only comparable for a common time
  origin.
- Positions are **unit-agnostic**: the caller's length unit is written [L]
  (mm in IMO production); rates are [L/yr].
- All evaluators are pure: float64 arithmetic, no I/O, inputs never
  mutated; scalar ``t`` yields a 0-d ndarray.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FloatArray",
    "TrajectoryParams",
    "exp_linear",
    "exp_linear_rate",
    "heaviside_steps",
    "linear",
    "lineperiodic",
    "periodic",
    "poly2",
    "poly2_peak_time",
    "poly2_peak_value",
    "poly2_rate",
]

FloatArray = NDArray[np.float64]
"""1-D or n-D float64 array — the working dtype of this package."""


def _as_float_array(t: ArrayLike) -> FloatArray:
    """Coerce input to a float64 ndarray without copying when possible."""
    return np.asarray(t, dtype=np.float64)


def linear(t: ArrayLike, offset: float, rate: float) -> FloatArray:
    """Evaluate the linear (secular) trajectory model x(t).

    Equation:
        ``x(t) = x₀ + v·t``

    Symbols → args:
        - ``t``  → ``t``: epoch, fractional years (``yearf``) [yr]
        - ``x₀`` → ``offset``: position at t = 0 [L]
        - ``v``  → ``rate``: secular rate [L/yr]

    Args:
        t: Epochs at which to evaluate the model [yr].
        offset: Position intercept x₀ at t = 0 [L].
        rate: Secular rate v [L/yr].

    Returns:
        Model positions x(t) [L], float64, same shape as ``t``
        (0-d for scalar ``t``).

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (1) with polynomial degree
        m = 1. Legacy source: ``line`` in ``detrend_rnes.py``.

    Numerical notes:
        Exact float64 affine map — no stability concerns. For absolute
        ``yearf`` inputs (t ≈ 2×10³) the intercept x₀ refers to year 0 and
        is strongly anti-correlated with v in a fit; re-reference t when
        the intercept itself matters.
    """
    tt = _as_float_array(t)
    return np.asarray(offset + rate * tt, dtype=np.float64)


def periodic(
    t: ArrayLike,
    cos_annual: float,
    sin_annual: float,
    cos_semiannual: float,
    sin_semiannual: float,
) -> FloatArray:
    """Evaluate the seasonal (annual + semiannual) oscillation s(t).

    Equation:
        ``s(t) = a·cos(2πt) + b·sin(2πt) + c·cos(4πt) + d·sin(4πt)``

    Symbols → args:
        - ``t`` → ``t``: epoch, fractional years (``yearf``) [yr]
        - ``a`` → ``cos_annual``: annual cosine amplitude [L]
        - ``b`` → ``sin_annual``: annual sine amplitude [L]
        - ``c`` → ``cos_semiannual``: semiannual cosine amplitude [L]
        - ``d`` → ``sin_semiannual``: semiannual sine amplitude [L]

    Args:
        t: Epochs at which to evaluate the model [yr].
        cos_annual: Annual cosine amplitude a [L].
        sin_annual: Annual sine amplitude b [L].
        cos_semiannual: Semiannual cosine amplitude c [L].
        sin_semiannual: Semiannual sine amplitude d [L].

    Returns:
        Seasonal displacement s(t) [L], float64, same shape as ``t``.

    Reference:
        Blewitt & Lavallée 2002, JGR 107(B7), eq. (2) (the annual +
        semiannual truncation of the seasonal series); Bevis & Brown 2014,
        eq. (1), n_F = 2. Legacy source: ``periodic`` in
        ``detrend_rnes.py``.

    Numerical notes:
        Phase convention: with absolute ``yearf``, phase zero is the
        calendar new year; re-referencing t rotates (a, b) and (c, d).
        Deliberate signature change from the legacy ``periodic(x, p0…p5)``,
        which accepted — and silently ignored — the intercept/rate
        parameters p0, p1 so that ``lineperiodic`` parameter vectors could
        be reused; callers holding a full 6-parameter vector ``p`` should
        pass ``p[2:]``. Total amplitudes are √(a²+b²) and √(c²+d²).
    """
    tt = _as_float_array(t)
    two_pi_t = 2.0 * np.pi * tt
    seasonal = (
        cos_annual * np.cos(two_pi_t)
        + sin_annual * np.sin(two_pi_t)
        + cos_semiannual * np.cos(2.0 * two_pi_t)
        + sin_semiannual * np.sin(2.0 * two_pi_t)
    )
    return np.asarray(seasonal, dtype=np.float64)


def lineperiodic(
    t: ArrayLike,
    offset: float,
    rate: float,
    cos_annual: float,
    sin_annual: float,
    cos_semiannual: float,
    sin_semiannual: float,
) -> FloatArray:
    """Evaluate the secular-plus-seasonal trajectory model x(t).

    Equation:
        ``x(t) = x₀ + v·t + a·cos(2πt) + b·sin(2πt) + c·cos(4πt) + d·sin(4πt)``

    i.e. :func:`linear` + :func:`periodic`.

    Symbols → args:
        - ``t``  → ``t``: epoch, fractional years (``yearf``) [yr]
        - ``x₀`` → ``offset``: position at t = 0 [L]
        - ``v``  → ``rate``: secular rate [L/yr]
        - ``a, b`` → ``cos_annual``, ``sin_annual``: annual amplitudes [L]
        - ``c, d`` → ``cos_semiannual``, ``sin_semiannual``: semiannual
          amplitudes [L]

    Args:
        t: Epochs at which to evaluate the model [yr].
        offset: Position intercept x₀ at t = 0 [L].
        rate: Secular rate v [L/yr].
        cos_annual: Annual cosine amplitude a [L].
        sin_annual: Annual sine amplitude b [L].
        cos_semiannual: Semiannual cosine amplitude c [L].
        sin_semiannual: Semiannual sine amplitude d [L].

    Returns:
        Model positions x(t) [L], float64, same shape as ``t``.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (1) (m = 1, n_F = 2);
        Blewitt & Lavallée 2002, JGR 107(B7), eq. (2) — co-estimating the
        seasonal terms with v is mandatory for windows ≲ 4.5 yr. Legacy
        source: ``lineperiodic`` in ``detrend_rnes.py`` (identical
        parameter order p0…p5). This is the default model of
        :func:`gps_analysis.fitting.detrend_fit`.

    Numerical notes:
        Implemented as ``linear(t, …) + periodic(t, …)`` — bitwise
        association differs from the legacy single expression by at most
        one float64 rounding (≤ 1 ulp); tests assert equality at
        ``rtol = 1e-15``. The seasonal/rate parameters are correlated for
        short windows — inspect the covariance from
        :func:`gps_analysis.fitting.fit_components`, not just the values.
    """
    trend = linear(t, offset, rate)
    seasonal = periodic(t, cos_annual, sin_annual, cos_semiannual, sin_semiannual)
    return np.asarray(trend + seasonal, dtype=np.float64)


def exp_linear(
    t: ArrayLike,
    offset: float,
    rate: float,
    amplitude: float,
    decay_rate: float,
) -> FloatArray:
    """Evaluate the linear-plus-exponential-relaxation trajectory x(t).

    Equation:
        ``x(t) = x₀ + v·t + A·exp(−k·t)``

    Symbols → args:
        - ``t``  → ``t``: time since the reference epoch [yr]
        - ``x₀`` → ``offset``: asymptotic intercept [L]
        - ``v``  → ``rate``: secular rate [L/yr]
        - ``A``  → ``amplitude``: transient amplitude at t = 0 [L]
          (negative for relaxation approaching x₀ + v·t from below)
        - ``k``  → ``decay_rate``: exponential decay constant 1/τ [1/yr];
          half-life t½ = ln 2 / k

    Args:
        t: Time since the reference (event) epoch [yr].
        offset: Asymptotic intercept x₀ [L].
        rate: Secular rate v [L/yr].
        amplitude: Transient amplitude A at t = 0 [L].
        decay_rate: Decay constant k = 1/τ [1/yr].

    Returns:
        Model positions x(t) [L], float64, same shape as ``t``.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (6)–(7) (exponential
        transient term of the extended trajectory model). Legacy sources:
        ``svartsengi_model.fitting.expf_long`` ≡
        ``exp_linear(t, p0, 0, p1, p2)`` (steady-state magma-inflow
        equilibration, Svartsengi half-life ≈ 90–120 d ⇒ k ≈ 2–3 yr⁻¹)
        and ``expf_short`` ≡ ``exp_linear(t, 0, 0, p0, 1/p1)`` (note the
        legacy short form is τ-parameterized, exp(−t/τ)).

    Numerical notes:
        Evaluate with t referenced to the event epoch (t = yearf − t_event):
        for absolute ``yearf`` (t ≈ 2×10³) the factor exp(−k·t) underflows
        and A becomes unidentifiable. k and A are strongly anti-correlated
        when the window is short relative to τ = 1/k; for t < 0 the
        exponential grows — the caller masks pre-event epochs.
    """
    tt = _as_float_array(t)
    value = offset + rate * tt + amplitude * np.exp(-decay_rate * tt)
    return np.asarray(value, dtype=np.float64)


def exp_linear_rate(
    t: ArrayLike, rate: float, amplitude: float, decay_rate: float
) -> FloatArray:
    """Evaluate the instantaneous rate dx/dt of :func:`exp_linear`.

    Equation:
        ``dx/dt = v − A·k·exp(−k·t)``

    Symbols → args:
        - ``t`` → ``t``: time since the reference epoch [yr]
        - ``v`` → ``rate``: secular rate [L/yr]
        - ``A`` → ``amplitude``: transient amplitude at t = 0 [L]
        - ``k`` → ``decay_rate``: decay constant 1/τ [1/yr]

    Args:
        t: Time since the reference (event) epoch [yr].
        rate: Secular rate v [L/yr].
        amplitude: Transient amplitude A [L].
        decay_rate: Decay constant k [1/yr].

    Returns:
        Instantaneous rate dx/dt [L/yr], float64, same shape as ``t``.

    Reference:
        Analytic derivative of :func:`exp_linear` (Bevis & Brown 2014,
        eq. 6). Legacy sources: ``svartsengi_model.fitting.dexpf`` ≡
        ``exp_linear_rate(t, 0, p1, p2)``; ``dexpf_short`` ≡
        ``exp_linear_rate(t, 0, p0, 1/p1)``.

    Numerical notes:
        Same domain caveats as :func:`exp_linear` (event-referenced t;
        growth for t < 0). Exact analytic form — prefer it over finite
        differences of :func:`exp_linear`.
    """
    tt = _as_float_array(t)
    value = rate - amplitude * decay_rate * np.exp(-decay_rate * tt)
    return np.asarray(value, dtype=np.float64)


def poly2(t: ArrayLike, offset: float, rate: float, curvature: float) -> FloatArray:
    """Evaluate the degree-2 polynomial transient x(t).

    Equation:
        ``x(t) = p₀ + p₁·t + p₂·t²``

    Symbols → args:
        - ``t``  → ``t``: time since the event epoch [yr]
        - ``p₀`` → ``offset``: position at t = 0 [L]
        - ``p₁`` → ``rate``: initial rate [L/yr]
        - ``p₂`` → ``curvature``: quadratic coefficient [L/yr²]
          (negative for a decelerating transient with a maximum)

    Args:
        t: Time since the event epoch [yr].
        offset: Position p₀ at t = 0 [L].
        rate: Initial rate p₁ [L/yr].
        curvature: Quadratic coefficient p₂ [L/yr²].

    Returns:
        Model positions x(t) [L], float64, same shape as ``t``.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (1) with polynomial degree
        m = 2, used here as the empirical proxy for post-intrusion
        viscoelastic relaxation (legacy source:
        ``svartsengi_model.fitting.polynomial_transient``, applied to the
        Sundhnúkur intrusions; cf. Segall 2010 ch. 6 for the physics this
        stands in for).

    Numerical notes:
        The legacy usage caps the model at its maximum (constant for
        t > t_peak); that composition is the caller's business — combine
        with :func:`poly2_peak_time` / :func:`poly2_peak_value` and
        ``np.where``. Evaluate with event-referenced t to keep the
        monomials well-scaled.
    """
    tt = _as_float_array(t)
    value = offset + rate * tt + curvature * tt**2
    return np.asarray(value, dtype=np.float64)


def poly2_rate(t: ArrayLike, rate: float, curvature: float) -> FloatArray:
    """Evaluate the instantaneous rate dx/dt of :func:`poly2`.

    Equation:
        ``dx/dt = p₁ + 2·p₂·t``

    Symbols → args:
        - ``t``  → ``t``: time since the event epoch [yr]
        - ``p₁`` → ``rate``: initial rate [L/yr]
        - ``p₂`` → ``curvature``: quadratic coefficient [L/yr²]

    Args:
        t: Time since the event epoch [yr].
        rate: Initial rate p₁ [L/yr].
        curvature: Quadratic coefficient p₂ [L/yr²].

    Returns:
        Instantaneous rate dx/dt [L/yr], float64, same shape as ``t``.

    Reference:
        Analytic derivative of :func:`poly2`. Legacy source:
        ``svartsengi_model.fitting.dpolynomial_transient``.

    Numerical notes:
        Exact analytic form; zero at t_peak = −p₁/(2p₂)
        (:func:`poly2_peak_time`).
    """
    tt = _as_float_array(t)
    return np.asarray(rate + 2.0 * curvature * tt, dtype=np.float64)


def poly2_peak_time(rate: float, curvature: float) -> float:
    """Compute the stationary-point epoch t_peak of :func:`poly2`.

    Equation:
        ``t_peak = −p₁ / (2·p₂)``   (root of dx/dt = p₁ + 2·p₂·t)

    Symbols → args:
        - ``p₁`` → ``rate``: initial rate [L/yr]
        - ``p₂`` → ``curvature``: quadratic coefficient [L/yr²]

    Args:
        rate: Initial rate p₁ [L/yr].
        curvature: Quadratic coefficient p₂ [L/yr²]; must be nonzero
            (a maximum additionally requires p₂ < 0).

    Returns:
        Stationary-point epoch t_peak, time since the event epoch [yr].

    Raises:
        ValueError: If ``curvature`` is (numerically) zero — the model is
            linear and has no stationary point.

    Reference:
        Vertex of the degree-2 trajectory polynomial (Bevis & Brown 2014,
        eq. 1, m = 2). Legacy source:
        ``svartsengi_model.fitting.polynomial_peak_time``.

    Numerical notes:
        Zero-curvature is detected with ``np.isclose(p₂, 0)`` (absolute
        tolerance 1e-8, matching the legacy check); near-degenerate
        curvature yields an ill-determined, far-future t_peak — check the
        fitted p₂ against its uncertainty before trusting t_peak.
    """
    if np.isclose(curvature, 0.0):
        raise ValueError("curvature is zero - poly2 has no stationary point")
    return -rate / (2.0 * curvature)


def poly2_peak_value(offset: float, rate: float, curvature: float) -> float:
    """Compute the stationary-point value x(t_peak) of :func:`poly2`.

    Equation:
        ``x_peak = p₀ − p₁² / (4·p₂)``   (x evaluated at t_peak = −p₁/(2p₂))

    Symbols → args:
        - ``p₀`` → ``offset``: position at t = 0 [L]
        - ``p₁`` → ``rate``: initial rate [L/yr]
        - ``p₂`` → ``curvature``: quadratic coefficient [L/yr²]

    Args:
        offset: Position p₀ at t = 0 [L].
        rate: Initial rate p₁ [L/yr].
        curvature: Quadratic coefficient p₂ [L/yr²]; must be nonzero.

    Returns:
        Stationary-point value x_peak [L] (a maximum when p₂ < 0).

    Raises:
        ValueError: If ``curvature`` is (numerically) zero.

    Reference:
        Vertex value of the degree-2 polynomial. Legacy source:
        ``svartsengi_model.fitting.polynomial_peak_value``. The legacy
        transient composition holds the model at x_peak for t > t_peak.

    Numerical notes:
        Same ``np.isclose`` zero-curvature guard as
        :func:`poly2_peak_time`; consistent by construction with
        ``poly2(poly2_peak_time(p₁, p₂), p₀, p₁, p₂)`` to float64
        rounding.
    """
    if np.isclose(curvature, 0.0):
        raise ValueError("curvature is zero - poly2 has no stationary point")
    return offset - rate**2 / (4.0 * curvature)


def heaviside_steps(
    t: ArrayLike, epochs: ArrayLike, amplitudes: ArrayLike
) -> FloatArray:
    """Evaluate the sum of Heaviside step terms x_step(t).

    Equation:
        ``x_step(t) = Σ_{k=1}^{K} a_k·H(t − t_k)``  with  ``H(0) = 1``

    — the step epoch t_k belongs to the **post-step side** (the daily
    solution of the step day already contains the offset). Convention
    documented here and pinned by test.

    Symbols → args:
        - ``t``   → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``t_k`` → ``epochs``: known step epochs, shape (K,) [yr] —
          fixed data, e.g. equipment changes from TOS or coseismic
          offsets from the deployed ``steps.csv`` (caller's business)
        - ``a_k`` → ``amplitudes``: step amplitudes, shape (K,) [L]

    Args:
        t: Epochs at which to evaluate the model [yr].
        epochs: Step epochs t_k, shape (K,) [yr]; K = 0 yields zeros.
        amplitudes: Step amplitudes a_k, shape (K,) [L].

    Returns:
        Step displacement x_step(t) [L], float64, same shape as ``t``
        (0-d for scalar ``t``).

    Raises:
        ValueError: If ``epochs`` is not 1-D, ``amplitudes`` does not
            match its shape, or either is non-finite.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88, eq. (1) (the Heaviside jump
        term of the standard trajectory model); Gazeaux et al. 2013,
        JGR 118 (why *known-step tables* beat automated detection).

    Numerical notes:
        Exact comparison ``t ≥ t_k`` — no tolerance; callers must supply
        step epochs on the same ``yearf`` grid convention as ``t`` when
        an exact epoch match matters. Evaluated as the (N, K) indicator
        matrix times a: O(N·K), no stability concerns.
    """
    tt = _as_float_array(t)
    ep = np.asarray(epochs, dtype=np.float64)
    am = np.asarray(amplitudes, dtype=np.float64)
    if ep.ndim != 1:
        raise ValueError(f"epochs must be 1-D, got shape {ep.shape}")
    if am.shape != ep.shape:
        raise ValueError(
            f"amplitudes shape {am.shape} does not match epochs shape {ep.shape}"
        )
    if not (np.all(np.isfinite(ep)) and np.all(np.isfinite(am))):
        raise ValueError("epochs and amplitudes must be finite")
    if ep.size == 0:
        return np.zeros(tt.shape, dtype=np.float64)
    indicator = (tt[..., np.newaxis] >= ep).astype(np.float64)
    return np.asarray((indicator @ am).reshape(tt.shape), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class TrajectoryParams:
    """Fitted trajectory-model parameters for one coordinate component.

    Typed container pairing a parameter vector p̂ with its covariance
    C_p̂ = cov(p̂) as returned by weighted least squares
    (:func:`gps_analysis.fitting.fit_components`). Merges the
    svartsengi-model ``FitParameters`` / ``LongTermParams`` /
    ``TransientParams`` dataclasses into one shape-validated type
    (``docs/CONSOLIDATION_MAP.md``); the parameter order is the positional
    argument order of the model callable that produced the fit (e.g. for
    :func:`lineperiodic`: ``[offset, rate, cos_annual, sin_annual,
    cos_semiannual, sin_semiannual]``, so ``params[1]`` is the secular
    rate v [L/yr] and ``uncertainties[1]`` its 1-σ formal error).

    Attributes:
        params: Fitted parameter vector p̂, shape (P,), float64. Units are
            model-specific (see the model function's docstring).
        covariance: Parameter covariance matrix C_p̂, shape (P, P), float64
            [units of pᵢ·pⱼ]. May contain ``inf`` when the fit could not
            estimate it (singular Jacobian — ``scipy.optimize.curve_fit``
            convention).
        component: Optional caller-supplied label (e.g. ``"north"``);
            purely descriptive.

    Numerical notes:
        Arrays are coerced to float64 and shape-validated at construction
        (``params`` 1-D, ``covariance`` (P, P)); the dataclass is frozen
        but ndarrays are not immutable — treat the contents as read-only.
    """

    params: FloatArray
    covariance: FloatArray
    component: str | None = None

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

    @property
    def uncertainties(self) -> FloatArray:
        """1-σ formal parameter errors σ_pᵢ = √(C_p̂)ᵢᵢ.

        Equation:
            ``σ_pᵢ = √(diag C_p̂)ᵢ``

        Returns:
            Standard errors, shape (P,), float64 [units of pᵢ]. ``inf``
            where the covariance is undefined; formal errors assume the
            weights were true 1-σ uncertainties and white noise — they are
            optimistic for temporally correlated GNSS series
            (Williams 2003, J. Geodesy 76).
        """
        return np.asarray(np.sqrt(np.diag(self.covariance)), dtype=np.float64)

    def __len__(self) -> int:
        """Number of parameters P."""
        return int(self.params.size)

    def to_record(self) -> dict[str, Any]:
        """Serialize (p̂, C_p̂) to a JSON-ready per-component record.

        Mapping (``docs/DESIGN_live_detrending.md`` §5.2 — the leaf owns
        this per-component shape; the station/document schema is the
        caller's):

            ``{"params": [p̂₀ … p̂_{P−1}],``
            `` "cov_upper": [C₀₀, C₀₁, …, C_{P−1,P−1}] | None,``
            `` "component": str | None}``

        ``cov_upper`` is the row-major **upper triangle** of C_p̂ —
        P(P+1)/2 numbers, exact linear error propagation into any
        derived view (JCGM 100:2008 GUM §5.1.2) at negligible cost —
        or ``None`` when the covariance contains non-finite entries
        (the ``inf``-filled "could not be estimated" convention of
        :func:`gps_analysis.fitting.fit_components`; JSON has no inf).

        Returns:
            Plain dict of Python floats/lists — full ``repr`` precision
            (17 significant digits) survives ``json.dumps``/``loads``
            bit-exactly, so store → load → apply equals fit → apply.
            **The full parameter vector is serialized, including the
            intercept** — fixing the legacy ``detrend_itrf2008.csv``
            5-of-6 defect that forced the ``vshift`` re-referencing
            pass (design §0.1 #1). Pure: no I/O — file writing is the
            caller's business (leaf rule R2).

        Reference:
            Design spec ``docs/DESIGN_live_detrending.md`` §3.2/§5.2.

        Numerical notes:
            Round trip via :meth:`from_record` reproduces ``params``
            bit-exactly and the covariance exactly on and above the
            diagonal; the lower triangle is re-mirrored from the upper
            one, so an input asymmetric at the 1-ulp level (matrix-
            product rounding) comes back exactly symmetric. Any
            non-finite covariance entry collapses to ``cov_upper:
            None`` → all-``inf`` on load (no partial-NaN paths, design
            §3.2 rules).
        """
        if bool(np.all(np.isfinite(self.covariance))):
            iu = np.triu_indices(self.params.size)
            cov_upper: list[float] | None = [float(v) for v in self.covariance[iu]]
        else:
            cov_upper = None
        return {
            "params": [float(v) for v in self.params],
            "cov_upper": cov_upper,
            "component": self.component,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrajectoryParams":
        """Reconstruct :class:`TrajectoryParams` from a :meth:`to_record` dict.

        Inverse of :meth:`to_record` — see there for the record shape.
        ``cov_upper: None`` loads as a covariance filled with ``+inf``
        (the "could not be estimated" convention); a **hand-pinned**
        record (operator-edited ``params``) loads unchanged — no
        renormalization, no re-fitting (design §0.7).

        Args:
            record: Mapping with ``"params"`` (length-P list of finite
                floats), optional ``"cov_upper"`` (length P(P+1)/2 list
                or ``None``) and optional ``"component"`` (str or
                ``None``).

        Returns:
            A validated :class:`TrajectoryParams`.

        Raises:
            ValueError: If ``params`` is missing/not 1-D/empty/non-finite
                (no silent NaN-tolerant paths — design §3.2 rules), or
                ``cov_upper`` has the wrong length.

        Reference:
            Design spec ``docs/DESIGN_live_detrending.md`` §3.2/§5.2.

        Numerical notes:
            The covariance is rebuilt exactly symmetric from the upper
            triangle; ``params`` round-trip bit-exactly (float64 ↔
            17-significant-digit decimal is lossless).
        """
        if "params" not in record:
            raise ValueError("record has no 'params' entry")
        params = np.asarray(record["params"], dtype=np.float64)
        if params.ndim != 1 or params.size == 0:
            raise ValueError(
                f"params must be 1-D and non-empty, got shape {params.shape}"
            )
        if not np.all(np.isfinite(params)):
            raise ValueError("params must be finite (no NaN/inf)")
        p = int(params.size)
        cov_upper = record.get("cov_upper")
        if cov_upper is None:
            covariance = np.full((p, p), np.inf, dtype=np.float64)
        else:
            cu = np.asarray(cov_upper, dtype=np.float64)
            expected = p * (p + 1) // 2
            if cu.shape != (expected,):
                raise ValueError(
                    f"cov_upper must have P(P+1)/2 = {expected} entries for "
                    f"{p} parameters, got shape {cu.shape}"
                )
            covariance = np.zeros((p, p), dtype=np.float64)
            covariance[np.triu_indices(p)] = cu
            covariance = covariance + np.triu(covariance, k=1).T
        component = record.get("component")
        if component is not None and not isinstance(component, str):
            raise ValueError(f"component must be a string or None, got {component!r}")
        return cls(params=params, covariance=covariance, component=component)
