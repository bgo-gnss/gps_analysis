"""Transient detection — GBIS4TS velocity break-point analysis.

Python port of **GBIS4TS** (Yang, Sigmundsson, Geirsson 2023, *2023GL103432*;
MATLAB/BSD-2, vendored under ``reference/gbis4ts/``). Per station, per component
(N/E/U), fit a piecewise-linear trajectory with 1–2 **velocity break points**
under a **power-law + white** colored-noise model, sampled by Metropolis-Hastings
MCMC + simulated annealing — yielding honest posterior uncertainties on the rate
change and the break epoch.

Derivation chain
----------------
For one coordinate component y(t) [mm] sampled at daily epochs t [fractional
years], the estimator composes the atomic pieces below (each function cites the
originating ``.m`` file; ``reference/gbis4ts/SOURCE_MAP.md`` pins the map):

1. **Forward model** — :func:`bpd1_forward` / :func:`bpd2_forward`
   (``BPD1.m`` / ``BPD2.m``): continuity-preserving piecewise-linear trajectory
   ``y = a − v·t₀ + v·t + Σ_k g_k·H(t − t_bk)·(t − t*_k)`` with Heaviside
   convention ``H(0) = 1`` and ``t*_k`` the first sample epoch ≥ t_bk.
   **Seasonal-aware variants** (this port; :func:`bpd1_seasonal_forward` /
   :func:`bpd2_seasonal_forward`, model codes ``"BPD1S"`` / ``"BPD2S"``) add an
   annual+semiannual term ``+ sa·cos2πt + sb·sin2πt + sc·cos4πt + sd·sin4πt``
   (:func:`_seasonal_design`; Blewitt & Lavallée 2002 eq. 2, mirroring
   :func:`gps_analysis.models.periodic`). The four amplitudes sit BETWEEN the
   trajectory and noise blocks — vector
   ``[a, v, (g_k, tb_k)…, sa, sb, sc, sd, κ, β]`` — so the noise pair stays
   last and the seasonal-blind BPD1/BPD2 path is byte-unchanged (validated
   against the MATLAB + TS14). The MCMC **co-estimates** the seasonal terms
   with the break/rate/noise parameters (joint identification): on real GNSS
   windows ≲ 4.5 yr a seasonal signal aliases into the trend and break unless
   fit jointly (Blewitt & Lavallée 2002 — pre-removal fails, the seasonal
   estimate is itself ramp-contaminated on short windows).
2. **Residual** — ``r = y − model`` (``runInversion_ts.m`` l.114).
3. **Colored-noise covariance** — :func:`noise_covariance`
   (``UniVarMatrix.m``; Williams 2003, J. Geodesy 76, eq. 4; Yang et al. 2023
   SI Text S1 eq. 1–3): ``C = σ_w²·I + β²·(T₁T₁ᵀ)`` with T₁ a scaled
   lower-triangular Toeplitz transform of white noise into power-law noise of
   spectral index κ. σ_w is **fixed per station** (from pre-processing), κ and
   β are sampled. C itself is **not** Toeplitz — power-law noise with κ ≤ −1
   is nonstationary (Hosking 1981), so the diagonals of T₁T₁ᵀ grow with the
   epoch index (pinned by test; 22–75 % deviation for κ ∈ [−1.5, −0.7]).
4. **Gaussian log-likelihood** — :func:`log_likelihood`
   (``runInversion_ts.m`` l.132; Bagnardi & Hooper 2018, G³, §3):
   ``ln P = −rᵀC⁻¹r/2 − ln det C/2 − n·ln 2π/2``. The MCMC hot loop evaluates
   it WITHOUT forming C, via the generalized Schur algorithm on the rank-2
   displacement structure of C (:func:`_schur_logdet_quad`, task H3) — exact
   O(N²) per sample; :func:`log_likelihood` remains the dense O(N³) reference
   path for callers who already hold a covariance matrix.
5. **Sampler** — :func:`run_inversion` (``runInversion_ts.m``): Metropolis
   accept rule ``exp((P − P_prev)/T) ≥ U(0,1)`` under the simulated-annealing
   schedule ``T = 10^{3, 2.8, …, 0}`` (1000 iterations each), with periodic
   single-parameter sensitivity sweeps that retune step sizes toward a 77 %
   rejection rate, uniform priors enforced by boundary reflection, and a
   one-day floor on the break-point step.
6. **Priors / entry point** — :func:`prepare_bounds` builds the uniform search
   ranges of ``prepareModel_ts.m`` (= Yang et al. 2023 SI Table S3) from
   preliminary estimates; :func:`detect_breakpoints` is the one-call
   per-station orchestrator that the precompute job uses.

Conventions and caveats (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------------------
- Time is numeric **fractional years**; displacements **mm**; rates **mm/yr**;
  κ dimensionless; β in mm/yr^(−κ/4) (unit depends on κ — see SI Text S4).
- The covariance recursion is **index-lagged**: it assumes uniform *daily*
  sampling (the ΔT = 1/365 yr scaling is hard-coded upstream,
  ``UniVarMatrix.m`` l.27). Data gaps are treated as if absent — flagged for
  the GBIS4TS authors.
- Working dtype float64 throughout (upstream stores the chain in single —
  deliberate deviation, documented in :class:`InversionResult`).
- Pure leaf: numpy/scipy only, no I/O, inputs never mutated.
- **Zero-reference input conditioning** (this port): the intercept prior is
  hard-coded to ±5 mm (``prepareModel_ts.m`` l.51–52 — SI Table S3), i.e.
  GBIS4TS presumes series referenced near zero; ``GBISrun_ts.m`` l.110 even
  carries a commented-out force-zero line
  ``timeseries(:,2) - timeseries(1,2)``. Real ``.NEU`` series can start tens
  of mm from zero, saturating the intercept and leaking the offset into the
  trend. :func:`run_inversion` therefore fits ``y − r`` with the start
  baseline ``r`` = :func:`_start_baseline` (median of the first
  ``baseline_epochs`` samples) and reports intercepts back in the input
  frame; ``r`` is surfaced as :attr:`InversionResult.y_ref`. This is
  **numerical conditioning, not a physics change** — a constant offset
  carries no deformation information and ``v, dv, tb, κ, β`` are invariant
  to it (the same precedent as ``velocity.estimate_velocity`` re-referencing
  ``t`` to ``t_ref``). A post-fit guard raises if the conditioned intercept
  optimum still saturates its prior (pathological input). The ±5 mm prior is
  the seasonal-blind value; the seasonal variants (below) widen it because
  the median baseline carries the seasonal window-mean.
- Runtime (task H3, closed): each MCMC pass costs one exact O(N²)
  generalized-Schur factorization+solve (:func:`_schur_logdet_quad`) — ≈ 6 ms
  at N = 1825 vs ≈ 160–600 ms for the pre-H3 dense build+Cholesky (≈ 27–70×);
  parity with the dense path ≤ 4e-12 absolute in ln P (test-pinned).

Fidelity flags (reproduced as-is for parity; raise with the NVC authors)
------------------------------------------------------------------------
- ``runInversion_ts.m`` l.258: the BPD2 ordering guard swaps ``trial(3)`` and
  ``trial(5)`` — the *trend changes*, not the break points (l.264 then spaces
  the trend changes, not the epochs). Likely an index slip; preserved.
- The one-day break-point step floor (l.244) applies only to parameter 4
  (Breakpoint / Breakpoint1) — never to Breakpoint2 in BPD2. Preserved.
- The hyperparameter appended by ``prepareModel_ts.m`` l.120–124 is inert
  (fixed to 1 in the likelihood) but random-walks as a dummy dimension and
  counts toward the sensitivity target ``0.5^(1/n_model)``. Preserved.

Status: task **H1/H2 first draft** (plan lane §3). Validated against
``reference/gbis4ts/Verification/TS14.txt`` (= the Yang et al. 2023 synthetic
scheme β = 4 mm/yr^0.25, Δv = −20 mm/yr; reference posterior in SI Table S4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import daxpy, drot

__all__ = [
    "BPD1Params",
    "BPD2Params",
    "BPD1SeasonalParams",
    "BPD2SeasonalParams",
    "PriorBounds",
    "InversionConfig",
    "InversionResult",
    "bpd1_forward",
    "bpd2_forward",
    "bpd1_seasonal_forward",
    "bpd2_seasonal_forward",
    "noise_covariance",
    "log_likelihood",
    "prepare_bounds",
    "run_inversion",
    "detect_breakpoints",
]

# Heaviside convention from GBIS4TS: H(0) == 1 (sympref HeavisideAtOrigin, 1).

_LOG_2PI = float(np.log(2.0 * np.pi))

#: Daily sampling interval ΔT [yr] hard-coded upstream (UniVarMatrix.m l.27).
_DELTA_T_YR = 1.0 / 365.0

#: Number of parameters of the trajectory(+seasonal)+noise function per model.
#: layout: [a, v, (g_k, tb_k)×n_breaks, (sa, sb, sc, sd)?, κ, β] — noise last.
#: n_func = 2 + 2·n_breaks + 4·seasonal + 2.
_N_FUNC = {"BPD1": 6, "BPD2": 8, "BPD1S": 10, "BPD2S": 12}

#: Velocity break points per model code.
_N_BREAKS = {"BPD1": 1, "BPD2": 2, "BPD1S": 1, "BPD2S": 2}

#: Model codes carrying the annual+semiannual seasonal block (this port; the
#: seasonal-blind BPD1/BPD2 stay byte-parity with the MATLAB — SOURCE_MAP.md).
_SEASONAL_MODELS = frozenset({"BPD1S", "BPD2S"})

#: Number of seasonal amplitude parameters (annual cos/sin + semiannual cos/sin).
_N_SEASONAL = 4

#: Symmetric prior half-range for each seasonal cos/sin amplitude [mm].
#: Icelandic GNSS seasonal amplitudes run a few mm (horizontal) to ~1 cm+
#: (vertical hydrological loading); ±15 mm per coefficient gives the joint
#: annual/semiannual amplitude √(a²+b²) up to ~21 mm of headroom while keeping
#: the search bounded. Symmetric about 0 (a sign flip is a half-cycle phase
#: shift — no negative-rate swap applies).
_SEASONAL_AMP_BOUND = 15.0

#: MCMC step for each seasonal amplitude [mm] (mm-scale parameter; cf. the
#: 1 mm intercept step of prepareModel_ts.m).
_SEASONAL_STEP = 0.5

#: Intercept prior half-range for the SEASONAL models [mm] (widened from the
#: ±5 mm of the seasonal-blind BPD1/BPD2). The zero-reference baseline
#: ``r = median(y[:baseline_epochs])`` (:func:`_start_baseline`) cannot be
#: seasonally neutral over a short leading window — it carries the seasonal
#: window-mean — so after conditioning the intercept absorbs ``a_true − r ≈
#: −(seasonal window-mean)``, whose magnitude for an *in-prior* seasonal fit
#: is bounded by ``√(sa²+sb²) + √(sc²+sd²) ≤ 2·√2·A_max`` (each cos/sin pair
#: at its ±A_max corner, both terms near-constant over the short window). The
#: bound is sized to that worst case plus the ±5 mm genuine-DC allowance, so
#: the saturation guard can never false-fire on legitimate high-amplitude
#: (e.g. vertical-loading) seasonal data — it still catches a true DC leak
#: beyond what the seasonal terms can explain. The intercept is tightly
#: data-constrained regardless, so the wide prior costs no accuracy.
_SEASONAL_INTERCEPT_BOUND = 5.0 + 2.0 * math.sqrt(2.0) * _SEASONAL_AMP_BOUND

# Inert hyperparameter slot appended by prepareModel_ts.m l.120-124.
_HYPER_START = 0.0
_HYPER_STEP = 1.0e-3
_HYPER_LOWER = -0.5
_HYPER_UPPER = 0.5

#: Simulated-annealing cooling schedule 10.^(3:-0.2:0) (GBISrun_ts.m l.125).
_T_SCHEDULE: NDArray[np.float64] = 10.0 ** np.linspace(3.0, 0.0, 16)

#: Safety cap on the BPD2 trial-regeneration loop (upstream can spin forever).
_MAX_TRIAL_REGEN = 100_000

#: Saturation-guard margin as a fraction of the intercept prior range: an
#: intercept optimum within 5 % of either bound (0.5 mm for the fixed ±5 mm
#: prior of prepareModel_ts.m) is treated as wedged against the prior.
_INTERCEPT_GUARD_FRAC = 0.05


@dataclass(frozen=True)
class BPD1Params:
    """One-break trajectory + noise parameters (``BPD1.m`` / ``prepareModel_ts.m``).

    Units: displacement mm, time fractional year (``yearf``), rates mm/yr.
    """

    intercept: float
    trend1: float  # mm/yr, secular rate before the break
    trend_change: float  # mm/yr, rate change at the break
    breakpoint: float  # yearf of the velocity break
    kappa: float  # power-law spectral index of the colored noise
    amp: float  # power-law noise amplitude

    def as_array(self) -> NDArray[np.float64]:
        """Parameter vector in MATLAB order (``BPD1.m`` header comment)."""
        return np.array(
            [
                self.intercept,
                self.trend1,
                self.trend_change,
                self.breakpoint,
                self.kappa,
                self.amp,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BPD2Params:
    """Two-break trajectory + noise parameters (``BPD2.m``)."""

    intercept: float
    trend1: float
    trend_change1: float
    breakpoint1: float
    trend_change2: float
    breakpoint2: float
    kappa: float
    amp: float

    def as_array(self) -> NDArray[np.float64]:
        """Parameter vector in MATLAB order (``BPD2.m`` header comment)."""
        return np.array(
            [
                self.intercept,
                self.trend1,
                self.trend_change1,
                self.breakpoint1,
                self.trend_change2,
                self.breakpoint2,
                self.kappa,
                self.amp,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BPD1SeasonalParams:
    """One-break trajectory + annual/semiannual seasonal + noise parameters.

    Seasonal-aware variant of :class:`BPD1Params` (model code ``"BPD1S"``):
    the four seasonal amplitudes ``(sa, sb, sc, sd)`` sit **between** the
    trajectory and noise blocks so the noise pair stays last (κ, β), matching
    the ``[a, v, g, tb, sa, sb, sc, sd, κ, β]`` sampler layout. Units:
    displacement mm, time fractional year, rates mm/yr, amplitudes mm.
    """

    intercept: float
    trend1: float
    trend_change: float
    breakpoint: float
    cos_annual: float  # a: annual cosine amplitude [mm]
    sin_annual: float  # b: annual sine amplitude [mm]
    cos_semiannual: float  # c: semiannual cosine amplitude [mm]
    sin_semiannual: float  # d: semiannual sine amplitude [mm]
    kappa: float
    amp: float

    def as_array(self) -> NDArray[np.float64]:
        """Parameter vector in sampler order ``[a,v,g,tb,sa,sb,sc,sd,κ,β]``."""
        return np.array(
            [
                self.intercept,
                self.trend1,
                self.trend_change,
                self.breakpoint,
                self.cos_annual,
                self.sin_annual,
                self.cos_semiannual,
                self.sin_semiannual,
                self.kappa,
                self.amp,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BPD2SeasonalParams:
    """Two-break trajectory + annual/semiannual seasonal + noise parameters.

    Seasonal-aware variant of :class:`BPD2Params` (model code ``"BPD2S"``);
    layout ``[a, v, g1, tb1, g2, tb2, sa, sb, sc, sd, κ, β]`` (noise last).
    """

    intercept: float
    trend1: float
    trend_change1: float
    breakpoint1: float
    trend_change2: float
    breakpoint2: float
    cos_annual: float
    sin_annual: float
    cos_semiannual: float
    sin_semiannual: float
    kappa: float
    amp: float

    def as_array(self) -> NDArray[np.float64]:
        """Parameter vector ``[a,v,g1,tb1,g2,tb2,sa,sb,sc,sd,κ,β]``."""
        return np.array(
            [
                self.intercept,
                self.trend1,
                self.trend_change1,
                self.breakpoint1,
                self.trend_change2,
                self.breakpoint2,
                self.cos_annual,
                self.sin_annual,
                self.cos_semiannual,
                self.sin_semiannual,
                self.kappa,
                self.amp,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PriorBounds:
    """Uniform search bounds + MCMC step per parameter (``prepareModel_ts.m``).

    Order must match the model's parameter vector. Table S3 of Yang et al. 2023
    SI is the reference for the ranges (see :func:`prepare_bounds`, which builds
    them from preliminary estimates). Vectors may be of length ``n_func`` (6 for
    BPD1, 8 for BPD2) or ``n_func + 1`` when the inert hyperparameter slot is
    already appended; :func:`run_inversion` appends it if missing.

    Numerical notes:
        Arrays are coerced to float64 1-D of equal length at construction; the
        dataclass is frozen but ndarrays are not immutable — treat as read-only.
        Steps inherit the sign of the preliminary rate (upstream does not take
        absolute values); the sampler only uses them symmetrically, so a
        negative step is equivalent to its magnitude.
    """

    start: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    step: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in ("start", "lower", "upper", "step")
        }
        n = arrays["start"].size
        for name, arr in arrays.items():
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
            if arr.size != n:
                raise ValueError(
                    f"{name} has {arr.size} entries, expected {n} (same as start)"
                )
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class InversionConfig:
    """Sampler schedule (``runInversion_ts.m`` / ``GBISrun_ts.m``).

    Defaults mirror the MATLAB: annealing ``10**arange(3, -0.2, -0.2)``,
    ``t_runs=1000`` per temperature, breakpoint step floor ``0.0027`` yr (~1 day),
    sensitivity test tuned to a 77% rejection rate.

    Attributes:
        n_runs: Number of kept MCMC iterations (``invpar.nRuns``).
        n_save: Upstream save/print block size (``invpar.nSave``) — accepted
            for parity, unused here (the port neither prints nor writes).
        t_runs: Kept iterations per annealing temperature (``invpar.TRuns``);
            the schedule has 16 temperatures, so annealing spans
            ``16*t_runs`` iterations before T = 1.
        rejection_target: Target rejection rate of the adaptive-step retune
            (``runInversion_ts.m`` l.150 hard-codes 0.77).
        breakpoint_step_floor: Minimum |random step| for the break-point
            parameter [yr] (l.244; 0.0027 yr ≈ 1 day). Also spaces the BPD2
            trend changes at ``20×`` this value (l.264 — see the fidelity flag).
        seed: Seed for :class:`numpy.random.Generator`; ``None`` draws entropy
            from the OS (non-reproducible).
    """

    n_runs: int
    n_save: int = 1000
    t_runs: int = 1000
    rejection_target: float = 0.77
    breakpoint_step_floor: float = 0.0027
    seed: int | None = None


@dataclass(frozen=True)
class InversionResult:
    """Posterior draws + optimum from one station/component inversion.

    Attributes:
        m_keep: ``(n_func + 1, n_runs)`` chain of kept parameter vectors
            (rejected iterations repeat the previous column — MATLAB ``mKeep``).
            The last row is the inert hyperparameter dimension. Stored float64
            (upstream stores single — deliberate precision upgrade).
        p_keep: ``(n_runs,)`` log-posterior (= log-likelihood, flat priors) per
            kept column (MATLAB ``PKeep``).
        optimal: Best-likelihood trajectory(+seasonal)+noise parameter vector,
            without the hyperparameter slot — MATLAB ``results.optimalmodel``.
            Length by model: 6 (BPD1) / 8 (BPD2) / 10 (BPD1S) / 12 (BPD2S);
            the seasonal variants carry ``(sa, sb, sc, sd)`` before ``(κ, β)``.
        model: ``"BPD1"`` | ``"BPD2"`` | ``"BPD1S"`` | ``"BPD2S"``.
        y_ref: Start baseline r [mm] subtracted internally before fitting
            (:func:`_start_baseline`) and added back to the reported
            intercepts (``optimal[0]`` and ``m_keep[0, :]``), so the chain
            and optimum are in the **input frame**. 0.0 when conditioning was
            disabled (``baseline_epochs=0``). Provenance field — the analogue
            of ``VelocityEstimate.t_ref`` (MATH_STANDARDS §6).
    """

    m_keep: NDArray[np.float64]  # (n_params, n_kept) accepted parameter chain
    p_keep: NDArray[np.float64]  # (n_kept,) log-probability per kept sample
    optimal: NDArray[np.float64]  # best-probability parameter vector
    model: str  # "BPD1" | "BPD2"
    y_ref: float = 0.0  # start baseline subtracted before the fit [mm]


def _as_time_array(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coerce epochs to a non-empty 1-D float64 array (no copy if possible)."""
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if tt.size == 0:
        raise ValueError("t must not be empty")
    return tt


def _break_term(
    t: NDArray[np.float64], rate_change: float, breakpoint: float
) -> NDArray[np.float64]:
    """Evaluate one continuity-preserving velocity-break ramp g·H(t−t_b)·(t−t*).

    Equation (``BPD1.m`` l.15, one break term):
        ``Δy(t) = g·H(t − t_b)·t − g·H(t − t_b)·t* = g·H(t − t_b)·(t − t*)``
        with ``t* = first t ≥ t_b`` (array order) and ``H(0) = 1``.

    Symbols → args:
        - ``t``   → ``t``: sample epochs [yr, fractional year]
        - ``g``   → ``rate_change``: velocity change at the break [mm/yr]
        - ``t_b`` → ``breakpoint``: break epoch [yr]

    Reference:
        Yang et al. 2023, 2023GL103432, eq. 4 (break term); ``BPD1.m`` l.15
        (``sympref('HeavisideAtOrigin', 1)``).

    Numerical notes:
        ``t*`` is the *first array element* with t ≥ t_b (MATLAB
        ``find(...,1,'first')``) — equal to the smallest such epoch only for
        ascending ``t``. If no epoch satisfies t ≥ t_b the term is identically
        zero (H = 0 everywhere; upstream would error on the empty ``find``).
    """
    active = t >= breakpoint  # H(0) = 1: equality is inside the break regime
    if not bool(active.any()):
        return np.zeros_like(t)
    t_star = float(t[int(np.argmax(active))])
    return np.where(active, rate_change * (t - t_star), 0.0)


def _trajectory(
    m_func: NDArray[np.float64], t: NDArray[np.float64], n_breaks: int
) -> NDArray[np.float64]:
    """Piecewise-linear trajectory shared by BPD1/BPD2 (array-parameter core).

    ``m_func`` in MATLAB order: ``[a, v, g1, tb1(, g2, tb2), κ, β]`` — the
    noise parameters (last two) do not enter the trajectory. Used by the MCMC
    hot loop; :func:`bpd1_forward` / :func:`bpd2_forward` are the typed wrappers.
    """
    y = m_func[0] - m_func[1] * t[0] + m_func[1] * t
    y = y + _break_term(t, float(m_func[2]), float(m_func[3]))
    if n_breaks == 2:
        y = y + _break_term(t, float(m_func[4]), float(m_func[5]))
    return np.asarray(y, dtype=np.float64)


def bpd1_forward(params: BPD1Params, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the one-break piecewise-linear trajectory y(t) (``BPD1.m`` l.15).

    Equation:
        ``y(t) = a − v·t₀ + v·t + g·H(t − t_b)·(t − t*)``
        with ``t* = first t ≥ t_b`` and Heaviside ``H(0) = 1``. The ramp is
        anchored at t*, so y is continuous across the break (no jump) and the
        post-break slope is v + g.

    Symbols → args:
        - ``t``   → ``t``: sample epochs [yr, fractional year]; ``t₀ = t[0]``
        - ``a``   → ``params.intercept``: position at the first epoch [mm]
        - ``v``   → ``params.trend1``: pre-break secular rate [mm/yr]
        - ``g``   → ``params.trend_change``: rate change at the break [mm/yr]
        - ``t_b`` → ``params.breakpoint``: break epoch [yr]
        - ``params.kappa``, ``params.amp``: noise parameters — carried in the
          vector for MATLAB parity, **not used** by the trajectory.

    Returns:
        Model positions y(t) [mm], float64, new array (inputs untouched).

    Reference:
        Yang et al. 2023, 2023GL103432, eq. 4; ``BPD1.m`` l.15
        (`sympref('HeavisideAtOrigin', 1)`).

    Numerical notes:
        Exact float64 affine arithmetic. ``t*`` uses the first element in array
        order (see :func:`_break_term`); pass ``t`` sorted ascending for the
        intended "first epoch at/after the break" semantics. A break beyond the
        last epoch degenerates to the pure line (guarded; upstream errors).
    """
    tt = _as_time_array(t)
    return _trajectory(params.as_array(), tt, n_breaks=1)


def bpd2_forward(params: BPD2Params, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the two-break piecewise-linear trajectory y(t) (``BPD2.m`` l.15).

    Equation:
        ``y(t) = a − v·t₀ + v·t + g₁·H(t − t_b1)·(t − t*₁) + g₂·H(t − t_b2)·(t − t*₂)``
        with ``t*_k = first t ≥ t_bk`` and ``H(0) = 1`` — :func:`bpd1_forward`
        plus one more anchored ramp; slope is v before t_b1, v + g₁ between the
        breaks, v + g₁ + g₂ after t_b2 (for ordered breaks).

    Symbols → args:
        - ``a, v`` → ``params.intercept`` [mm], ``params.trend1`` [mm/yr]
        - ``g₁, t_b1`` → ``params.trend_change1`` [mm/yr], ``params.breakpoint1`` [yr]
        - ``g₂, t_b2`` → ``params.trend_change2`` [mm/yr], ``params.breakpoint2`` [yr]
        - ``params.kappa``, ``params.amp``: noise parameters, unused here.

    Returns:
        Model positions y(t) [mm], float64, new array.

    Reference:
        Yang et al. 2023, 2023GL103432, eq. 4 extended to two breaks;
        ``BPD2.m`` l.15–17.

    Numerical notes:
        Same conventions and guards as :func:`bpd1_forward`. The model is
        symmetric under swapping the (g₁, t_b1) and (g₂, t_b2) pairs; the
        sampler's ordering guard (see :func:`run_inversion`) is what breaks the
        label degeneracy.
    """
    tt = _as_time_array(t)
    return _trajectory(params.as_array(), tt, n_breaks=2)


def _seasonal_design(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Annual + semiannual seasonal design matrix D, columns ``[cos2πt, sin2πt, cos4πt, sin4πt]``.

    Equation (Blewitt & Lavallée 2002, JGR 107(B7), eq. 2; Bevis & Brown 2014,
    J. Geodesy 88, eq. 1 with n_F = 2):
        ``D[:, 0] = cos(2πt)``, ``D[:, 1] = sin(2πt)``,
        ``D[:, 2] = cos(4πt)``, ``D[:, 3] = sin(4πt)``
    so that ``D · [a, b, c, d] = a·cos(2πt) + b·sin(2πt) + c·cos(4πt) +
    d·sin(4πt)`` — algebraically identical to
    :func:`gps_analysis.models.periodic` (the shared house convention;
    equivalence pinned to float eps in the test suite). The design is built
    ONCE per :func:`run_inversion` call (t is fixed) so the MCMC hot loop adds
    the seasonal mean as a single (n×4)·4 matvec.

    Symbols → args:
        - ``t`` → ``t``: sample epochs [yr, fractional year] — **absolute**
          ``yearf``, so phase zero is the calendar new year (the seasonal
          phase must NOT be re-referenced; the zero-reference conditioning of
          :func:`run_inversion` shifts only the displacement, not t).

    Returns:
        D, shape (n, 4), float64.

    Reference:
        Blewitt & Lavallée 2002, JGR 107(B7), eq. 2 (annual+semiannual
        truncation; co-estimation with v mandatory for windows ≲ 4.5 yr);
        Bevis & Brown 2014, J. Geodesy 88, eq. 1. Mirrors
        :func:`gps_analysis.models.periodic` for cross-module consistency.

    Numerical notes:
        Exact float64 trig on the absolute epochs; the two angular
        frequencies are 2π and 4π yr⁻¹ (``ω_semiannual = 2·ω_annual``,
        computed as ``2·(2πt)`` to share the multiply). No re-referencing —
        see the phase caveat above.
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


def bpd1_seasonal_forward(
    params: BPD1SeasonalParams, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the one-break trajectory plus annual/semiannual seasonal y(t).

    Equation (additive; :func:`bpd1_forward` + :func:`_seasonal_design`):
        ``y(t) = a − v·t₀ + v·t + g·H(t − t_b)·(t − t*)
                 + sa·cos(2πt) + sb·sin(2πt) + sc·cos(4πt) + sd·sin(4πt)``
    with ``t* = first t ≥ t_b`` and ``H(0) = 1``. The seasonal block is
    orthogonal to the break/rate/noise parameters in the model (a distinct
    deterministic-mean term), so setting ``sa = sb = sc = sd = 0`` reproduces
    :func:`bpd1_forward` **exactly**.

    Symbols → args:
        - ``a, v, g, t_b`` → as :func:`bpd1_forward`
        - ``sa, sb, sc, sd`` → ``params.cos_annual`` / ``sin_annual`` /
          ``cos_semiannual`` / ``sin_semiannual``: seasonal amplitudes [mm]
        - ``params.kappa``, ``params.amp``: noise parameters, unused here.

    Returns:
        Model positions y(t) [mm], float64, new array (inputs untouched).

    Reference:
        Yang et al. 2023, 2023GL103432, eq. 4 (break term); Blewitt &
        Lavallée 2002, JGR 107(B7), eq. 2 and Bevis & Brown 2014, J. Geodesy
        88, eq. 1 (annual+semiannual seasonal — why it must be co-estimated,
        not pre-removed, on short windows).

    Numerical notes:
        Same conventions/guards as :func:`bpd1_forward`; seasonal via
        :func:`_seasonal_design` (absolute-``yearf`` phase — do not
        re-reference t).
    """
    tt = _as_time_array(t)
    m = params.as_array()
    return _trajectory(m, tt, n_breaks=1) + _seasonal_design(tt) @ m[4:8]


def bpd2_seasonal_forward(
    params: BPD2SeasonalParams, t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the two-break trajectory plus annual/semiannual seasonal y(t).

    Equation (:func:`bpd2_forward` + :func:`_seasonal_design`):
        ``y(t) = a − v·t₀ + v·t + g₁·H(t − t_b1)·(t − t*₁)
                 + g₂·H(t − t_b2)·(t − t*₂)
                 + sa·cos(2πt) + sb·sin(2πt) + sc·cos(4πt) + sd·sin(4πt)``.
    ``sa = sb = sc = sd = 0`` reproduces :func:`bpd2_forward` exactly.

    Symbols → args:
        - ``a, v, g₁, t_b1, g₂, t_b2`` → as :func:`bpd2_forward`
        - ``sa, sb, sc, sd`` → seasonal amplitudes [mm]
        - ``params.kappa``, ``params.amp``: noise parameters, unused here.

    Returns:
        Model positions y(t) [mm], float64, new array.

    Reference:
        As :func:`bpd1_seasonal_forward`, break term extended to two breaks
        (``BPD2.m`` l.15–17).

    Numerical notes:
        Same conventions/guards as :func:`bpd2_forward`; seasonal amplitudes
        occupy vector indices 6–9 (noise pair stays last).
    """
    tt = _as_time_array(t)
    m = params.as_array()
    return _trajectory(m, tt, n_breaks=2) + _seasonal_design(tt) @ m[6:10]


def _powerlaw_psi(n: int, kappa: float) -> NDArray[np.float64]:
    """Fractional-integration coefficients ψ of the power-law transform T.

    Equation (``UniVarMatrix.m`` l.19–22; Williams 2003, J. Geodesy 76, eq. 3):
        ``ψ₀ = 1``, ``ψ_i = ((i − 1 − κ/2)/i)·ψ_{i−1}``  (0-based) —
        the Maclaurin coefficients of ``(1 − L)^{κ/2}``:
        ``ψ_i = Γ(i − κ/2)/(Γ(−κ/2)·i!)``, i.e. fractional integration of
        order ``d = −κ/2`` (Hosking 1981, Biometrika 68, eq. 2.4). The
        process ``T·w`` (w white) has power spectrum ∝ f^κ; for κ ≤ −1
        (d ≥ ½) it is **nonstationary**, which is why ``T·Tᵀ`` is not a
        Toeplitz (stationary-autocovariance) matrix.

    Symbols → args:
        - ``n`` → ``n``: number of coefficients (epochs), > 0
        - ``κ`` → ``kappa``: power-law spectral index (dimensionless; ≤ 0
          for GNSS noise, −1 = flicker, −2 = random walk)

    Returns:
        ψ, shape (n,), float64. ψ_i ≥ 0 for κ ≤ 0.

    Reference:
        Hosking 1981, Biometrika 68(1), eq. 2.4; Williams 2003, J. Geodesy
        76, eq. 3 (transformation-matrix entries); ``UniVarMatrix.m`` l.19–22.

    Numerical notes:
        Evaluated as one ``cumprod`` of the exact recursion ratios — the same
        float64 operation sequence as the upstream MATLAB loop, so the parity
        test against the literal ``UniVarMatrix.m`` build stays exact. |ratio|
        < 1 for i ≥ 1 and κ ∈ [−1.5, 0]: ψ decays monotonically (∼ i^(κ/2−1)
        asymptotically), no overflow. κ = 0 gives ψ = (1, 0, …).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    psi = np.empty(n, dtype=np.float64)
    psi[0] = 1.0
    if n > 1:
        i = np.arange(1.0, n)
        psi[1:] = np.cumprod((i - 1.0 - kappa / 2.0) / i)
    return psi


def noise_covariance(
    n: int, wn_amp: float, kappa: float, pln_amp: float
) -> NDArray[np.float64]:
    """Build the white + power-law data covariance C (``UniVarMatrix.m``).

    Equation (Williams 2003, J. Geodesy 76, eq. 4; Yang et al. 2023 SI Text S1
    eq. 1–3):
        ``C = σ_w²·I + β²·(T₁T₁ᵀ)``, ``T₁ = ΔT^(−κ/4)·T``, ``ΔT = 1/365 yr``
        where ``T`` is the unit lower-triangular Toeplitz matrix with first
        column ψ from the recursion
        ``ψ₀ = 1``, ``ψ_i = ((i − 1 − κ/2)/i)·ψ_{i−1}`` (0-based; MATLAB
        ``b(i) = ((i−2−κ/2)/(i−1))·b(i−1)``, 1-based).

    Symbols → args:
        - ``n``   → ``n``: number of epochs (dimension of C)
        - ``σ_w`` → ``wn_amp``: white-noise amplitude [mm], fixed per station
        - ``κ``   → ``kappa``: power-law spectral index (dimensionless; ≤ 0
          for GNSS, −1 = flicker; sampled on [−1.5, 0], SI Table S3)
        - ``β``   → ``pln_amp``: power-law amplitude [mm/yr^(−κ/4)]

    Returns:
        C, shape (n, n), float64, symmetric positive-definite for
        ``wn_amp > 0`` (σ_w²·I plus a PSD term).

    Reference:
        Williams 2003, J. Geodesy 76, eq. 4 (the disabled CATS/Williams-2008
        angle form in ``UniVarMatrix.m`` l.34–47 is *not* ported — commented
        out upstream); Yang et al. 2023 SI Text S1.

    Numerical notes:
        - ``β²·(T₁T₁ᵀ)`` is assembled diagonal-by-diagonal via
          ``(T Tᵀ)[j+d, j] = Σ_{m=0}^{j} ψ_{m+d}·ψ_m`` (cumulative sums) —
          algebraically identical to the upstream dense product, O(n²) instead
          of O(n³) (tests assert equality with the naive Toeplitz build to
          rtol 1e-12).
        - The sum above depends on j, not only on the diagonal offset d: C is
          **not Toeplitz** (nonstationary power-law noise — see
          :func:`_powerlaw_psi`; deviation pinned by test). The MCMC hot loop
          therefore never builds C — it uses the exact O(n²) displacement-
          structure path :func:`_schur_logdet_quad` (task H3); this dense
          builder remains for diagnostics and as the parity reference.
        - The recursion lag is the **sample index**, i.e. uniform daily
          sampling is assumed (ΔT hard-coded, ``UniVarMatrix.m`` l.27); gaps
          are silently treated as absent — fidelity-preserved, flagged.
        - ψ decays for κ ∈ [−1.5, 0] (|factor| < 1 for i ≥ 1), so no overflow;
          κ = 0 gives ψ = (1, 0, …) hence C = (σ_w² + β²)·I exactly.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    psi = _powerlaw_psi(n, kappa)
    # (ΔT^(−κ/4))² — the squared scaling of T1 (UniVarMatrix.m l.27).
    scale_sq = float(_DELTA_T_YR ** (-kappa / 2.0))
    cov = np.zeros((n, n), dtype=np.float64)
    for d in range(n):
        diag_vals = np.cumsum(psi[d:] * psi[: n - d])
        j = np.arange(n - d)
        cov[j + d, j] = diag_vals
        if d:
            cov[j, j + d] = diag_vals
    cov *= pln_amp * pln_amp * scale_sq
    cov[np.diag_indices(n)] += wn_amp * wn_amp
    return cov


def log_likelihood(residual: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    """Gaussian log-likelihood ln P of a residual vector under covariance C.

    Equation (``runInversion_ts.m`` l.132, citing Koulali & Clarke 2021 eq. 8):
        ``ln P = −rᵀC⁻¹r/2 − ln det C/2 − n·ln 2π/2``

    Symbols → args:
        - ``r`` → ``residual``: data-minus-model vector, shape (n,) [mm]
        - ``C`` → ``cov``: data covariance, shape (n, n), SPD [mm²]
          (from :func:`noise_covariance`)

    Returns:
        ln P (float, ≤ 0 up to the covariance normalization; the *hyper-*
        parameter of the original GBIS formulation is fixed to 1 upstream and
        therefore absent).

    Raises:
        numpy.linalg.LinAlgError / scipy LinAlgError: if C is not positive
        definite (cannot happen for :func:`noise_covariance` output with
        ``wn_amp > 0``).

    Reference:
        Bagnardi & Hooper 2018, G³ 19, §3 (GBIS posterior); ``logdet.m``
        (Minka): ``ln det C = 2·Σ ln diag(chol(C))``.

    Numerical notes:
        One Cholesky factorization serves both the quadratic form (triangular
        solves, ``cho_solve``) and the log-determinant — never an explicit
        inverse (upstream ``Cov^(-1)``, ``runInversion_ts.m`` l.113, is the
        flagged O(N³) inefficiency). This is the dense O(N³) *reference* path
        for callers holding an arbitrary covariance; the MCMC hot loop uses
        the exact O(N²) :func:`_log_likelihood_fast` instead (task H3).
        ``check_finite=False`` skips redundant validation; inputs are not
        mutated.
    """
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"residual must be 1-D, got shape {r.shape}")
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (r.size, r.size):
        raise ValueError(f"cov shape {c.shape} does not match {r.size} residuals")
    factor = cho_factor(c, lower=True, check_finite=False)
    quad = float(r @ cho_solve(factor, r, check_finite=False))
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    return -0.5 * (quad + logdet + r.size * _LOG_2PI)


def _schur_logdet_quad(
    residual: NDArray[np.float64], wn_amp: float, kappa: float, pln_amp: float
) -> tuple[float, float]:
    """Exact ``ln det C`` and ``rᵀC⁻¹r`` in O(n²) via the generalized Schur algorithm.

    For ``C = σ_w²·I + β̃²·(T Tᵀ)`` (:func:`noise_covariance`, with
    ``β̃ = β·ΔT^(−κ/4)`` the scaled power-law amplitude and T the unit
    lower-triangular Toeplitz matrix of ψ = :func:`_powerlaw_psi`), C is not
    Toeplitz, but it has **displacement rank 2 with positive generators**:

        ``C − Z·C·Zᵀ = σ_w²·e₀e₀ᵀ + β̃²·ψψᵀ = G·Gᵀ``,
        ``G = [σ_w·e₀, β̃·ψ] ∈ ℝ^{n×2}``,

    where Z is the down-shift matrix (Z_{ij} = δ_{i,j+1}). Proof:
    ``(T Tᵀ)_{ij} = Σ_{k=0}^{min(i,j)} ψ_{i−k}ψ_{j−k}`` ⇒ subtracting the
    shifted copy leaves only the k = 0 term ψ_i·ψ_j, and ``I − Z I Zᵀ =
    e₀e₀ᵀ`` (identity verified to machine ε in the test suite).

    The generalized Schur recursion then delivers the columns of the exact
    Cholesky factor L (C = L·Lᵀ) one at a time: at step k an orthogonal
    Givens rotation Θ_k zeroes the second generator column at row k, after
    which the first generator column *is* ``L[k:, k]``; the next generator is
    ``[Z·L[:, k], g₂]``. Because both generator columns are positive
    semidefinite contributions (J = I₂, no hyperbolic rotations), every Θ_k
    is orthogonal and the factorization is backward stable, comparable to
    dense Cholesky (Chandrasekaran & Sayed 1996). Fused into the same pass:

        ``z_k = (r_k − Σ_{j<k} L_{kj} z_j)/L_{kk}``  (forward substitution),
        ``rᵀC⁻¹r = zᵀz``,  ``ln det C = 2·Σ_k ln L_{kk}``,

    so neither C nor L is ever stored — O(n) memory, ~2n² flops (vs the
    O(n²) dense build + n³/3 Cholesky + n² solve it replaces).

    Symbols → args:
        - ``r``   → ``residual``: data-minus-model vector, shape (n,) [mm]
        - ``σ_w`` → ``wn_amp``: white-noise amplitude [mm], > 0
        - ``κ``   → ``kappa``: power-law spectral index (dimensionless)
        - ``β``   → ``pln_amp``: power-law amplitude [mm/yr^(−κ/4)]

    Returns:
        ``(ln det C, rᵀC⁻¹r)`` as floats, algebraically identical to the
        dense :func:`noise_covariance` + Cholesky evaluation.

    Raises:
        numpy.linalg.LinAlgError: if a Schur pivot is not positive (cannot
            happen for ``wn_amp > 0``: every Schur complement of
            ``σ_w²·I + PSD`` keeps the σ_w²·I floor, so ``L_kk ≥ σ_w``).

    Reference:
        Kailath, Kung & Morf 1979, J. Math. Anal. Appl. 68, 395–407
        (displacement rank); Kailath & Sayed 1995, SIAM Review 37(3), §1–4
        (generalized Schur algorithm); Chandrasekaran & Sayed 1996, SIAM J.
        Matrix Anal. Appl. 17(4) (stability, positive-definite case);
        Hosking 1981, Biometrika 68(1) (ψ, nonstationarity for κ ≤ −1);
        Williams 2003, J. Geodesy 76, eq. 4 (the covariance itself). Levinson/
        Trench (Golub & Van Loan §4.7) do NOT apply — C is not Toeplitz.

    Numerical notes:
        - Parity with the dense Cholesky path: ≤ 3e-15 relative on both
          outputs across N ≤ 1825, κ ∈ [−1.5, 0] (test-pinned at rel 1e-11);
          the resulting |Δ ln P| ≤ ~4e-12 absolute.
        - In-place BLAS kernels: the rotation is one ``drot`` over the two
          generator columns and the substitution update one ``daxpy``. The
          shift Z is realized implicitly by index bookkeeping — at step k the
          first generator column lives in ``u[0:n−k]`` (logical rows k…n−1)
          and is shortened from the END, while v and w advance from the
          FRONT; the arrays never move. ``w[k]`` is overwritten by z_k after
          its last read (w doubles as z).
        - The rotation is skipped when ``v[k] == 0`` exactly (κ = 0 tail,
          or ``pln_amp = 0`` ⇒ L = σ_w·I exactly, matching the white-noise
          closed form).
        - Inputs are not mutated (the residual is copied once).
    """
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"residual must be 1-D, got shape {r.shape}")
    n = r.size
    # Generators: u = σ_w·e0 (logical column 1), v = β̃·ψ (column 2).
    v = (pln_amp * float(_DELTA_T_YR ** (-kappa / 4.0))) * _powerlaw_psi(n, kappa)
    u = np.zeros(n, dtype=np.float64)
    u[0] = wn_amp
    w = r.copy()  # residual → forward-substituted z, in place
    diag = np.empty(n, dtype=np.float64)  # L_kk
    hyp = math.hypot
    for k in range(n):
        m = n - k
        b = float(v[k])
        if b != 0.0:
            a = float(u[0])
            h = hyp(a, b)
            # Θ_k: zero v at row k; u[0:m] becomes L[k:, k].
            u, v = drot(
                u, v, a / h, b / h, n=m, offy=k, overwrite_x=True, overwrite_y=True
            )
        d = float(u[0])
        if d <= 0.0:
            raise np.linalg.LinAlgError(
                f"covariance not positive definite: Schur pivot {d} at step {k}"
            )
        diag[k] = d
        z_k = w[k] / d
        w[k] = z_k
        if m > 1:
            # w[k+1:] -= z_k · L[k+1:, k]
            w = daxpy(u, w, n=m - 1, offx=1, offy=k + 1, a=-z_k)
    logdet = 2.0 * float(np.sum(np.log(diag)))
    quad = float(w @ w)
    return logdet, quad


def _log_likelihood_fast(
    residual: NDArray[np.float64], wn_amp: float, kappa: float, pln_amp: float
) -> float:
    """Gaussian ln P of a residual under C(σ_w, κ, β) without forming C.

    Equation (identical to :func:`log_likelihood`):
        ``ln P = −rᵀC⁻¹r/2 − ln det C/2 − n·ln 2π/2``
    with the quadratic form and log-determinant from the exact O(n²)
    generalized-Schur pass :func:`_schur_logdet_quad` instead of a dense
    O(n³) Cholesky — the MCMC hot-loop path (task H3).

    Symbols → args: as in :func:`_schur_logdet_quad`.

    Reference:
        Bagnardi & Hooper 2018, G³ 19, §3 (likelihood); Kailath & Sayed 1995
        (fast factorization — see :func:`_schur_logdet_quad`).

    Numerical notes:
        Agrees with ``log_likelihood(r, noise_covariance(n, σ_w, κ, β))`` to
        ≤ ~4e-12 absolute in ln P (test-pinned; both paths are exact
        factorizations of the same matrix, differing only in rounding order).
    """
    logdet, quad = _schur_logdet_quad(residual, wn_amp, kappa, pln_amp)
    return -0.5 * (quad + logdet + np.asarray(residual).size * _LOG_2PI)


def prepare_bounds(start: NDArray[np.float64], model: str = "BPD1") -> PriorBounds:
    """Build the Table-S3 uniform prior bounds + steps (``prepareModel_ts.m``).

    Given preliminary estimates ``[â, v̂, ĝ₁, t̂₁(, ĝ₂, t̂₂), κ̂, β̂]``, the
    uniform search ranges are (Yang et al. 2023 SI Table S3):

    ========================  ================  ================
    parameter                 lower             upper
    ========================  ================  ================
    intercept a [mm]          −5                5
    initial velocity v        −v̂               2·v̂
    velocity change g_k       −ĝ_k             2·ĝ_k
    break point t_bk [yr]     t̂_k − 1 (BPD1)   t̂_k + 1 (BPD1)
                              t̂_k − 0.5 (BPD2) t̂_k + 0.5 (BPD2)
    seasonal ampl. sa…sd [mm] −A_max            +A_max
    spectral index κ          −1.5              0
    PLN amplitude β           0                 1.5·β̂
    ========================  ================  ================

    with lower/upper swapped where a negative preliminary rate inverts them
    (``prepareModel_ts.m`` l.54–105). Steps: 1 mm (a), ``0.05·v̂`` (every rate
    parameter — *all* trend steps derive from v̂ upstream), 0.0027 yr (breaks),
    ``_SEASONAL_STEP`` mm (each seasonal amplitude), 0.05 (κ), 1 (β). The inert
    hyperparameter slot (start 0, step 1e-3, bounds ±0.5; ``prepareModel_ts.m``
    l.120–124) is appended.

    Seasonal models (``"BPD1S"`` / ``"BPD2S"``, this port) add four
    annual/semiannual amplitudes ``(sa, sb, sc, sd)`` between the trajectory
    and noise blocks (see :func:`_seasonal_design`). Their prior is symmetric
    ``±A_max`` = ``_SEASONAL_AMP_BOUND`` (15 mm) — physically motivated by
    Icelandic GNSS seasonal amplitudes (a few mm horizontal to ~1 cm+ vertical
    loading); symmetric about 0 so the negative-rate swap does not touch them.
    The seasonal intercept prior is widened to ``±_SEASONAL_INTERCEPT_BOUND``
    (``5 + 2√2·A_max`` ≈ 47 mm, vs ±5 for the blind models): the plain-median
    zero-reference baseline carries the seasonal window-mean, so the
    conditioned intercept absorbs ``−(seasonal window-mean)`` (magnitude up to
    ``2√2·A_max`` for an in-prior fit) and would saturate a ±5 prior on
    high-amplitude data. Sizing to the worst case + ±5 keeps the saturation
    guard from ever false-firing on legitimate seasonal series (the intercept
    stays tightly data-constrained, so the wider prior costs nothing).

    Symbols → args:
        - ``start``: preliminary estimates in the sampler layout, length 6
          (BPD1 ``[â, v̂, ĝ, t̂, κ̂, β̂]``) / 8 (BPD2
          ``[â, v̂, ĝ₁, t̂₁, ĝ₂, t̂₂, κ̂, β̂]``) / 10 (BPD1S, seasonal
          ``[â, v̂, ĝ, t̂, ŝa, ŝb, ŝc, ŝd, κ̂, β̂]``) / 12 (BPD2S); mm,
          mm/yr, yr as above.
        - ``model``: ``"BPD1"`` | ``"BPD2"`` | ``"BPD1S"`` | ``"BPD2S"``.

    Returns:
        :class:`PriorBounds` of length ``n_func + 1`` (hyperparameter slot
        included; its start is forced to 0 as upstream).

    Reference:
        Yang et al. 2023, 2023GL103432, SI Table S3; ``prepareModel_ts.m``.

    Numerical notes:
        A preliminary rate of exactly 0 degenerates its range to a point and
        its step to 0 (the parameter freezes) — upstream behaves identically;
        supply a nonzero preliminary. Steps inherit the sign of v̂ (no absolute
        value upstream); the sampler uses them symmetrically.
    """
    if model not in _N_FUNC:
        raise ValueError(f"model must be one of {sorted(_N_FUNC)}, got {model!r}")
    n_func = _N_FUNC[model]
    s = np.asarray(start, dtype=np.float64)
    if s.shape != (n_func,):
        raise ValueError(
            f"start must have shape ({n_func},) for {model}, got {s.shape}"
        )
    ss, sb, ib = _SEASONAL_STEP, _SEASONAL_AMP_BOUND, _SEASONAL_INTERCEPT_BOUND
    if model == "BPD1":
        step = np.array([1.0, s[1] * 0.05, s[1] * 0.05, 0.0027, 0.05, 1.0])
        lower = np.array([-5.0, -s[1], -s[2], s[3] - 1.0, -1.5, 0.0])
        upper = np.array([5.0, 2.0 * s[1], 2.0 * s[2], s[3] + 1.0, 0.0, 1.5 * s[5]])
        swap_indices: tuple[int, ...] = (1, 2)
    elif model == "BPD2":
        step = np.array(
            [1.0, s[1] * 0.05, s[1] * 0.05, 0.0027, s[1] * 0.05, 0.0027, 0.05, 1.0]
        )
        lower = np.array([-5.0, -s[1], -s[2], s[3] - 0.5, -s[4], s[5] - 0.5, -1.5, 0.0])
        upper = np.array(
            [
                5.0,
                2.0 * s[1],
                2.0 * s[2],
                s[3] + 0.5,
                2.0 * s[4],
                s[5] + 0.5,
                0.0,
                1.5 * s[7],
            ]
        )
        swap_indices = (1, 2, 3, 4, 5)
    elif model == "BPD1S":
        # [a, v, g, tb, sa, sb, sc, sd, κ, β] — seasonal before the noise pair.
        step = np.array(
            [1.0, s[1] * 0.05, s[1] * 0.05, 0.0027, ss, ss, ss, ss, 0.05, 1.0]
        )
        lower = np.array([-ib, -s[1], -s[2], s[3] - 1.0, -sb, -sb, -sb, -sb, -1.5, 0.0])
        upper = np.array(
            [ib, 2.0 * s[1], 2.0 * s[2], s[3] + 1.0, sb, sb, sb, sb, 0.0, 1.5 * s[9]]
        )
        swap_indices = (1, 2)
    else:  # BPD2S: [a, v, g1, tb1, g2, tb2, sa, sb, sc, sd, κ, β]
        step = np.array(
            [
                1.0,
                s[1] * 0.05,
                s[1] * 0.05,
                0.0027,
                s[1] * 0.05,
                0.0027,
                ss,
                ss,
                ss,
                ss,
                0.05,
                1.0,
            ]
        )
        lower = np.array(
            [
                -ib,
                -s[1],
                -s[2],
                s[3] - 0.5,
                -s[4],
                s[5] - 0.5,
                -sb,
                -sb,
                -sb,
                -sb,
                -1.5,
                0.0,
            ]
        )
        upper = np.array(
            [
                ib,
                2.0 * s[1],
                2.0 * s[2],
                s[3] + 0.5,
                2.0 * s[4],
                s[5] + 0.5,
                sb,
                sb,
                sb,
                sb,
                0.0,
                1.5 * s[11],
            ]
        )
        swap_indices = (1, 2, 3, 4, 5)
    for k in swap_indices:
        if lower[k] > upper[k]:
            lower[k], upper[k] = upper[k], lower[k]
    return PriorBounds(
        start=np.append(s, _HYPER_START),
        lower=np.append(lower, _HYPER_LOWER),
        upper=np.append(upper, _HYPER_UPPER),
        step=np.append(step, _HYPER_STEP),
    )


def _sensitivity_schedule(n_runs: int) -> frozenset[int]:
    """Kept-iteration counts that trigger a sensitivity sweep (GBISrun_ts.m l.120).

    MATLAB: ``[1:100:10000, 11000:1000:30000, 40000:10000:nRuns]``.
    """
    parts = (
        np.arange(1, 10001, 100),
        np.arange(11000, 30001, 1000),
        np.arange(40000, n_runs + 1, 10000),
    )
    return frozenset(int(v) for arr in parts for v in arr)


def _start_baseline(y: NDArray[np.float64], n_baseline: int) -> float:
    """Start-of-series displacement baseline ``r = median(y₀ … y_{k−1})``.

    Equation:
        ``r = median({y_i : 0 ≤ i < k})``, ``k = min(n_baseline, n)`` —
        the robust location of the series over its leading ``n_baseline``
        samples (≈ days for daily GNSS series).

    Symbols → args:
        - ``y`` → ``y``: displacement series [mm], 1-D, array order = time order
        - ``n_baseline`` → ``n_baseline``: number of leading samples, ≥ 1

    Returns:
        r [mm] — the zero-reference baseline that :func:`run_inversion`
        subtracts before fitting and adds back to the reported intercepts
        (surfaced as :attr:`InversionResult.y_ref`).

    Reference:
        GBIS4TS input contract: ``prepareModel_ts.m`` l.51–52 hard-codes the
        intercept prior to ±5 mm (Yang et al. 2023 SI Table S3), and
        ``GBISrun_ts.m`` l.110 carries the commented-out force-zero line
        ``timeseries(:,2) - timeseries(1,2)`` — referencing the series to ≈0
        near its start is the intended usage. The median over a short leading
        window generalizes the single first sample robustly.

    Numerical notes:
        The median (50 % breakdown point) is immune to a leading outlier or
        gap-fill artifact that would bias ``y[0]`` or a mean; averaging over
        k samples suppresses the σ_w-level daily scatter (∝ 1/√k for the
        mean-like central region). The window is kept *short* because the
        intercept is defined AT the first epoch (``y(t₀) = a``): the
        secular-drift bias of the baseline is ≈ ``|v|·(k/2)/365`` yr-scaled —
        ~2 mm at 50 mm/yr for the default k = 30 — well inside the ±5 mm
        prior. Exact NumPy median (mean of the two central order statistics
        for even k); series shorter than ``n_baseline`` use all samples.
    """
    if n_baseline < 1:
        raise ValueError(f"n_baseline must be >= 1, got {n_baseline}")
    k = min(int(n_baseline), y.size)
    return float(np.median(y[:k]))


def run_inversion(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    wn_amp: float,
    config: InversionConfig,
    bounds: PriorBounds,
    model: str = "BPD1",
    *,
    baseline_epochs: int = 30,
) -> InversionResult:
    """Metropolis-Hastings MCMC + simulated annealing (``runInversion_ts.m``).

    Samples the posterior of the BPD1/BPD2 parameters under flat priors
    (:class:`PriorBounds`) and the colored-noise Gaussian likelihood:

    - **Accept rule** (l.132–141): ``P_ratio = exp((ln P − ln P_prev)/T)``;
      accept if ``P_ratio ≥ U(0, 1)``. The GBIS hyperparameter prefactor
      ``(h_prev/h)^{n/2}`` is identically 1 (h fixed to 1 upstream).
    - **Annealing** (GBISrun_ts.m l.125): ``T = 10^{3, 2.8, …, 0}``, advanced
      every ``t_runs`` kept iterations; on each advance the chain restarts from
      the current optimum. After 16 steps T = 1 (plain Metropolis).
    - **Adaptive steps** (l.145–163): on the sensitivity schedule
      (``[1:100:10000, 11000:1000:30000, 40000:10000:n_runs]``) each parameter
      is perturbed alone by ``±step/2`` and its acceptance-probability compared
      to a target ``0.5^{1/n_model}``, itself retuned by the running rejection
      rate toward ``rejection_target`` (77 %); steps shrink/grow by
      ``exp(∓2·ΔP/·)`` and are capped at the prior range. Sweep trials are
      never kept.
    - **Bounds**: reflection at both limits (l.249–253); the break-point step
      has a one-day floor (l.244, parameter 4 only — fidelity flag).
    - **BPD2 ordering guard** (l.255–269): reproduced **exactly as written** —
      it swaps ``trial[2]``/``trial[4]`` (the *trend changes*; the break points
      ``trial[3]``/``trial[5]`` are untouched — likely an index slip, see the
      module fidelity flags) and regenerates while
      ``|trial[4] − trial[2]| < 20 × breakpoint_step_floor``.
    - **Zero-reference conditioning** (this port; module conventions): the
      fixed ±5 mm intercept prior (``prepareModel_ts.m`` l.51–52) presumes a
      series referenced near zero (``GBISrun_ts.m`` l.110 commented-out
      force-zero line). The sampler therefore fits ``y − r`` with
      ``r`` = :func:`_start_baseline` ``(y, baseline_epochs)`` and reports
      the intercept back in the **input frame** — ``optimal[0]`` and the
      chain row ``m_keep[0, :]`` get ``+r``; all other parameters
      (``v, dv, tb, κ, β``) are shift-invariant and untouched. This is
      numerical conditioning, not a physics change (precedent:
      ``velocity.estimate_velocity``'s ``t_ref``). The intercept entries of
      ``bounds`` (start and the ±5 range) refer to the **conditioned** frame.
    - **Saturation guard**: if the conditioned-frame intercept optimum lies
      within ``0.05 × range`` of either prior bound (0.5 mm for the ±5 mm
      blind prior; the seasonal models use the wider
      ``±_SEASONAL_INTERCEPT_BOUND``), the fit is rejected with
      :class:`ValueError` — the data is pathological even after referencing
      (e.g. a step or steep drift inside the baseline window) and the offset
      would silently leak into the trend.

    Symbols → args:
        - ``t``: epochs [yr], ``y``: displacements [mm], 1-D, same length
        - ``σ_w`` → ``wn_amp``: fixed white-noise amplitude [mm], > 0
        - ``config``: :class:`InversionConfig` (schedule + seed)
        - ``bounds``: :class:`PriorBounds`, length ``n_func`` (hyper slot
          appended automatically) or ``n_func + 1``
        - ``model``: ``"BPD1"`` | ``"BPD2"`` | ``"BPD1S"`` | ``"BPD2S"`` — the
          ``…S`` variants co-estimate an annual+semiannual seasonal term
          (:func:`_seasonal_design`) jointly with the break/rate/noise
          parameters; the seasonal-blind path is byte-unchanged.
        - ``baseline_epochs``: leading samples for the start-baseline median
          ``r`` (≈ days for daily series). Default 30: robust to a leading
          outlier, scatter-suppressing, yet short enough that secular drift
          stays well inside the ±5 mm prior (see :func:`_start_baseline`).
          ``0`` disables conditioning (``y_ref = 0``) — the exact upstream
          input contract, for callers who reference explicitly.

    Returns:
        :class:`InversionResult` — kept chain, kept log-posteriors, optimum
        (intercepts in the input frame), and the baseline ``y_ref = r``.

    Raises:
        ValueError: on shape mismatches, ``wn_amp ≤ 0``, unknown model,
            ``baseline_epochs < 0``, a start vector outside the bounds
            (upstream errors identically), or a saturated intercept optimum
            (the guard above — never silently returned).
        RuntimeError: if the BPD2 regeneration loop exceeds a safety cap
            (upstream would spin forever — deliberate guarded deviation).

    Reference:
        Bagnardi & Hooper 2018, G³ 19, §3 (GBIS sampler); Yang et al. 2023,
        2023GL103432 (time-series adaptation); ``runInversion_ts.m``.

    Numerical notes:
        - Per kept iteration: one exact O(n²) generalized-Schur likelihood
          (:func:`_log_likelihood_fast`; task H3) — ~6 ms at n = 1825 vs
          ~160–600 ms for the pre-H3 dense build+Cholesky, i.e. a production
          ``n_runs = 1e6`` chain drops from days to ~2–4 h.
        - ``exp((P − P_prev)/T)`` is capped at ``exp(700)`` to avoid float64
          overflow; the accept decision is unchanged (both compare ≥ U < 1).
        - MCMC is stochastic: results are reproducible only for a fixed
          ``config.seed`` and identical numpy/scipy builds; cross-
          implementation parity is statistical (posterior location/spread),
          not bitwise.
    """
    tt = _as_time_array(t)
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != tt.shape:
        raise ValueError(f"y shape {yy.shape} does not match t shape {tt.shape}")
    if not wn_amp > 0.0:
        raise ValueError(f"wn_amp must be > 0, got {wn_amp}")
    if model not in _N_FUNC:
        raise ValueError(f"model must be one of {sorted(_N_FUNC)}, got {model!r}")
    if baseline_epochs < 0:
        raise ValueError(f"baseline_epochs must be >= 0, got {baseline_epochs}")
    n_func = _N_FUNC[model]
    n_breaks = _N_BREAKS[model]
    has_seasonal = model in _SEASONAL_MODELS
    n_traj = 2 + 2 * n_breaks  # a, v, (g_k, tb_k)×n_breaks — seasonal follows

    # --- Zero-reference conditioning (module conventions; input untouched) ---
    y_ref = _start_baseline(yy, baseline_epochs) if baseline_epochs else 0.0
    if y_ref != 0.0:
        yy = yy - y_ref  # new array — the caller's y is never mutated

    # Seasonal design D (annual+semiannual), built ONCE from the FIXED epochs:
    # conditioning shifts yy only, so the seasonal phase (absolute yearf) is
    # preserved. The hot loop adds the seasonal mean as a single D·[sa..sd].
    seasonal_d = _seasonal_design(tt) if has_seasonal else None

    # --- Parameter vectors (hyper slot appended if absent) -------------------
    if bounds.start.size == n_func:
        m = np.append(bounds.start, _HYPER_START)
        lower = np.append(bounds.lower, _HYPER_LOWER)
        upper = np.append(bounds.upper, _HYPER_UPPER)
        step = np.append(bounds.step, _HYPER_STEP)
    elif bounds.start.size == n_func + 1:
        m = bounds.start.copy()
        lower = bounds.lower.copy()
        upper = bounds.upper.copy()
        step = bounds.step.copy()
    else:
        raise ValueError(
            f"bounds must have {n_func} or {n_func + 1} entries for {model}, "
            f"got {bounds.start.size}"
        )
    if bool(np.any(m > upper)) or bool(np.any(m < lower)):
        bad = np.nonzero((m > upper) | (m < lower))[0]
        raise ValueError(f"starting model out of bounds at indices {bad.tolist()}")

    n_model = m.size
    n_obs = tt.size
    prm_range = upper - lower  # model.range
    prob_target = 0.5 ** (1.0 / n_model)
    prob_sens = np.zeros(n_model, dtype=np.float64)
    sens_schedule = _sensitivity_schedule(config.n_runs)
    bp_floor = config.breakpoint_step_floor

    rng = np.random.default_rng(config.seed)

    m_keep = np.zeros((n_model, config.n_runs), dtype=np.float64)
    p_keep = np.zeros(config.n_runs, dtype=np.float64)

    i_keep = 0
    i_reject = 0
    i_keep_save = 0
    i_reject_save = 0
    p_opt = -1.0e99
    p_prev = -np.inf  # set on first (always accepted) iteration
    hyper_prev = 1.0  # idem
    optimal_full = m.copy()
    func_opt = m[:n_func].copy()
    i_temp = 0
    n_temp = _T_SCHEDULE.size
    temperature = _T_SCHEDULE[0]
    sensitivity_test = 0
    set_hyper = False
    trial = m.copy()

    while i_keep < config.n_runs:
        # -- Annealing schedule (runInversion_ts.m l.74-88) -------------------
        if i_keep % config.t_runs == 0 and i_temp < n_temp:
            temperature = float(_T_SCHEDULE[i_temp])
            i_temp += 1
            if i_keep > 0:
                trial = optimal_full.copy()  # restart from current optimum
            set_hyper = temperature == 1.0

        if i_keep in sens_schedule:
            sensitivity_test = 1

        # -- Forward model + colored-noise likelihood (l.95-132) --------------
        m_func = trial[:n_func]
        u = _trajectory(m_func, tt, n_breaks)
        if seasonal_d is not None:
            # Co-estimated seasonal mean D·[sa,sb,sc,sd] (indices n_traj..+4);
            # orthogonal to the covariance (deterministic-mean term only).
            u = u + seasonal_d @ m_func[n_traj : n_traj + _N_SEASONAL]
        residual = yy - u
        if set_hyper:
            hyper_prev = 1.0  # l.121: hyperparameter pinned to 1 at T = 1
            trial[-1] = 0.0  # log10(1)
            set_hyper = False
        hyper_param = 1.0
        # Exact O(n²) evaluation of ln P under C(σ_w, κ, β) — never forms C
        # (generalized Schur on the displacement structure; task H3).
        log_p = _log_likelihood_fast(
            residual, wn_amp, float(trial[-3]), float(trial[-2])
        )

        if i_keep > 0:
            # (hyper_prev/hyper_param)^(n/2) ≡ 1; exp capped against overflow.
            p_ratio = (hyper_prev / hyper_param) ** (n_obs / 2.0) * math.exp(
                min((log_p - p_prev) / temperature, 700.0)
            )
        else:
            p_ratio = 1.0  # first iteration always kept (l.140)

        # -- Sensitivity bookkeeping / accept-reject (l.145-225) --------------
        if sensitivity_test > 1:
            idx = sensitivity_test - 2  # parameter whose perturbation this was
            if idx < n_model:
                # (Beyond n_model MATLAB grows probSens and later crashes on
                # the step update — the BPD2 regen loop can over-increment.
                # We drop the overflow entry instead; flagged deviation.)
                prob_sens[idx] = p_ratio
            if sensitivity_test > n_model:  # sweep complete: retune steps
                if i_keep_save > 0:
                    rejection_ratio = (i_reject - i_reject_save) / (
                        i_keep - i_keep_save
                    )
                    prob_target = max(
                        prob_target * rejection_ratio / config.rejection_target,
                        1.0e-6,
                    )
                sensitivity_test = 0
                ps = prob_sens.copy()
                above = ps > 1.0
                ps[above] = 1.0 / ps[above]
                p_diff = prob_target - ps
                shrink = p_diff > 0.0
                step[shrink] *= np.exp(-p_diff[shrink] / prob_target * 2.0)
                grow = p_diff < 0.0
                step[grow] *= np.exp(-p_diff[grow] / (1.0 - prob_target) * 2.0)
                too_big = step > prm_range
                step[too_big] = prm_range[too_big]
                i_keep_save = i_keep
                i_reject_save = i_reject
        else:
            i_keep += 1
            if p_ratio >= rng.random():
                m = trial.copy()
                m_keep[:, i_keep - 1] = m
                p_keep[i_keep - 1] = log_p
                p_prev = log_p
                hyper_prev = hyper_param
                if log_p > p_opt:
                    optimal_full = m.copy()
                    func_opt = m_func.copy()
                    p_opt = log_p
            else:
                i_reject += 1
                m_keep[:, i_keep - 1] = m_keep[:, i_keep - 2]
                p_keep[i_keep - 1] = p_keep[i_keep - 2]

        # -- Next trial (l.229-269) -------------------------------------------
        for _ in range(_MAX_TRIAL_REGEN):
            if sensitivity_test > 0:
                # Single-parameter sensitivity perturbation of ±step/2.
                k = sensitivity_test - 1
                trial = m.copy()
                if k < n_model:
                    trial[k] += step[k] * float(np.sign(rng.standard_normal())) / 2.0
                    if trial[k] > upper[k]:  # upper only, as upstream (l.235)
                        trial[k] -= step[k]
                sensitivity_test += 1
            else:
                random_step = step * (rng.random(n_model) - 0.5) * 2.0
                # One-day floor on the break-point step (parameter 4 only).
                if abs(random_step[3]) < bp_floor:
                    random_step[3] = bp_floor * float(np.sign(random_step[3]))
                trial = m + random_step
                over = trial > upper  # reflection at the bounds (l.249-253)
                trial[over] = 2.0 * upper[over] - trial[over]
                under = trial < lower
                trial[under] = 2.0 * lower[under] - trial[under]
            if n_breaks == 1:
                break
            # BPD2 ordering guard — fidelity flag: swaps the TREND CHANGES
            # (indices 2/4 == MATLAB trial(3)/trial(5)), not the break points.
            # Trajectory params lead the vector, so 2/4 are g1/g2 for BPD2S too.
            if trial[2] > trial[4]:
                trial[2], trial[4] = trial[4], trial[2]
            if abs(trial[4] - trial[2]) >= bp_floor * 20.0:
                break
        else:
            raise RuntimeError(
                "BPD2 trial regeneration exceeded the safety cap - the prior "
                "ranges force |trend_change2 - trend_change1| below the "
                f"minimum separation {bp_floor * 20.0}"
            )

    # --- Saturation guard (conditioned frame) --------------------------------
    a_lo, a_hi = float(lower[0]), float(upper[0])
    if a_hi > a_lo:  # a zero-width intercept prior means "deliberately pinned"
        margin = _INTERCEPT_GUARD_FRAC * (a_hi - a_lo)
        a_opt = float(func_opt[0])
        if a_opt <= a_lo + margin or a_opt >= a_hi - margin:
            raise ValueError(
                f"intercept optimum {a_opt:.3f} mm saturates its prior "
                f"[{a_lo:g}, {a_hi:g}] mm (margin {margin:.3f}) even after "
                f"zero-reference conditioning (y_ref = {y_ref:.3f} mm, "
                f"baseline_epochs = {baseline_epochs}): the residual offset "
                "would leak into the trend. Inspect the series start "
                "(step/outliers/steep drift inside the baseline window?) or "
                "reference it explicitly before calling."
            )

    # --- Intercept frame round-trip: report in the input frame ---------------
    if y_ref != 0.0:
        func_opt[0] += y_ref
        m_keep[0, :] += y_ref

    return InversionResult(
        m_keep=m_keep, p_keep=p_keep, optimal=func_opt, model=model, y_ref=y_ref
    )


def _preliminary_start(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    wn_amp: float,
    n_breaks: int,
    seasonal: bool = False,
) -> NDArray[np.float64]:
    """Coarse OLS grid seed ``[â, v̂, ĝ.., t̂.., (ŝa..ŝd,) κ̂, β̂]`` for :func:`prepare_bounds`.

    Stand-in for the GBIS4TS pre-processing (WLS + variogram noise estimation,
    ``Variogram/*`` — a later slice): break epochs are grid-searched over ≤ 96
    interior sample epochs (5th–95th index percentile) minimizing the OLS RSS
    of the trajectory design ``[1, t − t₀, H·(t − t*)…]``; for two breaks the
    second epoch is scanned greedily given the first (min separation 20 days),
    then the pair is relabeled ascending. Noise seeds are heuristic:
    ``κ̂ = −1`` (flicker) and ``β̂ = √max(var(r) − σ_w², σ_w²/4)`` [mm ≈
    mm/yr^0.25 at κ = −1]. The intercept is clipped into the fixed ±5 mm prior.
    For production parity pass explicit ``start`` values instead
    (:func:`detect_breakpoints`).

    Seasonal (``seasonal=True``): the four annual+semiannual design columns
    (:func:`_seasonal_design`) are appended to the OLS design so the seasonal
    amplitudes are **co-estimated with** the break/rate at every grid epoch
    (joint, never pre-removed — Blewitt & Lavallée 2002: a pre-removed
    seasonal is ramp-contaminated on short windows). The four ŝ estimates are
    inserted between the trajectory and noise seeds, matching the sampler
    layout ``[â, v̂, ĝ.., t̂.., ŝa, ŝb, ŝc, ŝd, κ̂, β̂]``.
    """
    n = t.size
    lo = max(int(0.05 * n), 1)
    hi = min(int(0.95 * n), n - 2)
    if hi <= lo:
        raise ValueError(f"series too short for break detection: {n} epochs")
    candidates = np.unique(np.linspace(lo, hi, 96).astype(np.int64))
    ones = np.ones(n, dtype=np.float64)
    t_rel = t - t[0]
    # Seasonal design columns (co-estimated in the seed); empty if seasonal-blind.
    seas_cols = list(_seasonal_design(t).T) if seasonal else []
    n_seas = len(seas_cols)

    def _fit(
        columns: list[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], float]:
        g_mat = np.column_stack(columns + seas_cols)
        coef, _, _, _ = np.linalg.lstsq(g_mat, y, rcond=None)
        res = y - g_mat @ coef
        return np.asarray(coef, dtype=np.float64), float(res @ res)

    best_rss = np.inf
    best_tb1 = float(t[candidates[0]])
    best_coef = np.zeros(3 + n_seas)
    for idx in candidates:
        tb = float(t[idx])
        coef, rss = _fit([ones, t_rel, _break_term(t, 1.0, tb)])
        if rss < best_rss:
            best_rss, best_tb1, best_coef = rss, tb, coef

    if n_breaks == 1:
        a_hat, v_hat, g_hat = best_coef[:3]
        seasonal_hat = best_coef[3 : 3 + n_seas]
        residual = y - (a_hat + v_hat * t_rel + g_hat * _break_term(t, 1.0, best_tb1))
        trajectory = np.array(
            [float(np.clip(a_hat, -5.0, 5.0)), v_hat, g_hat, best_tb1]
        )
    else:
        col1 = _break_term(t, 1.0, best_tb1)
        best_rss2 = np.inf
        best_tb2 = float("nan")
        best_coef4 = np.zeros(4 + n_seas)
        for idx in candidates:
            tb2 = float(t[idx])
            if abs(tb2 - best_tb1) < 20.0 * 0.0027:
                continue
            coef, rss = _fit([ones, t_rel, col1, _break_term(t, 1.0, tb2)])
            if rss < best_rss2:
                best_rss2, best_tb2, best_coef4 = rss, tb2, coef
        if not np.isfinite(best_tb2):
            raise ValueError("no admissible second break epoch found")
        a_hat, v_hat, g1_hat, g2_hat = best_coef4[:4]
        seasonal_hat = best_coef4[4 : 4 + n_seas]
        tb1, tb2 = best_tb1, best_tb2
        if tb2 < tb1:  # relabel ascending (the model is pair-symmetric)
            tb1, tb2 = tb2, tb1
            g1_hat, g2_hat = g2_hat, g1_hat
        residual = y - (
            a_hat
            + v_hat * t_rel
            + g1_hat * _break_term(t, 1.0, tb1)
            + g2_hat * _break_term(t, 1.0, tb2)
        )
        trajectory = np.array(
            [float(np.clip(a_hat, -5.0, 5.0)), v_hat, g1_hat, tb1, g2_hat, tb2]
        )

    if seasonal:  # subtract the co-estimated seasonal from the noise residual
        residual = residual - _seasonal_design(t) @ seasonal_hat
    amp_hat = math.sqrt(
        max(float(residual.var()) - wn_amp * wn_amp, 0.25 * wn_amp * wn_amp)
    )
    return np.concatenate([trajectory, seasonal_hat, [-1.0, amp_hat]])


def detect_breakpoints(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    wn_amp: float,
    *,
    n_breaks: int = 1,
    n_runs: int = 1_000_000,
    seed: int | None = None,
    start: NDArray[np.float64] | None = None,
    t_runs: int = 1000,
    baseline_epochs: int = 30,
    seasonal: bool = False,
) -> InversionResult:
    """High-level one-call entry: build bounds + config, run the inversion.

    Convenience wrapper mirroring one iteration of the ``GBISrun_ts.m`` station
    loop: seed → :func:`prepare_bounds` (SI Table S3 ranges) →
    :class:`InversionConfig` → :func:`run_inversion`. This is what the
    precompute job calls per station/component.

    Symbols → args:
        - ``t``: epochs [yr, fractional year, sorted ascending, uniform daily]
        - ``y``: displacements [mm] — raw ``.NEU``-frame series are fine: the
          leaf auto-references to the start baseline
          (:func:`_start_baseline`) so the fixed ±5 mm intercept prior of
          ``prepareModel_ts.m`` is honored; intercepts are reported back in
          the input frame (see :func:`run_inversion` and
          :attr:`InversionResult.y_ref`)
        - ``σ_w`` → ``wn_amp``: fixed white-noise amplitude [mm] (upstream
          ``WNlist``, from pre-processing noise estimation)
        - ``n_breaks``: 1 (BPD1) or 2 (BPD2)
        - ``n_runs``: kept MCMC iterations (paper: 1.2e5 synthetic, 1e6 real)
        - ``seed``: RNG seed for reproducibility
        - ``start``: optional preliminary estimates in the sampler layout
          (length 6/8, or 10/12 when ``seasonal`` — see :func:`prepare_bounds`)
          — the upstream ``startPara`` route. If omitted, a coarse OLS grid
          seed is derived from the data (:func:`_preliminary_start`; documented
          heuristic, not the GBIS4TS variogram pre-processing; seasonal
          amplitudes co-estimated in the seed when ``seasonal``).
        - ``seasonal``: co-estimate an annual+semiannual term jointly with the
          break/rate/noise (model ``"BPD1S"``/``"BPD2S"``; new API surface,
          default ``False`` keeps the exact seasonal-blind behavior). Use on
          real GNSS series, where an un-modeled seasonal signal biases v/dv/tb
          on windows ≲ 4.5 yr (Blewitt & Lavallée 2002).
        - ``t_runs``: kept iterations per annealing temperature (default 1000
          = upstream; reduce for short exploratory chains so the 16-step
          cooling still completes within ``n_runs``).
        - ``baseline_epochs``: leading samples for the zero-reference
          baseline (default 30, 0 disables — see :func:`run_inversion`).
          Also conditions the auto-seed: :func:`_preliminary_start` is run on
          ``y − r`` so its intercept estimate lives in the conditioned frame
          where the ±5 mm prior applies (``r`` is deterministic, so the value
          recomputed inside :func:`run_inversion` is identical). An explicit
          ``start`` intercept must likewise be given in the conditioned
          frame (near zero — as upstream ``startPara``).

    Returns:
        :class:`InversionResult` for the requested model.

    Reference:
        ``GBISrun_ts.m`` (driver); Yang et al. 2023, 2023GL103432, SI Table S3.

    Numerical notes:
        Discard the burn-in before summarizing marginals — Yang et al. use
        20 000 iterations. The annealing phase spans ``16·t_runs`` iterations;
        ``n_runs`` must comfortably exceed that for a T = 1 posterior chain.
    """
    tt = _as_time_array(t)
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != tt.shape:
        raise ValueError(f"y shape {yy.shape} does not match t shape {tt.shape}")
    if n_breaks not in (1, 2):
        raise ValueError(f"n_breaks must be 1 or 2, got {n_breaks}")
    if baseline_epochs < 0:
        raise ValueError(f"baseline_epochs must be >= 0, got {baseline_epochs}")
    if n_breaks == 1:
        model = "BPD1S" if seasonal else "BPD1"
    else:
        model = "BPD2S" if seasonal else "BPD2"
    if start is None:
        # Seed on the conditioned series: run_inversion recomputes the same
        # deterministic baseline, so seed frame == sampling frame.
        y_seed = yy
        if baseline_epochs:
            y_seed = yy - _start_baseline(yy, baseline_epochs)
        seed_params = _preliminary_start(tt, y_seed, wn_amp, n_breaks, seasonal)
    else:
        seed_params = np.asarray(start, dtype=np.float64)
    bounds = prepare_bounds(seed_params, model)
    config = InversionConfig(n_runs=n_runs, t_runs=t_runs, seed=seed)
    return run_inversion(
        tt, yy, wn_amp, config, bounds, model, baseline_epochs=baseline_epochs
    )
