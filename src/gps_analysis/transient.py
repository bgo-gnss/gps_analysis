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
    "PriorBounds",
    "InversionConfig",
    "InversionResult",
    "bpd1_forward",
    "bpd2_forward",
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

#: Number of parameters of the trajectory+noise function per model.
_N_FUNC = {"BPD1": 6, "BPD2": 8}

# Inert hyperparameter slot appended by prepareModel_ts.m l.120-124.
_HYPER_START = 0.0
_HYPER_STEP = 1.0e-3
_HYPER_LOWER = -0.5
_HYPER_UPPER = 0.5

#: Simulated-annealing cooling schedule 10.^(3:-0.2:0) (GBISrun_ts.m l.125).
_T_SCHEDULE: NDArray[np.float64] = 10.0 ** np.linspace(3.0, 0.0, 16)

#: Safety cap on the BPD2 trial-regeneration loop (upstream can spin forever).
_MAX_TRIAL_REGEN = 100_000


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
        optimal: Best-likelihood trajectory+noise parameter vector, length 6
            (BPD1) or 8 (BPD2), without the hyperparameter slot — MATLAB
            ``results.optimalmodel``.
        model: ``"BPD1"`` | ``"BPD2"``.
    """

    m_keep: NDArray[np.float64]  # (n_params, n_kept) accepted parameter chain
    p_keep: NDArray[np.float64]  # (n_kept,) log-probability per kept sample
    optimal: NDArray[np.float64]  # best-probability parameter vector
    model: str  # "BPD1" | "BPD2"


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
    spectral index κ          −1.5              0
    PLN amplitude β           0                 1.5·β̂
    ========================  ================  ================

    with lower/upper swapped where a negative preliminary rate inverts them
    (``prepareModel_ts.m`` l.54–105). Steps: 1 mm (a), ``0.05·v̂`` (every rate
    parameter — *all* trend steps derive from v̂ upstream), 0.0027 yr (breaks),
    0.05 (κ), 1 (β). The inert hyperparameter slot (start 0, step 1e-3, bounds
    ±0.5; ``prepareModel_ts.m`` l.120–124) is appended.

    Symbols → args:
        - ``start``: preliminary estimates in MATLAB order, length 6 (BPD1:
          ``[â, v̂, ĝ, t̂, κ̂, β̂]``) or 8 (BPD2: ``[â, v̂, ĝ₁, t̂₁, ĝ₂, t̂₂,
          κ̂, β̂]``); mm, mm/yr, yr as above.
        - ``model``: ``"BPD1"`` | ``"BPD2"``.

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
    if model == "BPD1":
        step = np.array([1.0, s[1] * 0.05, s[1] * 0.05, 0.0027, 0.05, 1.0])
        lower = np.array([-5.0, -s[1], -s[2], s[3] - 1.0, -1.5, 0.0])
        upper = np.array([5.0, 2.0 * s[1], 2.0 * s[2], s[3] + 1.0, 0.0, 1.5 * s[5]])
        swap_indices: tuple[int, ...] = (1, 2)
    else:
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


def run_inversion(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    wn_amp: float,
    config: InversionConfig,
    bounds: PriorBounds,
    model: str = "BPD1",
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

    Symbols → args:
        - ``t``: epochs [yr], ``y``: displacements [mm], 1-D, same length
        - ``σ_w`` → ``wn_amp``: fixed white-noise amplitude [mm], > 0
        - ``config``: :class:`InversionConfig` (schedule + seed)
        - ``bounds``: :class:`PriorBounds`, length ``n_func`` (hyper slot
          appended automatically) or ``n_func + 1``
        - ``model``: ``"BPD1"`` | ``"BPD2"``

    Returns:
        :class:`InversionResult` — kept chain, kept log-posteriors, optimum.

    Raises:
        ValueError: on shape mismatches, ``wn_amp ≤ 0``, unknown model, or a
            start vector outside the bounds (upstream errors identically).
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
    n_func = _N_FUNC[model]
    n_breaks = 1 if model == "BPD1" else 2

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
            if model == "BPD1":
                break
            # BPD2 ordering guard — fidelity flag: swaps the TREND CHANGES
            # (indices 2/4 == MATLAB trial(3)/trial(5)), not the break points.
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

    return InversionResult(m_keep=m_keep, p_keep=p_keep, optimal=func_opt, model=model)


def _preliminary_start(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    wn_amp: float,
    n_breaks: int,
) -> NDArray[np.float64]:
    """Coarse OLS grid seed ``[â, v̂, ĝ..., t̂..., κ̂, β̂]`` for :func:`prepare_bounds`.

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
    """
    n = t.size
    lo = max(int(0.05 * n), 1)
    hi = min(int(0.95 * n), n - 2)
    if hi <= lo:
        raise ValueError(f"series too short for break detection: {n} epochs")
    candidates = np.unique(np.linspace(lo, hi, 96).astype(np.int64))
    ones = np.ones(n, dtype=np.float64)
    t_rel = t - t[0]

    def _fit(
        columns: list[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], float]:
        g_mat = np.column_stack(columns)
        coef, _, _, _ = np.linalg.lstsq(g_mat, y, rcond=None)
        res = y - g_mat @ coef
        return np.asarray(coef, dtype=np.float64), float(res @ res)

    best_rss = np.inf
    best_tb1 = float(t[candidates[0]])
    best_coef = np.zeros(3)
    for idx in candidates:
        tb = float(t[idx])
        coef, rss = _fit([ones, t_rel, _break_term(t, 1.0, tb)])
        if rss < best_rss:
            best_rss, best_tb1, best_coef = rss, tb, coef

    if n_breaks == 1:
        a_hat, v_hat, g_hat = best_coef
        residual = y - (a_hat + v_hat * t_rel + g_hat * _break_term(t, 1.0, best_tb1))
        trajectory = np.array(
            [float(np.clip(a_hat, -5.0, 5.0)), v_hat, g_hat, best_tb1]
        )
    else:
        col1 = _break_term(t, 1.0, best_tb1)
        best_rss2 = np.inf
        best_tb2 = float("nan")
        best_coef4 = np.zeros(4)
        for idx in candidates:
            tb2 = float(t[idx])
            if abs(tb2 - best_tb1) < 20.0 * 0.0027:
                continue
            coef, rss = _fit([ones, t_rel, col1, _break_term(t, 1.0, tb2)])
            if rss < best_rss2:
                best_rss2, best_tb2, best_coef4 = rss, tb2, coef
        if not np.isfinite(best_tb2):
            raise ValueError("no admissible second break epoch found")
        a_hat, v_hat, g1_hat, g2_hat = best_coef4
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

    amp_hat = math.sqrt(
        max(float(residual.var()) - wn_amp * wn_amp, 0.25 * wn_amp * wn_amp)
    )
    return np.concatenate([trajectory, [-1.0, amp_hat]])


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
) -> InversionResult:
    """High-level one-call entry: build bounds + config, run the inversion.

    Convenience wrapper mirroring one iteration of the ``GBISrun_ts.m`` station
    loop: seed → :func:`prepare_bounds` (SI Table S3 ranges) →
    :class:`InversionConfig` → :func:`run_inversion`. This is what the
    precompute job calls per station/component.

    Symbols → args:
        - ``t``: epochs [yr, fractional year, sorted ascending, uniform daily]
        - ``y``: displacements [mm] (typically detrended/zero-referenced so the
          intercept fits the fixed ±5 mm prior)
        - ``σ_w`` → ``wn_amp``: fixed white-noise amplitude [mm] (upstream
          ``WNlist``, from pre-processing noise estimation)
        - ``n_breaks``: 1 (BPD1) or 2 (BPD2)
        - ``n_runs``: kept MCMC iterations (paper: 1.2e5 synthetic, 1e6 real)
        - ``seed``: RNG seed for reproducibility
        - ``start``: optional preliminary estimates in MATLAB order (length 6
          or 8, see :func:`prepare_bounds`) — the upstream ``startPara`` route.
          If omitted, a coarse OLS grid seed is derived from the data
          (:func:`_preliminary_start`; documented heuristic, not the GBIS4TS
          variogram pre-processing).
        - ``t_runs``: kept iterations per annealing temperature (default 1000
          = upstream; reduce for short exploratory chains so the 16-step
          cooling still completes within ``n_runs``).

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
    model = "BPD1" if n_breaks == 1 else "BPD2"
    if start is None:
        seed_params = _preliminary_start(tt, yy, wn_amp, n_breaks)
    else:
        seed_params = np.asarray(start, dtype=np.float64)
    bounds = prepare_bounds(seed_params, model)
    config = InversionConfig(n_runs=n_runs, t_runs=t_runs, seed=seed)
    return run_inversion(tt, yy, wn_amp, config, bounds, model)
