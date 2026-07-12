"""Shared GBIS-style Metropolis-Hastings MCMC core (annealing + adaptive steps).

The Bayesian samplers of this package — the GBIS4TS velocity-break-point
inversion (:mod:`gps_analysis.transient`, ``runInversion_ts.m``) and the
deformation-source inversion (:func:`gps_analysis.deformation.mogi_invert_bayes`)
— use one and the same sampling scheme, the **GBIS** engine of Bagnardi &
Hooper 2018 (G³ 19, doi:10.1029/2018GC007585, §3.2–3.3):

1. **Metropolis accept rule** — a trial parameter vector m' is kept when
   ``exp((ln P(m') − ln P(m))/T) ≥ U(0, 1)`` under flat (uniform) priors, so
   only the log-posterior difference enters (Metropolis et al. 1953;
   Mosegaard & Tarantola 1995, JGR 100).
2. **Simulated annealing** — the temperature ladder :data:`T_SCHEDULE`
   ``T = 10^{3, 2.8, …, 0}`` (16 rungs) is advanced every ``t_runs`` kept
   iterations, the chain restarting from the running optimum on each advance;
   after the last rung T = 1 and the sampler is plain Metropolis, so only the
   post-annealing chain samples the posterior (Bagnardi & Hooper 2018 §3.3:
   annealed burn-in to escape local maxima).
3. **Adaptive step sizes** — on the iteration counts of
   :func:`sensitivity_schedule` each parameter is perturbed alone by
   ``±step/2`` and its acceptance probability compared to a per-parameter
   target ``0.5^{1/n}`` (joint acceptance ≈ 50 %), itself retuned by the
   running rejection rate toward ``rejection_target``; steps shrink/grow by
   ``exp(∓2·ΔP/·)`` and cap at the prior range (Bagnardi & Hooper 2018 §3.2,
   "sensitivity test"; ``runInversion_ts.m`` l.145–163). Sensitivity trials
   are never kept in the chain.
4. **Uniform priors by reflection** — random-walk proposals reflect at both
   prior limits, keeping proposals inside the support without wasting draws.

This module is the single implementation of that loop
(:func:`metropolis`): the caller supplies the log-likelihood/posterior
callable, the prior box + start (:class:`PriorBounds`) and the schedule
(:class:`InversionConfig`). Model-specific behavior stays in the callers and
enters through three optional hooks (``adjust_step``, ``constrain_trial``,
``on_anneal``) so that the byte-parity-verified GBIS4TS port in
:mod:`gps_analysis.transient` reproduces its MATLAB reference **bit-for-bit**
(same seed → identical chain; the RNG call sequence and float64 operation
order of the pre-consolidation implementation are preserved exactly and
pinned by test).

Conventions (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------
- Pure leaf: numpy only, no I/O; caller inputs are never mutated (start /
  bounds / step vectors are copied on entry).
- Working dtype float64 throughout.
- Reproducibility: fixed ``config.seed`` + identical numpy builds give
  identical chains; MCMC convergence is the caller's to assess.

Reference:
    Bagnardi & Hooper 2018, *Inversion of surface deformation data for rapid
    estimates of source parameters and uncertainties: A Bayesian approach*,
    G³ 19, 2194–2211 (doi:10.1029/2018GC007585) — the GBIS engine.
    Yang, Sigmundsson & Geirsson 2023, GRL 2023GL103432 (``GBIS4TS``,
    ``runInversion_ts.m`` / ``GBISrun_ts.m`` — the vendored MATLAB this loop
    was ported from, ``reference/gbis4ts/``). Kirkpatrick, Gelatt & Vecchi
    1983, Science 220 (simulated annealing).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "T_SCHEDULE",
    "InversionConfig",
    "MetropolisResult",
    "PriorBounds",
    "metropolis",
    "sensitivity_schedule",
]

#: Simulated-annealing cooling schedule ``T = 10^{3, 2.8, …, 0}`` (16 rungs;
#: ``GBISrun_ts.m`` l.125, ``10.^(3:-0.2:0)``) — Bagnardi & Hooper 2018 §3.3.
T_SCHEDULE: NDArray[np.float64] = 10.0 ** np.linspace(3.0, 0.0, 16)

#: Safety cap on the trial-regeneration loop when ``constrain_trial`` keeps
#: rejecting proposals (the upstream MATLAB can spin forever — guarded
#: deviation, flagged in :mod:`gps_analysis.transient`).
_MAX_TRIAL_REGEN = 100_000


@dataclass(frozen=True)
class PriorBounds:
    """Uniform prior box + MCMC step per parameter (GBIS ``model`` struct).

    One entry per sampled parameter, in the caller's parameter-vector order:
    ``start`` (initial model, must lie inside the box), ``lower``/``upper``
    (uniform-prior support — the sampler reflects proposals at these limits)
    and ``step`` (initial random-walk half-range; retuned adaptively, see
    :func:`metropolis`).

    For the GBIS4TS break-point models the box is built by
    :func:`gps_analysis.transient.prepare_bounds` (Yang et al. 2023 SI
    Table S3, ``prepareModel_ts.m``); deformation-source callers supply
    physical ranges directly (e.g. :func:`~gps_analysis.deformation.mogi_invert_bayes`).

    Numerical notes:
        Arrays are coerced to float64 1-D of equal length at construction; the
        dataclass is frozen but ndarrays are not immutable — treat as read-only
        (:func:`metropolis` copies them on entry). Steps inherit the sign of
        the preliminary rate upstream (no absolute values taken); the sampler
        only uses them symmetrically, so a negative step is equivalent to its
        magnitude.
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
    """Sampler schedule (GBIS ``invpar``; ``runInversion_ts.m`` / ``GBISrun_ts.m``).

    Defaults mirror the GBIS4TS MATLAB: annealing over :data:`T_SCHEDULE`,
    ``t_runs=1000`` kept iterations per temperature, sensitivity test tuned
    to a 77 % rejection rate, break-point step floor ``0.0027`` yr (~1 day).

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
            parameter [yr] (l.244; 0.0027 yr ≈ 1 day). Consumed by the
            :mod:`gps_analysis.transient` hooks only — inert for generic
            callers of :func:`metropolis` (e.g. the Mogi Bayesian lane).
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
class MetropolisResult:
    """Raw sampler output of :func:`metropolis` (GBIS ``mKeep``/``PKeep``).

    Attributes:
        m_keep: ``(n_params, n_runs)`` chain of kept parameter vectors —
            rejected iterations repeat the previous column (GBIS convention).
            Stored float64 (upstream stores single — deliberate upgrade).
        p_keep: ``(n_runs,)`` log-posterior (= log-likelihood under the flat
            priors) per kept column.
        optimal: Best-posterior parameter vector seen by the chain
            (including annealing-phase samples).
        p_opt: Log-posterior at ``optimal``.
    """

    m_keep: NDArray[np.float64]
    p_keep: NDArray[np.float64]
    optimal: NDArray[np.float64]
    p_opt: float


def sensitivity_schedule(n_runs: int) -> frozenset[int]:
    """Kept-iteration counts that trigger an adaptive-step sensitivity sweep.

    Equation (``GBISrun_ts.m`` l.120, ``invpar.sensitivityTest``):
        ``S = {1, 101, …, 9901, 10001} ∪ {11000, 12000, …, 30000}
        ∪ {40000, 50000, …, ≤ n_runs}``
    (MATLAB ``[1:100:10000, 11000:1000:30000, 40000:10000:nRuns]``) — dense
    early retuning while the steps are far from tuned, sparse late.

    Symbols → args:
        - ``n_runs`` → ``n_runs``: total kept iterations of the chain.

    Returns:
        The set S (frozenset of ints) — membership-tested once per kept
        iteration by :func:`metropolis`.

    Reference:
        Bagnardi & Hooper 2018, G³ 19, §3.2 (step-size sensitivity test);
        ``GBISrun_ts.m`` l.120 (the schedule itself).

    Numerical notes:
        Exact integer arithmetic; O(n_runs/100) memory. ``arange`` upper
        limits are inclusive as in the MATLAB colon operator.
    """
    parts = (
        np.arange(1, 10001, 100),
        np.arange(11000, 30001, 1000),
        np.arange(40000, n_runs + 1, 10000),
    )
    return frozenset(int(v) for arr in parts for v in arr)


def metropolis(
    log_post: Callable[[NDArray[np.float64]], float],
    bounds: PriorBounds,
    config: InversionConfig,
    *,
    adjust_step: Callable[[NDArray[np.float64]], None] | None = None,
    constrain_trial: Callable[[NDArray[np.float64]], bool] | None = None,
    on_anneal: Callable[[float, NDArray[np.float64]], None] | None = None,
    max_trial_regen: int = _MAX_TRIAL_REGEN,
    regen_message: str | None = None,
) -> MetropolisResult:
    """Metropolis–Hastings sampler with GBIS annealing and step adaptation.

    Samples ``P(m | d) ∝ exp(ln P(m))`` over the uniform prior box of
    ``bounds`` (Bagnardi & Hooper 2018, G³ 19, §3.2–3.3; ``runInversion_ts.m``):

    - **Accept rule** (l.132–141): ``P_ratio = exp((ln P − ln P_prev)/T)``;
      accept if ``P_ratio ≥ U(0, 1)``. The first iteration is always kept.
    - **Annealing** (``GBISrun_ts.m`` l.125): T walks :data:`T_SCHEDULE`,
      advanced every ``config.t_runs`` kept iterations; on each advance the
      chain restarts from the running optimum. After 16 rungs T = 1 (plain
      Metropolis; only that tail samples the true posterior).
    - **Adaptive steps** (l.145–163): on the :func:`sensitivity_schedule`
      counts each parameter is perturbed alone by ``±step/2`` and its
      acceptance probability compared to a target ``0.5^{1/n}``, itself
      retuned by the running rejection rate toward
      ``config.rejection_target``; steps shrink/grow by ``exp(∓2·ΔP/·)``
      and are capped at the prior range. Sweep trials are never kept.
    - **Bounds**: uniform priors; random-walk proposals reflect at both
      limits (l.249–253); the single-parameter sensitivity kick folds back
      at the upper limit only (l.235 — upstream asymmetry, preserved).

    Symbols → args:
        - ``ln P`` → ``log_post``: callable m ↦ log-posterior (flat priors ⇒
          log-likelihood up to a constant; only differences enter the accept
          ratio). Must not mutate its argument.
        - ``bounds`` → :class:`PriorBounds`: start m₀, prior box, initial
          steps (length n = number of sampled parameters).
        - ``config`` → :class:`InversionConfig`: schedule + seed
          (``breakpoint_step_floor``/``n_save`` are not read here).
        - ``adjust_step``: optional in-place hook on the random-walk step
          vector *before* it is added to m (transient: one-day break-point
          step floor, ``runInversion_ts.m`` l.244).
        - ``constrain_trial``: optional in-place hook on the proposed trial
          *after* reflection; returns True to accept the proposal, False to
          regenerate it (transient: the BPD2 ordering guard l.255–269 —
          note it also re-perturbs sensitivity trials, upstream fidelity).
          ``None`` ⇒ every proposal is admissible (single pass, no extra
          RNG draws — parity with the hook-free callers).
        - ``on_anneal``: optional hook ``(T, trial) → None`` called after
          every temperature advance, with the (possibly optimum-restarted)
          trial mutable in place (transient: pin the inert GBIS
          hyperparameter slot at T = 1, ``runInversion_ts.m`` l.121).
        - ``max_trial_regen`` / ``regen_message``: safety cap on the
          ``constrain_trial`` regeneration loop and the RuntimeError text on
          exhaustion (upstream would spin forever — guarded deviation).

    Returns:
        :class:`MetropolisResult` — kept chain ``(n, n_runs)``, kept
        log-posteriors, best model, best log-posterior.

    Raises:
        ValueError: if the start vector lies outside the prior box (upstream
            errors identically).
        RuntimeError: if ``constrain_trial`` rejects ``max_trial_regen``
            consecutive proposals.

    Reference:
        Bagnardi & Hooper 2018, G³ 19, 2194–2211 (doi:10.1029/2018GC007585),
        §3.2–3.3 (sampler, sensitivity test, annealed burn-in); Yang et al.
        2023, GRL 2023GL103432 (``runInversion_ts.m`` — the line numbers
        above; vendored under ``gps_analysis``'s ``reference/gbis4ts/``);
        Kirkpatrick et al. 1983, Science 220 (annealing); Mosegaard &
        Tarantola 1995, JGR 100 (Bayesian sampling with flat priors).

    Numerical notes:
        - **Bit-parity contract**: the RNG call sequence (one ``U(0,1)`` per
          accept test, one ``U(0,1)ⁿ`` block per random-walk proposal, one
          standard normal per sensitivity kick) and the float64 operation
          order are exactly those of the verified GBIS4TS port — a fixed
          ``config.seed`` reproduces the pre-consolidation chains of both
          :func:`gps_analysis.transient.run_inversion` and
          :func:`gps_analysis.deformation.mogi_invert_bayes` bit-for-bit
          (test-pinned).
        - ``exp((ln P − ln P_prev)/T)`` is capped at ``exp(700)`` against
          float64 overflow; the accept decision is unchanged (both sides
          compare ≥ U < 1).
        - Inputs are copied on entry; ``bounds``/``config`` are never
          mutated. Chain memory is ``8·n·n_runs`` bytes.
    """
    m = bounds.start.copy()
    lower = bounds.lower.copy()
    upper = bounds.upper.copy()
    step = bounds.step.copy()
    if bool(np.any(m > upper)) or bool(np.any(m < lower)):
        bad = np.nonzero((m > upper) | (m < lower))[0]
        raise ValueError(f"starting model out of bounds at indices {bad.tolist()}")

    n_model = m.size
    prm_range = upper - lower  # model.range
    prob_target = 0.5 ** (1.0 / n_model)
    prob_sens = np.zeros(n_model, dtype=np.float64)
    sens_schedule = sensitivity_schedule(config.n_runs)
    rng = np.random.default_rng(config.seed)

    m_keep = np.zeros((n_model, config.n_runs), dtype=np.float64)
    p_keep = np.zeros(config.n_runs, dtype=np.float64)

    i_keep = 0
    i_reject = 0
    i_keep_save = 0
    i_reject_save = 0
    p_opt = -1.0e99
    p_prev = -np.inf  # set on first (always accepted) iteration
    optimal = m.copy()
    i_temp = 0
    n_temp = T_SCHEDULE.size
    sensitivity_test = 0
    temperature = float(T_SCHEDULE[0])
    trial = m.copy()

    while i_keep < config.n_runs:
        # -- Annealing schedule (runInversion_ts.m l.74-88) -------------------
        if i_keep % config.t_runs == 0 and i_temp < n_temp:
            temperature = float(T_SCHEDULE[i_temp])
            i_temp += 1
            if i_keep > 0:
                trial = optimal.copy()  # restart from current optimum
            if on_anneal is not None:
                on_anneal(temperature, trial)

        if i_keep in sens_schedule:
            sensitivity_test = 1

        # -- Posterior of the trial + Metropolis ratio (l.95-141) -------------
        log_p = log_post(trial)
        if i_keep > 0:
            # exp capped against float64 overflow; accept decision unchanged.
            p_ratio = math.exp(min((log_p - p_prev) / temperature, 700.0))
        else:
            p_ratio = 1.0  # first iteration always kept (l.140)

        # -- Sensitivity bookkeeping / accept-reject (l.145-225) --------------
        if sensitivity_test > 1:
            idx = sensitivity_test - 2  # parameter whose perturbation this was
            if idx < n_model:
                # (Beyond n_model MATLAB grows probSens and later crashes on
                # the step update — the constrained regeneration loop can
                # over-increment. We drop the overflow entry instead;
                # flagged deviation.)
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
                if log_p > p_opt:
                    optimal = m.copy()
                    p_opt = log_p
            else:
                i_reject += 1
                m_keep[:, i_keep - 1] = m_keep[:, i_keep - 2]
                p_keep[i_keep - 1] = p_keep[i_keep - 2]

        # -- Next trial (l.229-269) -------------------------------------------
        for _ in range(max_trial_regen):
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
                if adjust_step is not None:
                    adjust_step(random_step)  # e.g. break-point step floor
                trial = m + random_step
                over = trial > upper  # reflection at the bounds (l.249-253)
                trial[over] = 2.0 * upper[over] - trial[over]
                under = trial < lower
                trial[under] = 2.0 * lower[under] - trial[under]
            if constrain_trial is None or constrain_trial(trial):
                break
        else:
            raise RuntimeError(
                regen_message
                or "trial regeneration exceeded the safety cap "
                f"({max_trial_regen}) - constrain_trial rejects every proposal"
            )

    return MetropolisResult(m_keep=m_keep, p_keep=p_keep, optimal=optimal, p_opt=p_opt)
