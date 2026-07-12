"""Tests for gps_analysis._mcmc — the shared GBIS Metropolis/annealing core.

Two concerns:

1. **Consolidation parity** — the transient (GBIS4TS, byte-parity-verified
   port) and deformation (Mogi Bayesian) lanes now delegate to one shared
   sampler. The pinned-chain tests reproduce seeded inversions whose
   reference values were captured from the PRE-consolidation implementations
   (git main @ 2592eba, 2026-07-12) — same seed must give the same chain.
   At refactor time full-size chains (BPD1/BPD2/BPD1S/BPD2S + Mogi Bayes,
   17 arrays) were verified **bit-identical**; these smaller pins keep that
   contract alive in CI.
2. **Generic sampler behavior** — posterior recovery on an analytic Gaussian
   target, prior-box validation, reflection, and the hook surface
   (adjust_step / constrain_trial / on_anneal / regeneration cap).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import gps_analysis._mcmc as _mcmc
import gps_analysis.deformation as deformation
import gps_analysis.transient as transient
from gps_analysis._mcmc import (
    T_SCHEDULE,
    InversionConfig,
    PriorBounds,
    metropolis,
    sensitivity_schedule,
)
from gps_analysis.deformation import MogiSource, mogi_forward, mogi_invert_bayes
from gps_analysis.transient import detect_breakpoints

# =====================================================================
# Shared-core identity: both lanes use the SAME objects, not copies
# =====================================================================


def test_transient_reexports_shared_sampler_types() -> None:
    """transient.PriorBounds/InversionConfig ARE the _mcmc classes."""
    assert transient.PriorBounds is _mcmc.PriorBounds
    assert transient.InversionConfig is _mcmc.InversionConfig


def test_deformation_uses_shared_sampler() -> None:
    """deformation imports the shared metropolis loop (no private copy)."""
    # getattr: runtime identity of module-internal names, not the typed API
    assert getattr(deformation, "metropolis") is _mcmc.metropolis  # noqa: B009
    assert getattr(deformation, "PriorBounds") is _mcmc.PriorBounds  # noqa: B009
    assert not hasattr(deformation, "_gbis_metropolis")


def test_t_schedule_is_gbis_ladder() -> None:
    """T = 10^{3, 2.8, …, 0} — 16 rungs, ending at plain Metropolis T = 1."""
    assert T_SCHEDULE.shape == (16,)
    np.testing.assert_allclose(T_SCHEDULE, 10.0 ** np.linspace(3.0, 0.0, 16))
    assert float(T_SCHEDULE[-1]) == 1.0


def test_sensitivity_schedule_matches_matlab_counts() -> None:
    """[1:100:10000, 11000:1000:30000, 40000:10000:nRuns] — GBISrun_ts.m l.120."""
    s = sensitivity_schedule(60_000)
    assert {1, 101, 9901} <= s
    assert {11000, 30000, 40000, 60000} <= s
    assert 10001 not in s  # MATLAB 1:100:10000 ends at 9901
    assert 10101 not in s and 31000 not in s and 61000 not in s


# =====================================================================
# Consolidation parity pins (references from pre-refactor main @ 2592eba)
# =====================================================================


def test_parity_pin_transient_bpd1_chain_unchanged() -> None:
    """Seeded BPD1 inversion reproduces the pre-consolidation chain.

    Reference values captured from git main @ 2592eba (the inline
    runInversion_ts.m loop) with this exact synthetic + seeds; the shared
    _mcmc core preserves the RNG call sequence and float64 op order, so the
    chain is bit-identical (asserted here to 1e-12 relative to tolerate
    nothing but printing round-off of the pins).
    """
    rng = np.random.default_rng(1234)
    n = 200
    t = 2018.0 + np.arange(n) / 365.0
    y = (
        2.0
        + 15.0 * (t - t[0])
        + np.where(t >= t[120], -25.0 * (t - t[120]), 0.0)
        + rng.normal(0.0, 1.0, n)
    )
    res = detect_breakpoints(t, y, 1.0, n_breaks=1, n_runs=600, seed=11, t_runs=30)
    pinned_optimal = np.array(
        [
            1.7001372830928545e00,
            1.8200957525110962e01,
            -2.7652975807677478e01,
            2.0182942597468011e03,
            -2.9063266455096881e-01,
            3.9547755960961672e-01,
        ]
    )
    np.testing.assert_allclose(res.optimal, pinned_optimal, rtol=1e-12)
    assert float(res.p_keep[-1]) == pytest.approx(-299.33382590207054, rel=1e-12)
    # whole-chain fingerprint (every kept column contributes)
    assert float(res.m_keep.sum()) == pytest.approx(1205985.5590102463, rel=1e-12)


def test_parity_pin_mogi_bayes_chain_unchanged() -> None:
    """Seeded Mogi Bayesian inversion reproduces the pre-consolidation chain.

    Reference values captured from git main @ 2592eba (the private
    _gbis_metropolis copy in deformation.py) — the hook-free shared core
    issues the identical RNG sequence.
    """
    grng = np.random.default_rng(99)
    e = grng.uniform(-6000.0, 6000.0, 12)
    n = grng.uniform(-6000.0, 6000.0, 12)
    src = MogiSource(x=400.0, y=-300.0, depth=3500.0, dv=1.5e6)
    obs = mogi_forward(e, n, src) + grng.normal(0.0, 0.002, (3, 12))
    sig = np.full((3, 12), 0.002)
    bounds = PriorBounds(
        start=np.array([0.0, 0.0, 3000.0, 1.0e6]),
        lower=np.array([-5000.0, -5000.0, 500.0, 1.0e5]),
        upper=np.array([5000.0, 5000.0, 8000.0, 6.0e6]),
        step=np.array([250.0, 250.0, 250.0, 1.5e5]),
    )
    post = mogi_invert_bayes(
        e, n, obs, sig, bounds, InversionConfig(n_runs=800, t_runs=40, seed=5)
    )
    pinned_optimal = np.array(
        [
            5.5052021117130903e02,
            -4.0107631934752868e02,
            3.4170723098060921e03,
            1.3926876975809715e06,
        ]
    )
    np.testing.assert_allclose(post.optimal.as_array(), pinned_optimal, rtol=1e-12)
    assert post.p_opt == pytest.approx(-14.482806817479878, rel=1e-12)
    assert float(post.m_keep.sum()) == pytest.approx(1187428316.9162793, rel=1e-12)


# =====================================================================
# Generic sampler behavior
# =====================================================================


def _gaussian_bounds() -> PriorBounds:
    return PriorBounds(
        start=np.array([2.0, -2.0]),
        lower=np.array([-10.0, -10.0]),
        upper=np.array([10.0, 10.0]),
        step=np.array([0.5, 0.5]),
    )


def _gaussian_log_post(m: NDArray[np.float64]) -> float:
    """ln P of independent N(1, 0.5²) × N(-1, 0.25²) (up to a constant)."""
    return -0.5 * float(((m[0] - 1.0) / 0.5) ** 2 + ((m[1] + 1.0) / 0.25) ** 2)


def test_metropolis_samples_gaussian_target() -> None:
    """Post-annealing chain recovers mean and spread of an analytic Gaussian."""
    config = InversionConfig(n_runs=30_000, t_runs=200, seed=42)
    res = metropolis(_gaussian_log_post, _gaussian_bounds(), config)
    burn = 16 * config.t_runs
    chain = res.m_keep[:, burn:]
    mean = chain.mean(axis=1)
    std = chain.std(axis=1)
    # sampling error of an autocorrelated Metropolis chain: generous bands
    np.testing.assert_allclose(mean, [1.0, -1.0], atol=0.05)
    np.testing.assert_allclose(std, [0.5, 0.25], rtol=0.25)
    # MAP near the analytic mode, best posterior near 0
    np.testing.assert_allclose(res.optimal, [1.0, -1.0], atol=0.05)
    assert res.p_opt <= 0.0 and res.p_opt > -0.01


def test_metropolis_seed_reproducibility_and_chain_shape() -> None:
    cfg = InversionConfig(n_runs=500, t_runs=25, seed=3)
    r1 = metropolis(_gaussian_log_post, _gaussian_bounds(), cfg)
    r2 = metropolis(_gaussian_log_post, _gaussian_bounds(), cfg)
    assert r1.m_keep.shape == (2, 500) and r1.p_keep.shape == (500,)
    np.testing.assert_array_equal(r1.m_keep, r2.m_keep)
    np.testing.assert_array_equal(r1.p_keep, r2.p_keep)


def test_metropolis_respects_prior_box() -> None:
    """Reflection keeps every kept sample inside the uniform-prior support.

    Holds for step ≤ prior range (a single reflection then always lands
    inside; the adaptive retune caps steps at the range, preserving it) —
    the upstream single-reflection scheme, runInversion_ts.m l.249-253.
    """
    bounds = PriorBounds(
        start=np.array([0.9]),
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        step=np.array([0.8]),  # large vs the box: reflection is exercised
    )
    res = metropolis(
        lambda m: 0.0, bounds, InversionConfig(n_runs=2000, t_runs=100, seed=8)
    )
    assert bool(np.all(res.m_keep >= -1.0)) and bool(np.all(res.m_keep <= 1.0))
    # reflection actually fired: some accepted proposals sit in the outer
    # band only reachable by reflecting a > upper / < lower trial
    assert float(np.abs(res.m_keep).max()) > 0.9


def test_metropolis_rejects_out_of_bounds_start() -> None:
    bounds = PriorBounds(
        start=np.array([3.0]),
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        step=np.array([0.1]),
    )
    with pytest.raises(ValueError, match="out of bounds at indices \\[0\\]"):
        metropolis(lambda m: 0.0, bounds, InversionConfig(n_runs=10))


def test_metropolis_does_not_mutate_bounds() -> None:
    bounds = _gaussian_bounds()
    before = [
        bounds.start.copy(),
        bounds.lower.copy(),
        bounds.upper.copy(),
        bounds.step.copy(),
    ]
    metropolis(
        _gaussian_log_post, bounds, InversionConfig(n_runs=300, t_runs=20, seed=1)
    )
    for orig, arr in zip(
        before, (bounds.start, bounds.lower, bounds.upper, bounds.step), strict=True
    ):
        np.testing.assert_array_equal(orig, arr)


# ---------------------------------------------------------------------
# Hook surface
# ---------------------------------------------------------------------


def test_adjust_step_hook_floors_a_component() -> None:
    """An adjust_step floor keeps successive kept samples of that parameter
    at least `floor` apart whenever they move (the transient break-point
    one-day floor semantics)."""
    floor = 0.05

    def _floor(random_step: NDArray[np.float64]) -> None:
        if abs(random_step[0]) < floor:
            random_step[0] = floor * float(np.sign(random_step[0]))

    # Bounds far wider than any adaptively grown step, so reflection never
    # fires (a reflected trial may legitimately move < floor upstream) and
    # every accepted move IS the floored random step.
    bounds = PriorBounds(
        start=np.array([0.0]),
        lower=np.array([-1.0e6]),
        upper=np.array([1.0e6]),
        step=np.array([0.01]),  # would propose |δ| < floor without the hook
    )
    res = metropolis(
        lambda m: 0.0,
        bounds,
        InversionConfig(n_runs=400, t_runs=400, seed=2),  # single temperature
        adjust_step=_floor,
    )
    moves = np.abs(np.diff(res.m_keep[0]))
    moved = moves[moves > 0.0]
    assert moved.size > 0
    assert bool(np.all(moved >= floor - 1e-12))


def test_constrain_trial_hook_enforces_ordering() -> None:
    """A constrain hook (swap + separation, the BPD2 guard shape) holds on
    every kept sample."""
    sep = 0.2

    def _guard(trial: NDArray[np.float64]) -> bool:
        if trial[0] > trial[1]:
            trial[0], trial[1] = trial[1], trial[0]
        return bool(abs(trial[1] - trial[0]) >= sep)

    bounds = PriorBounds(
        start=np.array([-0.5, 0.5]),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        step=np.array([0.3, 0.3]),
    )
    res = metropolis(
        lambda m: 0.0,
        bounds,
        InversionConfig(n_runs=1000, t_runs=50, seed=4),
        constrain_trial=_guard,
    )
    kept = res.m_keep
    # ordering + separation hold for every kept column except possibly the
    # (unconstrained) start vector column itself, which satisfies both anyway
    assert bool(np.all(kept[1] - kept[0] >= sep - 1e-12))


def test_constrain_trial_regen_cap_raises() -> None:
    bounds = _gaussian_bounds()
    with pytest.raises(RuntimeError, match="impossible constraint"):
        metropolis(
            _gaussian_log_post,
            bounds,
            InversionConfig(n_runs=50, t_runs=10, seed=6),
            constrain_trial=lambda trial: False,
            max_trial_regen=100,
            regen_message="impossible constraint",
        )


def test_on_anneal_hook_sees_full_ladder() -> None:
    """on_anneal fires once per rung with the T_SCHEDULE temperatures."""
    seen: list[float] = []

    def _record(temperature: float, trial: NDArray[np.float64]) -> None:
        seen.append(temperature)

    metropolis(
        _gaussian_log_post,
        _gaussian_bounds(),
        InversionConfig(n_runs=16 * 20 + 40, t_runs=20, seed=9),
        on_anneal=_record,
    )
    assert len(seen) == 16
    np.testing.assert_allclose(seen, T_SCHEDULE)
    assert math.isclose(seen[-1], 1.0)
