"""Tests for gps_analysis.shared_tau (one τ shared by many series)."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

from gps_analysis.models import FloatArray, exp_linear_rate
from gps_analysis.shared_tau import (
    LAG1_RHO_MAX,
    lag1_scale,
    model_rate,
    profile_shared_tau,
    transient_rate,
)
from gps_analysis.terms import (
    ExpTransient,
    Polynomial,
    TrajectoryModel,
    profile_transient_tau,
)

T0 = 2025.6
TAU = 0.25


def _model(tau: float = 0.5) -> TrajectoryModel:
    # Any: the Term protocol's kind/group are settable, the dataclasses' are
    # not (an old mypy mismatch, shared by test_terms_transient.py)
    terms: list[Any] = [Polynomial(1), ExpTransient(epoch=T0, tau=tau)]
    return TrajectoryModel(terms)


def _series(
    rng: np.random.Generator,
    rate: float,
    phi: float,
    *,
    n: int = 450,
    sigma: float = 2.0,
    span: float = 1.25,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    t = T0 + np.linspace(0.0, span, n)
    y = 3.0 + rate * (t - T0) + phi * (1 - np.exp(-(t - T0) / TAU))
    return t, y + rng.normal(0.0, sigma, n), np.full(n, sigma)


def test_recovers_shared_tau() -> None:
    rng = np.random.default_rng(1)
    series = [_series(rng, r, p) for r, p in [(100, 120), (-30, -60), (60, 40), (5, 0)]]
    fit = profile_shared_tau(_model(), series, tau_bounds=(0.02, 5.0))
    assert abs(fit.tau - TAU) < 3 * fit.tau_sigma
    assert fit.tau_sigma < 0.02
    assert fit.n_series == 4
    assert fit.tau_interval[0] < fit.tau < fit.tau_interval[1]
    assert not (fit.interval_open_lower or fit.interval_open_upper)
    assert fit.scale_sq == pytest.approx(1.0, abs=0.15)


def test_single_series_matches_varpro_profile() -> None:
    """One series: the summed profile is varpro's (τ̂ and bordered σ)."""
    rng = np.random.default_rng(2)
    t, y, s = _series(rng, 80, 150)
    shared = profile_shared_tau(_model(), [(t, y, s)], tau_bounds=(0.02, 5.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single = profile_transient_tau(_model(), t, y, sigma=s, tau_bounds=(0.02, 5.0))
    assert shared.tau == pytest.approx(single.tau, rel=1e-4)
    assert shared.tau_sigma == pytest.approx(single.tau_sigma, rel=0.02)


def test_more_series_tighten_tau() -> None:
    rng = np.random.default_rng(3)
    one = [_series(rng, 50, 80)]
    many = one + [_series(rng, 50, 80) for _ in range(8)]
    a = profile_shared_tau(_model(), one)
    b = profile_shared_tau(_model(), many)
    assert b.tau_sigma < a.tau_sigma / 2


def test_rate_matches_exp_linear_rate() -> None:
    """dx/dt = rate + Φ/τ·e^(−Δt/τ) = exp_linear_rate with A = −Φ, k = 1/τ."""
    m = _model(TAU)
    params = np.array([3.0, 100.0, 120.0])
    at = np.array([T0 + 0.01, T0 + 0.5, T0 + 1.2])
    r, sig = model_rate(m, params, np.zeros((3, 3)), at)
    expect = exp_linear_rate(at - T0, 100.0, -120.0, 1 / TAU)
    np.testing.assert_allclose(r, expect, rtol=1e-6)
    np.testing.assert_allclose(sig, 0.0)


def test_transient_rate_noise_free_is_exact() -> None:
    t = T0 + np.linspace(0, 1.25, 300)
    y = 1.0 + 70 * (t - T0) + 90 * (1 - np.exp(-(t - T0) / TAU))
    s = np.ones_like(t)
    y[::7] += 1e-9  # χ² > 0, so ŝ² is defined
    tr = transient_rate(_model(), t, y, s, [T0 + 1.0], tau=TAU)
    assert tr.rate[0] == pytest.approx(70 + 90 / TAU * math.exp(-1.0 / TAU), rel=1e-6)
    assert tr.params[1] == pytest.approx(70, rel=1e-6)


def test_tau_sigma_propagates_linearly() -> None:
    """σ² = σ_fixed² + (∂r/∂θ·σ_τ/τ)²: zero σ_τ adds nothing, 2σ_τ adds 4×."""
    rng = np.random.default_rng(4)
    t, y, s = _series(rng, 100, 120)
    at = [T0 + 0.1, T0 + 1.2]
    r0 = transient_rate(_model(), t, y, s, at, tau=TAU)
    r1 = transient_rate(_model(), t, y, s, at, tau=TAU, tau_sigma=0.01)
    r2 = transient_rate(_model(), t, y, s, at, tau=TAU, tau_sigma=0.02)
    np.testing.assert_allclose(r0.sigma, r0.sigma_fixed_tau)
    assert np.all(r1.sigma > r0.sigma)
    extra1 = r1.sigma**2 - r1.sigma_fixed_tau**2
    extra2 = r2.sigma**2 - r2.sigma_fixed_tau**2
    np.testing.assert_allclose(extra2, 4 * extra1, rtol=1e-9)


def test_nan_rows_dropped_and_errors() -> None:
    rng = np.random.default_rng(5)
    t, y, s = _series(rng, 10, 20)
    y[:5] = np.nan
    tr = transient_rate(_model(), t, y, s, T0 + 1, tau=TAU)
    assert tr.n_obs == len(t) - 5
    with pytest.raises(ValueError):
        profile_shared_tau(_model(), [])
    with pytest.raises(ValueError):
        profile_shared_tau(_model(), [(t, y, s)], tau_bounds=(1.0, 0.5))
    with pytest.raises(ValueError):
        transient_rate(_model(), t, y, s, T0 + 1, tau=-1.0)
    with pytest.raises(ValueError):
        profile_shared_tau(TrajectoryModel([cast(Any, Polynomial(1))]), [(t, y, s)])


def test_short_span_warns() -> None:
    rng = np.random.default_rng(6)
    series = [_series(rng, 50, 80, span=0.4)]
    with pytest.warns(UserWarning, match="3·tau"):
        profile_shared_tau(_model(), series, tau_bounds=(0.05, 2.0))


def _ar1(rng: np.random.Generator, n: int, rho: float) -> FloatArray:
    e = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + math.sqrt(1 - rho**2) * e[i]
    return x


def test_lag1_scale_recovers_ar1() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(4000.0)
    white = lag1_scale(t, rng.normal(0, 1, t.size), step=1.0)
    assert abs(white.rho) < 0.05 and white.k == pytest.approx(1.0, abs=0.06)
    red = lag1_scale(t, _ar1(rng, t.size, 0.6), step=1.0)
    assert red.rho == pytest.approx(0.6, abs=0.04)
    assert red.k == pytest.approx(2.0, rel=0.08)  # √(1.6/0.4)
    assert red.n_pairs == t.size - 1


def test_lag1_scale_k_widens_a_slope_as_ar1_does() -> None:
    # the k it gives is the ratio of the true slope scatter to the WLS σ
    rng = np.random.default_rng(5)
    t = np.arange(400.0)
    slopes = [np.polyfit(t, _ar1(rng, t.size, 0.5), 1)[0] for _ in range(400)]
    white_sd = 1.0 / math.sqrt(np.sum((t - t.mean()) ** 2))
    measured = float(np.std(slopes)) / white_sd
    assert measured == pytest.approx(math.sqrt(3.0), rel=0.15)


def test_lag1_scale_skips_gaps_and_short_series() -> None:
    t = np.array([0.0, 1.0, 5.0, 6.0, 20.0])
    few = lag1_scale(t, np.ones(5), step=1.0)
    assert few.n_pairs == 2 and math.isnan(few.rho) and few.k == 1.0
    rng = np.random.default_rng(1)
    t = np.arange(200.0)
    r = np.cumsum(rng.normal(0, 1, 200))  # a random walk: ρ → 1, clipped
    assert lag1_scale(t, r, step=1.0).rho == LAG1_RHO_MAX
    with pytest.raises(ValueError):
        lag1_scale(t, r[:-1], step=1.0)
    with pytest.raises(ValueError):
        lag1_scale(t, r, step=0.0)


def test_transient_rate_returns_its_residuals() -> None:
    rng = np.random.default_rng(2)
    t, y, s = _series(rng, 60.0, 80.0)
    y[5] = np.nan
    tr = transient_rate(_model(), t, y, s, [T0 + 1.0], tau=TAU)
    assert tr.t_used.size == tr.residuals.size == tr.n_obs == t.size - 1
    assert float(np.mean(tr.residuals**2)) * tr.n_obs / (
        tr.n_obs - tr.model.n_params
    ) == pytest.approx(tr.chi2nu, rel=1e-9)
