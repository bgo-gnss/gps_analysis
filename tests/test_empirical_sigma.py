"""gps_analysis.empirical_sigma: daily scatter, σ scaling, distance-dependent common mode."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gps_analysis import empirical_sigma as es

DAY = es.DAY_YEARS


def _days(n: int, start: float = 2023.0) -> np.ndarray:
    return start + np.arange(n) * DAY


def test_differences_skip_gaps_and_never_bridge() -> None:
    t = np.r_[_days(5), _days(5, 2023.1)]
    y = np.arange(10.0)
    dy, i = es.consecutive_differences(t, y)
    assert i.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    np.testing.assert_array_equal(dy, np.ones((1, 8)))


def test_daily_scatter_ignores_trend_steps_and_seasonal() -> None:
    """Differencing removes slow signal: σ̂ recovers the white σ (±5 %, N=3000)."""
    rng = np.random.default_rng(1)
    t = _days(3000)
    signal = 12.0 * (t - t[0]) + 4 * np.sin(2 * np.pi * t) + 30.0 * (t > 2025.0)
    y = np.stack([signal + rng.normal(0, s, t.size) for s in (1.0, 1.5, 4.0)])
    np.testing.assert_allclose(es.daily_scatter(t, y), [1.0, 1.5, 4.0], rtol=0.05)


def test_daily_scatter_is_nan_below_min_pairs() -> None:
    t = _days(20)
    assert np.isnan(es.daily_scatter(t, np.zeros(20))).all()
    assert np.isfinite(
        es.daily_scatter(t, np.random.default_rng(0).normal(size=20), min_pairs=10)
    ).all()


def test_sigma_scale_factor_with_time_varying_sigma() -> None:
    rng = np.random.default_rng(2)
    t = _days(4000)
    formal = np.where(t < 2027.0, 2.0, 1.0) * np.ones((3, t.size))
    true = formal * np.array([[1.0], [0.5], [1.7]])
    y = rng.normal(0.0, true)
    np.testing.assert_allclose(
        es.sigma_scale_factor(t, y, formal), [1.0, 0.5, 1.7], rtol=0.05
    )


def test_shared_fraction_recovers_the_common_part() -> None:
    rng = np.random.default_rng(3)
    t = _days(5000)
    rho, s = 0.8, 2.0
    common = rng.normal(0, s * np.sqrt(rho), t.size)
    a = common + rng.normal(0, s * np.sqrt(1 - rho), t.size)
    b = common + rng.normal(0, s * np.sqrt(1 - rho), t.size)
    est = es.shared_fraction(
        es.daily_scatter(t, a), es.daily_scatter(t, b), es.daily_scatter(t, a - b)
    )
    assert est[0] == pytest.approx(rho, abs=0.03)


def test_model_limits_and_record_round_trip() -> None:
    m = es.CommonModeModel((0.96, 0.92, 0.96), (0.53, 0.49, 0.50), (11.7, 30.4, 7.8))
    r = es.shared_fraction_model([0.0, 1e6], m)
    np.testing.assert_allclose(r[:, 0], m.rho0)
    np.testing.assert_allclose(r[:, 1], m.rho_inf, atol=1e-12)
    assert es.shared_fraction_model(np.zeros((2, 4)), m).shape == (3, 2, 4)
    back = es.CommonModeModel.from_record(json.loads(json.dumps(m.to_record())))
    assert back == m


def test_fit_recovers_a_synthetic_law() -> None:
    rng = np.random.default_rng(4)
    true = es.CommonModeModel((0.95, 0.9, 0.97), (0.5, 0.45, 0.55), (10.0, 30.0, 6.0))
    d = rng.uniform(0.05, 60.0, 1500)
    rho = es.shared_fraction_model(d, true) + rng.normal(0, 0.03, (3, d.size))
    rho[:, ::50] -= 0.6  # gross outliers: the soft-L1 loss must shrug them off
    fit = es.fit_common_mode(d, rho)
    np.testing.assert_allclose(fit.rho0, true.rho0, atol=0.02)
    np.testing.assert_allclose(fit.rho_inf, true.rho_inf, atol=0.02)
    np.testing.assert_allclose(fit.length, true.length, rtol=0.2)


def test_baseline_sigma_limits() -> None:
    np.testing.assert_allclose(es.baseline_sigma(3.0, 4.0, 0.0), 5.0)
    np.testing.assert_allclose(es.baseline_sigma(2.0, 2.0, 1.0), 0.0)
    np.testing.assert_allclose(es.baseline_sigma(2.0, 2.0, 0.75), 2.0 * np.sqrt(0.5))
    assert es.baseline_sigma(
        np.ones((3, 5)), np.ones((3, 5)), np.full((3, 1), 0.5)
    ).shape == (3, 5)


def test_end_to_end_baseline_sigma_matches_the_baseline_scatter() -> None:
    """Scale + ρ predict the A − B scatter that quadrature overstates."""
    rng = np.random.default_rng(5)
    t = _days(3000)
    rho, s = 0.9, 2.0
    formal = np.full(t.size, 3.0)  # formal σ 1.5× too large, as for IMO Up
    common = rng.normal(0, s * np.sqrt(rho), t.size)
    a = common + rng.normal(0, s * np.sqrt(1 - rho), t.size)
    b = common + rng.normal(0, s * np.sqrt(1 - rho), t.size)
    ka = es.sigma_scale_factor(t, a, formal)[0]
    kb = es.sigma_scale_factor(t, b, formal)[0]
    pred = es.baseline_sigma(ka * 3.0, kb * 3.0, rho)
    obs = es.daily_scatter(t, a - b)[0]
    assert pred == pytest.approx(obs, rel=0.07)
    assert es.baseline_sigma(3.0, 3.0, 0.0) / obs > 4  # what quadrature would claim


def test_refusals() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        es.consecutive_differences([1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="y must be"):
        es.daily_scatter(_days(3), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="sigma shape"):
        es.sigma_scale_factor(_days(3), np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="rho0"):
        es.CommonModeModel((0.4,), (0.5,), (1.0,), ("N",))
    with pytest.raises(ValueError, match="equal length"):
        es.CommonModeModel((0.9,), (0.5,), (1.0,))
    with pytest.raises(ValueError, match="unknown"):
        es.CommonModeModel.from_record({"model": "gauss"})
    m = es.CommonModeModel((0.9,), (0.5,), (1.0,), ("N",))
    with pytest.raises(ValueError, match=">= 0"):
        es.shared_fraction_model(-1.0, m)
    with pytest.raises(ValueError, match="5 finite"):
        es.fit_common_mode([1.0, 2.0], [[0.9, 0.8]], components=("N",))
    with pytest.raises(ValueError, match="rho must"):
        es.baseline_sigma(1.0, 1.0, 1.5)
