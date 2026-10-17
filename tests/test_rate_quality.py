"""rate_quality: the empirical rate scale k(T) and the window gates."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis import empirical_rate_scale, largest_gap, window_gates


def _daily(years: float) -> np.ndarray:
    return 2010.0 + np.arange(0.0, years, 1 / 365.25)


def test_white_noise_with_true_sigma_gives_k_near_one() -> None:
    rng = np.random.default_rng(0)
    t = _daily(8.0)
    sigma = np.full((3, t.size), 2.0)
    y = rng.normal(0.0, 2.0, size=(3, t.size))
    rs = empirical_rate_scale(t, y, sigma, window_years=0.25)
    assert rs.n_windows.min() >= 40
    np.testing.assert_allclose(rs.k, 1.0, atol=0.25)


def test_correlated_noise_inflates_k() -> None:
    rng = np.random.default_rng(1)
    t = _daily(8.0)
    white = rng.normal(0.0, 1.0, t.size)
    walk = np.cumsum(rng.normal(0.0, 0.4, t.size))  # random walk: rate noise
    y = white + walk
    rs = empirical_rate_scale(t, y, np.ones_like(y), window_years=0.5)
    assert rs.k.shape == (1,)
    assert rs.k[0] > 3.0  # formal σ assumes white noise: far too small


def test_k_is_never_below_one_and_nan_without_enough_windows() -> None:
    rng = np.random.default_rng(2)
    t = _daily(2.0)
    y = rng.normal(0.0, 1.0, t.size)
    # σ over-stated tenfold: the spread is smaller than formal, k floors at 1
    rs = empirical_rate_scale(t, y, np.full(t.size, 10.0), window_years=0.1)
    assert rs.k[0] == 1.0
    # windows as long as the series: one window, below min_windows
    rs = empirical_rate_scale(t, y, np.ones(t.size), window_years=1.9)
    assert np.isnan(rs.k[0]) and rs.n_windows[0] < 8


def test_nan_epochs_are_dropped_per_component() -> None:
    rng = np.random.default_rng(3)
    t = _daily(6.0)
    y = rng.normal(0.0, 1.0, size=(2, t.size))
    y[1, ::3] = np.nan
    rs = empirical_rate_scale(t, y, np.ones_like(y), window_years=0.5)
    assert np.all(np.isfinite(rs.k))


def test_bad_arguments_raise() -> None:
    t = _daily(1.0)
    with pytest.raises(ValueError, match="window_years"):
        empirical_rate_scale(t, np.zeros(t.size), np.ones(t.size), window_years=0)
    with pytest.raises(ValueError, match="shapes differ"):
        empirical_rate_scale(t, np.zeros(3), np.ones(3), window_years=0.1)


def test_window_gates_name_each_reason() -> None:
    t = np.linspace(2020.0, 2020.05, 4)
    reasons = window_gates(t, (2020.0, 2020.5), step_epochs=[2020.2, 2021.0])
    assert reasons == ("n_obs=4<5", "cover=0.10<0.8", "step_in_window@2020.2000")
    # a week of daily epochs passes: the inter-episode windows of an unrest
    week = 2020.0 + np.arange(8) / 365.25
    assert window_gates(week, (2020.0, 2020.0 + 7 / 365.25)) == ()
    full = np.linspace(2020.0, 2020.5, 100)
    assert window_gates(full, (2020.0, 2020.5)) == ()
    # a step on the window edge is not inside it
    assert window_gates(full, (2020.0, 2020.5), step_epochs=[2020.5]) == ()
    with pytest.raises(ValueError, match="not after"):
        window_gates(full, (2020.5, 2020.0))


def test_largest_gap() -> None:
    assert largest_gap([2020.0, 2020.1, 2020.6, np.nan]) == pytest.approx(0.5)
    assert largest_gap([2020.0]) == 0.0


def test_rate_scale_at_interpolates_in_log_t_and_holds_the_ends() -> None:
    from gps_analysis import rate_scale_at

    years = [0.1, 1.0]
    k = [[2.0, 4.0], [4.0, np.nan]]
    np.testing.assert_allclose(rate_scale_at(np.sqrt(0.1), years, k), [3.0, 4.0])
    np.testing.assert_allclose(rate_scale_at(0.01, years, k), [2.0, 4.0])
    np.testing.assert_allclose(rate_scale_at(10.0, years, k), [4.0, 4.0])
    assert np.isnan(rate_scale_at(1.0, [1.0], [[np.nan]]))[0]
    with pytest.raises(ValueError, match="window_years"):
        rate_scale_at(0.0, years, k)
    with pytest.raises(ValueError, match="table shapes"):
        rate_scale_at(1.0, [1.0], k)


def test_line_fit_matches_the_wls_estimator_and_scales_sigma() -> None:
    from gps_analysis import estimate_velocity, line_fit

    rng = np.random.default_rng(4)
    t = _daily(0.1)
    s = np.full(t.size, 2.0)
    y = 30.0 * (t - t[0]) + rng.normal(0.0, 3.0, t.size)
    fit = line_fit(t, y, s)
    est = estimate_velocity(t, y, s, model="linear")
    assert fit.rate == pytest.approx(float(est.rates[0]), rel=1e-9)
    assert fit.sigma_scaled == pytest.approx(float(est.sigmas[0]), rel=1e-6)
    assert fit.chi2nu == pytest.approx(9.0 / 4.0, rel=0.4)
    assert fit.sigma_short >= fit.sigma_apriori


def test_line_fit_on_two_epochs_and_a_blunder() -> None:
    from gps_analysis import line_fit

    two = line_fit([2020.0, 2020.01], [0.0, 5.0], [2.0, 2.0])
    assert two.rate == pytest.approx(500.0)
    assert two.sigma_apriori == pytest.approx(np.sqrt(8.0) / 0.01)
    assert np.isnan(two.chi2nu) and two.sigma_short == two.sigma_apriori
    # a perfect line has χ² = 0: the short σ does not shrink below a priori
    three = line_fit([2020.0, 2020.01, 2020.02], [0.0, 5.0, 10.0], [2.0] * 3)
    assert three.sigma_scaled < 1e-6 and three.sigma_short == three.sigma_apriori
    # an exact line on more epochs: the quadratic's F is round-off, not a flag
    exact = line_fit(2020.0 + np.arange(6) / 365.25, 7.0 * np.arange(6), [2.0] * 6)
    assert np.isnan(exact.f_quadratic)
    # a blunder at the end inflates the short σ
    bad = line_fit([2020.0, 2020.01, 2020.02], [0.0, 0.0, 30.0], [2.0] * 3)
    assert bad.sigma_short > 2 * bad.sigma_apriori
    with pytest.raises(ValueError, match="2 epochs"):
        line_fit([2020.0], [0.0], [1.0])


def test_quadratic_f_flags_a_decaying_pulse_not_noise() -> None:
    from gps_analysis import NONLINEAR_F, line_fit

    rng = np.random.default_rng(5)
    t = _daily(30 / 365.25)
    s = np.full(t.size, 2.0)
    quiet = line_fit(t, rng.normal(0.0, 2.0, t.size), s)
    decay = 100.0 * (1 - np.exp(-(t - t[0]) * 365.25 / 5.0))
    pulse = line_fit(t, decay + rng.normal(0.0, 2.0, t.size), s)
    assert quiet.f_quadratic < NONLINEAR_F < pulse.f_quadratic


def test_signal_to_noise_of_the_horizontal_pair_and_up() -> None:
    from gps_analysis import signal_to_noise

    h, u = signal_to_noise([3.0, 4.0, -10.0], [1.0, 1.0, 2.0])
    assert h == pytest.approx(5.0) and u == pytest.approx(5.0)
    h, u = signal_to_noise([0.0, 0.0, 1.0], [1.0, 1.0, np.nan])
    assert np.isnan(h) and np.isnan(u)
