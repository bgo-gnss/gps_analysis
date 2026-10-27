"""Tests for gps_analysis.displacement: end levels, σ, neighbour check, conversions."""

import math

import numpy as np
import pytest

from gps_analysis.displacement import (
    EndLevel,
    displacement_from_rate,
    end_displacement,
    end_level,
    level_sigma,
    line_value_at,
    neighbour_check,
    rate_from_displacement,
)
from gps_analysis.rate_quality import line_fit

DAY = 1.0 / 365.25


def _days(n: int, start: float = 2024.0, per_day: int = 1) -> np.ndarray:
    return start + np.arange(n * per_day) * DAY / per_day


def test_end_level_closed_form() -> None:
    t = _days(5)
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    s = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    lv = end_level(t, y, s, window=(t[0], t[-1] + DAY))
    w = 1 / s**2
    assert lv.level == pytest.approx(np.sum(w * y) / np.sum(w))
    assert lv.sigma_apriori == pytest.approx(1 / math.sqrt(np.sum(w)))
    chi = np.sum(w * (y - lv.level) ** 2) / 4
    assert lv.chi2nu == pytest.approx(chi)
    assert lv.n == 5
    assert lv.t_mean == pytest.approx(np.sum(w * t) / np.sum(w))


def test_window_is_half_open_and_drops_nan() -> None:
    t = _days(6)
    y = np.array([0.0, 1.0, np.nan, 1.0, 1.0, 9.0])
    s = np.ones(6)
    lv = end_level(t, y, s, window=(t[1], t[5]))  # t[5] excluded
    assert lv.n == 3
    assert lv.level == pytest.approx(1.0)


def test_empty_window_raises() -> None:
    t = _days(3)
    with pytest.raises(ValueError, match="no epochs"):
        end_level(t, np.ones(3), np.ones(3), window=(2030.0, 2031.0))


def test_level_sigma_is_a_max_never_a_product() -> None:
    # the three estimates; the answer is the largest, not k·√χ²·…
    assert level_sigma(1.0, 4.0, 4, k_day=1.5, s_day=1.0) == pytest.approx(2.0)
    assert level_sigma(1.0, 1.0, 4, k_day=1.5, s_day=1.0) == pytest.approx(1.5)
    assert level_sigma(1.0, 1.0, 1, k_day=1.0, s_day=3.0) == pytest.approx(3.0)
    # NaN inputs fall back (k → 1, s_day → 0, χ² skipped)
    assert level_sigma(2.0, math.nan, 1, k_day=math.nan, s_day=math.nan) == 2.0


def test_single_day_end_uses_floor() -> None:
    t = _days(1)
    lv = end_level(t, [5.0], [1.0], window=(t[0], t[0] + DAY), k_day=1.2, s_day=2.5)
    assert lv.n == 1 and math.isnan(lv.chi2nu)
    assert lv.sigma == pytest.approx(2.5)


def test_blunder_inflates_through_chi2() -> None:
    t = _days(5)
    y = np.zeros(5)
    y[2] = 30.0
    lv = end_level(t, y, np.ones(5), window=(t[0], t[-1] + DAY), k_day=1.0)
    assert lv.sigma > 5 * lv.sigma_apriori


def test_end_displacement_quadrature() -> None:
    a = EndLevel(1.0, 0.5, 3.0, 5, 2024.0, (2024.0, 2024.01), 1.0)
    b = EndLevel(11.0, 0.5, 4.0, 5, 2025.0, (2025.0, 2025.01), 1.0)
    d, s = end_displacement(a, b)
    assert d == 10.0 and s == pytest.approx(5.0)


def test_linear_motion_displacement_equals_rate_times_period() -> None:
    rng = np.random.default_rng(1)
    t = _days(366)
    v = 20.0  # mm/yr
    y = v * (t - t[0]) + rng.normal(0, 1.0, t.size)
    s = np.ones_like(t)
    n = 5
    a = end_level(t, y, s, window=(t[0], t[0] + n * DAY))
    b = end_level(t, y, s, window=(t[-1] + DAY - n * DAY, t[-1] + DAY))
    d, sd = end_displacement(a, b)
    expected = v * (b.t_mean - a.t_mean)
    assert abs(d - expected) < 3 * sd


def test_line_fit_level_fields() -> None:
    t = _days(10)
    y = 3.0 + 0.0 * t
    s = np.full(10, 2.0)
    f = line_fit(t, y, s)
    assert f.level == pytest.approx(3.0)
    assert f.sigma_level_apriori == pytest.approx(2.0 / math.sqrt(10))
    assert f.t_mean == pytest.approx(t.mean())


def test_line_value_at_grows_with_distance() -> None:
    rng = np.random.default_rng(2)
    t = _days(30)
    y = 50.0 * (t - t[0]) + rng.normal(0, 1, 30)
    f = line_fit(t, y, np.ones(30))
    v0, s0 = line_value_at(f, f.t_mean)
    v1, s1 = line_value_at(f, f.t_mean + 0.5, k_rate=3.0)
    assert v0 == pytest.approx(f.level)
    assert s1 > s0
    # the level σ is scaled by k_day (or √χ²/ν), never left formal
    _, s_k = line_value_at(f, f.t_mean, k_day=3.0)
    assert s_k == pytest.approx(3.0 * f.sigma_level_apriori)


def test_neighbour_check_flags_a_blunder() -> None:
    rng = np.random.default_rng(3)
    t = _days(10)
    y = rng.normal(0, 1, 10)
    s = np.ones(10)
    y[-1] = 30.0
    end = end_level(t, y, s, window=(t[-1], t[-1] + DAY))
    chk = neighbour_check(t, y, s, end=end, neighbour=(t[-6], t[-1]))
    assert chk.disputed is True


def test_neighbour_check_accepts_fast_steady_motion() -> None:
    rng = np.random.default_rng(4)
    t = _days(10)
    y = 20.0 * 365.25 * (t - t[0]) + rng.normal(0, 1, 10)  # 20 mm/day
    s = np.ones(10)
    end = end_level(t, y, s, window=(t[-1], t[-1] + DAY))
    chk = neighbour_check(t, y, s, end=end, neighbour=(t[-6], t[-1]))
    assert chk.disputed is False


def test_neighbour_check_unchecked_with_too_few() -> None:
    t = _days(3)
    y = np.zeros(3)
    end = end_level(t, y, np.ones(3), window=(t[-1], t[-1] + DAY))
    chk = neighbour_check(t, y, np.ones(3), end=end, neighbour=(t[0], t[-1]))
    assert chk.disputed is None and "n=2" in chk.reason


def test_conversions_round_trip() -> None:
    v = np.array([10.0, -3.0, 2.0])
    sv = np.array([0.5, 0.4, 1.0])
    d, sd = displacement_from_rate(v, sv, 2.5)
    np.testing.assert_allclose(d, v * 2.5)
    v2, sv2 = rate_from_displacement(d, sd, 2.5)
    np.testing.assert_allclose(v2, v, rtol=1e-12)
    np.testing.assert_allclose(sv2, sv, rtol=1e-12)
    with pytest.raises(ValueError):
        rate_from_displacement(d, sd, 0.0)


def test_eight_hour_sessions_window_in_days() -> None:
    """A window in days holds 3N 8-h epochs; the level matches 24 h."""
    rng = np.random.default_rng(5)
    t24 = _days(5)
    t8 = _days(5, per_day=3)
    lv24 = end_level(
        t24, 7 + rng.normal(0, 1, 5), np.ones(5), window=(2024.0, 2024.0 + 5 * DAY)
    )
    lv8 = end_level(
        t8, 7 + rng.normal(0, 1, 15), np.ones(15), window=(2024.0, 2024.0 + 5 * DAY)
    )
    assert lv8.n == 15 and lv24.n == 5
    assert abs(lv8.level - lv24.level) < 3 * math.hypot(lv8.sigma, lv24.sigma)


def test_step_displacement_recovers_a_step() -> None:
    from gps_analysis.displacement import step_displacement

    rng = np.random.default_rng(6)
    t = _days(120)
    event = t[0] + 60 * DAY  # a midnight between two noon epochs
    y = 10.0 * (t - t[0]) + np.where(t > event, 50.0, 0.0) + rng.normal(0, 1, t.size)
    est = step_displacement(
        t,
        y,
        np.ones(t.size),
        before=(event - 30 * DAY, event),
        after=(event, event + 30 * DAY),
    )
    assert abs(est.step - 50.0) < 3 * est.sigma
    assert est.at == (event - 30 * DAY + 30 * DAY, event)
    # the event's own day left out of both lines: same answer, wider gap
    gap = step_displacement(
        t,
        y,
        np.ones(t.size),
        before=(event - 30 * DAY, event),
        after=(event + DAY, event + 31 * DAY),
    )
    assert abs(gap.step - 50.0) < 3 * gap.sigma


def test_step_sigma_grows_with_k_rate_and_distance() -> None:
    from gps_analysis.displacement import step_displacement

    rng = np.random.default_rng(7)
    t = _days(80)
    y = rng.normal(0, 1, t.size)
    b = (t[0], t[0] + 30 * DAY)
    a = (t[0] + 50 * DAY, t[0] + 80 * DAY)
    near = step_displacement(t, y, np.ones(t.size), before=b, after=a)
    k = step_displacement(t, y, np.ones(t.size), k_rate=(5.0, 5.0), before=b, after=a)
    far = step_displacement(
        t, y, np.ones(t.size), at=(t[0] + 40 * DAY, t[0] + 40 * DAY), before=b, after=a
    )
    assert k.sigma > near.sigma
    assert far.sigma > near.sigma
    with pytest.raises(ValueError):
        step_displacement(t, y, np.ones(t.size), before=(2030.0, 2030.1), after=a)


def test_gls_mean_matches_the_weighted_mean_for_diagonal_covariance() -> None:
    from gps_analysis.displacement import gls_mean

    rng = np.random.default_rng(13)
    base = np.array([2602350.0, -1074405.0, 5704035.0])
    pts = base + rng.normal(0, 0.003, (10, 3))
    sig = rng.uniform(0.001, 0.004, (10, 3))
    cov = np.zeros((10, 3, 3))
    idx = np.arange(3)
    cov[:, idx, idx] = sig**2
    g = gls_mean(pts, cov)
    w = 1 / sig**2
    np.testing.assert_allclose(g.mean, (w * pts).sum(0) / w.sum(0), rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(g.cov)), 1 / np.sqrt(w.sum(0)))
    assert g.n == 10 and 0.2 < g.chi2nu < 3.0


def test_gls_mean_uses_the_correlations_and_one_point() -> None:
    from gps_analysis.displacement import gls_mean

    c = np.array(
        [
            [[1.0, 0.9, 0], [0.9, 1.0, 0], [0, 0, 1.0]],
            [[1.0, -0.9, 0], [-0.9, 1.0, 0], [0, 0, 1.0]],
        ]
    )
    g = gls_mean(np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]), c)
    assert g.mean[1] == pytest.approx(0.0, abs=1e-12)
    one = gls_mean(np.array([[5.0, 6.0, 7.0]]), np.eye(3)[None] * 4.0)
    assert math.isnan(one.chi2nu) and one.mean.tolist() == [5.0, 6.0, 7.0]
    with pytest.raises(ValueError):
        gls_mean(np.zeros((0, 3)), np.zeros((0, 3, 3)))
