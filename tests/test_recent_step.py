"""recent_step: the step of a short window against the epochs before it."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis.recent_step import YEAR_DAYS, daily_noise, recent_step


def _days(n: int, start: float = 2025.0) -> np.ndarray:
    return start + np.arange(n) / YEAR_DAYS


def test_daily_noise_recovers_white_noise_and_ignores_a_trend() -> None:
    rng = np.random.default_rng(1)
    t = _days(2000)
    y = 2.0 * rng.standard_normal(2000) + 30.0 * (t - t[0])  # 30 mm/yr trend
    assert daily_noise(y)[0] == pytest.approx(2.0, rel=0.08)


def test_a_slip_in_the_window_stands_out_of_the_noise() -> None:
    rng = np.random.default_rng(2)
    t = _days(60)
    y = 1.5 * rng.standard_normal((2, 60))
    y[0, 55:] += 10.0  # 10 mm north in the last week
    r = recent_step(t, y, end=t[-1], noise=[1.5, 1.5])
    assert r is not None
    assert r.n_after == 7 and r.n_before == 14
    assert r.step[0] == pytest.approx(10.0 * 5 / 7, abs=2.5)  # median of 7 days
    assert abs(r.step[0]) / r.sigma[0] > 4
    assert abs(r.step[1]) / r.sigma[1] < 3


def test_a_quiet_week_gives_a_step_within_its_sigma() -> None:
    rng = np.random.default_rng(3)
    t = _days(60)
    y = 1.5 * rng.standard_normal((3, 60))
    r = recent_step(t, y, end=t[-1])
    assert r is not None
    assert np.all(np.abs(r.step) < 3 * r.sigma)
    assert r.rate == pytest.approx(r.step / r.span_years)


def test_a_step_across_a_data_gap_is_measured() -> None:
    t = np.concatenate([_days(30), _days(10, start=2025.0 + 90 / YEAR_DAYS)])
    y = np.concatenate([np.zeros(30), np.full(10, 8.0)])
    r = recent_step(t, y, end=t[-1], noise=[1.0])
    assert r is not None and r.step[0] == pytest.approx(8.0)
    assert r.span_years > 60 / YEAR_DAYS  # the rate is over the gap


def test_too_few_epochs_give_none() -> None:
    t = _days(10)
    assert recent_step(t, np.zeros(10), end=t[-1]) is None  # 3 before
    assert recent_step(t[:12], np.zeros(10)[:12], end=t[-1] + 30) is None
