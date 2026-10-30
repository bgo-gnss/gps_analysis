"""gps_analysis.direction: motion direction from robust end levels."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis import direction


def _series(az_deg: float, plunge_deg: float, size: float, n: int = 200,
            noise: float = 0.0, seed: int = 0):  # fmt: skip
    """A ramp of ``size`` toward azimuth/plunge over t = 0…1, with noise."""
    t = np.linspace(0.0, 1.0, n)
    a, p = np.radians(az_deg), np.radians(plunge_deg)
    unit = np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), -np.sin(p)])
    y = unit[:, None] * size * t[None, :]
    y = y + np.random.default_rng(seed).normal(0.0, noise, y.shape)
    return t, y, np.full_like(y, max(noise, 1e-3))


@pytest.mark.parametrize("az", [10.0, 95.0, 205.0, 340.0])
def test_direction_of_a_clean_ramp(az: float) -> None:
    t, y, s = _series(az, 20.0, 50.0)
    d = direction.motion_direction(t, y, s, 0.0, 1.0, 0.05)
    assert d.azimuth == pytest.approx(az, abs=1e-6)
    assert d.plunge == pytest.approx(20.0, abs=1e-6)
    # the end medians sit half a window in: |d| is a bit short of 50
    assert 0.9 * 50 < np.linalg.norm(d.displacement) < 50
    assert d.significant


def test_sigma_follows_the_noise_and_outliers_do_not_move_it() -> None:
    t, y, s = _series(205.0, 0.0, 30.0, n=400, noise=2.0)
    clean = direction.motion_direction(t, y, s, 0.0, 1.0, 0.1)
    y2 = y.copy()
    y2[:, 3] += 500.0  # one wild epoch in the first window
    spiked = direction.motion_direction(t, y2, s, 0.0, 1.0, 0.1)
    assert abs(spiked.azimuth - clean.azimuth) < clean.sigma_azimuth
    assert abs(clean.azimuth - 205.0) < 3 * clean.sigma_azimuth
    assert 0.5 < clean.sigma_azimuth < 5.0


def test_noise_alone_has_no_significant_direction() -> None:
    t, y, s = _series(0.0, 0.0, 0.0, noise=2.0)
    assert not direction.motion_direction(t, y, s, 0.0, 1.0, 0.1).significant


def test_azimuth_sigma_matches_propagation() -> None:
    t = np.array([0.0, 1.0])
    y = np.array([[0.0, 3.0], [0.0, 4.0], [0.0, 0.0]])
    s = np.array([[0.3, 0.4], [0.3, 0.4], [1.0, 1.0]])
    d = direction.motion_direction(t, y, s, 0.0, 1.0, 0.5)
    sn = se = 0.5  # √(0.3² + 0.4²)
    want = np.degrees(np.sqrt((4 * sn) ** 2 + (3 * se) ** 2) / 25.0)
    assert d.sigma_azimuth == pytest.approx(want)
    assert d.azimuth == pytest.approx(np.degrees(np.arctan2(4, 3)))


@pytest.mark.parametrize(
    "start, end, window", [(1.0, 0.0, 0.1), (0.0, 1.0, 0.0), (0.0, 1.0, 0.6)]
)
def test_bad_intervals(start: float, end: float, window: float) -> None:
    t, y, s = _series(0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        direction.motion_direction(t, y, s, start, end, window)


def test_an_empty_end_is_an_error() -> None:
    t, y, s = _series(0.0, 0.0, 1.0)
    y[:, t > 0.9] = np.nan
    with pytest.raises(ValueError, match="no samples"):
        direction.motion_direction(t, y, s, 0.0, 1.0, 0.05)
