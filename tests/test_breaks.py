"""Tests for gps_analysis.breaks (rate-change placement)."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis.breaks import (
    LONG_GAP,
    SHORT_GAP,
    bin_medians,
    default_min_gap,
    rate_breaks,
    segment_edges,
)


def _series(
    t0: float,
    t1: float,
    rates: list[float],
    breaks: list[float],
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily continuous piecewise-linear series [mm] with white noise."""
    t = np.arange(t0, t1, 1 / 365.25)
    y = rates[0] * (t - t0)
    for b, (r_before, r_after) in zip(
        breaks, zip(rates[:-1], rates[1:], strict=True), strict=True
    ):
        y = y + (r_after - r_before) * np.maximum(t - b, 0.0)
    rng = np.random.default_rng(seed)
    return t, y + rng.normal(0.0, noise, t.size)


def test_straight_line_has_no_break() -> None:
    t, y = _series(2018.0, 2026.0, [15.0], [], 3.0, 1)
    got = rate_breaks(t, y)
    assert got.breaks == ()
    assert len(got.bic) == 1


def test_finds_known_breaks_within_a_month() -> None:
    truth = [2019.6, 2024.1, 2026.2]
    t, y = _series(2018.0, 2026.85, [15.0, 5.0, 18.0, 30.0], truth, 3.0, 2)
    got = rate_breaks(t, y)
    assert len(got.breaks) == 3
    for b, want in zip(got.breaks, truth, strict=True):
        assert abs(b - want) <= 1 / 12 + 1e-9


def test_short_span_uses_short_gap_and_finds_onset() -> None:
    # the VONC 2014 shape: flat, then 27 mm/yr from early Feb 2014
    t, y = _series(2013.67, 2014.62, [0.0, 27.0], [2014.09], 1.5, 3)
    got = rate_breaks(t, y)
    assert got.min_gap == SHORT_GAP
    assert len(got.breaks) == 1
    assert abs(got.breaks[0] - 2014.09) <= 1 / 12 + 1e-9


def test_min_gap_is_respected() -> None:
    t, y = _series(
        2010.0, 2020.0, [0.0, 20.0, 0.0, 20.0], [2013.0, 2013.3, 2016.0], 1.0, 4
    )
    got = rate_breaks(t, y, min_gap=1.0)
    b = np.array(got.breaks)
    assert np.all(np.diff(b) >= 1.0 - 1e-9)
    assert b.min() >= t.min() + 1.0 - 1e-9
    assert b.max() <= t.max() - 1.0 + 1e-9


def test_bic_decreases_by_the_gain() -> None:
    t, y = _series(2018.0, 2026.0, [5.0, 30.0], [2022.0], 2.0, 5)
    got = rate_breaks(t, y, min_bic_gain=10.0)
    assert len(got.bic) == len(got.breaks) + 1
    assert all(a - b >= 10.0 for a, b in zip(got.bic[:-1], got.bic[1:], strict=True))
    assert len(got.order) == len(got.breaks)


def test_max_breaks_and_gain_switch() -> None:
    t, y = _series(2018.0, 2026.0, [5.0, 30.0, 0.0], [2020.0, 2023.0], 1.0, 6)
    assert len(rate_breaks(t, y, max_breaks=1).breaks) == 1
    assert rate_breaks(t, y, min_bic_gain=1e9).breaks == ()


def test_offset_and_scale_do_not_move_breaks() -> None:
    t, y = _series(2018.0, 2026.0, [5.0, 30.0], [2022.0], 2.0, 7)
    a = rate_breaks(t, y)
    b = rate_breaks(t, 3.0 * y + 100.0)
    assert a.breaks == b.breaks


def test_nan_dropped_and_inputs_untouched() -> None:
    t, y = _series(2018.0, 2026.0, [5.0, 30.0], [2022.0], 2.0, 8)
    y[::17] = np.nan
    t_copy, y_copy = t.copy(), y.copy()
    got = rate_breaks(t, y)
    assert len(got.breaks) == 1
    np.testing.assert_array_equal(t, t_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_too_short_series() -> None:
    got = rate_breaks([2020.0, 2020.01], [1.0, 2.0])
    assert got.breaks == () and got.bic == ()
    assert rate_breaks([], []).n_bins == 0


def test_bin_medians() -> None:
    t = np.array([0.0, 0.1, 0.2, 1.05, 1.1, 2.5])
    y = np.array([1.0, 100.0, 3.0, 5.0, 7.0, 9.0])
    tb, yb = bin_medians(t, y, width=1.0)
    np.testing.assert_allclose(tb, [0.1, 1.075, 2.5])
    np.testing.assert_allclose(yb, [3.0, 6.0, 9.0])
    with pytest.raises(ValueError):
        bin_medians(t, y, width=0.0)
    with pytest.raises(ValueError):
        bin_medians(t, y[:-1])


def test_default_min_gap() -> None:
    assert default_min_gap(8.0) == LONG_GAP
    assert default_min_gap(1.0) == SHORT_GAP


def test_segment_edges() -> None:
    assert segment_edges(0.0, 10.0, [3.0, 7.0]) == [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0)]
    assert segment_edges(0.0, 10.0, [-1.0, 12.0]) == [(0.0, 10.0)]
    with pytest.raises(ValueError):
        segment_edges(1.0, 1.0, [])
    with pytest.raises(ValueError):
        rate_breaks([0.0], [0.0], candidate_step=0.0)
