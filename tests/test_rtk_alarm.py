"""rtk_alarm: the window offset of an RTK baseline and its alarm level."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis.rtk_alarm import alarm_level, robust_scatter, window_offset


def _series(step: float, at: float, n: int = 600, sd: float = 2.0) -> np.ndarray:
    """One-minute epochs 0 … n−1, white noise sd, a step of ``step`` at ``at``."""
    rng = np.random.default_rng(1)
    t = np.arange(n, dtype=float)
    y = rng.normal(0.0, sd, (3, n))
    y[:, t >= at] += step
    return t, y


def test_offset_finds_a_step_inside_the_window() -> None:
    t, y = _series(20.0, at=560)
    w = window_offset(t, y, end=599, window=30, reference=360)
    assert w is not None and w.n_after == 30 and w.n_before == 360
    np.testing.assert_allclose(w.offset, 20.0, atol=1.5)
    # σ_Δ ≈ 1.2533·2·√(1/360 + 1/30) ≈ 0.48
    np.testing.assert_allclose(w.sigma, 0.48, rtol=0.25)


def test_no_step_gives_an_offset_near_zero() -> None:
    t, y = _series(0.0, at=10_000)
    w = window_offset(t, y, end=599, window=60, reference=360)
    assert w is not None
    assert np.all(np.abs(w.offset) < 4 * w.sigma)


def test_gap_keeps_the_onset_out_of_the_reference() -> None:
    t, y = _series(20.0, at=500)  # the step is 100 min before the end
    no_gap = window_offset(t, y, end=599, window=60, reference=360)
    gapped = window_offset(t, y, end=599, window=60, reference=360, gap=60)
    assert no_gap is not None and gapped is not None
    np.testing.assert_allclose(gapped.offset, 20.0, atol=1.5)
    assert np.all(no_gap.offset < gapped.offset)  # its reference holds the step


def test_too_few_epochs_and_nan_epochs() -> None:
    t, y = _series(0.0, at=10_000, n=50)
    assert window_offset(t, y, end=49, window=10, reference=360, min_before=100) is None
    y[1, 45] = np.nan  # an epoch with any NaN component is left out
    w = window_offset(t, y, end=49, window=10, reference=30)
    assert w is not None and w.n_after == 9


def test_robust_scatter_is_sigma_at_the_gaussian() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 3.0, (2, 20_000))
    y[0, :100] = 1e6  # blunders hardly move it
    np.testing.assert_allclose(robust_scatter(y), 3.0, rtol=0.03)
    assert np.isnan(robust_scatter([np.nan, np.nan]))[0]


@pytest.mark.parametrize(
    ("value", "want"),
    [(0.0, 0), (9.9, 0), (10.0, 1), (-25.0, 2), (40.0, 3), (1e9, 3), (np.nan, 0)],
)
def test_alarm_level(value: float, want: int) -> None:
    assert alarm_level(value, (10.0, 20.0, 40.0)) == want


def test_alarm_level_refuses_unsorted_edges() -> None:
    with pytest.raises(ValueError):
        alarm_level(1.0, (20.0, 10.0, 40.0))
