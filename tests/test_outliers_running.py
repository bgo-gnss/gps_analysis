"""§15 — prior-free running-window detection (``method="running"``)."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis import (
    PROTECT_FLOOR,
    PROTECT_RUN,
    PROTECT_STEP,
    PROTECT_WINDOW,
    OutlierParams,
    detect_outliers_running,
    one_sided_line_predictions,
    rolling_difference_scale,
)

DAY = 1.0 / 365.25
NOISE = 1.0  # mm


def _series(n_days: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Daily epochs, a 30 mm/yr line plus white noise [mm]."""
    rng = np.random.default_rng(seed)
    t = 2020.0 + np.arange(n_days) * DAY
    return t, 30.0 * (t - t[0]) + rng.normal(0.0, NOISE, t.size)


def test_an_isolated_blunder_is_flagged() -> None:
    t, y = _series()
    y[200] += 15.0
    det = detect_outliers_running(t, y)
    assert det.flags.shape == y.shape  # 1-D in, 1-D out
    assert np.flatnonzero(det.flags).tolist() == [200]
    assert det.z[200] > 4.5


def test_a_step_is_not_flagged() -> None:
    t, y = _series()
    y[200:] += 60.0  # coseismic / dike step, not declared anywhere
    det = detect_outliers_running(t, y)
    assert not det.flags[190:210].any()


def test_a_ramp_between_the_sides_is_a_transition() -> None:
    t, y = _series()
    y[200] += 20.0  # day of the dike: halfway
    y[201:] += 40.0
    det = detect_outliers_running(t, y)
    assert not det.flags[195:206].any()


def test_one_sided_verdicts_are_never_flags() -> None:
    t, y = _series()
    y[2] += 25.0  # too close to the series start for a left side
    y[-2] += 25.0  # the newest days: no right side yet
    det = detect_outliers_running(t, y)
    assert not det.flags[2] and not det.flags[-2]
    assert det.candidates[2] and det.candidates[-2]
    assert det.protected[-2] & PROTECT_STEP
    last = [e for e in det.suspected_events if e.i_start <= t.size - 2 <= e.i_end]
    assert last and last[0].kind == "step" and np.isnan(last[0].step_evidence)


@pytest.mark.parametrize("days", [1, 2, 3])
def test_short_blunder_clusters_are_flagged(days: int) -> None:
    t, y = _series()
    y[200 : 200 + days] += 15.0
    det = detect_outliers_running(t, y)
    assert np.flatnonzero(det.flags).tolist() == list(range(200, 200 + days))


@pytest.mark.parametrize("days", [4, 6, 12])
def test_a_run_of_bad_days_is_signal(days: int) -> None:
    t, y = _series()
    y[200 : 200 + days] += 15.0  # off and back: a transient, not a blunder
    det = detect_outliers_running(t, y)
    assert not det.flags.any()
    if days == 4:
        assert (det.protected[200:204] & PROTECT_RUN).any()
        assert any(e.kind == "transient_run" for e in det.suspected_events)


@pytest.mark.parametrize(("after", "flagged"), [(2, True), (7, True), (12, True)])
def test_a_blunder_next_to_a_step(after: int, flagged: bool) -> None:
    """Right after an undeclared step the sides disagree and the step's
    days make a one-sided run around the blunder; a run without a core no
    longer shields a dominant (15 σ) blunder, so it is caught at 2 days too."""
    t, y = _series()
    y[200:] += 60.0
    y[200 + after] += 15.0
    det = detect_outliers_running(t, y)
    assert np.flatnonzero(det.flags).tolist() == ([200 + after] if flagged else [])


def test_floor_and_protect_window() -> None:
    t, y = _series()
    y[100] += 8.0
    y[300] += 15.0
    det = detect_outliers_running(
        t, y, min_outlier=10.0, protect_windows=[(t[295], t[305])]
    )
    assert not det.flags[100] and det.protected[100] & PROTECT_FLOOR
    assert not det.flags[300] and det.protected[300] & PROTECT_WINDOW


def test_threshold_is_the_tolerance_in_sigma() -> None:
    t, y = _series()
    y[200] += 6.0  # ~6 noise units off both sides
    assert detect_outliers_running(
        t, y, params=OutlierParams(running_n_sigma=3.0)
    ).flags[200]
    assert not detect_outliers_running(
        t, y, params=OutlierParams(running_n_sigma=12.0)
    ).flags[200]


def test_too_many_candidates_abort_the_component() -> None:
    t, y = _series()
    y2 = np.vstack([y, y.copy()])
    y2[1, 20:380:6] += 20.0  # 60 isolated blunders = 15 % of the epochs
    det = detect_outliers_running(t, y2)
    assert det.component_abort.tolist() == [False, True]
    assert not det.flags[1].any() and det.excess_flag_abort


def test_the_noise_level_ignores_steps() -> None:
    t, y = _series(seed=3)
    y[200:] += 80.0
    s = rolling_difference_scale(t, y, half_window=60 * DAY, max_gap=1.5 * DAY)
    assert np.nanmax(np.abs(s - NOISE)) < 0.35


def test_one_sided_predictions_exclude_the_epoch() -> None:
    t = 2020.0 + np.arange(40) * DAY
    x = 2.0 * (t - t[0])
    x[20] = 99.0
    before, after = one_sided_line_predictions(t, x, half_window=10 * DAY, min_count=5)
    assert before[20] == pytest.approx(2.0 * (t[20] - t[0]), abs=1e-9)
    assert after[20] == pytest.approx(2.0 * (t[20] - t[0]), abs=1e-9)
    assert np.isnan(before[0]) and np.isnan(after[-1])


@pytest.mark.parametrize(
    ("kw", "match"),
    [({"method": "kalman"}, "method"), ({"running_min_count": 2}, "running_min_count"),
     ({"running_n_sigma": 0.0}, "running_n_sigma")],
)  # fmt: skip
def test_bad_params_are_refused(kw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OutlierParams(**kw)  # type: ignore[arg-type]


def test_running_is_the_default_method() -> None:
    """BGÓ 2026-10-19: the cleaned view's default detector."""
    assert OutlierParams().method == "running"


def test_flags_are_nested_in_the_tolerance() -> None:
    """Lowering --outlier-sigma only ever adds flags (the run rule's
    excursion level is fixed, not tied to k)."""
    t, y = _series(seed=5)
    y[100] += 9.0
    y[250:253] += 12.0
    y[320:330] += 8.0  # a transient
    prev = None
    for k in (6.0, 5.0, 4.5, 4.0, 3.5, 3.0):
        f = detect_outliers_running(t, y, params=OutlierParams(running_n_sigma=k)).flags
        if prev is not None:
            assert not (prev & ~f).any(), k
        prev = f


def test_a_side_reaches_past_missing_days() -> None:
    """THOB 2026-09-19: three missing days left 7 of 8 epochs before a
    46 mm blunder — a one-sided verdict, never flagged. A side now takes
    the nearest running_min_count epochs within running_max_reach_days."""
    t, y = _series()
    keep = np.ones(t.size, dtype=bool)
    keep[[192, 195, 198]] = False  # three missing days in the 10 before
    t, y = t[keep], y[keep]
    i = int(np.searchsorted(t, 2020.0 + 200 * DAY))
    y[i] += 40.0
    assert np.flatnonzero(detect_outliers_running(t, y).flags).tolist() == [i]
    no_reach = OutlierParams(running_max_reach_days=10.0)
    assert not detect_outliers_running(t, y, params=no_reach).flags[i]


def test_reach_is_bounded() -> None:
    """Beyond running_max_reach_days a side gives no verdict (a long gap)."""
    t = np.r_[2020.0 + np.arange(40) * DAY, 2020.0 + (np.arange(40) + 80) * DAY]
    x = np.zeros(t.size)
    before, after = one_sided_line_predictions(
        t, x, half_window=10 * DAY, min_count=8, max_reach=30 * DAY
    )
    assert np.isnan(before[40]) and np.isfinite(after[40])  # 41 d gap behind


# HRAC East, 90 days to 2026-10-20, plate-fixed with the seasonal removed
# [mm] (yearf = 2026 + (doy + 0.5)/365, as the .pos files): −42 and +33 mm
# on doy 258/261, the data back at its level in between and after.
HRAC_DOY = [201, *range(203, 253), *range(254, 290), 291, 292]
HRAC_E = [
    0.8, -2.2, 0.6, 0.1, 0.6, 0.6, 1.1, -1.2, 2.0, -0.3, 3.8, 2.2, 2.8, 2.8,
    0.7, -0.4, -0.7, 0.5, -0.4, 0.1, 0.8, 0.2, 1.4, 1.2, 0.4, 2.0, 1.9, -0.2,
    0.9, 0.5, -3.6, -4.9, -4.5, -3.9, -2.7, -4.4, -3.6, -2.4, -2.6, -3.1, -1.2,
    -1.6, -2.8, -2.9, -2.9, -2.6, -2.6, -0.3, -2.3, -2.2, -2.9, -1.8, -2.2,
    -3.2, -42.0, -3.6, -0.4, 33.5, -1.7, -2.7, -6.8, -5.6, -6.7, -2.9, -0.5,
    -1.4, -1.1, -1.3, -4.9, -2.7, 0.7, -4.3, -3.0, -2.2, -2.2, -1.6, -2.7,
    -2.8, 1.1, -2.1, -0.1, -5.2, -7.8, -2.4, -3.3, -3.1, -1.7, -7.5, -8.0,
]  # fmt: skip


def test_lone_blunders_in_a_coreless_run_are_flagged() -> None:
    """HRAC E 2026-09-16/19: the neighbours of each blunder are off only the
    far-side line (its extrapolation wanders on this noise), so the 3-day
    "run" around them has no core and must not protect them."""
    t = 2026.0 + (np.array(HRAC_DOY) + 0.5) / 365.0
    y = np.array(HRAC_E)
    det = detect_outliers_running(t, y, params=OutlierParams(running_n_sigma=3.5))
    assert np.flatnonzero(det.flags).tolist() == [54, 57]


def test_a_peaked_hump_is_signal() -> None:
    """A hump off both sides on several days has a core: kept, peak too."""
    t, y = _series()
    y[200:207] += np.array([5.0, 12.0, 25.0, 30.0, 25.0, 12.0, 5.0])
    det = detect_outliers_running(t, y)
    assert not det.flags[198:209].any()
