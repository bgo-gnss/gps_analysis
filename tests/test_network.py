"""Network days: whole-solution bad days from per-station running verdicts."""

from __future__ import annotations

import numpy as np
import pytest

from gps_analysis import detect_outliers, detect_outliers_running
from gps_analysis.models import linear
from gps_analysis.network import day_scores, network_days

DAY = 1.0 / 365.25


def _station(seed: int, n_days: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = 2020.0 + np.arange(n_days) * DAY
    return t, rng.normal(0.0, 1.0, (3, n_days))


def test_day_scores_vote_only_with_two_sides() -> None:
    t, y = _station(1)
    y[1, 150] += 20.0  # a blunder in east
    s = day_scores(detect_outliers_running(t, y))
    assert s.shape == y.shape
    assert s[1, 150] > 6.0
    assert np.isnan(s[:, 0]).all() and np.isnan(s[:, -1]).all()  # one side only
    assert np.nanmax(np.delete(s[1], 150)) < 6.0


def test_day_scores_refuse_the_model_method() -> None:
    t, y = _station(2)
    with pytest.raises(ValueError, match="running"):
        day_scores(detect_outliers(linear, t, y[0]))


def _scores(n_sta: int, n_days: int, bad: dict[int, int]) -> np.ndarray:
    """Synthetic (S, 3, D) scores: day -> number of stations with a 10 σ vote."""
    sc = np.zeros((n_sta, 3, n_days))
    for day, k in bad.items():
        sc[:k, 1, day] = 10.0
    return sc


def test_decided_partial_and_quiet_days() -> None:
    sc = _scores(40, 10, {2: 39, 5: 16, 7: 3})
    nd = network_days(sc)
    assert np.flatnonzero(nd.decided).tolist() == [2]
    assert np.flatnonzero(nd.partial).tolist() == [5]  # 40 %
    assert nd.n_bad[7] == 3 and not nd.decided[7]
    assert nd.n_bad_component[1, 2] == 39 and nd.n_bad_component[0, 2] == 0
    assert nd.n_voting.tolist() == [40] * 10


def test_no_vote_is_not_a_quiet_vote() -> None:
    """Stations without a two-sided verdict leave the denominator."""
    sc = _scores(40, 4, {1: 15})
    sc[15:, :, 1] = np.nan  # 25 stations without a verdict that day
    nd = network_days(sc, min_stations=10)
    assert nd.n_voting[1] == 15 and nd.share[1] == 1.0 and nd.decided[1]


def test_too_few_voters_decide_nothing() -> None:
    sc = _scores(15, 3, {1: 15})
    nd = network_days(sc)  # n_min 20
    assert not nd.decided.any() and not nd.partial.any()
    assert nd.share[1] == 1.0


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"share": 0.2}, "partial_share"),
        ({"partial_share": 0.0}, "partial_share"),
        ({"min_stations": 0}, "min_stations"),
    ],
)
def test_bad_levels_are_refused(kw: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        network_days(_scores(3, 2, {}), **kw)  # type: ignore[arg-type]


def test_scores_must_be_three_dimensional() -> None:
    with pytest.raises(ValueError, match="stations, components, days"):
        network_days(np.zeros((3, 4)))


def test_a_common_blunder_day_is_found_end_to_end() -> None:
    """25 stations, one day shifted 15 mm east at all of them."""
    stations = [_station(seed) for seed in range(25)]
    for _, y in stations:
        y[1, 120] += 15.0
    sc = np.stack([day_scores(detect_outliers_running(t, y)) for t, y in stations])
    nd = network_days(sc)
    assert np.flatnonzero(nd.decided).tolist() == [120]


def test_midpoint_residuals_keep_only_the_fast_part() -> None:
    """A rate and an annual term leave (almost) nothing; a one-day offset
    stays whole, and the ends are one-sided."""
    from gps_analysis.network import midpoint_residuals

    t = 2020.0 + np.arange(400) * DAY
    y = np.vstack([30.0 * (t - t[0]), 5.0 * np.sin(2 * np.pi * t), np.zeros(t.size)])
    y[2, 200] += 4.0
    r, one = midpoint_residuals(t, y)
    assert np.nanmax(np.abs(r[:2, 20:-20])) < 0.3
    assert abs(r[2, 200] - 4.0) < 1e-9
    assert one[:, -1].all() and one[:, 0].all() and not one[:, 200].any()


def test_common_mode_is_the_median_with_a_voter_floor() -> None:
    from gps_analysis.network import common_mode

    r = np.zeros((20, 3, 4))
    r[:, 1, 2] = 3.0  # every station +3 east on day 2
    r[0, 1, 2] = 50.0  # one station's own blunder does not move it
    r[5:, :, 3] = np.nan  # day 3: only 5 voters
    cm = common_mode(r)
    assert cm.value[1, 2] == 3.0 and cm.value[0, 2] == 0.0
    assert np.isnan(cm.value[:, 3]).all() and cm.n[0, 3] == 5
    with pytest.raises(ValueError, match="stations, components, days"):
        common_mode(np.zeros((3, 4)))


def test_common_mode_marks_one_sided_days_provisional() -> None:
    from gps_analysis.network import common_mode

    r = np.zeros((20, 3, 3))
    one = np.zeros(r.shape, dtype=bool)
    one[:15, :, 2] = True
    assert common_mode(r, one).provisional.tolist() == [False, False, True]
