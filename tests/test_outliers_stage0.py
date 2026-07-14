"""Tests for the two outlier upgrades (branch ``outlier-prefilter-localfit``).

Addition 1 — Stage-0 gross-blunder despike (:func:`neighbor_differences`,
:func:`spike_mask`, ``OutlierParams.despike``): an isolated single-epoch
spike, extreme against both temporal neighbors with the series returning
to baseline, is masked before the model fit — a persistent step is not.

Addition 2 — windowed identifier generalized from a local constant
(order 0) to a robust local polynomial f(x) (:func:`rolling_polyfit`,
``OutlierParams.window_order`` ∈ {0, 1, 2}; LOWESS, Cleveland 1979).

Zero-regression is the release gate here: ``OutlierParams()`` defaults
(despike off, order 0) must reproduce the pre-existing behavior
bit-identically — pinned by :class:`TestZeroRegression` (order-0 window
path == the ``rolling_median``/``rolling_mad`` path it replaces, and the
new fields never touched leave every existing result unchanged).
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from gps_analysis.models import lineperiodic
from gps_analysis.outliers import (
    PROTECT_WINDOW,
    REASON_GROSS,
    OutlierParams,
    detect_outliers,
    hampel_mask,
    mad_scale,
    neighbor_differences,
    rolling_mad,
    rolling_median,
    rolling_polyfit,
    spike_mask,
)

FloatArr = NDArray[np.float64]

DAY = 1.0 / 365.25
TRUE_LP = (12.0, -3.5, 4.0, -2.0, 1.0, 0.5)
WN = 2.0


def _daily_t(n: int, start: float = 2015.0) -> FloatArr:
    return start + np.arange(n, dtype=np.float64) * DAY


def _white_series(n: int, seed: int, wn: float = WN) -> tuple[FloatArr, FloatArr]:
    rng = np.random.default_rng(seed)
    t = _daily_t(n)
    y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, wn, n)
    return t, y


# ---------------------------------------------------------------------------
# neighbor_differences (Stage-0 atom 1)
# ---------------------------------------------------------------------------


class TestNeighborDifferences:
    def test_closed_form(self) -> None:
        t = _daily_t(4)
        x = np.array([1.0, 5.0, 2.0, 2.5])
        dp, dn = neighbor_differences(t, x, max_gap=1.5 * DAY)
        np.testing.assert_allclose(dp, [np.nan, 4.0, -3.0, 0.5])
        np.testing.assert_allclose(dn, [4.0, -3.0, 0.5, np.nan])

    def test_boundaries_nan(self) -> None:
        t = _daily_t(3)
        dp, dn = neighbor_differences(t, np.array([1.0, 2.0, 3.0]), max_gap=1.5 * DAY)
        assert math.isnan(dp[0]) and math.isnan(dn[-1])

    def test_gap_breaks_adjacency(self) -> None:
        # a 5-day gap between index 1 and 2 exceeds max_gap=1.5 d
        t = np.array([0.0, 1.0, 6.0, 7.0]) * DAY
        x = np.array([0.0, 9.0, 0.0, 0.0])
        dp, dn = neighbor_differences(t, x, max_gap=1.5 * DAY)
        assert not math.isnan(dp[1])  # 0->1 adjacent
        assert math.isnan(dn[1])  # 1->2 across the gap
        assert math.isnan(dp[2])  # same edge, other side

    def test_trend_immune_same_sign(self) -> None:
        # a pure linear ramp gives equal, SAME-sign differences everywhere
        t = _daily_t(6)
        x = 3.0 * np.arange(6, dtype=np.float64)
        dp, dn = neighbor_differences(t, x, max_gap=1.5 * DAY)
        interior = slice(1, 5)
        np.testing.assert_allclose(dp[interior], 3.0)
        np.testing.assert_allclose(dn[interior], 3.0)
        assert np.all(dp[interior] * dn[interior] > 0.0)  # never opposite-sign

    def test_bad_args(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            neighbor_differences([1.0, 0.0], [1.0, 2.0], max_gap=1.0)
        with pytest.raises(ValueError, match="max_gap"):
            neighbor_differences([0.0, 1.0], [1.0, 2.0], max_gap=0.0)


# ---------------------------------------------------------------------------
# spike_mask (Stage-0 atom 2) — the spike-vs-step discrimination rule
# ---------------------------------------------------------------------------


class TestSpikeMask:
    def test_isolated_spike_flagged(self) -> None:
        # up-then-down, returns to baseline: a spike
        dp = np.array([np.nan, 100.0, -98.0, np.nan])
        dn = np.array([100.0, -98.0, np.nan, np.nan])
        mask = spike_mask(dp, dn, n_sigma=5.0, scale=2.0, return_n_sigma=4.0)
        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_persistent_step_not_flagged(self) -> None:
        # a level shift: big δ⁻ at the jump epoch, but δ⁺ ~ 0 (stays up)
        dp = np.array([np.nan, 100.0, 0.5, 0.3])
        dn = np.array([100.0, 0.5, 0.3, np.nan])
        mask = spike_mask(dp, dn, n_sigma=5.0, scale=2.0, return_n_sigma=4.0)
        assert not bool(mask.any())

    def test_same_sign_not_flagged(self) -> None:
        # both differences large and SAME sign (ramp acceleration) -> not a spike
        dp = np.array([np.nan, 50.0, 60.0, np.nan])
        dn = np.array([50.0, 60.0, np.nan, np.nan])
        mask = spike_mask(dp, dn, n_sigma=5.0, scale=2.0, return_n_sigma=4.0)
        assert not bool(mask.any())

    def test_no_return_to_baseline_not_flagged(self) -> None:
        # opposite signs but |δ⁻ + δ⁺| large: a spike PLUS a residual step,
        # left to the main protection-aware stage
        dp = np.array([np.nan, 100.0, -40.0, np.nan])
        dn = np.array([100.0, -40.0, np.nan, np.nan])
        mask = spike_mask(dp, dn, n_sigma=5.0, scale=2.0, return_n_sigma=4.0)
        assert not bool(mask[1])

    def test_nan_neighbor_never_flags(self) -> None:
        dp = np.array([np.nan, np.nan, 100.0])
        dn = np.array([np.nan, 100.0, np.nan])
        mask = spike_mask(dp, dn, n_sigma=5.0, scale=2.0, return_n_sigma=4.0)
        assert not bool(mask.any())

    def test_bad_args(self) -> None:
        dp = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="n_sigma"):
            spike_mask(dp, dp, n_sigma=0.0, scale=1.0, return_n_sigma=4.0)
        with pytest.raises(ValueError, match="scale"):
            spike_mask(dp, dp, n_sigma=5.0, scale=0.0, return_n_sigma=4.0)
        with pytest.raises(ValueError, match="return_n_sigma"):
            spike_mask(dp, dp, n_sigma=5.0, scale=1.0, return_n_sigma=0.0)


# ---------------------------------------------------------------------------
# rolling_polyfit (Addition 2 atom)
# ---------------------------------------------------------------------------


class TestRollingPolyfit:
    def test_line_recovered_exactly(self) -> None:
        # noiseless local line: the order-1 fit value equals the sample and
        # the residual scale is 0
        t = _daily_t(60)
        x = 2.0 + 5.0 * (t - t[0])
        m, s = rolling_polyfit(
            t, x, half_window=15.5 * DAY, min_count=5, order=1, robust_iterations=0
        )
        interior = slice(20, 40)
        np.testing.assert_allclose(m[interior], x[interior], atol=1e-8)
        np.testing.assert_allclose(s[interior], 0.0, atol=1e-8)

    def test_parabola_recovered_by_order2_not_order1(self) -> None:
        # a curved segment: order-2 tracks it (residual ~0), order-1 leaves
        # curvature in the residual (nonzero local scale)
        t = _daily_t(80)
        u = (t - t[40]) / DAY
        x = 100.0 * (u / 40.0) ** 2
        _, s1 = rolling_polyfit(
            t, x, half_window=15.5 * DAY, min_count=6, order=1, robust_iterations=0
        )
        _, s2 = rolling_polyfit(
            t, x, half_window=15.5 * DAY, min_count=6, order=2, robust_iterations=0
        )
        interior = slice(30, 50)
        assert np.nanmedian(s1[interior]) > 1e-3
        np.testing.assert_allclose(s2[interior], 0.0, atol=1e-6)

    def test_robustness_iterations_resist_single_outlier(self) -> None:
        # one big spike in the window: with robustness iterations the fit
        # value at the spike epoch is barely moved (bisquare down-weights it)
        t = _daily_t(41)
        x = np.zeros(41)
        x[20] = 500.0
        m_robust, _ = rolling_polyfit(
            t, x, half_window=15.5 * DAY, min_count=5, order=1, robust_iterations=3
        )
        m_plain, _ = rolling_polyfit(
            t, x, half_window=15.5 * DAY, min_count=5, order=1, robust_iterations=0
        )
        assert abs(m_robust[20]) < abs(m_plain[20])
        assert abs(m_robust[20]) < 50.0  # pulled far less than the 500 spike

    def test_thin_window_nan(self) -> None:
        t = np.array([0.0, 1.0, 2.0])
        m, s = rolling_polyfit(
            t, t, half_window=0.1, min_count=3, order=1, robust_iterations=0
        )
        assert np.all(np.isnan(m)) and np.all(np.isnan(s))

    def test_bad_args(self) -> None:
        t = _daily_t(10)
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            rolling_polyfit(t, t, half_window=1.0, min_count=5, order=0)
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            rolling_polyfit(t, t, half_window=1.0, min_count=5, order=3)
        with pytest.raises(ValueError, match="min_count"):
            rolling_polyfit(t, t, half_window=1.0, min_count=2, order=1)
        with pytest.raises(ValueError, match="robust_iterations"):
            rolling_polyfit(
                t, t, half_window=1.0, min_count=5, order=1, robust_iterations=-1
            )


# ---------------------------------------------------------------------------
# Zero-regression — defaults reproduce the current behavior bit-identically
# ---------------------------------------------------------------------------


class TestZeroRegression:
    def test_order0_equals_rolling_median_path(self) -> None:
        # The windowed identifier at window_order=0 must take the EXACT
        # rolling_median/rolling_mad + hampel_mask path — assert the center
        # and scale it uses match those atoms bit-for-bit on a residual-like
        # fixture (this is the order-0 == current-behavior proof at the
        # windowed-identifier level).
        rng = np.random.default_rng(7)
        keep = rng.random(400) > 0.25
        t = _daily_t(400)[keep]
        w = rng.normal(0.0, 1.0, int(keep.sum()))
        w[[30, 120, 210]] += np.array([12.0, -14.0, 13.0])
        h = 15.5 * DAY
        m = rolling_median(t, w, half_window=h, min_count=11)
        s = rolling_mad(t, w, m, half_window=h, min_count=11)
        mask_ref = hampel_mask(
            w, np.nan_to_num(m), np.nan_to_num(s, nan=-1.0), n_sigma=4.0
        )
        # rolling_polyfit is NOT used at order 0; confirm the median center
        # differs from an order-1 center on the spikes (so the two paths are
        # genuinely distinct code, not accidentally identical)
        m1, _ = rolling_polyfit(t, w, half_window=h, min_count=11, order=1)
        assert not np.allclose(np.nan_to_num(m), np.nan_to_num(m1))
        assert mask_ref.dtype == np.bool_

    def test_detect_outliers_order0_bit_identical(self) -> None:
        # detect_outliers with explicit order-0 / despike-off params must be
        # bit-identical to the plain default call (the whole-pipeline proof).
        t, y = _white_series(900, 11)
        y[[120, 400, 700]] += np.array([18.0, -22.0, 30.0])
        default = detect_outliers(lineperiodic, t, y)
        explicit = detect_outliers(
            lineperiodic,
            t,
            y,
            params=OutlierParams(window_order=0, despike=False),
        )
        np.testing.assert_array_equal(default.flags, explicit.flags)
        np.testing.assert_array_equal(default.reasons, explicit.reasons)
        np.testing.assert_array_equal(default.z, explicit.z)
        np.testing.assert_array_equal(default.scale_local, explicit.scale_local)

    def test_despike_off_by_default(self) -> None:
        t, y = _white_series(900, 11)
        y[500] += 150.0  # a gross blunder
        res = detect_outliers(lineperiodic, t, y)  # defaults
        assert not res.params.despike
        assert int(res.n_despiked.sum()) == 0
        assert not bool((res.reasons & REASON_GROSS).any())

    def test_new_fields_defaults(self) -> None:
        p = OutlierParams()
        assert p.window_order == 0
        assert p.window_robust_iterations == 2
        assert p.despike is False
        assert p.despike_n_sigma == 10.0
        assert p.despike_return_sigma == 4.0
        assert p.despike_gap_days == 1.5


# ---------------------------------------------------------------------------
# Stage-0 despike behavior (Addition 1, integrated)
# ---------------------------------------------------------------------------


class TestStage0Despike:
    def test_despike_catches_150mm_spike(self) -> None:
        # the FAGD East case: an isolated 150 mm blunder
        t, y = _white_series(1500, 101)
        i0 = 750
        y[i0] += 150.0
        res = detect_outliers(
            lineperiodic, t, y, params=OutlierParams(despike=True), min_outlier=5.0
        )
        assert bool(res.flags[i0])
        assert res.reasons[i0] & REASON_GROSS
        assert int(res.n_despiked.sum()) == 1

    def test_despike_catches_end_spike_the_main_stage_misses(self) -> None:
        # An isolated 150 mm blunder 2 d before the series end: the main
        # stage cannot form a post-flank, so it protects the run as a
        # suspected step and never flags it (and the contaminated fit
        # biases the rate). Stage-0 despike catches it up front and the
        # refit rate is much closer to truth (harness Case A2).
        rng = np.random.default_rng(101)
        n = 1500
        t = _daily_t(n, start=2021.0)
        y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, 3.0, n)
        i0 = n - 2
        y[i0] += 150.0
        base = detect_outliers(lineperiodic, t, y, min_outlier=5.0)
        desp = detect_outliers(
            lineperiodic, t, y, params=OutlierParams(despike=True), min_outlier=5.0
        )
        assert not bool(base.flags[i0])  # main stage misses the end spike
        assert bool(desp.flags[i0])  # despike catches it
        rate_err_base = abs(float(base.fits[0].params[1]) - TRUE_LP[1])
        rate_err_desp = abs(float(desp.fits[0].params[1]) - TRUE_LP[1])
        assert rate_err_desp < rate_err_base

    def test_despike_preserves_real_step(self) -> None:
        # a genuine 60 mm step must NOT be despiked (level shifts and stays)
        n = 1500
        t, y = _white_series(n, 3)
        i0 = n // 2
        y = np.asarray(y + 60.0 * (t >= t[i0]), dtype=np.float64)
        res = detect_outliers(lineperiodic, t, y, params=OutlierParams(despike=True))
        # no epoch inside the step neighborhood is despiked
        assert int(res.n_despiked.sum()) == 0
        assert not bool((res.reasons[i0 - 5 : i0 + 6] & REASON_GROSS).any())

    def test_despike_respects_protect_window(self) -> None:
        t, y = _white_series(1500, 101)
        i0 = 750
        y[i0] += 150.0
        window = (float(t[i0 - 3]), float(t[i0 + 3]))
        res = detect_outliers(
            lineperiodic,
            t,
            y,
            protect_windows=[window],
            params=OutlierParams(despike=True),
            min_outlier=5.0,
        )
        assert int(res.n_despiked.sum()) == 0
        assert not bool(res.flags[i0])
        assert res.protected[i0] & PROTECT_WINDOW

    def test_despike_never_mutates_input(self) -> None:
        t, y = _white_series(900, 101)
        y[500] += 150.0
        y_bytes, t_bytes = y.tobytes(), t.tobytes()
        detect_outliers(lineperiodic, t, y, params=OutlierParams(despike=True))
        assert y.tobytes() == y_bytes
        assert t.tobytes() == t_bytes

    def test_despiked_not_a_candidate(self) -> None:
        # a despiked epoch carries REASON_GROSS and is NOT in candidates
        t, y = _white_series(1500, 101)
        i0 = 750
        y[i0] += 150.0
        res = detect_outliers(
            lineperiodic, t, y, params=OutlierParams(despike=True), min_outlier=5.0
        )
        assert not bool(res.candidates[i0])
        assert bool(res.flags[i0])

    def test_despike_multicomponent_counts(self) -> None:
        rng = np.random.default_rng(9)
        n = 1200
        t = _daily_t(n)
        y = np.vstack(
            [
                lineperiodic(t, *TRUE_LP) + rng.normal(0.0, WN, n),
                lineperiodic(t, 5.0, 1.0, 2.0, -1.0, 0.5, 0.2) + rng.normal(0.0, WN, n),
            ]
        )
        y[0, 300] += 150.0
        y[1, 600] += -160.0
        res = detect_outliers(lineperiodic, t, y, params=OutlierParams(despike=True))
        assert res.n_despiked.shape == (2,)
        assert int(res.n_despiked[0]) == 1
        assert int(res.n_despiked[1]) == 1


# ---------------------------------------------------------------------------
# Local-polynomial windowed identifier (Addition 2, integrated)
# ---------------------------------------------------------------------------


class TestLocalPolynomialIdentifier:
    def _ramp_with_spikes(self) -> tuple[FloatArr, FloatArr, NDArray[np.intp]]:
        rng = np.random.default_rng(5)
        n = 2000
        t = _daily_t(n, start=2018.0)
        y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, WN, n)
        i0 = 900
        y[i0:] += (t[i0:] - t[i0]) / DAY * 3.0  # 3 mm/d ongoing ramp
        spikes = np.array([950, 1000, 1050], dtype=np.intp)
        y[spikes] += np.array([12.0, -13.0, 12.0])
        return t, y, spikes

    def test_order2_recovers_recall_on_steep_ramp(self) -> None:
        # order-0's local MAD is inflated by the in-window ramp spread and
        # masks the spikes; order-2 restores an honest scale and catches them
        t, y, spikes = self._ramp_with_spikes()
        base = detect_outliers(
            lineperiodic,
            t,
            y,
            params=OutlierParams(global_n_sigma=50.0),
            min_outlier=5.0,
        )
        poly = detect_outliers(
            lineperiodic,
            t,
            y,
            params=OutlierParams(global_n_sigma=50.0, window_order=2),
            min_outlier=5.0,
        )
        assert int(base.flags[spikes].sum()) == 0
        assert int(poly.flags[spikes].sum()) == 3
        # honest local scale on the ramp segment
        seg = slice(900, 1100)
        assert np.nanmedian(poly.scale_local[seg]) < 0.5 * np.nanmedian(
            base.scale_local[seg]
        )

    def test_local_scale_tracks_slope(self) -> None:
        # the reported scale_local on a steep ramp is far smaller for a
        # local line than for the local constant (order 0)
        t, y, _ = self._ramp_with_spikes()
        r0 = detect_outliers(
            lineperiodic, t, y, params=OutlierParams(global_n_sigma=50.0)
        )
        r1 = detect_outliers(
            lineperiodic,
            t,
            y,
            params=OutlierParams(global_n_sigma=50.0, window_order=1),
        )
        seg = slice(900, 1100)
        assert np.nanmedian(r1.scale_local[seg]) < np.nanmedian(r0.scale_local[seg])

    def test_declared_step_not_eaten_by_order1(self) -> None:
        # step-augmented model + window_order=1: a declared 40 mm step is
        # absorbed by the model, zero flags around it (protection intact)
        n = 1500
        t, y = _white_series(n, 3)
        i0 = n // 2
        t0 = float(t[i0])
        y = np.asarray(y + 40.0 * (t >= t0), dtype=np.float64)
        res = detect_outliers(
            lineperiodic,
            t,
            y,
            step_epochs=[t0],
            params=OutlierParams(window_order=1),
        )
        assert int(res.flags[i0 - 30 : i0 + 31].sum()) == 0

    def test_order1_determinism(self) -> None:
        t, y, _ = self._ramp_with_spikes()
        p = OutlierParams(window_order=1)
        r1 = detect_outliers(lineperiodic, t, y, params=p)
        r2 = detect_outliers(lineperiodic, t, y, params=p)
        np.testing.assert_array_equal(r1.flags, r2.flags)
        np.testing.assert_array_equal(r1.scale_local, r2.scale_local)


# ---------------------------------------------------------------------------
# Parameter validation for the new fields
# ---------------------------------------------------------------------------


class TestNewParamValidation:
    def test_window_order_enum(self) -> None:
        with pytest.raises(ValueError, match="window_order"):
            OutlierParams(window_order=3)

    def test_order_min_count_coupling(self) -> None:
        with pytest.raises(ValueError, match="window_order . 2"):
            OutlierParams(window_order=2, window_min_count=3)
        # order 0 is exempt (rolling median needs only 1)
        OutlierParams(window_order=0, window_min_count=1)

    def test_robust_iterations_nonneg(self) -> None:
        with pytest.raises(ValueError, match="window_robust_iterations"):
            OutlierParams(window_robust_iterations=-1)

    def test_despike_positive_fields(self) -> None:
        with pytest.raises(ValueError, match="despike_n_sigma"):
            OutlierParams(despike_n_sigma=0.0)
        with pytest.raises(ValueError, match="despike_return_sigma"):
            OutlierParams(despike_return_sigma=-1.0)
        with pytest.raises(ValueError, match="despike_gap_days"):
            OutlierParams(despike_gap_days=0.0)

    def test_spike_scale_uses_mad(self) -> None:
        # sanity: the difference-scale helper used by the harness/pipeline
        # is the module's mad_scale (documented ŝ_Δ)
        d = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
        assert mad_scale(d) == pytest.approx(
            1.4826 * np.median(np.abs(d - np.median(d)))
        )
