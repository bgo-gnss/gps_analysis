"""Tests for gps_analysis.baseline (MATH_STANDARDS §4).

Covers legacy parity (``slice_window`` mask vs the ``dPeriod`` delete
logic reproduced inline; ``estimate_offset`` vs the
``geo_dataread.gps_read.estimate_offset`` 1/sigma weighted average),
analytic checks (step offset between two known lines), purity, and the
end-to-end segment-fit -> midpoint-difference pipeline of the legacy
``find_line_offsets``.
"""

import numpy as np
import pytest

from gps_analysis.baseline import (
    estimate_offset,
    estimate_step_offset,
    remove_offset,
    slice_window,
)
from gps_analysis.fitting import fit_components
from gps_analysis.models import TrajectoryParams, linear

RNG = np.random.default_rng(20260711)


def _legacy_dperiod_keep(
    yearf: np.ndarray, startyear: float | None, endyear: float | None
) -> np.ndarray:
    """The keep-set of legacy dPeriod (geo_dataread.gps_read), inline."""
    kept = yearf.copy()
    if startyear:
        kept = np.delete(kept, np.where(kept <= startyear - 0.001))
    if endyear:
        kept = np.delete(kept, np.where(kept >= endyear + 0.001))
    return kept


class TestSliceWindow:
    def test_legacy_dperiod_parity(self) -> None:
        t = np.sort(RNG.uniform(2018.0, 2026.0, size=400))
        for start, end in [
            (2019.5, 2024.25),
            (None, 2022.0),
            (2020.0, None),
            (2025.9, 2026.5),
        ]:
            mask = slice_window(t, start, end)
            np.testing.assert_array_equal(t[mask], _legacy_dperiod_keep(t, start, end))

    def test_boundary_tolerance(self) -> None:
        t = np.array([1.998, 1.999, 2.0, 3.0, 3.001, 3.002])
        mask = slice_window(t, 2.0, 3.0)
        # keep t > start - 0.001 and t < end + 0.001 (strict, legacy)
        np.testing.assert_array_equal(mask, [False, False, True, True, False, False])

    def test_open_bounds_keep_everything(self) -> None:
        t = np.linspace(2015.0, 2020.0, 50)
        assert slice_window(t).all()

    def test_nan_epochs_are_excluded(self) -> None:
        t = np.array([2020.0, np.nan, 2021.0])
        np.testing.assert_array_equal(
            slice_window(t, 2019.0, 2022.0), [True, False, True]
        )


class TestEstimateOffset:
    def test_unweighted_is_plain_mean_over_window(self) -> None:
        t = np.linspace(2020.0, 2021.0, 366)
        y = RNG.normal(5.0, 1.0, size=t.size)
        window = slice_window(t, 2020.0, 2020.1)
        level = estimate_offset(t, y, start=2020.0, end=2020.1)
        assert level.shape == ()
        assert float(level) == pytest.approx(float(np.mean(y[window])))

    def test_legacy_inverse_sigma_parity(self) -> None:
        # geo_dataread.gps_read.estimate_offset (no refdate):
        #   np.average(data[0:3, 0:P], 1, weights=1 / Ddata[0:3, 0:P])
        n, period = 200, 5
        t = 2020.0 + np.arange(n) / 365.25
        data = RNG.normal(0.0, 10.0, size=(3, n))
        ddata = RNG.uniform(0.5, 3.0, size=(3, n))
        legacy = np.average(data[0:3, 0:period], 1, weights=1 / ddata[0:3, 0:period])
        end = t[period - 1]  # time window covering the first `period` samples
        level = estimate_offset(t, data, ddata, start=t[0], end=end)
        assert level.shape == (3,)
        np.testing.assert_allclose(level, legacy, rtol=1e-12)

    def test_inverse_variance_weighting(self) -> None:
        t = np.array([0.0, 1.0])
        y = np.array([0.0, 3.0])
        sigma = np.array([1.0, 2.0])
        level = estimate_offset(t, y, sigma, weighting="inverse_variance")
        # w = [1, 1/4] -> (0*1 + 3/4) / (5/4) = 0.6
        assert float(level) == pytest.approx(0.6)

    def test_empty_window_raises(self) -> None:
        t = np.linspace(2020.0, 2021.0, 100)
        with pytest.raises(ValueError, match="no samples"):
            estimate_offset(t, np.zeros_like(t), start=2025.0, end=2026.0)

    def test_bad_weighting_and_shapes_raise(self) -> None:
        t = np.linspace(0.0, 1.0, 10)
        y = np.zeros(10)
        with pytest.raises(ValueError, match="weighting"):
            estimate_offset(t, y, np.ones(10), weighting="magic")
        with pytest.raises(ValueError, match="sigma"):
            estimate_offset(t, y, np.ones(9))
        with pytest.raises(ValueError, match="t must be 1-D"):
            estimate_offset(t[:5], y)


class TestRemoveOffset:
    def test_per_component_shift(self) -> None:
        y = np.arange(6.0).reshape(2, 3)
        shifted = remove_offset(y, np.array([1.0, 10.0]))
        np.testing.assert_array_equal(shifted, [[-1.0, 0.0, 1.0], [-7.0, -6.0, -5.0]])

    def test_scalar_offset_and_1d(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(remove_offset(y, 2.0), [-1.0, 0.0, 1.0])

    def test_roundtrip_with_estimate_offset(self) -> None:
        t = np.linspace(2020.0, 2022.0, 300)
        y = RNG.normal(0.0, 1.0, size=(3, t.size)) + np.array([[5.0], [-3.0], [40.0]])
        level = estimate_offset(t, y, start=2020.0, end=2020.2)
        shifted = remove_offset(y, level)
        relevel = estimate_offset(t, shifted, start=2020.0, end=2020.2)
        np.testing.assert_allclose(relevel, np.zeros(3), atol=1e-12)

    def test_wrong_component_count_raises(self) -> None:
        with pytest.raises(ValueError, match="components"):
            remove_offset(np.zeros((3, 5)), np.zeros(2))

    def test_does_not_mutate_input(self) -> None:
        y = np.ones((2, 4))
        y0 = y.copy()
        remove_offset(y, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(y, y0)


class TestEstimateStepOffset:
    def test_two_parallel_lines_analytic(self) -> None:
        # x_before = 0 + 1*t, x_after = 5 + 1*t: step = +5 at any epoch
        step = estimate_step_offset(linear, [0.0, 1.0], [5.0, 1.0], epoch=3.7)
        assert step == pytest.approx(5.0)

    def test_sign_convention_after_minus_before(self) -> None:
        step = estimate_step_offset(linear, [5.0, 0.0], [0.0, 0.0], epoch=0.0)
        assert step == pytest.approx(-5.0)

    def test_differing_rates_at_given_epoch(self) -> None:
        # delta(t) = (1 + 2t) - (0 + 1t) = 1 + t
        step = estimate_step_offset(linear, [0.0, 1.0], [1.0, 2.0], epoch=2.0)
        assert step == pytest.approx(3.0)

    def test_accepts_trajectory_params(self) -> None:
        before = TrajectoryParams(params=np.array([0.0, 1.0]), covariance=np.eye(2))
        after = TrajectoryParams(params=np.array([5.0, 1.0]), covariance=np.eye(2))
        assert estimate_step_offset(linear, before, after, epoch=1.0) == pytest.approx(
            5.0
        )

    def test_pipeline_recovers_synthetic_step(self) -> None:
        # legacy find_line_offsets pattern: fit the segment before and the
        # segment after an event, difference the fits at the gap midpoint.
        rng = np.random.default_rng(3)
        t = 2020.0 + np.arange(730) / 365.25
        true_step = -12.5
        y = linear(t - 2020.0, 2.0, 4.0) + rng.normal(0.0, 0.5, size=t.size)
        y[t >= 2021.0] += true_step
        before_mask = slice_window(t, 2020.0, 2020.995)
        after_mask = slice_window(t, 2021.005, 2022.0)
        (fit_before,) = fit_components(linear, t[before_mask] - 2020.0, y[before_mask])
        (fit_after,) = fit_components(linear, t[after_mask] - 2020.0, y[after_mask])
        step = estimate_step_offset(linear, fit_before, fit_after, epoch=1.0)
        assert step == pytest.approx(true_step, abs=0.3)
