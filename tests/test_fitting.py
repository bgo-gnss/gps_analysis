"""Tests for gps_analysis.fitting (MATH_STANDARDS §4).

Covers exact parameter recovery on noise-free synthetics (analytic check),
noisy recovery within stated tolerances, weighted-fit covariance scaling,
purity (no input mutation), detrend/remove-trend invertibility, and the
robust outlier-rejection loop (injected gross outliers are flagged, clean
data survives, both ``soft_l1`` and ``huber`` losses).

Tolerances: noise-free fits recover parameters at ``rtol = 1e-6``
(optimizer convergence, not float eps); noisy fits are checked against
their own 3-sigma formal errors (seeded rng, so deterministic).
"""

import numpy as np
import pytest

from gps_analysis.fitting import (
    OutlierRejection,
    detrend_fit,
    fit_components,
    reject_outliers,
    remove_trend,
)
from gps_analysis.models import TrajectoryParams, linear, lineperiodic

TRUE_LP = np.array([12.0, -3.5, 4.0, -2.0, 1.0, 0.5])
"""offset, rate, cos_annual, sin_annual, cos_semiannual, sin_semiannual."""


def _synthetic_lineperiodic(
    n: int = 1500, noise: float = 0.0, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(n) / 365.25
    y = lineperiodic(t, *TRUE_LP)
    if noise > 0.0:
        y = y + rng.normal(0.0, noise, size=n)
    return t, y


class TestFitComponents:
    def test_exact_recovery_noise_free(self) -> None:
        t, y = _synthetic_lineperiodic()
        (fit,) = fit_components(lineperiodic, t, y, p0=np.zeros(6))
        np.testing.assert_allclose(fit.params, TRUE_LP, rtol=1e-6)

    def test_noisy_recovery_within_formal_errors(self) -> None:
        t, y = _synthetic_lineperiodic(noise=2.0, seed=7)
        (fit,) = fit_components(lineperiodic, t, y, p0=np.zeros(6))
        # rate recovered within 3 x its own formal 1-sigma error
        assert abs(fit.params[1] - TRUE_LP[1]) < 3.0 * fit.uncertainties[1]

    def test_default_p0_is_ones(self) -> None:
        t = np.linspace(0.0, 10.0, 100)
        y = linear(t, 2.0, 0.5)
        (fit,) = fit_components(linear, t, y)
        np.testing.assert_allclose(fit.params, [2.0, 0.5], rtol=1e-8)

    def test_multi_component_with_names(self) -> None:
        t, _ = _synthetic_lineperiodic(n=400)
        rates = [-3.5, 1.2, 8.0]
        y = np.vstack([linear(t - 2015.0, 10.0 * i, r) for i, r in enumerate(rates)])
        fits = fit_components(linear, t - 2015.0, y, names=("north", "east", "up"))
        assert [f.component for f in fits] == ["north", "east", "up"]
        for fit, rate in zip(fits, rates, strict=True):
            assert fit.params[1] == pytest.approx(rate, rel=1e-8)

    def test_absolute_sigma_covariance_scaling(self) -> None:
        # With absolute_sigma=True, doubling sigma quadruples C_p.
        t, y = _synthetic_lineperiodic(n=300, noise=1.0, seed=3)
        (fit1,) = fit_components(
            lineperiodic,
            t,
            y,
            sigma=np.ones_like(y),
            p0=np.zeros(6),
            absolute_sigma=True,
        )
        (fit2,) = fit_components(
            lineperiodic,
            t,
            y,
            sigma=2.0 * np.ones_like(y),
            p0=np.zeros(6),
            absolute_sigma=True,
        )
        # rtol 1e-3: the uniform sigma rescaling leaves the minimizer
        # unchanged analytically, but the two curve_fit runs stop at
        # slightly different iterates, so the Jacobians differ at ~5e-4.
        np.testing.assert_allclose(fit2.covariance, 4.0 * fit1.covariance, rtol=1e-3)

    def test_sigma_downweights_bad_segment(self) -> None:
        t = np.linspace(0.0, 10.0, 200)
        y = linear(t, 0.0, 1.0)
        y2 = y.copy()
        y2[:50] += 40.0  # corrupted segment ...
        sigma = np.ones_like(y2)
        sigma[:50] = 1e4  # ... assigned huge uncertainty
        (fit,) = fit_components(linear, t, y2, sigma=sigma)
        assert fit.params[1] == pytest.approx(1.0, abs=1e-3)

    def test_does_not_mutate_inputs(self) -> None:
        t, y = _synthetic_lineperiodic(n=200, noise=1.0)
        t0, y0 = t.copy(), y.copy()
        fit_components(lineperiodic, t, y, p0=np.zeros(6))
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)

    def test_shape_validation(self) -> None:
        t = np.linspace(0.0, 1.0, 10)
        y = np.zeros(10)
        with pytest.raises(ValueError, match="t must be 1-D"):
            fit_components(linear, t[:5], y)
        with pytest.raises(ValueError, match="sigma"):
            fit_components(linear, t, y, sigma=np.ones(5))
        with pytest.raises(ValueError, match="p0"):
            fit_components(linear, t, np.zeros((2, 10)), p0=np.zeros(2))
        with pytest.raises(ValueError, match="names"):
            fit_components(linear, t, y, names=("north", "east"))
        with pytest.raises(ValueError, match="y must be 1-D or 2-D"):
            fit_components(linear, t, np.zeros((2, 2, 10)))


class TestRemoveTrendAndDetrendFit:
    def test_remove_trend_is_exact_subtraction(self) -> None:
        t, y = _synthetic_lineperiodic(n=300, noise=1.5, seed=11)
        fit = TrajectoryParams(params=TRUE_LP, covariance=np.eye(6))
        detrended = remove_trend(lineperiodic, t, y, fit)
        np.testing.assert_allclose(detrended, y - lineperiodic(t, *TRUE_LP))

    def test_detrended_series_has_no_remaining_rate(self) -> None:
        t, y = _synthetic_lineperiodic(noise=2.0, seed=13)
        detrended, (fit,) = detrend_fit(lineperiodic, t, y, p0=np.zeros(6))
        (refit,) = fit_components(lineperiodic, t, detrended, p0=np.zeros(6))
        assert abs(refit.params[1]) < 3.0 * fit.uncertainties[1]

    def test_detrend_fit_is_invertible(self) -> None:
        t, y = _synthetic_lineperiodic(n=400, noise=1.0, seed=5)
        detrended, (fit,) = detrend_fit(lineperiodic, t, y, p0=np.zeros(6))
        np.testing.assert_allclose(
            detrended + lineperiodic(t, *fit.params), y, rtol=0, atol=1e-9
        )

    def test_multi_component_and_purity(self) -> None:
        t, y1 = _synthetic_lineperiodic(n=300, noise=1.0, seed=2)
        y = np.vstack([y1, 2.0 * y1])
        y0 = y.copy()
        detrended, fits = detrend_fit(lineperiodic, t, y, p0=np.zeros((2, 6)))
        assert detrended.shape == y.shape
        assert len(fits) == 2
        np.testing.assert_array_equal(y, y0)  # input untouched

    def test_remove_trend_fit_count_mismatch(self) -> None:
        t, y = _synthetic_lineperiodic(n=50)
        fit = TrajectoryParams(params=TRUE_LP, covariance=np.eye(6))
        with pytest.raises(ValueError, match="parameter sets"):
            remove_trend(lineperiodic, t, np.vstack([y, y]), [fit])


class TestRejectOutliers:
    @staticmethod
    def _line_with_outliers(
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 5.0, 250)
        y = linear(t, 1.0, 2.0) + rng.normal(0.0, 1.0, size=t.size)
        bad = np.zeros(t.size, dtype=bool)
        bad[[10, 60, 61, 120, 200, 240]] = True
        y[bad] += 50.0
        return t, y, bad

    def test_flags_injected_gross_outliers(self) -> None:
        t, y, bad = self._line_with_outliers()
        result = reject_outliers(linear, t, y, n_sigma=4.0)
        assert isinstance(result, OutlierRejection)
        assert not result.inliers[bad].any()  # every injected point flagged
        # false-rejection rate on the clean points stays small
        clean = ~bad
        assert result.inliers[clean].mean() > 0.98
        # and the final WLS fit is unbiased by the outliers
        assert result.fits[0].params[1] == pytest.approx(2.0, abs=0.05)

    def test_huber_loss_also_works(self) -> None:
        t, y, bad = self._line_with_outliers(seed=9)
        result = reject_outliers(linear, t, y, loss="huber", n_sigma=4.0)
        assert not result.inliers[bad].any()

    def test_clean_noise_free_data_keeps_everything(self) -> None:
        # perfect fit => robust scale 0 => rejection stops, all inliers
        t = np.linspace(0.0, 5.0, 100)
        y = linear(t, 1.0, 2.0)
        result = reject_outliers(linear, t, y)
        assert result.inliers.all()
        assert result.n_iterations >= 1

    def test_with_sigma_and_two_components(self) -> None:
        t, y1, bad = self._line_with_outliers(seed=21)
        y = np.vstack([y1, y1])
        sigma = np.ones_like(y)
        result = reject_outliers(
            linear, t, y, sigma=sigma, n_sigma=4.0, names=("north", "east")
        )
        assert result.inliers.shape == y.shape
        assert not result.inliers[0][bad].any()
        assert [f.component for f in result.fits] == ["north", "east"]

    def test_max_iterations_validated_and_respected(self) -> None:
        t, y, _ = self._line_with_outliers()
        with pytest.raises(ValueError, match="max_iterations"):
            reject_outliers(linear, t, y, max_iterations=0)
        result = reject_outliers(linear, t, y, max_iterations=1)
        assert result.n_iterations == 1

    def test_does_not_mutate_inputs(self) -> None:
        t, y, _ = self._line_with_outliers()
        t0, y0 = t.copy(), y.copy()
        reject_outliers(linear, t, y)
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)
