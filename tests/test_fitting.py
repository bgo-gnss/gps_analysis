"""Tests for gps_analysis.fitting (MATH_STANDARDS §4).

Covers exact parameter recovery on noise-free synthetics (analytic check),
noisy recovery within stated tolerances, weighted-fit covariance scaling,
purity (no input mutation), detrend/remove-trend invertibility, and the
robust outlier-rejection loop (injected gross outliers are flagged, clean
data survives, both ``soft_l1`` and ``huber`` losses).

Tolerances: noise-free fits recover parameters at ``rtol = 1e-6``
(optimizer convergence, not float eps); noisy fits are checked against
their own 3-sigma formal errors (seeded rng, so deterministic).

``TestClosedFormLinearPath`` pins the closed-form WLS fast path (linear /
periodic / lineperiodic) against a direct ``curve_fit`` LM reference:
params + formal sigmas at rtol 1e-7 on centered epochs, |dp| <= 1e-4
sigma_p and sigma rtol 1e-5 at absolute yearf (where LM's own
conditioning error dominates; the closed form is chi-square-verified as
the better optimum), plus the intercept/rate centering round-trip and the
curve_fit inf-covariance/OptimizeWarning degeneracy conventions.
"""

import numpy as np
import pytest
from scipy import optimize

from gps_analysis.fitting import (
    OutlierRejection,
    detrend_fit,
    fit_components,
    reject_outliers,
    remove_trend,
)
from gps_analysis.models import (
    TrajectoryParams,
    exp_linear,
    linear,
    lineperiodic,
    periodic,
)

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
        # rtol tightened 1e-3 -> 1e-9 with the closed-form WLS path:
        # lineperiodic no longer goes through curve_fit, so there is no
        # iterate-to-iterate Jacobian wobble - the uniform sigma rescaling
        # scales the covariance by exactly 4 up to SVD rounding. (Not a
        # pinned-LM-value change; the asserted identity is analytic.)
        np.testing.assert_allclose(fit2.covariance, 4.0 * fit1.covariance, rtol=1e-9)

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


def _lineperiodic_via_curve_fit(
    t: np.ndarray,
    offset: float,
    rate: float,
    cos_annual: float,
    sin_annual: float,
    cos_semiannual: float,
    sin_semiannual: float,
) -> np.ndarray:
    """lineperiodic wrapper NOT in the linear-design registry.

    Same math, different callable identity — forces ``fit_components``
    onto the iterative ``curve_fit`` path, giving an in-process reference
    for the closed-form/LM equivalence tests.
    """
    return lineperiodic(
        t, offset, rate, cos_annual, sin_annual, cos_semiannual, sin_semiannual
    )


class TestClosedFormLinearPath:
    """LM (curve_fit) vs closed-form WLS equivalence (perf finding #2).

    The registry models (linear / periodic / lineperiodic) now solve via
    SVD weighted lstsq instead of Levenberg-Marquardt. These tests pin
    the numerical equivalence of parameters AND formal sigmas against a
    direct ``scipy.optimize.curve_fit`` reference (the previous
    implementation), plus the conditioning round-trip and the
    curve_fit-compatible degeneracy conventions.
    """

    @staticmethod
    def _noisy_series(
        n: int = 730, seed: int = 7
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        t = 2015.0 + np.arange(n) / 365.25  # absolute yearf epochs
        y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, 2.0, size=n)
        sigma = np.abs(rng.normal(2.0, 0.3, size=n)) + 0.5
        return t, y, sigma

    @pytest.mark.parametrize("absolute_sigma", [True, False])
    def test_matches_curve_fit_centered_epochs(self, absolute_sigma: bool) -> None:
        # Well-conditioned (centered) epochs: LM and the closed form agree
        # to ~1e-8 (audit measured 4.8e-8); pinned at rtol 1e-7 for both
        # the parameters and the formal 1-sigma errors. The series is
        # generated in the centered frame so all parameters keep their
        # nominal O(1-10) magnitudes (LM's relative ftol/xtol stopping
        # rule limits ITS accuracy on a huge-intercept parameterization).
        rng = np.random.default_rng(7)
        t, _, sigma = self._noisy_series()
        tc = t - t.mean()
        y = lineperiodic(tc, *TRUE_LP) + rng.normal(0.0, 2.0, size=t.size)
        (fit,) = fit_components(
            lineperiodic,
            tc,
            y,
            sigma=sigma,
            p0=np.zeros(6),
            absolute_sigma=absolute_sigma,
        )
        popt, pcov = optimize.curve_fit(
            lineperiodic,
            tc,
            y,
            p0=np.zeros(6),
            sigma=sigma,
            absolute_sigma=absolute_sigma,
        )
        np.testing.assert_allclose(fit.params, popt, rtol=1e-7)
        np.testing.assert_allclose(fit.uncertainties, np.sqrt(np.diag(pcov)), rtol=1e-7)

    def test_matches_curve_fit_absolute_yearf_epochs(self) -> None:
        # Absolute yearf (t ~ 2e3): the [1, t] columns are near-collinear
        # (condition ~1e7) and LM's own answer carries conditioning error,
        # so equivalence is asserted in the statistically meaningful
        # metric |dp| <= 1e-4 sigma_p (measured 3.3e-5) with sigmas at
        # rtol 1e-5 (measured 2.1e-6) - and the closed form is proven at
        # least as optimal via chi-square. This is not a loosened gate
        # hiding a discrepancy: the discrepancy is LM's, quantified.
        t, y, sigma = self._noisy_series()
        (fit,) = fit_components(
            lineperiodic, t, y, sigma=sigma, p0=np.zeros(6), absolute_sigma=True
        )
        popt, pcov = optimize.curve_fit(
            lineperiodic, t, y, p0=np.zeros(6), sigma=sigma, absolute_sigma=True
        )
        sig_lm = np.sqrt(np.diag(pcov))
        np.testing.assert_array_less(np.abs(fit.params - popt), 1e-4 * sig_lm)
        np.testing.assert_allclose(fit.uncertainties, sig_lm, rtol=1e-5)

        def chisq(p: np.ndarray) -> float:
            r = (y - lineperiodic(t, *p)) / sigma
            return float(r @ r)

        assert chisq(fit.params) <= chisq(popt) + 1e-9  # exact WLS optimum

    def test_linear_and_periodic_match_curve_fit(self) -> None:
        rng = np.random.default_rng(3)
        t = 2015.0 + np.arange(500) / 365.25
        sigma = np.full(t.size, 1.5)
        y_lin = linear(t, 5.0, -2.0) + rng.normal(0.0, 1.0, size=t.size)
        y_per = periodic(t, 4.0, -2.0, 1.0, 0.5) + rng.normal(0.0, 1.0, size=t.size)

        (fit_l,) = fit_components(linear, t, y_lin, sigma=sigma)
        popt_l, pcov_l = optimize.curve_fit(
            linear, t, y_lin, p0=np.ones(2), sigma=sigma
        )
        sig_l = np.sqrt(np.diag(pcov_l))
        np.testing.assert_array_less(np.abs(fit_l.params - popt_l), 1e-4 * sig_l)
        np.testing.assert_allclose(fit_l.uncertainties, sig_l, rtol=1e-5)

        # periodic: all columns bounded, no conditioning issue -> 1e-7
        (fit_p,) = fit_components(periodic, t, y_per, sigma=sigma, absolute_sigma=True)
        popt_p, pcov_p = optimize.curve_fit(
            periodic, t, y_per, p0=np.ones(4), sigma=sigma, absolute_sigma=True
        )
        np.testing.assert_allclose(fit_p.params, popt_p, rtol=1e-7)
        np.testing.assert_allclose(
            fit_p.uncertainties, np.sqrt(np.diag(pcov_p)), rtol=1e-7
        )

    def test_registry_and_curve_fit_paths_agree_via_wrapper(self) -> None:
        # Same model math through both dispatch branches of the CURRENT
        # code (wrapper callable is not in the registry): end-to-end
        # equivalence without depending on scipy internals.
        t, y, sigma = self._noisy_series(seed=21)
        (fast,) = fit_components(
            lineperiodic, t, y, sigma=sigma, p0=np.zeros(6), absolute_sigma=True
        )
        (slow,) = fit_components(
            _lineperiodic_via_curve_fit,
            t,
            y,
            sigma=sigma,
            p0=np.zeros(6),
            absolute_sigma=True,
        )
        np.testing.assert_array_less(
            np.abs(fast.params - slow.params), 1e-4 * slow.uncertainties
        )
        np.testing.assert_allclose(fast.uncertainties, slow.uncertainties, rtol=1e-5)

    def test_intercept_rate_roundtrip_absolute_epochs(self) -> None:
        # Conditioning round-trip proof: the solve centers t internally
        # (t -> t - mean t) and maps the intercept back exactly; on a
        # noise-free series at absolute yearf the true absolute-t
        # parameters are recovered essentially to float precision.
        t = 2015.0 + np.arange(1500) / 365.25
        y = lineperiodic(t, *TRUE_LP)
        (fit,) = fit_components(lineperiodic, t, y)
        np.testing.assert_allclose(fit.params, TRUE_LP, rtol=1e-9)
        np.testing.assert_allclose(lineperiodic(t, *fit.params), y, rtol=0, atol=1e-9)

    def test_covariance_matches_analytic_normal_equations(self) -> None:
        # C = (A^T W A)^{-1} computed independently via explicit normal
        # equations on a small well-conditioned problem.
        rng = np.random.default_rng(5)
        t = np.linspace(-2.0, 2.0, 60)
        y = linear(t, 1.0, 3.0) + rng.normal(0.0, 0.5, size=t.size)
        sigma = np.abs(rng.normal(0.5, 0.1, size=t.size)) + 0.1
        (fit,) = fit_components(linear, t, y, sigma=sigma, absolute_sigma=True)
        a = np.column_stack((np.ones_like(t), t))
        w = np.diag(1.0 / sigma**2)
        cov_ref = np.linalg.inv(a.T @ w @ a)
        params_ref = cov_ref @ (a.T @ w @ y)
        np.testing.assert_allclose(fit.params, params_ref, rtol=1e-10)
        np.testing.assert_allclose(fit.covariance, cov_ref, rtol=1e-9)

    def test_singular_design_warns_and_fills_inf(self) -> None:
        # curve_fit convention preserved: singular design -> inf
        # covariance + OptimizeWarning (message identical to scipy's).
        t = np.full(10, 2020.0)  # constant epoch: rank-1 [1, t] design
        y = np.ones(10)
        with pytest.warns(optimize.OptimizeWarning, match="could not be estimated"):
            (fit,) = fit_components(linear, t, y)
        assert np.isinf(fit.covariance).all()
        assert np.all(np.isfinite(fit.params))  # minimum-norm solution

    def test_zero_dof_warns_and_fills_inf(self) -> None:
        # N == P without absolute_sigma: no dof for the chi-square
        # rescale -> inf covariance + OptimizeWarning (curve_fit parity).
        t = np.array([0.0, 1.0])
        y = linear(t, 1.0, 2.0)
        with pytest.warns(optimize.OptimizeWarning, match="could not be estimated"):
            (fit,) = fit_components(linear, t, y)
        assert np.isinf(fit.covariance).all()
        np.testing.assert_allclose(fit.params, [1.0, 2.0], rtol=1e-12)

    def test_nonfinite_y_raises(self) -> None:
        # curve_fit's check_finite behavior preserved on the fast path.
        t = np.linspace(0.0, 1.0, 10)
        y = linear(t, 1.0, 2.0)
        y[3] = np.nan
        with pytest.raises(ValueError, match="must not contain infs or NaNs"):
            fit_components(linear, t, y)

    def test_p0_wrong_length_raises(self) -> None:
        t = np.linspace(0.0, 1.0, 10)
        y = linear(t, 1.0, 2.0)
        with pytest.raises(ValueError, match="parameters"):
            fit_components(linear, t, y, p0=np.zeros(3))

    def test_nonlinear_model_still_fits_via_curve_fit(self) -> None:
        # exp_linear is NOT in the registry - the iterative path must
        # keep working unchanged for genuinely nonlinear models.
        t = np.linspace(0.0, 6.0, 400)
        true = (2.0, 1.5, -8.0, 1.2)
        y = exp_linear(t, *true)
        (fit,) = fit_components(exp_linear, t, y, p0=[1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(fit.params, true, rtol=1e-6)


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
