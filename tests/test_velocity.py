"""Tests for gps_analysis.velocity (MATH_STANDARDS §4).

Covers exact rate recovery on noise-free synthetics (analytic check), the
closed-form WLS slope variance sigma_v = sigma0 / sqrt(sum (t - tbar)^2)
(reference parity with the Gauss-Markov formula), formal-sigma scaling with
the observation sigma (absolute_sigma=True: exactly linear) and with the
noise level (chi-square rescaling: residuals scaled by k scale sigma_v by
k), horizontal magnitude/azimuth analytic axis + quadrant cases with
delta-method sigma propagation, sliding-window recovery of piecewise-linear
segment rates, the min-obs/gap policy (NaN with counts recorded), guard
validation, purity (no input mutation), the wls method tag, the
colored-noise MLE velocity (method="mle": noise-param + rate recovery,
sigma_v inflation vs WLS, white-noise limit) and the detectability floor.

Tolerances: noise-free/linear-in-parameters fits recover values at
rtol <= 1e-6 (optimizer convergence, not float eps); analytic sigma
identities at rtol 1e-6; delta-method formulas checked exactly against
their own closed forms (rtol 1e-12). MLE recovery is a stochastic check
(fixed-seed synthetics; loose tolerances documented at each assert).
"""

import math

import numpy as np
import pytest
from scipy import stats

from gps_analysis.models import linear, lineperiodic
from gps_analysis.noise import estimate_noise_mle, powerlaw_rate_sigma
from gps_analysis.transient import _DELTA_T_YR, _powerlaw_psi, noise_covariance
from gps_analysis.velocity import (
    SlidingVelocity,
    VelocityEstimate,
    VelocityEstimateMLE,
    detectability_floor,
    estimate_velocity,
    estimate_velocity_mle,
    horizontal_azimuth,
    horizontal_azimuth_sigma,
    horizontal_magnitude,
    horizontal_magnitude_sigma,
    sliding_velocity,
)


def _synthetic_colored(
    n: int,
    sigma_white: float,
    beta: float,
    kappa: float,
    rate: float,
    seed: int,
    *,
    seasonal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily series: line (+ seasonal) + white + power-law noise.

    Power-law noise is white noise passed through the fractional-integration
    filter psi = (1-L)^(kappa/2) (Hosking 1981; the same transform used by
    transient.noise_covariance), scaled by beta*ΔT^(-kappa/4) so beta matches
    the Williams-2003 amplitude normalization the estimator reports.
    """
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(n) / 365.0
    psi = _powerlaw_psi(n, kappa)
    scale = beta * float(_DELTA_T_YR ** (-kappa / 4.0))
    colored = scale * np.convolve(psi, rng.standard_normal(n))[:n]
    tt = np.arange(n) * _DELTA_T_YR
    y = 3.0 + rate * tt + sigma_white * rng.standard_normal(n) + colored
    if seasonal:
        y = y + 4.0 * np.cos(2 * np.pi * t) - 2.0 * np.sin(2 * np.pi * t)
    return t, y


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


def _piecewise_linear(
    rate1: float = 1.0, rate2: float = -2.0, t_break: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous piecewise-linear series on t in [0, 10]."""
    t = np.linspace(0.0, 10.0, 501)
    y = np.where(t < t_break, rate1 * t, rate1 * t_break + rate2 * (t - t_break))
    return t, y


class TestEstimateVelocity:
    def test_exact_rate_recovery_noise_free(self) -> None:
        t, y = _synthetic_lineperiodic()
        result = estimate_velocity(t, y)
        assert isinstance(result, VelocityEstimate)
        assert result.rates.shape == (1,)
        assert result.rates[0] == pytest.approx(TRUE_LP[1], rel=1e-7)
        assert result.method == "wls"
        assert result.n_obs == t.size
        assert result.span == (t[0], t[-1])
        assert result.t_ref == pytest.approx(t.mean())
        # fit is in re-referenced time: rate slot matches, full params carried
        assert result.fits[0].params[1] == result.rates[0]

    def test_linear_model_exact_recovery(self) -> None:
        t = np.linspace(0.0, 8.0, 300)
        y = linear(t, 3.0, 2.5)
        result = estimate_velocity(t, y, model="linear")
        assert result.rates[0] == pytest.approx(2.5, rel=1e-8)

    def test_custom_callable_model(self) -> None:
        t = np.linspace(0.0, 8.0, 300)
        y = linear(t, 3.0, 2.5)
        result = estimate_velocity(t, y, model=linear)
        assert result.rates[0] == pytest.approx(2.5, rel=1e-8)

    def test_formal_sigma_matches_gauss_markov_closed_form(self) -> None:
        # For the linear model with absolute sigma s0:
        #   sigma_v = s0 / sqrt(sum (t - tbar)^2)   (Strang & Borre ch. 9)
        t = np.linspace(0.0, 4.0, 201)
        y = linear(t, 1.0, 2.0)
        s0 = 0.3
        result = estimate_velocity(
            t, y, sigma=s0 * np.ones_like(y), model="linear", absolute_sigma=True
        )
        expected = s0 / np.sqrt(np.sum((t - t.mean()) ** 2))
        assert result.sigmas[0] == pytest.approx(expected, rel=1e-6)

    def test_formal_sigma_scales_with_observation_sigma(self) -> None:
        # absolute_sigma=True: doubling sigma exactly doubles sigma_v.
        t, y = _synthetic_lineperiodic(n=400, noise=1.0, seed=3)
        r1 = estimate_velocity(t, y, sigma=np.ones_like(y), absolute_sigma=True)
        r2 = estimate_velocity(t, y, sigma=2.0 * np.ones_like(y), absolute_sigma=True)
        assert r2.sigmas[0] == pytest.approx(2.0 * r1.sigmas[0], rel=1e-3)

    def test_formal_sigma_scales_with_noise_level(self) -> None:
        # Default chi-square rescaling: scaling the noise (residuals) by k
        # scales sigma_v by k (linear model => exact linear algebra).
        rng = np.random.default_rng(11)
        t = np.linspace(0.0, 6.0, 400)
        clean = linear(t, 1.0, -4.0)
        eps = rng.normal(0.0, 1.0, size=t.size)
        r1 = estimate_velocity(t, clean + eps, model="linear")
        r2 = estimate_velocity(t, clean + 2.0 * eps, model="linear")
        assert r2.sigmas[0] == pytest.approx(2.0 * r1.sigmas[0], rel=1e-6)

    def test_window_selects_segment_rate(self) -> None:
        t, y = _piecewise_linear(rate1=1.0, rate2=-2.0)
        r1 = estimate_velocity(t, y, model="linear", window=(0.0, 4.99))
        r2 = estimate_velocity(t, y, model="linear", window=(5.0, None))
        assert r1.rates[0] == pytest.approx(1.0, rel=1e-8)
        assert r2.rates[0] == pytest.approx(-2.0, rel=1e-8)
        assert r1.n_obs < t.size

    def test_horizontal_products_from_named_components(self) -> None:
        t = np.linspace(0.0, 6.0, 300)
        # north 3, east 4, up -1 [L/yr] => magnitude 5, azimuth atan2(4,3)
        y = np.vstack([linear(t, 0.0, 3.0), linear(t, 1.0, 4.0), linear(t, -2.0, -1.0)])
        result = estimate_velocity(
            t,
            y,
            sigma=0.5 * np.ones_like(y),
            model="linear",
            names=("north", "east", "up"),
            absolute_sigma=True,
        )
        assert result.rates.shape == (3,)
        assert result.components == ("north", "east", "up")
        assert result.magnitude == pytest.approx(5.0, rel=1e-8)
        assert result.azimuth == pytest.approx(np.degrees(np.arctan2(4.0, 3.0)))
        # equal component sigmas => sigma_mag equals the component sigma
        assert result.magnitude_sigma == pytest.approx(result.sigmas[0], rel=1e-6)
        assert result.azimuth_sigma is not None and result.azimuth_sigma > 0.0

    def test_no_horizontal_products_without_north_east_names(self) -> None:
        t = np.linspace(0.0, 6.0, 200)
        y = linear(t, 0.0, 3.0)
        result = estimate_velocity(t, y, model="linear")
        assert result.magnitude is None
        assert result.azimuth is None
        assert result.magnitude_sigma is None
        assert result.azimuth_sigma is None
        assert result.components is None

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 6.0, 200)
        y = linear(t, 0.0, 3.0)
        with pytest.raises(ValueError, match="unknown model"):
            estimate_velocity(t, y, model="cubic")
        with pytest.raises(ValueError, match="at least 2 parameters"):
            estimate_velocity(t, y, model=lambda tt, a: a * np.asarray(tt))
        with pytest.raises(ValueError, match="need at least"):
            estimate_velocity(t, y, model="linear", window=(0.0, 0.01))
        with pytest.raises(ValueError, match="t must be finite"):
            estimate_velocity(np.array([0.0, np.nan, 2.0]), np.zeros(3))
        with pytest.raises(ValueError, match="sigma"):
            estimate_velocity(t, y, sigma=np.ones(5), model="linear")
        with pytest.raises(ValueError, match="names"):
            estimate_velocity(t, y, model="linear", names=("north", "east"))

    def test_does_not_mutate_inputs(self) -> None:
        t, y = _synthetic_lineperiodic(n=300, noise=1.0, seed=5)
        sigma = np.full_like(y, 1.5)
        t0, y0, s0 = t.copy(), y.copy(), sigma.copy()
        estimate_velocity(t, y, sigma=sigma)
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(sigma, s0)


class TestHorizontalProducts:
    @pytest.mark.parametrize(
        ("v_east", "v_north", "expected"),
        [
            (0.0, 1.0, 0.0),  # due north
            (1.0, 0.0, 90.0),  # due east
            (0.0, -1.0, 180.0),  # due south
            (-1.0, 0.0, 270.0),  # due west
            (1.0, 1.0, 45.0),  # NE quadrant
            (1.0, -1.0, 135.0),  # SE quadrant
            (-1.0, -1.0, 225.0),  # SW quadrant
            (-1.0, 1.0, 315.0),  # NW quadrant
        ],
    )
    def test_azimuth_axes_and_quadrants(
        self, v_east: float, v_north: float, expected: float
    ) -> None:
        assert float(horizontal_azimuth(v_east, v_north)) == pytest.approx(expected)

    def test_azimuth_is_wrapped_to_0_360(self) -> None:
        az = horizontal_azimuth([-1.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        assert np.all((az >= 0.0) & (az < 360.0))

    def test_magnitude_pythagorean_and_array(self) -> None:
        assert float(horizontal_magnitude(3.0, 4.0)) == pytest.approx(5.0)
        np.testing.assert_allclose(
            horizontal_magnitude([3.0, 0.0], [4.0, 2.0]), [5.0, 2.0]
        )

    def test_magnitude_sigma_closed_form(self) -> None:
        # sigma_mag = sqrt(vE^2 sE^2 + vN^2 sN^2) / |v|
        v_e, v_n, s_e, s_n = 4.0, 3.0, 0.4, 0.1
        expected = np.sqrt(v_e**2 * s_e**2 + v_n**2 * s_n**2) / 5.0
        got = float(horizontal_magnitude_sigma(v_e, v_n, s_e, s_n))
        assert got == pytest.approx(expected, rel=1e-12)
        # equal sigmas collapse to that sigma
        assert float(horizontal_magnitude_sigma(4.0, 3.0, 0.2, 0.2)) == pytest.approx(
            0.2, rel=1e-12
        )

    def test_azimuth_sigma_closed_form(self) -> None:
        # due-east vector: only sigma_north rotates the azimuth
        s_n = 0.5
        expected = np.degrees(s_n / 1.0)
        got = float(horizontal_azimuth_sigma(1.0, 0.0, 0.1, s_n))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_zero_magnitude_gives_nan_sigmas(self) -> None:
        assert np.isnan(horizontal_magnitude_sigma(0.0, 0.0, 0.1, 0.1))
        assert np.isnan(horizontal_azimuth_sigma(0.0, 0.0, 0.1, 0.1))


class TestSlidingVelocity:
    def test_recovers_piecewise_segment_rates(self) -> None:
        t, y = _piecewise_linear(rate1=1.0, rate2=-2.0, t_break=5.0)
        result = sliding_velocity(
            t, y, window_years=2.0, step_years=1.0, model="linear"
        )
        assert isinstance(result, SlidingVelocity)
        np.testing.assert_allclose(result.centers, np.arange(1.0, 10.0))
        assert result.rates.shape == (1, 9)
        assert result.method == "wls"
        # windows fully inside segment 1 (centres 1-4: the t=5 sample lies
        # on both branches of the continuous series) recover rate1 exactly
        np.testing.assert_allclose(result.rates[0, :4], 1.0, rtol=1e-7)
        # windows fully inside segment 2 (centres 6-9) recover rate2
        np.testing.assert_allclose(result.rates[0, 5:], -2.0, rtol=1e-7)
        # the break-straddling window (centre 5) lies between the rates
        assert -2.0 < result.rates[0, 4] < 1.0
        assert np.all(result.counts > 0)
        assert np.all(np.isfinite(result.sigmas))

    def test_gap_windows_are_nan_with_counts_recorded(self) -> None:
        t = np.concatenate([np.linspace(0.0, 3.0, 151), np.linspace(7.0, 10.0, 151)])
        y = linear(t, 1.0, 2.0)
        result = sliding_velocity(
            t, y, window_years=2.0, step_years=1.0, model="linear"
        )
        np.testing.assert_allclose(result.centers, np.arange(1.0, 10.0))
        # windows centred 4, 5, 6 fall in/around the gap: <= 1 sample each,
        # below the default min_obs = 2 P = 4 -> NaN, counts still recorded
        assert np.all(np.isnan(result.rates[0, 3:6]))
        assert np.all(np.isnan(result.sigmas[0, 3:6]))
        assert result.counts[4] == 0
        assert result.counts[3] <= 1 and result.counts[5] <= 1
        # populated windows are fine and exact
        np.testing.assert_allclose(result.rates[0, :2], 2.0, rtol=1e-7)
        np.testing.assert_allclose(result.rates[0, 7:], 2.0, rtol=1e-7)

    def test_min_obs_guard_skips_sparse_windows(self) -> None:
        t, y = _piecewise_linear()
        result = sliding_velocity(
            t, y, window_years=2.0, step_years=1.0, model="linear", min_obs=1000
        )
        assert np.all(np.isnan(result.rates))  # every window is too sparse
        assert np.all(result.counts > 0)  # but the counts are recorded

    def test_multi_component_with_names_and_sigma(self) -> None:
        t = np.linspace(0.0, 6.0, 300)
        y = np.vstack([linear(t, 0.0, 3.0), linear(t, 1.0, 4.0)])
        sigma = np.full_like(y, 0.5)
        result = sliding_velocity(
            t,
            y,
            sigma=sigma,
            window_years=2.0,
            step_years=2.0,
            model="linear",
            names=("north", "east"),
            absolute_sigma=True,
        )
        assert result.components == ("north", "east")
        assert result.rates.shape == (2, result.centers.size)
        np.testing.assert_allclose(result.rates[0], 3.0, rtol=1e-7)
        np.testing.assert_allclose(result.rates[1], 4.0, rtol=1e-7)

    def test_window_geometry_recorded(self) -> None:
        t, y = _piecewise_linear()
        result = sliding_velocity(
            t, y, window_years=3.0, step_years=0.5, model="linear"
        )
        assert result.window_years == 3.0
        assert result.step_years == 0.5
        assert result.centers[0] == pytest.approx(1.5)
        np.testing.assert_allclose(np.diff(result.centers), 0.5)

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 6.0, 100)
        y = linear(t, 0.0, 1.0)
        with pytest.raises(ValueError, match="window_years must be > 0"):
            sliding_velocity(t, y, window_years=0.0, step_years=1.0)
        with pytest.raises(ValueError, match="step_years must be > 0"):
            sliding_velocity(t, y, window_years=2.0, step_years=0.0)
        with pytest.raises(ValueError, match="min_obs"):
            sliding_velocity(
                t, y, window_years=2.0, step_years=1.0, model="linear", min_obs=2
            )
        with pytest.raises(ValueError, match="exceeds the data span"):
            sliding_velocity(t, y, window_years=10.0, step_years=1.0)
        with pytest.raises(ValueError, match="t must be finite"):
            sliding_velocity(
                np.array([0.0, np.inf]), np.zeros(2), window_years=1.0, step_years=1.0
            )

    def test_does_not_mutate_inputs(self) -> None:
        t, y = _piecewise_linear()
        t0, y0 = t.copy(), y.copy()
        sliding_velocity(t, y, window_years=2.0, step_years=1.0, model="linear")
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)

    def test_precomputed_design_matches_iterative_reference(self) -> None:
        # Perf finding #5 equivalence: the named lineperiodic model now
        # solves each window from a row slice of one precomputed design
        # (absolute-t trig basis, per-window re-centered trend column);
        # a wrapper callable with identical math is NOT in the linear
        # registry, so it takes the old per-window curve_fit route in
        # window-local time. Rates and formal sigmas must agree - the
        # two bases span the same column space, so v and sigma_v are
        # mathematically identical; the tolerance budget is entirely
        # LM's relative ftol/xtol stopping error on the huge-|y|
        # absolute-yearf series (measured ~2e-5 sigma_v on rates and
        # ~1.4e-6 relative on the chi-square-rescaled sigmas), hence the
        # sigma-scaled rate metric (|dv| <= 1e-4 sigma_v) and sigma
        # rtol 1e-5 - matching the fit_components absolute-t gates.
        rng = np.random.default_rng(17)
        t, y = _synthetic_lineperiodic(n=1500, noise=2.0, seed=17)
        sigma = np.abs(rng.normal(2.0, 0.3, size=t.size)) + 0.5

        def lineperiodic_wrapper(
            tt: np.ndarray,
            offset: float,
            rate: float,
            cos_annual: float,
            sin_annual: float,
            cos_semiannual: float,
            sin_semiannual: float,
        ) -> np.ndarray:
            return lineperiodic(
                tt, offset, rate, cos_annual, sin_annual, cos_semiannual, sin_semiannual
            )

        kwargs = dict(sigma=sigma, window_years=2.0, step_years=0.25, p0=np.zeros(6))
        fast = sliding_velocity(t, y, model="lineperiodic", **kwargs)
        slow = sliding_velocity(t, y, model=lineperiodic_wrapper, **kwargs)
        np.testing.assert_array_equal(fast.centers, slow.centers)
        np.testing.assert_array_equal(fast.counts, slow.counts)
        np.testing.assert_array_equal(np.isnan(fast.rates), np.isnan(slow.rates))
        finite = np.isfinite(slow.rates)
        assert finite.any()
        np.testing.assert_array_less(
            np.abs(fast.rates[finite] - slow.rates[finite]),
            1e-4 * slow.sigmas[finite],
        )
        np.testing.assert_allclose(fast.sigmas, slow.sigmas, rtol=1e-5)


class TestPowerlawRateSigma:
    """Exact colored-noise GLS rate uncertainty (Williams 2003, eqs. 23-30)."""

    def test_white_matches_closed_form(self) -> None:
        # White noise (kappa=0, beta=0): sigma_v = sigma_w / sqrt(Sum (t-tbar)^2)
        # with centered t = ΔT*(i-(n-1)/2), Sum (t-tbar)^2 = ΔT^2*n*(n^2-1)/12.
        n, sw = 366, 1.3
        got = powerlaw_rate_sigma(sw, 0.0, 0.0, n)
        exact = sw / (_DELTA_T_YR * math.sqrt(n * (n**2 - 1) / 12.0))
        assert got == pytest.approx(exact, rel=1e-10)

    def test_matches_dense_gls_covariance(self) -> None:
        # Reference parity: the Schur-based sigma_v equals the dense
        # (A^T C^-1 A)^-1 built from transient.noise_covariance.
        n, sw, beta, kappa = 200, 1.0, 4.0, -1.0
        c = noise_covariance(n, sw, kappa, beta)
        t = _DELTA_T_YR * (np.arange(n) - (n - 1) / 2.0)
        a = np.column_stack((np.ones_like(t), t))
        cov = np.linalg.inv(a.T @ np.linalg.solve(c, a))
        assert powerlaw_rate_sigma(sw, beta, kappa, n) == pytest.approx(
            math.sqrt(cov[1, 1]), rel=1e-8
        )

    @pytest.mark.parametrize(
        ("kappa", "expected_exponent"),
        [(0.0, -3.0), (-1.0, -2.0), (-2.0, -1.0)],
    )
    def test_span_scaling(self, kappa: float, expected_exponent: float) -> None:
        # Williams 2003: sigma_v^2 ∝ T^(-3-kappa) at fixed ΔT. Doubling the
        # span (~doubling n) scales sigma_v by 2^((-3-kappa)/2).
        white, pln = (1.0, 0.0) if kappa == 0.0 else (0.0, 4.0)
        n1, n2 = 2 * 365 + 1, 4 * 365 + 1
        s1 = powerlaw_rate_sigma(white, pln, kappa, n1)
        s2 = powerlaw_rate_sigma(white, pln, kappa, n2)
        assert s2 / s1 == pytest.approx(2.0 ** (expected_exponent / 2.0), rel=0.02)

    def test_guards(self) -> None:
        with pytest.raises(ValueError, match="both be zero"):
            powerlaw_rate_sigma(0.0, 0.0, -1.0, 100)
        with pytest.raises(ValueError, match=">= 0"):
            powerlaw_rate_sigma(-1.0, 1.0, -1.0, 100)
        with pytest.raises(ValueError, match="spectral_index"):
            powerlaw_rate_sigma(1.0, 1.0, -4.0, 100)
        with pytest.raises(ValueError, match="n_epochs"):
            powerlaw_rate_sigma(1.0, 1.0, -1.0, 2)


class TestEstimateNoiseMLE:
    """Joint trajectory + white/power-law noise MLE (module gps_analysis.noise)."""

    def test_recovers_flicker_noise_and_rate(self) -> None:
        # Injected flicker (kappa=-1) + white; MLE recovers noise triple and
        # rate. Single fixed seed -> loose tolerances (finite-sample MLE
        # scatter); the multi-seed calibration lives in the class docstring.
        n = 1461  # 4 yr daily
        t, y = _synthetic_colored(n, 1.0, 4.0, -1.0, -3.5, seed=6)
        tt = np.arange(n) * _DELTA_T_YR
        a = np.column_stack(
            (
                np.ones(n),
                tt - tt.mean(),
                np.cos(2 * np.pi * t),
                np.sin(2 * np.pi * t),
                np.cos(4 * np.pi * t),
                np.sin(4 * np.pi * t),
            )
        )
        fit = estimate_noise_mle(a, y)
        assert fit.noise.spectral_index == pytest.approx(-1.0, abs=0.35)
        assert fit.noise.amplitude_powerlaw == pytest.approx(4.0, rel=0.35)
        assert fit.noise.sigma_white == pytest.approx(1.0, abs=0.5)
        assert fit.params[1] == pytest.approx(
            -3.5, abs=3.0 * math.sqrt(fit.covariance[1, 1])
        )
        assert fit.noise.n_obs == n

    def test_white_series_gives_negligible_powerlaw(self) -> None:
        # A pure white series: the MLE should not manufacture power-law noise,
        # so sigma_v stays close to the white-noise WLS formal error.
        n = 1200
        t, y = _synthetic_colored(n, 1.5, 0.0, -1.0, 8.0, seed=3, seasonal=False)
        a = np.column_stack((np.ones(n), np.arange(n) * _DELTA_T_YR))
        fit = estimate_noise_mle(a, y)
        white_cov = np.linalg.inv(a.T @ a) * (
            float(((y - a @ np.linalg.lstsq(a, y, rcond=None)[0]) ** 2).sum()) / (n - 2)
        )
        inflation = math.sqrt(fit.covariance[1, 1] / white_cov[1, 1])
        assert inflation == pytest.approx(1.0, abs=0.25)

    def test_guards(self) -> None:
        a = np.column_stack((np.ones(10), np.arange(10.0)))
        with pytest.raises(ValueError, match="P \\+ 3"):
            estimate_noise_mle(a[:4], np.arange(4.0))
        with pytest.raises(ValueError, match="finite"):
            estimate_noise_mle(a, np.r_[np.arange(9.0), np.nan])
        with pytest.raises(ValueError, match="kappa_bounds"):
            estimate_noise_mle(a, np.arange(10.0), kappa_bounds=(-4.0, 0.0))


class TestEstimateVelocityMLE:
    """Fixed-window colored-noise MLE velocity (method='mle')."""

    def test_sigma_inflated_over_wls(self) -> None:
        # The honest colored-noise sigma_v must exceed the optimistic WLS
        # formal error for a flicker-dominated series (Williams et al. 2004).
        n = 1461
        t, y = _synthetic_colored(n, 1.0, 4.0, -1.0, -3.5, seed=6)
        mle = estimate_velocity_mle(t, y, model="lineperiodic")
        wls = estimate_velocity(t, y, model="lineperiodic")
        assert isinstance(mle, VelocityEstimateMLE)
        assert mle.method == "mle"
        inflation = float(mle.sigmas[0] / wls.sigmas[0])
        assert inflation > 3.0  # several-x inflation for flicker (n=4 yr daily)
        # Rate agrees with the truth within a few honest sigma.
        assert mle.rates[0] == pytest.approx(-3.5, abs=3.0 * float(mle.sigmas[0]))
        # Noise model is attached and flicker-like.
        assert mle.noise[0].spectral_index == pytest.approx(-1.0, abs=0.4)

    def test_honest_sigma_covers_empirical_scatter(self) -> None:
        # Across seeds the MLE sigma_v should be the right order as the true
        # GLS rate uncertainty (the empirical rate scatter), unlike WLS which
        # is ~7x too small. Compare mean MLE sigma to the dense-GLS truth.
        n = 1461
        rates, mle_sigmas, wls_sigmas = [], [], []
        for seed in range(1, 7):
            t, y = _synthetic_colored(n, 1.0, 4.0, -1.0, -3.5, seed=seed)
            mle = estimate_velocity_mle(t, y, model="lineperiodic")
            wls = estimate_velocity(t, y, model="lineperiodic")
            rates.append(float(mle.rates[0]))
            mle_sigmas.append(float(mle.sigmas[0]))
            wls_sigmas.append(float(wls.sigmas[0]))
        scatter = float(np.std(rates, ddof=1))
        mean_mle = float(np.mean(mle_sigmas))
        mean_wls = float(np.mean(wls_sigmas))
        # MLE sigma is within a factor ~2 of the empirical scatter ...
        assert 0.4 * scatter < mean_mle < 2.5 * scatter
        # ... while WLS underestimates it by a large factor.
        assert mean_wls < 0.3 * scatter

    def test_horizontal_products_and_provenance(self) -> None:
        n = 900
        t, ye = _synthetic_colored(n, 1.0, 3.0, -1.0, 2.0, seed=11, seasonal=False)
        _, yn = _synthetic_colored(n, 1.0, 3.0, -1.0, 5.0, seed=12, seasonal=False)
        y = np.vstack((yn, ye))
        mle = estimate_velocity_mle(t, y, model="linear", names=("north", "east"))
        assert mle.magnitude is not None and mle.azimuth is not None
        assert mle.magnitude == pytest.approx(
            math.hypot(mle.rates[0], mle.rates[1]), rel=1e-12
        )
        assert len(mle.noise) == 2
        assert mle.n_obs == n

    def test_rejects_nonlinear_model(self) -> None:
        from gps_analysis.models import exp_linear

        n = 500
        t, y = _synthetic_colored(n, 1.0, 2.0, -1.0, 1.0, seed=1, seasonal=False)
        with pytest.raises(ValueError, match="linear-in-parameters"):
            estimate_velocity_mle(t, y, model=exp_linear)

    def test_does_not_mutate_inputs(self) -> None:
        n = 500
        t, y = _synthetic_colored(n, 1.0, 2.0, -1.0, 1.0, seed=2, seasonal=False)
        t0, y0 = t.copy(), y.copy()
        estimate_velocity_mle(t, y, model="linear")
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)


class TestDetectabilityFloor:
    """Minimum detectable velocity change under colored noise (Williams 2003)."""

    def test_hand_computed_flicker_case(self) -> None:
        # Hand check: T = 4 yr daily (n = 4*365+1 = 1461), flicker kappa=-1,
        # beta=4 mm/yr^0.25, no white noise, 95% two-sided => z=1.959964.
        # Dv_min = z*sqrt(2)*sigma_v with sigma_v the exact GLS rate error.
        sigma_v = powerlaw_rate_sigma(0.0, 4.0, -1.0, 1461)
        z = float(stats.norm.ppf(0.975))
        expected = z * math.sqrt(2.0) * sigma_v
        got = detectability_floor(0.0, 4.0, -1.0, 4.0)
        assert got == pytest.approx(expected, rel=1e-9)
        # And cross-check sigma_v itself against the dense GLS covariance.
        n = 1461
        c = noise_covariance(n, 0.0, -1.0, 4.0)
        t = _DELTA_T_YR * (np.arange(n) - (n - 1) / 2.0)
        a = np.column_stack((np.ones_like(t), t))
        cov = np.linalg.inv(a.T @ np.linalg.solve(c, a))
        assert sigma_v == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-8)

    def test_single_window_drops_sqrt2(self) -> None:
        two = detectability_floor(1.0, 4.0, -1.0, 3.0)
        one = detectability_floor(1.0, 4.0, -1.0, 3.0, single_window=True)
        assert two / one == pytest.approx(math.sqrt(2.0), rel=1e-12)

    def test_confidence_scales_the_z_quantile(self) -> None:
        d95 = detectability_floor(1.0, 4.0, -1.0, 3.0, confidence=0.95)
        d99 = detectability_floor(1.0, 4.0, -1.0, 3.0, confidence=0.99)
        ratio = float(stats.norm.ppf(0.995) / stats.norm.ppf(0.975))
        assert d99 / d95 == pytest.approx(ratio, rel=1e-9)

    def test_longer_window_lowers_floor(self) -> None:
        # More data => tighter rate => smaller detectable change.
        assert detectability_floor(1.0, 4.0, -1.0, 6.0) < detectability_floor(
            1.0, 4.0, -1.0, 3.0
        )

    def test_guards(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            detectability_floor(1.0, 4.0, -1.0, 3.0, confidence=1.0)
        with pytest.raises(ValueError, match="window_years"):
            detectability_floor(1.0, 4.0, -1.0, 0.0)
        with pytest.raises(ValueError, match="3 epochs"):
            detectability_floor(1.0, 4.0, -1.0, 0.001)
