"""Tests for gps_analysis.velocity (MATH_STANDARDS §4).

Covers exact rate recovery on noise-free synthetics (analytic check), the
closed-form WLS slope variance sigma_v = sigma0 / sqrt(sum (t - tbar)^2)
(reference parity with the Gauss-Markov formula), formal-sigma scaling with
the observation sigma (absolute_sigma=True: exactly linear) and with the
noise level (chi-square rescaling: residuals scaled by k scale sigma_v by
k), horizontal magnitude/azimuth analytic axis + quadrant cases with
delta-method sigma propagation, sliding-window recovery of piecewise-linear
segment rates, the min-obs/gap policy (NaN with counts recorded), guard
validation, purity (no input mutation), and the wls method tag +
detectability-floor stub.

Tolerances: noise-free/linear-in-parameters fits recover values at
rtol <= 1e-6 (optimizer convergence, not float eps); analytic sigma
identities at rtol 1e-6; delta-method formulas checked exactly against
their own closed forms (rtol 1e-12).
"""

import numpy as np
import pytest

from gps_analysis.models import linear, lineperiodic
from gps_analysis.velocity import (
    SlidingVelocity,
    VelocityEstimate,
    detectability_floor,
    estimate_velocity,
    horizontal_azimuth,
    horizontal_azimuth_sigma,
    horizontal_magnitude,
    horizontal_magnitude_sigma,
    sliding_velocity,
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


class TestDetectabilityFloor:
    def test_is_a_documented_stub(self) -> None:
        with pytest.raises(NotImplementedError, match="GBIS4TS"):
            detectability_floor(1.0, 1.0, -1.0, 2.0)
