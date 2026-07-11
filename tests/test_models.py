"""Tests for gps_analysis.models (MATH_STANDARDS §4).

Covers analytic checks (zero seasonal ⇒ pure line, derivative identities,
polynomial vertex identities), legacy parity against inline copies of the
``detrend_rnes.py`` / ``svartsengi_model.fitting`` formulas (golden
behavior), and property tests (annual periodicity, container validation).
Tolerances: parity at ``rtol = 1e-12`` (association differences ≤ a few
ulp); derivative checks against central differences at ``rtol = 1e-6``
(truncation error O(h²), h = 1e-6 yr).
"""

import numpy as np
import pytest

from gps_analysis.models import (
    TrajectoryParams,
    exp_linear,
    exp_linear_rate,
    linear,
    lineperiodic,
    periodic,
    poly2,
    poly2_peak_time,
    poly2_peak_value,
    poly2_rate,
)

RNG = np.random.default_rng(20260710)


# --- legacy formulas, copied verbatim for golden parity -------------------
# detrend-reykjanes/detrend_rnes.py (line/periodic/lineperiodic)


def _legacy_line(x: np.ndarray, p0: float, p1: float) -> np.ndarray:
    return p0 + p1 * x


def _legacy_lineperiodic(
    x: np.ndarray, p0: float, p1: float, p2: float, p3: float, p4: float, p5: float
) -> np.ndarray:
    return (
        p0
        + p1 * x
        + p2 * np.cos(2 * np.pi * x)
        + p3 * np.sin(2 * np.pi * x)
        + p4 * np.cos(4 * np.pi * x)
        + p5 * np.sin(4 * np.pi * x)
    )


# svartsengi_model/fitting.py (expf_long/expf_short/dexpf/dexpf_short,
# polynomial_transient/dpolynomial_transient)


def _legacy_expf_long(x: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    return p0 + p1 * np.exp(-p2 * np.asarray(x))


def _legacy_expf_short(x: np.ndarray, p0: float, p1: float) -> np.ndarray:
    return p0 * np.exp(-np.asarray(x) / p1)


def _legacy_polynomial_transient(
    x: np.ndarray, p0: float, p1: float, p2: float
) -> np.ndarray:
    x_arr = np.asarray(x)
    return p0 + p1 * x_arr + p2 * x_arr**2


def _yearf(n: int = 500) -> np.ndarray:
    return np.sort(RNG.uniform(2015.0, 2026.0, size=n))


class TestLinear:
    def test_analytic_values(self) -> None:
        t = np.array([0.0, 1.0, 2.5])
        np.testing.assert_array_equal(linear(t, 3.0, -2.0), [3.0, 1.0, -2.0])

    def test_legacy_parity(self) -> None:
        t = _yearf()
        np.testing.assert_allclose(
            linear(t, 12.3, -4.5), _legacy_line(t, 12.3, -4.5), rtol=1e-12
        )

    def test_scalar_t_gives_0d_float64(self) -> None:
        out = linear(2020.5, 1.0, 2.0)
        assert out.shape == ()
        assert out.dtype == np.float64


class TestPeriodic:
    def test_zero_amplitudes_give_zero(self) -> None:
        t = _yearf()
        np.testing.assert_array_equal(periodic(t, 0.0, 0.0, 0.0, 0.0), 0.0 * t)

    def test_one_year_periodicity(self) -> None:
        t = _yearf(200)
        a = periodic(t, 3.0, -1.0, 0.5, 2.0)
        b = periodic(t + 1.0, 3.0, -1.0, 0.5, 2.0)
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_legacy_parity_ignoring_dropped_params(self) -> None:
        # legacy periodic(x, p0..p5) silently ignored p0, p1 — the new
        # signature drops them; equality holds for the seasonal terms.
        t = _yearf()
        legacy = _legacy_lineperiodic(t, 0.0, 0.0, 3.0, -1.0, 0.5, 2.0)
        np.testing.assert_allclose(
            periodic(t, 3.0, -1.0, 0.5, 2.0), legacy, rtol=1e-12, atol=1e-12
        )


class TestLineperiodic:
    def test_zero_seasonal_reduces_to_pure_line(self) -> None:
        t = _yearf()
        np.testing.assert_array_equal(
            lineperiodic(t, 5.0, -1.2, 0.0, 0.0, 0.0, 0.0), linear(t, 5.0, -1.2)
        )

    def test_is_sum_of_linear_and_periodic(self) -> None:
        t = _yearf()
        total = lineperiodic(t, 5.0, -1.2, 3.0, -1.0, 0.5, 2.0)
        parts = linear(t, 5.0, -1.2) + periodic(t, 3.0, -1.0, 0.5, 2.0)
        np.testing.assert_array_equal(total, parts)

    def test_legacy_parity(self) -> None:
        t = _yearf()
        for _ in range(5):
            p = RNG.normal(0.0, 10.0, size=6)
            np.testing.assert_allclose(
                lineperiodic(t, *p), _legacy_lineperiodic(t, *p), rtol=1e-12
            )


class TestExpLinear:
    def test_value_at_zero_and_asymptote(self) -> None:
        assert float(exp_linear(0.0, 10.0, 0.0, -5.0, 2.0)) == pytest.approx(5.0)
        # rate = 0: x(t) -> offset as t -> inf
        assert float(exp_linear(50.0, 10.0, 0.0, -5.0, 2.0)) == pytest.approx(10.0)

    def test_legacy_parity_expf_long(self) -> None:
        t = np.linspace(0.0, 3.0, 300)
        np.testing.assert_allclose(
            exp_linear(t, 89.0, 0.0, -1625.0, 2.9),
            _legacy_expf_long(t, 89.0, -1625.0, 2.9),
            rtol=1e-12,
        )

    def test_legacy_parity_expf_short_tau_form(self) -> None:
        # expf_short(t, A, tau) == exp_linear(t, 0, 0, A, 1/tau)
        t = np.linspace(0.0, 0.2, 100)
        tau = 8.0 / 365.25
        np.testing.assert_allclose(
            exp_linear(t, 0.0, 0.0, -120.0, 1.0 / tau),
            _legacy_expf_short(t, -120.0, tau),
            rtol=1e-12,
        )

    def test_rate_matches_central_difference(self) -> None:
        t = np.linspace(0.05, 3.0, 50)
        h = 1e-6
        numeric = (
            exp_linear(t + h, 1.0, -2.0, -5.0, 2.5)
            - exp_linear(t - h, 1.0, -2.0, -5.0, 2.5)
        ) / (2 * h)
        np.testing.assert_allclose(
            exp_linear_rate(t, -2.0, -5.0, 2.5), numeric, rtol=1e-6
        )

    def test_rate_legacy_parity_dexpf(self) -> None:
        # dexpf(x, p1, p2) = -p1 * p2 * exp(-p2 * x) == exp_linear_rate(x, 0, p1, p2)
        t = np.linspace(0.0, 3.0, 100)
        p1 = -1625.0
        legacy = -p1 * 2.9 * np.exp(-2.9 * t)
        np.testing.assert_allclose(exp_linear_rate(t, 0.0, p1, 2.9), legacy, rtol=1e-12)


class TestPoly2:
    def test_legacy_parity(self) -> None:
        t = np.linspace(0.0, 2.0, 200)
        np.testing.assert_allclose(
            poly2(t, 1.0, 10.0, -4.0),
            _legacy_polynomial_transient(t, 1.0, 10.0, -4.0),
            rtol=1e-12,
        )

    def test_rate_matches_central_difference(self) -> None:
        t = np.linspace(0.0, 2.0, 50)
        h = 1e-6
        numeric = (poly2(t + h, 1.0, 10.0, -4.0) - poly2(t - h, 1.0, 10.0, -4.0)) / (
            2 * h
        )
        np.testing.assert_allclose(poly2_rate(t, 10.0, -4.0), numeric, rtol=1e-5)

    def test_vertex_identities(self) -> None:
        offset, rate, curvature = 1.0, 10.0, -4.0
        t_peak = poly2_peak_time(rate, curvature)
        assert t_peak == pytest.approx(1.25)
        # dx/dt vanishes at the vertex
        assert float(poly2_rate(t_peak, rate, curvature)) == pytest.approx(0.0)
        # x(t_peak) equals the closed-form vertex value
        assert float(poly2(t_peak, offset, rate, curvature)) == pytest.approx(
            poly2_peak_value(offset, rate, curvature)
        )

    def test_zero_curvature_raises(self) -> None:
        with pytest.raises(ValueError, match="curvature"):
            poly2_peak_time(10.0, 0.0)
        with pytest.raises(ValueError, match="curvature"):
            poly2_peak_value(1.0, 10.0, 0.0)


class TestTrajectoryParams:
    def test_uncertainties_are_sqrt_of_diagonal(self) -> None:
        cov = np.diag([4.0, 9.0, 0.25])
        fit = TrajectoryParams(params=np.zeros(3), covariance=cov)
        np.testing.assert_allclose(fit.uncertainties, [2.0, 3.0, 0.5])

    def test_coerces_integer_arrays_to_float64(self) -> None:
        fit = TrajectoryParams(params=np.array([1, 2]), covariance=np.eye(2, dtype=int))
        assert fit.params.dtype == np.float64
        assert fit.covariance.shape == (2, 2)
        assert len(fit) == 2

    def test_component_label(self) -> None:
        fit = TrajectoryParams(
            params=np.zeros(2), covariance=np.eye(2), component="north"
        )
        assert fit.component == "north"

    def test_rejects_non_1d_params(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            TrajectoryParams(params=np.zeros((2, 2)), covariance=np.eye(2))

    def test_rejects_mismatched_covariance(self) -> None:
        with pytest.raises(ValueError, match="covariance"):
            TrajectoryParams(params=np.zeros(3), covariance=np.eye(2))
