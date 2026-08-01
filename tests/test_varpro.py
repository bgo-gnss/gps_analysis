"""Tests for :mod:`gps_analysis.varpro` (MATH_STANDARDS §4).

Analytic checks: noise-free exact recovery of both the nonlinear
parameter and the amplitudes; parity of the concentrated amplitude solve
with :func:`gps_analysis.fitting._wls_solve` (the canonical WLS path).
Reference parity: the eq.-(8) Jacobian against central differences (the
prior math review measured 6e-9; asserted at 1e-6 — FD truncation is
O(h²) ≈ 1e-12 relative and roundoff ≈ eps/h ≈ 1e-10, so 1e-6 carries
two orders of margin over both). Property tests: covariance symmetry and
positive-definiteness, σ-scaling invariance. Corrections pinned: the
B term is kept (Kaufman 1975 rejected) and the bordered σ_θ strictly
exceeds the JᵀJ value. Degeneracy guards raise informatively.

Test model family (term-agnostic on purpose — no transient import):
``Φ(θ) = [1, t, 1 − exp(−t·e^{−θ})]`` with θ = ln τ, the analytic
derivative ``∂Φ₃/∂θ = −u·e^{−u}``, ``u = t·e^{−θ}``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from gps_analysis import varpro
from gps_analysis.fitting import _wls_solve
from gps_analysis.models import FloatArray
from gps_analysis.varpro import (
    VarproFit,
    _bordered_covariance,
    _design_svd,
    _projection_residual,
    _svd_solve,
    _varpro_jacobian,
    _whiten,
    estimate_varpro,
)

THETA_TRUE = float(np.log(0.5))  # tau = 0.5 yr over a 4 yr span: T/tau = 8
COEFFS_TRUE = np.array([3.0, -2.0, 12.0])  # mm: offset, rate*t, transient amp
BOUNDS = (-3.0, 1.6)


def _time_axis(n: int = 400, span: float = 4.0) -> FloatArray:
    return np.linspace(0.0, span, n)


def _builders(
    t: FloatArray,
) -> tuple[Callable[[float], FloatArray], Callable[[float], FloatArray]]:
    """Design and analytic-derivative builders for the test family."""

    def design(theta: float) -> FloatArray:
        u = t * np.exp(-theta)
        return np.column_stack((np.ones_like(t), t, 1.0 - np.exp(-u)))

    def d_design(theta: float) -> FloatArray:
        u = t * np.exp(-theta)
        zeros = np.zeros_like(t)
        return np.column_stack((zeros, zeros, -u * np.exp(-u)))

    return design, d_design


def _noisy_data(
    t: FloatArray,
    theta: float = THETA_TRUE,
    noise: float = 1.5,
    seed: int = 20260801,
) -> tuple[FloatArray, FloatArray]:
    """Observations y = Phi(theta)·c* + N(0, noise²) and their sigma."""
    design, _ = _builders(t)
    rng = np.random.default_rng(seed)
    y = design(theta) @ COEFFS_TRUE + rng.normal(0.0, noise, t.size)
    sigma = np.full(t.size, noise)
    return np.asarray(y, dtype=np.float64), sigma


def _jacobian_pieces(
    t: FloatArray,
    y: FloatArray,
    sigma: FloatArray,
    theta: float,
    *,
    kaufman: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """(residual, Jacobian) of the concentrated problem at theta."""
    design, d_design = _builders(t)
    a_w, y_w = _whiten(design(theta), y, sigma)
    svd = _design_svd(a_w, y_w)
    coeffs = _svd_solve(svd)
    residual = _projection_residual(svd)
    d_w = np.asarray(d_design(theta) / sigma[:, np.newaxis], dtype=np.float64)
    jac = _varpro_jacobian(svd, d_w, coeffs, residual, kaufman=kaufman)
    return residual, jac


class TestExactRecovery:
    def test_noise_free_recovers_theta_and_amplitudes(self) -> None:
        # Analytic check: noise-free separable data => the profile optimum
        # and the concentrated amplitudes are exact to the objective's
        # conditioning floor. chi2 bottoms out at roundoff (~5e-16 here),
        # so theta resolves to ~sqrt(chi2_floor / curvature) ≈ 2e-10 —
        # measured 1.6e-9; asserted at 1e-8 (search xatol is 1e-10, so
        # the limit is float64 conditioning, not the search).
        t = _time_axis()
        design, d_design = _builders(t)
        y = design(THETA_TRUE) @ COEFFS_TRUE
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            d_design=d_design,
            absolute_sigma=True,
        )
        assert isinstance(fit, VarproFit)
        assert abs(fit.theta - THETA_TRUE) < 1e-8
        np.testing.assert_allclose(fit.params, COEFFS_TRUE, atol=1e-8)
        assert fit.n_obs == t.size
        assert fit.scale_sq == 1.0

    def test_amplitudes_match_wls_solve_at_optimum(self) -> None:
        # Parity pin: the shared-SVD amplitude solve must agree with the
        # package's canonical _wls_solve on the same design — ties the
        # one-factorization path to the established solver.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
            absolute_sigma=True,
        )
        params_wls, _ = _wls_solve(design(fit.theta), y, sigma, True)
        np.testing.assert_allclose(fit.params, params_wls, rtol=0.0, atol=1e-10)

    def test_finite_difference_fallback_matches_analytic(self) -> None:
        # The documented FD fallback for D must reproduce the analytic-D
        # covariance to well inside its own error (~eps^(2/3) ≈ 4e-11).
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        kwargs: dict[str, Any] = {
            "theta_bounds": BOUNDS,
            "sigma": sigma,
            "absolute_sigma": True,
        }
        fit_fd = estimate_varpro(design, y, **kwargs)
        fit_an = estimate_varpro(design, y, d_design=d_design, **kwargs)
        assert fit_fd.theta == fit_an.theta  # theta never touches D
        # atol covers the theta-amplitude cross terms, which are ~1e-10
        # here (near zero at the optimum) and meaningless in relative
        # terms; every O(1) element must agree to 1e-6 relative.
        np.testing.assert_allclose(
            fit_fd.covariance, fit_an.covariance, rtol=1e-6, atol=1e-12
        )


class TestJacobian:
    def test_eq8_matches_central_differences(self) -> None:
        # Reference parity for O'Leary & Rust 2013 eq. (8) (p. 585,
        # verified against the primary source), evaluated away from the
        # optimum (nonzero gradient). Prior review measured 6e-9; 1e-6
        # leaves two orders of margin over FD truncation + roundoff.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, _ = _builders(t)

        def resid(theta: float) -> FloatArray:
            a_w, y_w = _whiten(design(theta), y, sigma)
            return _projection_residual(_design_svd(a_w, y_w))

        for theta0 in (THETA_TRUE - 0.4, THETA_TRUE + 0.3):
            _, jac = _jacobian_pieces(t, y, sigma, theta0)
            h = 1e-6
            jac_fd = (resid(theta0 + h) - resid(theta0 - h)) / (2.0 * h)
            rel = float(np.max(np.abs(jac - jac_fd)) / np.max(np.abs(jac)))
            assert rel < 1e-6

    def test_b_term_kept_but_gradient_blind_to_it(self) -> None:
        # WHY Kaufman 1975 is tempting: the objective gradient r·J is
        # bit-identical (to roundoff) with the B term dropped, because
        # U'r = 0 exactly — so a gradient check can never expose the
        # approximation. It is rejected anyway: O'Leary & Rust 2013 §3.1
        # (pp. 588-589) measure iterations rising 3->7 and function
        # evaluations 8->12 (three stalls instead of one) with ||B||/||J||
        # never above 2 %, and the covariance border needs the full J.
        # This test stops a future reader from "optimizing" the B term
        # away: the full and Kaufman Jacobians must DIFFER while their
        # gradients agree.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        for theta0 in (THETA_TRUE - 0.5, THETA_TRUE + 0.4):
            residual, jac_full = _jacobian_pieces(t, y, sigma, theta0)
            _, jac_kaufman = _jacobian_pieces(t, y, sigma, theta0, kaufman=True)
            # The Jacobians themselves differ materially (B term nonzero).
            assert float(np.max(np.abs(jac_full - jac_kaufman))) > 1e-8
            # ... yet the gradients agree to roundoff (measured ~4e-13).
            g_full = 2.0 * float(residual @ jac_full)
            g_kaufman = 2.0 * float(residual @ jac_kaufman)
            assert abs(g_full - g_kaufman) < 1e-11 * max(1.0, abs(g_full))


class TestCovariance:
    def test_bordered_sigma_strictly_exceeds_jtj(self) -> None:
        # Pins the review correction: sigma_theta from the bordered matrix
        # H = [Phi_w, J] (O'Leary & Rust §2.5, p. 587), never from J'J —
        # the latter conditions on the amplitudes and understates
        # sigma_theta. tau comparable to
        # the span makes the transient column nearly collinear with the
        # rate column (correlated amplitudes), where the gap is material.
        t = _time_axis()
        theta_corr = float(np.log(1.0))
        y, sigma = _noisy_data(t, theta=theta_corr, noise=2.0)
        design, d_design = _builders(t)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
            absolute_sigma=True,
        )
        _, jac = _jacobian_pieces(t, y, sigma, fit.theta)
        sigma_naive = float(np.sqrt(fit.scale_sq / float(jac @ jac)))
        assert fit.theta_sigma > sigma_naive

    def test_border_column_is_j_and_variances_sign_invariant(self) -> None:
        # The paper borders with J itself — H = W[Phi, J] (§2.5, p. 587)
        # — and this pins that convention: the returned covariance must
        # equal _bordered_covariance(a_w, J). All variances are invariant
        # to the border column's sign (similarity by diag(1, ..., 1, -1))
        # so only the amplitude-theta cross terms distinguish J from -J;
        # they must follow the paper's sign.
        t = _time_axis()
        y, sigma = _noisy_data(t, theta=float(np.log(1.0)), noise=2.0)
        design, d_design = _builders(t)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
            absolute_sigma=True,
        )
        a_w = np.asarray(design(fit.theta)) / sigma[:, np.newaxis]
        _, jac = _jacobian_pieces(t, y, sigma, fit.theta)
        cov_j = _bordered_covariance(a_w, jac, fit.scale_sq)
        np.testing.assert_allclose(fit.covariance, cov_j, rtol=0.0, atol=1e-14)
        cov_neg = _bordered_covariance(a_w, -jac, fit.scale_sq)
        np.testing.assert_allclose(np.diag(cov_neg), np.diag(cov_j), rtol=1e-12)
        np.testing.assert_allclose(cov_neg[:-1, -1], -cov_j[:-1, -1], rtol=1e-12)

    def test_covariance_symmetric_positive_definite(self) -> None:
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
        )
        cov = fit.covariance
        assert cov.shape == (COEFFS_TRUE.size + 1, COEFFS_TRUE.size + 1)
        np.testing.assert_array_equal(cov, cov.T)  # symmetrized exactly
        assert float(np.min(np.linalg.eigvalsh(cov))) > 0.0

    def test_sigma_scaling_invariance(self) -> None:
        # Property: doubling all sigma (absolute_sigma=True) leaves theta
        # untouched, quarters chi-square and quadruples the covariance —
        # the plain Aitken scaling.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        kwargs: dict[str, Any] = {
            "theta_bounds": BOUNDS,
            "d_design": d_design,
            "absolute_sigma": True,
        }
        fit1 = estimate_varpro(design, y, sigma=sigma, **kwargs)
        fit2 = estimate_varpro(design, y, sigma=2.0 * sigma, **kwargs)
        assert abs(fit1.theta - fit2.theta) < 1e-9
        assert fit2.chisq == pytest.approx(fit1.chisq / 4.0, rel=1e-9)
        np.testing.assert_allclose(fit2.covariance, 4.0 * fit1.covariance, rtol=1e-6)


class TestProfileInterval:
    def test_near_symmetric_in_log_tau_when_identified(self) -> None:
        # Well-identified regime (T_post = 8·tau, amplitude SNR >> 5):
        # the Delta-chi2 = 1 interval in ln tau is near-symmetric — the
        # regime-dependence the spec records (and what ratifies searching
        # in ln tau). Both sides must close inside the bounds.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
            absolute_sigma=True,
        )
        assert not fit.interval_open_lower
        assert not fit.interval_open_upper
        lower, upper = fit.theta_interval
        assert lower < fit.theta < upper
        half_lo = fit.theta - lower
        half_up = upper - fit.theta
        assert 0.7 < half_up / half_lo < 1.4
        # In this regime the interval half-width agrees with the bordered
        # sigma to first order (Venzon & Moolgavkar's quadratic limit).
        assert 0.5 * (half_lo + half_up) == pytest.approx(fit.theta_sigma, rel=0.25)

    def test_bound_hit_reports_one_sided_and_warns(self) -> None:
        # Optimum clamped to the upper edge of theta_bounds: the upper
        # side cannot close, so the fit reports "lower bound only" (the
        # noise.py kappa-edge convention) and warns the operator to
        # publish a bound, not an estimate.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)
        capped = (-4.0, THETA_TRUE - 1.0)
        with pytest.warns(UserWarning, match="one-sided bound"):
            fit = estimate_varpro(
                design,
                y,
                theta_bounds=capped,
                sigma=sigma,
                d_design=d_design,
                absolute_sigma=True,
            )
        assert fit.theta == capped[1]
        assert fit.interval_open_upper
        assert not fit.interval_open_lower
        assert fit.theta_interval[1] == capped[1]
        assert fit.theta_interval[0] < fit.theta


class TestSearch:
    def test_failed_polish_never_worsens_grid_best(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The noise.py:534 guarantee, reproduced: the polish result is
        # accepted only if it strictly improves on the coarse grid, so a
        # pathological optimizer return cannot degrade the estimate.
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, d_design = _builders(t)

        class _FailedPolish:
            fun = np.inf
            x = -999.0  # nonsense the estimator must ignore

        def _fake_minimize_scalar(*args: Any, **kwargs: Any) -> _FailedPolish:
            return _FailedPolish()

        monkeypatch.setattr(varpro.optimize, "minimize_scalar", _fake_minimize_scalar)
        fit = estimate_varpro(
            design,
            y,
            theta_bounds=BOUNDS,
            sigma=sigma,
            d_design=d_design,
            absolute_sigma=True,
        )
        grid = np.linspace(BOUNDS[0], BOUNDS[1], varpro._N_GRID_DEFAULT)
        assert float(np.min(np.abs(grid - fit.theta))) < 1e-12
        spacing = (BOUNDS[1] - BOUNDS[0]) / (varpro._N_GRID_DEFAULT - 1)
        assert abs(fit.theta - THETA_TRUE) <= spacing


class TestDegeneracyGuards:
    def test_inverted_theta_bounds(self) -> None:
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, _ = _builders(t)
        with pytest.raises(ValueError, match="theta_bounds"):
            estimate_varpro(design, y, theta_bounds=(1.0, -1.0), sigma=sigma)

    def test_fewer_observations_than_parameters(self) -> None:
        t = _time_axis(n=4)
        design, _ = _builders(t)
        y = design(THETA_TRUE) @ COEFFS_TRUE
        with pytest.raises(ValueError, match="observations"):
            estimate_varpro(design, y, theta_bounds=BOUNDS)

    def test_rank_deficient_linear_design(self) -> None:
        # Duplicated column: the amplitudes are not identifiable, and the
        # estimator raises (the noise._profile_fit precedent for
        # propagating _wls_solve's failure convention) instead of silently
        # returning the minimum-norm solution.
        t = _time_axis()
        rng = np.random.default_rng(7)
        y = 1.0 + 2.0 * t + rng.normal(0.0, 0.5, t.size)

        def degenerate(_theta: float) -> FloatArray:
            return np.column_stack((np.ones_like(t), t, t))

        with pytest.raises(ValueError, match="rank deficient"):
            estimate_varpro(degenerate, y, theta_bounds=BOUNDS)

    def test_theta_independent_design_not_identifiable(self) -> None:
        # D = 0 (the design never moves with theta): the bordered matrix
        # is singular and theta has no covariance — informative raise.
        t = _time_axis()
        rng = np.random.default_rng(8)
        y = 1.0 + 2.0 * t + rng.normal(0.0, 0.5, t.size)

        def fixed(_theta: float) -> FloatArray:
            return np.column_stack((np.ones_like(t), t, t**2))

        with pytest.raises(ValueError, match="not identifiable"):
            estimate_varpro(fixed, y, theta_bounds=BOUNDS)

    def test_zero_residual_without_absolute_sigma(self) -> None:
        # chi2 = 0 exactly => no s² is estimable; the estimator refuses
        # rather than returning a zero covariance (noise.py precedent).
        t = _time_axis()
        design, _ = _builders(t)
        with pytest.raises(ValueError, match="scale"):
            estimate_varpro(design, np.zeros(t.size), theta_bounds=BOUNDS)

    def test_nonfinite_y(self) -> None:
        t = _time_axis()
        design, _ = _builders(t)
        y = design(THETA_TRUE) @ COEFFS_TRUE
        y[10] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimate_varpro(design, y, theta_bounds=BOUNDS)

    def test_nonpositive_sigma(self) -> None:
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, _ = _builders(t)
        sigma[0] = 0.0
        with pytest.raises(ValueError, match="positive"):
            estimate_varpro(design, y, theta_bounds=BOUNDS, sigma=sigma)

    def test_design_wrong_shape(self) -> None:
        t = _time_axis()
        y, _ = _noisy_data(t)

        def bad(_theta: float) -> FloatArray:
            return np.ones_like(t)  # 1-D, not (N, P)

        with pytest.raises(ValueError, match="shape"):
            estimate_varpro(bad, y, theta_bounds=BOUNDS)

    def test_d_design_wrong_shape(self) -> None:
        t = _time_axis()
        y, sigma = _noisy_data(t)
        design, _ = _builders(t)

        def bad_d(_theta: float) -> FloatArray:
            return np.zeros((t.size, 2))

        with pytest.raises(ValueError, match="d_design"):
            estimate_varpro(design, y, theta_bounds=BOUNDS, sigma=sigma, d_design=bad_d)

    def test_n_grid_too_small(self) -> None:
        t = _time_axis()
        y, _ = _noisy_data(t)
        design, _ = _builders(t)
        with pytest.raises(ValueError, match="n_grid"):
            estimate_varpro(design, y, theta_bounds=BOUNDS, n_grid=1)

    def test_bordered_covariance_guard_direct(self) -> None:
        # Unit-level: a border column inside the design span is refused.
        rng = np.random.default_rng(9)
        a_w = np.column_stack((np.ones(50), np.linspace(0, 1, 50)))
        del rng
        m = np.asarray(a_w[:, 0] + a_w[:, 1], dtype=np.float64)
        with pytest.raises(ValueError, match="not identifiable"):
            _bordered_covariance(a_w, m, 1.0)
