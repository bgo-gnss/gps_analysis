"""Tests for gps_analysis.joint (MATH_STANDARDS §4).

Coverage:
- LOS geometry: unit norm, Sentinel-1 ascending/descending sign checks
  (Hanssen 2001 ch. 2; Fialko et al. 2001 eq. 1), the binding sign
  convention (positive = motion toward the satellite / range decrease),
  projection identities (vertical → cos θ, orthogonal → 0, linearity).
- Nuisance model: ramp design-matrix shapes and mean-centering.
- VCE: Helmert variance components on an analytic two-group problem
  (understated σ → component ≈ ratio²); partial-redundancy guard.
- Joint inversion: exact noiseless round-trip (source + offset + ramp),
  noisy recovery within the formal covariance, dense-covariance whitening
  equivalence, VCE reweighting behavior.
- THE EXIT GATE — the depth–ΔV trade-off break (plan §9b): on synthetic
  Svartsengi-like data the joint GPS+InSAR fit must shrink σ_depth and the
  depth–ΔV covariance markedly vs GPS-only, and a second viewing geometry
  (ascending + descending, Wright et al. 2004) must recover the full source
  tightly.

Tolerances: analytic identities at rtol 1e-12 (a few ulp); noiseless
round-trips at rtol 1e-6 (optimizer convergence tolerance); statistical
checks (VCE components, noisy recovery) at fixed seed within ranges wide
enough for the sampling error of the finite synthetic data.
"""

import math

import numpy as np
import pytest

from gps_analysis.deformation import (
    MogiSource,
    mogi_forward,
    mogi_invert,
)
from gps_analysis.joint import (
    InsarLos,
    los_project,
    los_unit_vector,
    mogi_invert_joint,
    param_correlation,
    ramp_design,
    variance_components,
)

RNG = np.random.default_rng(20260712)


# =====================================================================
# LOS unit vector (Hanssen 2001 ch. 2; Fialko et al. 2001 eq. 1)
# =====================================================================


def test_los_unit_vector_is_unit_norm() -> None:
    inc = np.array([0.0, 23.0, 35.0, 46.0, 89.0])
    head = np.array([192.0, 348.0, 0.0, 90.0, 270.0])
    u = los_unit_vector(inc, head)
    assert u.shape == (3, 5)
    assert np.allclose(np.sqrt((u * u).sum(axis=0)), 1.0, rtol=1e-14)


def test_los_unit_vector_zero_incidence_is_vertical() -> None:
    u = los_unit_vector(0.0, 123.0)
    assert u.shape == (3,)
    assert np.allclose(u, [0.0, 0.0, 1.0], atol=1e-16)


def test_los_unit_vector_descending_sentinel1_geometry() -> None:
    """Descending (heading ≈ 192°), right-looking: satellite sits ESE of the
    target — u_E > 0, u_N < 0 (slightly), u_U = cos θ."""
    theta, alpha = 35.0, 192.0
    u = los_unit_vector(theta, alpha)
    assert u[0] > 0.0
    assert u[1] < 0.0
    assert abs(u[0]) > abs(u[1])  # mostly east-looking
    assert u[2] == pytest.approx(math.cos(math.radians(theta)), rel=1e-14)
    # explicit formula check: u = (−sinθ cosα, sinθ sinα, cosθ)
    st = math.sin(math.radians(theta))
    assert u[0] == pytest.approx(-st * math.cos(math.radians(alpha)), rel=1e-14)
    assert u[1] == pytest.approx(st * math.sin(math.radians(alpha)), rel=1e-14)


def test_los_unit_vector_ascending_looks_from_west() -> None:
    """Ascending (heading ≈ 348°), right-looking: satellite west of target."""
    u = los_unit_vector(39.0, 348.0)
    assert u[0] < 0.0


def test_los_unit_vector_left_look_flips_horizontal() -> None:
    ur = los_unit_vector(35.0, 192.0, look="right")
    ul = los_unit_vector(35.0, 192.0, look="left")
    assert ul[0] == pytest.approx(-ur[0], rel=1e-14)
    assert ul[1] == pytest.approx(-ur[1], rel=1e-14)
    assert ul[2] == pytest.approx(ur[2], rel=1e-14)


def test_los_unit_vector_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="incidence"):
        los_unit_vector(90.0, 190.0)
    with pytest.raises(ValueError, match="incidence"):
        los_unit_vector(-1.0, 190.0)
    with pytest.raises(ValueError, match="look"):
        los_unit_vector(35.0, 190.0, look="up")  # type: ignore[arg-type]


# =====================================================================
# LOS projection — d_LOS = u·d, positive toward the satellite
# =====================================================================


def test_los_project_pure_uplift_gives_cos_incidence() -> None:
    """Vertical motion projects as cos θ — and uplift is POSITIVE (toward
    the satellite / range decrease): the binding sign convention."""
    theta = 35.0
    u = los_unit_vector(theta, 192.0)
    d = np.array([[0.0, 0.0], [0.0, 0.0], [0.010, -0.010]])  # up, down
    dlos = los_project(d, u)
    expected = 0.010 * math.cos(math.radians(theta))
    assert dlos[0] == pytest.approx(expected, rel=1e-14)
    assert dlos[1] == pytest.approx(-expected, rel=1e-14)
    assert dlos[0] > 0.0  # uplift toward a right-looking SAR: range decrease


def test_los_project_orthogonal_motion_is_blind() -> None:
    """Motion perpendicular to the LOS is invisible (the 1-component nature
    of a single interferogram — Wright et al. 2004)."""
    u = los_unit_vector(35.0, 192.0)
    # along-track-ish horizontal direction orthogonal to u
    d_perp = np.cross(u, [0.0, 0.0, 1.0])
    d_perp /= np.linalg.norm(d_perp)
    assert float(los_project(d_perp[:, None], u)[0]) == pytest.approx(0.0, abs=1e-15)


def test_los_project_is_linear_and_broadcasts() -> None:
    u_shared = los_unit_vector(35.0, 192.0)
    d1 = RNG.standard_normal((3, 7))
    d2 = RNG.standard_normal((3, 7))
    out = los_project(2.0 * d1 - 3.0 * d2, u_shared)
    assert out.shape == (7,)
    assert np.allclose(
        out, 2.0 * los_project(d1, u_shared) - 3.0 * los_project(d2, u_shared)
    )
    # per-point unit vectors
    u_per = los_unit_vector(np.full(7, 35.0), np.full(7, 192.0))
    assert np.allclose(los_project(d1, u_per), los_project(d1, u_shared))


def test_los_project_rejects_wrong_rows() -> None:
    with pytest.raises(ValueError, match="3 .E, N, U. rows"):
        los_project(np.zeros((2, 4)), np.array([0.0, 0.0, 1.0]))


# =====================================================================
# Ramp design matrix
# =====================================================================


def test_ramp_design_shapes_and_centering() -> None:
    e = RNG.uniform(-5000, 5000, 12)
    n = RNG.uniform(-5000, 5000, 12)
    assert ramp_design(e, n, "none").shape == (12, 0)
    a1 = ramp_design(e, n, "offset")
    assert a1.shape == (12, 1) and np.all(a1 == 1.0)
    a3 = ramp_design(e, n, "linear")
    assert a3.shape == (12, 3)
    assert np.all(a3[:, 0] == 1.0)
    # linear columns mean-centered (decorrelated from the offset)
    assert abs(float(a3[:, 1].mean())) < 1e-9
    assert abs(float(a3[:, 2].mean())) < 1e-9
    with pytest.raises(ValueError, match="ramp"):
        ramp_design(e, n, "quadratic")


# =====================================================================
# param_correlation
# =====================================================================


def test_param_correlation_reads_off_covariance() -> None:
    cov = np.array([[4.0, 3.0], [3.0, 9.0]])
    assert param_correlation(cov, 0, 1) == pytest.approx(3.0 / 6.0, rel=1e-15)
    assert param_correlation(cov, 0, 0) == pytest.approx(1.0, rel=1e-15)
    with pytest.raises(ValueError, match="positive"):
        param_correlation(np.array([[0.0, 0.0], [0.0, 1.0]]), 0, 1)
    with pytest.raises(ValueError, match="square"):
        param_correlation(np.zeros((2, 3)), 0, 1)


# =====================================================================
# InsarLos contract
# =====================================================================


def _small_track(**kw: object) -> InsarLos:
    n = 5
    base: dict[str, object] = {
        "e": np.arange(n, dtype=float),
        "n": np.zeros(n),
        "d_los": np.zeros(n),
        "los_unit": los_unit_vector(35.0, 192.0),
    }
    base.update(kw)
    return InsarLos(**base)  # type: ignore[arg-type]


def test_insarlos_broadcasts_shared_unit_vector() -> None:
    trk = _small_track()
    assert trk.los_unit.shape == (3, 5)
    assert trk.n_points == 5
    assert trk.n_nuisance == 1  # default "offset"
    assert _small_track(ramp="linear").n_nuisance == 3
    assert _small_track(ramp="none").n_nuisance == 0


def test_insarlos_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        _small_track(los_unit=np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="not both"):
        _small_track(sigma=np.ones(5), cov=np.eye(5))
    with pytest.raises(ValueError, match="strictly positive"):
        _small_track(sigma=np.zeros(5))
    with pytest.raises(ValueError, match="symmetric"):
        cov = np.eye(5)
        cov[0, 1] = 0.5
        _small_track(cov=cov)
    with pytest.raises(ValueError, match="equal-length"):
        _small_track(d_los=np.zeros(4))
    with pytest.raises(ValueError, match="ramp"):
        _small_track(ramp="cubic")
    with pytest.raises(ValueError, match="finite"):
        _small_track(d_los=np.array([0.0, np.nan, 0.0, 0.0, 0.0]))


# =====================================================================
# Variance components (Helmert VCE — Koch 1999 ch. 3;
# Sudhaus & Jónsson 2009)
# =====================================================================


def test_variance_components_detect_understated_group_noise() -> None:
    """Two groups measure one constant; group 2's true noise is 2× its
    stated σ ⇒ its variance component ≈ 4, group 1's ≈ 1."""
    rng = np.random.default_rng(7)
    n1, n2 = 400, 400
    truth = 5.0
    y1 = truth + rng.standard_normal(n1)  # σ = 1, honest
    y2 = truth + 2.0 * rng.standard_normal(n2)  # σ stated 1, true 2
    # whitened (σ_stated = 1) linear model y = a: solve jointly
    a_hat = float(np.concatenate((y1, y2)).mean())
    r1, r2 = y1 - a_hat, y2 - a_hat
    j1 = np.ones((n1, 1))
    j2 = np.ones((n2, 1))
    comp = variance_components([r1, r2], [j1, j2])
    assert comp[0] == pytest.approx(1.0, abs=0.25)
    assert comp[1] == pytest.approx(4.0, rel=0.25)


def test_variance_components_redundancy_guard() -> None:
    """A group whose observations are fully absorbed by the parameters
    (n_k = tr(N⁻¹N_k)) carries no redundancy — not estimable."""
    with pytest.raises(ValueError, match="redundancy"):
        variance_components([np.zeros(1)], [np.ones((1, 1))])


def test_variance_components_validates_blocks() -> None:
    with pytest.raises(ValueError, match=">= 1 group"):
        variance_components([], [])
    with pytest.raises(ValueError, match="group blocks"):
        variance_components([np.zeros(3)], [np.ones((4, 1))])


# =====================================================================
# Joint inversion — synthetic Svartsengi-like scenario
# =====================================================================

TRUE = MogiSource(x=500.0, y=-300.0, depth=4000.0, dv=2.0e6)

# Sparse near-field GNSS network (5 stations, 1.5–3 km from the source):
# realistic monitoring geometry, deliberately aperture-limited so the
# depth–ΔV trade-off is strong in the GPS-only fit.
_ANG = np.deg2rad(np.array([20.0, 95.0, 170.0, 250.0, 320.0]))
_RAD = np.array([1500.0, 2500.0, 2000.0, 3000.0, 1800.0])
GPS_E = TRUE.x + _RAD * np.sin(_ANG)
GPS_N = TRUE.y + _RAD * np.cos(_ANG)
GPS_SIGMA = np.vstack(
    (np.full(5, 0.003), np.full(5, 0.003), np.full(5, 0.008))
)  # 3 mm horizontal / 8 mm vertical — daily GNSS displacement level

# InSAR sampling: 20×20 grid over ±12 km (the leaf receives downsampled
# points — quadtree/covariance generation is the reader's job).
_G = np.linspace(-12000.0, 12000.0, 20)
_ME, _MN = np.meshgrid(_G, _G)
INS_E = _ME.ravel() + TRUE.x
INS_N = _MN.ravel() + TRUE.y
LOS_SIGMA = 0.004  # 4 mm downsampled-point noise
U_DESC = los_unit_vector(35.0, 192.0)  # Sentinel-1 descending
U_ASC = los_unit_vector(39.0, 348.0)  # Sentinel-1 ascending


def _gps_obs(rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(rng.standard_normal((3, 5)) * GPS_SIGMA, dtype=np.float64)
    return mogi_forward(GPS_E, GPS_N, TRUE) + noise


def _insar_track(
    rng: np.random.Generator | None,
    los_unit: np.ndarray,
    offset: float,
    sigma_stated: float = LOS_SIGMA,
    sigma_true: float = LOS_SIGMA,
    ramp_true: tuple[float, float] = (0.0, 0.0),
    ramp: str = "offset",
) -> InsarLos:
    d = los_project(mogi_forward(INS_E, INS_N, TRUE), los_unit) + offset
    d = (
        d
        + ramp_true[0] * (INS_E - INS_E.mean())
        + ramp_true[1] * (INS_N - INS_N.mean())
    )
    if rng is not None:
        d = d + rng.standard_normal(INS_E.size) * sigma_true
    return InsarLos(
        e=INS_E,
        n=INS_N,
        d_los=d,
        los_unit=los_unit,
        sigma=np.full(INS_E.size, sigma_stated),
        ramp=ramp,  # type: ignore[arg-type]
    )


def test_joint_noiseless_round_trip_is_exact() -> None:
    track = _insar_track(None, U_DESC, offset=0.005)
    fit = mogi_invert_joint(
        GPS_E,
        GPS_N,
        mogi_forward(GPS_E, GPS_N, TRUE),
        GPS_SIGMA,
        insar=[track],
        vce=False,
    )
    assert fit.source.x == pytest.approx(TRUE.x, abs=1e-3)
    assert fit.source.y == pytest.approx(TRUE.y, abs=1e-3)
    assert fit.source.depth == pytest.approx(TRUE.depth, rel=1e-6)
    assert fit.source.dv == pytest.approx(TRUE.dv, rel=1e-6)
    assert fit.nuisance[0][0] == pytest.approx(0.005, rel=1e-6)
    assert fit.rms_gps == pytest.approx(0.0, abs=1e-9)
    assert fit.rms_los[0] == pytest.approx(0.0, abs=1e-9)
    assert fit.param_names == ("x", "y", "depth", "dv", "los0_offset")


def test_joint_recovers_linear_ramp_nuisance() -> None:
    slope_e, slope_n = 2.0e-7, -1.5e-7  # 0.2 / −0.15 mm per km
    track = _insar_track(
        None, U_DESC, offset=0.003, ramp_true=(slope_e, slope_n), ramp="linear"
    )
    fit = mogi_invert_joint(
        GPS_E,
        GPS_N,
        mogi_forward(GPS_E, GPS_N, TRUE),
        GPS_SIGMA,
        insar=[track],
        vce=False,
    )
    assert fit.source.depth == pytest.approx(TRUE.depth, rel=1e-5)
    assert fit.source.dv == pytest.approx(TRUE.dv, rel=1e-5)
    offset, ramp_e, ramp_n = fit.nuisance[0]
    assert offset == pytest.approx(0.003, rel=1e-4)
    assert ramp_e == pytest.approx(slope_e, rel=1e-4)
    assert ramp_n == pytest.approx(slope_n, rel=1e-4)
    assert fit.param_names[4:] == ("los0_offset", "los0_ramp_e", "los0_ramp_n")


def test_joint_noisy_recovery_within_formal_errors() -> None:
    rng = np.random.default_rng(20260712)
    fit = mogi_invert_joint(
        GPS_E,
        GPS_N,
        _gps_obs(rng),
        GPS_SIGMA,
        insar=[_insar_track(rng, U_DESC, offset=0.006)],
    )
    truth = TRUE.as_array()
    est = fit.source.as_array()
    for i in range(4):
        assert abs(est[i] - truth[i]) < 3.5 * fit.sigma[i], fit.param_names[i]
    assert fit.chi2_reduced == pytest.approx(1.0, abs=0.35)


def test_joint_dense_covariance_matches_diagonal_sigma() -> None:
    """A diagonal dense cov must reproduce the per-point-σ path exactly
    (whitening equivalence)."""
    rng = np.random.default_rng(3)
    obs = _gps_obs(rng)
    d = los_project(mogi_forward(INS_E, INS_N, TRUE), U_DESC) + 0.004
    d = d + rng.standard_normal(INS_E.size) * LOS_SIGMA
    t_sig = InsarLos(
        e=INS_E,
        n=INS_N,
        d_los=d,
        los_unit=U_DESC,
        sigma=np.full(INS_E.size, LOS_SIGMA),
    )
    t_cov = InsarLos(
        e=INS_E,
        n=INS_N,
        d_los=d,
        los_unit=U_DESC,
        cov=np.eye(INS_E.size) * LOS_SIGMA**2,
    )
    f_sig = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[t_sig], vce=False)
    f_cov = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[t_cov], vce=False)
    assert np.allclose(f_sig.source.as_array(), f_cov.source.as_array(), rtol=1e-8)
    assert np.allclose(f_sig.covariance, f_cov.covariance, rtol=1e-6)


def test_joint_correlated_covariance_widens_uncertainty() -> None:
    """An exponential covariogram (correlated atmosphere — Lohman & Simons
    2005 §3) carries less independent information than the same variance
    taken diagonal: σ_depth must grow."""
    rng = np.random.default_rng(11)
    obs = _gps_obs(rng)
    dx = INS_E[:, None] - INS_E[None, :]
    dy = INS_N[:, None] - INS_N[None, :]
    dist = np.hypot(dx, dy)
    cov = LOS_SIGMA**2 * np.exp(-dist / 3000.0)  # 3 km e-folding
    d = los_project(mogi_forward(INS_E, INS_N, TRUE), U_DESC) + 0.004
    d = d + np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0])) @ (
        rng.standard_normal(INS_E.size)
    )
    t_corr = InsarLos(e=INS_E, n=INS_N, d_los=d, los_unit=U_DESC, cov=cov)
    t_diag = InsarLos(
        e=INS_E,
        n=INS_N,
        d_los=d,
        los_unit=U_DESC,
        sigma=np.full(INS_E.size, LOS_SIGMA),
    )
    f_corr = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[t_corr], vce=False)
    f_diag = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[t_diag], vce=False)
    assert f_corr.sigma[2] > f_diag.sigma[2]


def test_joint_vce_detects_understated_insar_sigma() -> None:
    """Stated LOS σ understated by 2× ⇒ that track's variance component ≈ 4
    and its influence is correspondingly down-weighted (Sudhaus & Jónsson
    2009: estimate the relative weight, never fix it)."""
    rng = np.random.default_rng(20260712)
    track = _insar_track(
        rng,
        U_DESC,
        offset=0.006,
        sigma_stated=LOS_SIGMA / 2.0,
        sigma_true=LOS_SIGMA,
    )
    fit = mogi_invert_joint(GPS_E, GPS_N, _gps_obs(rng), GPS_SIGMA, insar=[track])
    assert fit.variance_components[1] == pytest.approx(4.0, rel=0.35)
    assert fit.variance_components[0] == pytest.approx(1.0, abs=0.6)
    assert fit.vce_iterations >= 2
    # after reweighting the fit is again consistent
    assert fit.chi2_reduced == pytest.approx(1.0, abs=0.35)


# =====================================================================
# THE EXIT GATE — depth–ΔV trade-off break (plan §9b demonstration)
# =====================================================================


def test_joint_breaks_depth_dv_tradeoff_formal() -> None:
    """GPS-only vs joint GPS+InSAR on the same synthetic truth: the dense
    LOS field must collapse the depth uncertainty and the depth–ΔV
    covariance (the GPS-only degeneracy — VALIDATION_svartsengi §6).

    Money numbers, formal covariances at the pinned seed (20260712):
    σ_depth 523 m → 215 m (×0.41), cov(depth, ΔV) ×0.14,
    ρ(depth, ΔV) 0.972 → 0.917.
    """
    rng = np.random.default_rng(20260712)
    obs = _gps_obs(rng)
    fit_gps = mogi_invert(GPS_E, GPS_N, obs, GPS_SIGMA)
    fit_joint = mogi_invert_joint(
        GPS_E,
        GPS_N,
        obs,
        GPS_SIGMA,
        insar=[_insar_track(rng, U_DESC, offset=0.006)],
    )
    sd_gps = float(fit_gps.sigma[2])
    sd_joint = float(fit_joint.sigma[2])
    corr_gps = param_correlation(fit_gps.covariance, 2, 3)
    corr_joint = fit_joint.depth_dv_correlation
    cov_gps = float(fit_gps.covariance[2, 3])
    cov_joint = float(fit_joint.covariance[2, 3])

    # the trade-off is REAL in the GPS-only fit ...
    assert corr_gps > 0.95, f"GPS-only depth-dv corr {corr_gps:.3f}"
    # ... and the joint fit breaks it: σ_depth shrinks markedly,
    assert sd_joint < 0.5 * sd_gps, (
        f"sigma_depth joint {sd_joint:.0f} m vs GPS-only {sd_gps:.0f} m"
    )
    # the depth–ΔV covariance collapses,
    assert abs(cov_joint) < 0.20 * abs(cov_gps), (
        f"cov(depth,dv) joint {cov_joint:.3g} vs GPS-only {cov_gps:.3g}"
    )
    # and the correlation itself decreases.
    assert corr_joint < corr_gps - 0.04, (
        f"corr joint {corr_joint:.3f} vs GPS-only {corr_gps:.3f}"
    )
    # the joint solution is also closer to (and consistent with) the truth
    assert abs(fit_joint.source.depth - TRUE.depth) < 3.0 * sd_joint


def test_joint_breaks_depth_dv_tradeoff_ensemble() -> None:
    """Monte-Carlo version of the exit gate: the EMPIRICAL scatter of the
    estimator over 30 noise realizations (realization-independent, unlike a
    single formal covariance whose χ² scaling fluctuates with the 11-dof
    GPS-only sample).

    Money numbers (seed 20260712, 30 realizations): empirical σ_depth
    986 m (GPS-only) → 252 m (joint), ×0.26; both estimators unbiased
    (mean depth 4101 / 4003 m vs true 4000 m); ensemble
    ρ(deptĥ, ΔV̂) 0.95 → 0.93 (the residual correlation is the intrinsic
    single-LOS Mogi geometry — a second geometry lowers it further, see
    :func:`test_joint_two_geometries_near_full_3d`).
    """
    rng = np.random.default_rng(20260712)
    clean_gps = mogi_forward(GPS_E, GPS_N, TRUE)
    clean_los = los_project(mogi_forward(INS_E, INS_N, TRUE), U_DESC) + 0.006
    depth_gps, dv_gps, depth_joint, dv_joint = [], [], [], []
    for _ in range(30):
        obs = clean_gps + rng.standard_normal((3, 5)) * GPS_SIGMA
        d = clean_los + rng.standard_normal(INS_E.size) * LOS_SIGMA
        track = InsarLos(
            e=INS_E,
            n=INS_N,
            d_los=d,
            los_unit=U_DESC,
            sigma=np.full(INS_E.size, LOS_SIGMA),
        )
        fg = mogi_invert(GPS_E, GPS_N, obs, GPS_SIGMA)
        fj = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[track], vce=False)
        depth_gps.append(fg.source.depth)
        dv_gps.append(fg.source.dv)
        depth_joint.append(fj.source.depth)
        dv_joint.append(fj.source.dv)
    sd_gps = float(np.std(depth_gps, ddof=1))
    sd_joint = float(np.std(depth_joint, ddof=1))
    # the joint estimator's depth scatter collapses (the trade-off break) ...
    assert sd_joint < 0.45 * sd_gps, (
        f"empirical sigma_depth joint {sd_joint:.0f} m vs GPS-only {sd_gps:.0f} m"
    )
    # ... without bias (both means consistent with the truth)
    assert abs(float(np.mean(depth_joint)) - TRUE.depth) < 3.0 * sd_joint / math.sqrt(
        30.0
    )
    assert abs(float(np.mean(depth_gps)) - TRUE.depth) < 3.0 * sd_gps / math.sqrt(30.0)
    # the GPS-only estimate errors ride the depth–ΔV trade-off line
    corr_gps = float(np.corrcoef(depth_gps, dv_gps)[0, 1])
    corr_joint = float(np.corrcoef(depth_joint, dv_joint)[0, 1])
    assert corr_gps > 0.9, f"ensemble GPS-only corr {corr_gps:.3f}"
    assert corr_joint < corr_gps, (
        f"ensemble corr joint {corr_joint:.3f} vs GPS-only {corr_gps:.3f}"
    )


def test_joint_two_geometries_near_full_3d() -> None:
    """Ascending + descending LOS (Wright et al. 2004): the second viewing
    geometry tightens the source further — σ_depth below the single-track
    value and every parameter recovered within its (small) uncertainty."""
    rng = np.random.default_rng(20260712)
    obs = _gps_obs(rng)
    desc = _insar_track(rng, U_DESC, offset=0.006)
    asc = _insar_track(rng, U_ASC, offset=-0.004)
    fit_one = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[desc])
    fit_two = mogi_invert_joint(GPS_E, GPS_N, obs, GPS_SIGMA, insar=[asc, desc])
    assert fit_two.sigma[2] < fit_one.sigma[2]
    assert len(fit_two.nuisance) == 2
    truth = TRUE.as_array()
    est = fit_two.source.as_array()
    for i in range(4):
        assert abs(est[i] - truth[i]) < 3.5 * fit_two.sigma[i], fit_two.param_names[i]
    # depth pinned to a few percent by the dual-geometry joint data
    assert fit_two.sigma[2] < 0.06 * TRUE.depth


# =====================================================================
# Error handling
# =====================================================================


def test_joint_requires_at_least_one_track() -> None:
    with pytest.raises(ValueError, match="insar"):
        mogi_invert_joint(
            GPS_E, GPS_N, mogi_forward(GPS_E, GPS_N, TRUE), GPS_SIGMA, insar=[]
        )


def test_joint_validates_vce_controls() -> None:
    track = _insar_track(None, U_DESC, offset=0.0)
    obs = mogi_forward(GPS_E, GPS_N, TRUE)
    with pytest.raises(ValueError, match="vce_tol"):
        mogi_invert_joint(GPS_E, GPS_N, obs, insar=[track], vce_tol=0.0)
    with pytest.raises(ValueError, match="vce_max_iter"):
        mogi_invert_joint(GPS_E, GPS_N, obs, insar=[track], vce_max_iter=0)
    with pytest.raises(ValueError, match="length-4"):
        mogi_invert_joint(
            GPS_E,
            GPS_N,
            obs,
            insar=[track],
            bounds=(np.zeros(3), np.ones(3)),
        )


def test_joint_underdetermined_raises() -> None:
    """1 GPS station + 2-point track: 5 obs for 4 source + 1 nuisance."""
    u = los_unit_vector(35.0, 192.0)
    tiny = InsarLos(
        e=np.array([0.0, 100.0]),
        n=np.array([0.0, 0.0]),
        d_los=np.zeros(2),
        los_unit=u,
    )
    with pytest.raises(ValueError, match="underdetermined"):
        mogi_invert_joint(
            np.array([1000.0]),
            np.array([0.0]),
            np.zeros((3, 1)),
            None,
            insar=[tiny],
        )
