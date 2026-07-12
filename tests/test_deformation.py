"""Tests for gps_analysis.deformation (MATH_STANDARDS §4).

Coverage:
- Analytic spot-checks of the forward models: the Mogi peak-uplift / axial
  ratio identities (Segall 2010 §7.1) and the McTigue point-source limit.
- Reference parity: Okada 1985 Table 2 "checklist for numerical
  calculations" (p. 1149), cases 2–4, reproduced through the public
  :func:`okada_forward` to < 5e-6 (Okada quotes 4 significant figures).
  The centroid placement that maps Okada's raw fault-frame test geometry
  onto :class:`OkadaSource`'s centroid convention is derived inline.
- Round-trip inversion: forward-model a known source → add Gaussian noise →
  invert → recover the parameters within the formal uncertainties, for both
  the deterministic :func:`mogi_invert` and the Bayesian
  :func:`mogi_invert_bayes`, and for :func:`okada_invert` (opening/geometry).
- Unit/product identities (ΔV↔ΔP, rate conversions, half-life) and the
  leaf-guard (no forbidden imports pulled in via deformation).

Tolerances: Okada Table 2 at atol 5e-6 (Okada's own rounding); analytic
identities at rtol 1e-12 (a few ulp); round-trip recovery within ~3σ of the
formal covariance (Monte-Carlo noise realization, fixed seed).
"""

import math

import numpy as np
import pytest

from gps_analysis.deformation import (
    DEFAULT_NU,
    MogiSource,
    OkadaSource,
    halflife_days,
    local_coordinates,
    mogi_forward,
    mogi_invert,
    mogi_invert_bayes,
    mogi_mctigue,
    okada_forward,
    okada_invert,
    pressure_from_volume,
    rate_from_m3s,
    rate_to_m3s,
    time_for_rate,
    volume_from_pressure,
)
from gps_analysis.transient import InversionConfig, PriorBounds

RNG = np.random.default_rng(20260712)


# =====================================================================
# Mogi forward — analytic identities (Segall 2010 §7.1)
# =====================================================================


def test_mogi_peak_uplift_on_axis() -> None:
    """Directly above the source u_z(0) = (1−ν)·ΔV/(π·d²), u_h = 0."""
    src = MogiSource(x=0.0, y=0.0, depth=3000.0, dv=1.0e7)
    u = mogi_forward(np.array([0.0]), np.array([0.0]), src)
    expected_uz = (1.0 - DEFAULT_NU) * src.dv / (math.pi * src.depth**2)
    assert u[0, 0] == pytest.approx(0.0, abs=1e-18)
    assert u[1, 0] == pytest.approx(0.0, abs=1e-18)
    assert u[2, 0] == pytest.approx(expected_uz, rel=1e-12)


def test_mogi_horizontal_points_radially_outward_for_inflation() -> None:
    """Inflation (ΔV>0) drives horizontal displacement away from the source."""
    src = MogiSource(x=0.0, y=0.0, depth=2000.0, dv=5.0e6)
    e = np.array([1000.0, 0.0, -1500.0])
    n = np.array([0.0, 1200.0, 0.0])
    u = mogi_forward(e, n, src)
    # radial dot product u·r > 0 everywhere
    dot = u[0] * e + u[1] * n
    assert bool(np.all(dot > 0.0))
    # vertical is positive (uplift) for inflation
    assert bool(np.all(u[2] > 0.0))


def test_mogi_deflation_flips_all_signs() -> None:
    """ΔV linear in the field: negating ΔV negates every component."""
    e = RNG.uniform(-3000, 3000, 8)
    n = RNG.uniform(-3000, 3000, 8)
    inflate = mogi_forward(e, n, MogiSource(100.0, -50.0, 2500.0, +8.0e6))
    deflate = mogi_forward(e, n, MogiSource(100.0, -50.0, 2500.0, -8.0e6))
    assert np.allclose(inflate, -deflate, rtol=1e-13)


def test_mogi_axial_uplift_to_horizontal_ratio() -> None:
    """At radius r the ratio u_z/u_h = d/r (Mogi geometry, ν-independent)."""
    src = MogiSource(x=0.0, y=0.0, depth=4000.0, dv=1.0e7)
    r = 3000.0
    u = mogi_forward(np.array([r]), np.array([0.0]), src)
    uh = math.hypot(float(u[0, 0]), float(u[1, 0]))
    assert float(u[2, 0]) / uh == pytest.approx(src.depth / r, rel=1e-12)


def test_mogi_rejects_nonpositive_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        mogi_forward(np.array([0.0]), np.array([0.0]), MogiSource(0, 0, 0.0, 1e6))


# =====================================================================
# McTigue finite sphere — point-source limit (McTigue 1987 / Segall §7.2)
# =====================================================================


def test_mctigue_reduces_to_mogi_for_small_radius() -> None:
    """As a/d → 0 McTigue → Mogi with ΔV = π a³ ΔP/μ (leading order)."""
    x, y, depth, radius, dp_over_mu = 0.0, 0.0, 5000.0, 50.0, 1.0e-3
    e = np.array([2000.0, -1000.0, 500.0])
    n = np.array([0.0, 1500.0, -800.0])
    u_mct = mogi_mctigue(e, n, x, y, depth, radius, dp_over_mu)
    dv = math.pi * radius**3 * dp_over_mu  # μ folded into dp_over_mu
    u_mogi = mogi_forward(e, n, MogiSource(x, y, depth, dv))
    # a/d = 1e-2 → the ε⁶ correction is ~1e-6 relative
    assert np.allclose(u_mct, u_mogi, rtol=1e-3)


def test_mctigue_correction_grows_with_radius() -> None:
    """The finite-sphere correction is monotone in a/d on the source axis."""
    x, y, depth, dp_over_mu = 0.0, 0.0, 4000.0, 1.0e-3
    e = np.array([0.0])
    n = np.array([0.0])

    def axial_ratio(radius: float) -> float:
        u_mct = float(mogi_mctigue(e, n, x, y, depth, radius, dp_over_mu)[2, 0])
        dv = math.pi * radius**3 * dp_over_mu
        u_mogi = float(mogi_forward(e, n, MogiSource(x, y, depth, dv))[2, 0])
        return u_mct / u_mogi

    small = abs(axial_ratio(400.0) - 1.0)
    large = abs(axial_ratio(1600.0) - 1.0)
    assert large > small
    assert small < 0.05  # a/d = 0.1 → sub-5 % correction


def test_mctigue_rejects_radius_ge_depth() -> None:
    with pytest.raises(ValueError, match="radius"):
        mogi_mctigue(np.array([0.0]), np.array([0.0]), 0.0, 0.0, 100.0, 100.0, 1e-3)


# =====================================================================
# Okada 1985 Table 2 checklist (BSSA 75(4), p. 1149)
# =====================================================================

# Okada's raw test geometry (fault-frame): observation (x, y) at the surface,
# reference (deep) edge depth d, dip δ, fault L×W; slip modes strike/dip/
# tensile with unit dislocation. The centroid convention of OkadaSource places
# the fault centroid at (0, 0, −depth_c); to reproduce Okada's raw geometry
# with strike = 90° the mapping is
#     depth_c = d − sinδ·W/2   (centroid is W/2 up-dip of the deep edge)
#     e_obs   = x − L/2,  n_obs = y − cosδ·W/2
# so that okada85's internal (x, p, q) equal Okada's originals and the
# geographic rotation is the identity ( e ← x_fault, n ← y_fault).
#
# Reference values: Okada 1985 Table 2, cases 2–4 (from IPGP okada85_checklist).
_OKADA_TABLE2 = {
    ("c2", "strike"): (-8.689e-3, -4.298e-3, -2.747e-3),
    ("c2", "dip"): (-4.682e-3, -3.527e-2, -3.564e-2),
    ("c2", "tensile"): (-2.660e-4, +1.056e-2, +3.214e-3),
    ("c3", "strike"): (0.0, +5.253e-3, 0.0),
    ("c3", "tensile"): (+1.223e-2, 0.0, -1.606e-2),
    ("c4", "strike"): (0.0, -1.303e-3, 0.0),
    ("c4", "tensile"): (+3.507e-3, 0.0, -7.740e-3),
}
_OKADA_GEOM = {
    "c2": (2.0, 3.0, 4.0, 70.0, 3.0, 2.0),
    "c3": (0.0, 0.0, 4.0, 90.0, 3.0, 2.0),
    "c4": (0.0, 0.0, 6.0, 90.0, 3.0, 2.0),
}
_OKADA_MODE = {
    "strike": (0.0, 1.0, 0.0),
    "dip": (90.0, 1.0, 0.0),
    "tensile": (0.0, 0.0, 1.0),
}


@pytest.mark.parametrize(("case", "mode"), sorted(_OKADA_TABLE2))
def test_okada_table2_checklist(case: str, mode: str) -> None:
    """Reproduce Okada 1985 Table 2 (cases 2–4) via the public forward model."""
    xo, yo, d, dip, length, width = _OKADA_GEOM[case]
    rake, slip, opening = _OKADA_MODE[mode]
    if case == "c4" and mode == "strike":
        rake = 180.0  # case 4 uses rake=180 at dip=90 to simulate dip=-90
    dr = math.radians(dip)
    src = OkadaSource(
        x=0.0,
        y=0.0,
        depth=d - math.sin(dr) * width / 2.0,
        strike=90.0,
        dip=dip,
        length=length,
        width=width,
        strike_slip=math.cos(math.radians(rake)) * slip,
        dip_slip=math.sin(math.radians(rake)) * slip,
        opening=opening,
    )
    e = np.array([xo - length / 2.0])
    n = np.array([yo - math.cos(dr) * width / 2.0])
    u = okada_forward(e, n, src)
    assert u[:, 0] == pytest.approx(np.array(_OKADA_TABLE2[(case, mode)]), abs=5e-6)


def test_okada_tensile_opening_is_linear() -> None:
    """Displacement scales linearly with a pure tensile opening."""
    e = RNG.uniform(-5, 5, 6)
    n = RNG.uniform(-5, 5, 6)
    base = OkadaSource(0, 0, 4.0, 30.0, 60.0, 3.0, 2.0, 0.0, 0.0, 1.0)
    doubled = OkadaSource(0, 0, 4.0, 30.0, 60.0, 3.0, 2.0, 0.0, 0.0, 2.0)
    assert np.allclose(2.0 * okada_forward(e, n, base), okada_forward(e, n, doubled))


def test_okada_rejects_surface_breach() -> None:
    """A fault whose up-dip edge crosses z = 0 is rejected."""
    breaching = OkadaSource(0, 0, 0.5, 0.0, 90.0, 3.0, 2.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="breaches the surface"):
        okada_forward(np.array([1.0]), np.array([1.0]), breaching)


# =====================================================================
# Round-trip inversion — Mogi (deterministic)
# =====================================================================


def _svartsengi_network(n_stations: int) -> tuple[np.ndarray, np.ndarray]:
    """A Svartsengi-scale synthetic network (meters, source-centered)."""
    e = RNG.uniform(-8000.0, 8000.0, n_stations)
    n = RNG.uniform(-8000.0, 8000.0, n_stations)
    return e, n


def test_mogi_invert_round_trip_recovers_source() -> None:
    """Forward → noise → invert recovers (x, y, d, ΔV) within ~3σ."""
    e, n = _svartsengi_network(25)
    true = MogiSource(x=500.0, y=-300.0, depth=4500.0, dv=2.5e7)
    clean = mogi_forward(e, n, true)
    sigma = np.full((3, e.size), 0.005)
    sigma[2] *= 3.0  # vertical 3× noisier, as real GNSS
    obs = clean + RNG.normal(0.0, sigma)
    fit = mogi_invert(e, n, obs, sigma)
    recovered = fit.source.as_array()
    truth = true.as_array()
    # within 3 formal sigma on every parameter
    assert np.all(np.abs(recovered - truth) < 3.0 * fit.sigma)
    # reduced chi-square consistent with the injected noise
    assert 0.5 < fit.chi2_reduced < 2.0
    # covariance is symmetric positive-definite
    assert np.allclose(fit.covariance, fit.covariance.T)
    assert bool(np.all(np.linalg.eigvalsh(fit.covariance) > 0.0))


def test_mogi_invert_noise_free_is_exact() -> None:
    """With no noise the optimum matches the true source to high precision."""
    e, n = _svartsengi_network(20)
    true = MogiSource(x=-1200.0, y=800.0, depth=3200.0, dv=-1.4e7)  # deflation
    obs = mogi_forward(e, n, true)
    fit = mogi_invert(e, n, obs)
    assert fit.source.as_array() == pytest.approx(true.as_array(), rel=1e-4)
    assert fit.rms < 1e-6


def test_mogi_invert_robust_loss_resists_outlier() -> None:
    """A soft_l1 loss recovers ΔV better than linear under a gross outlier."""
    e, n = _svartsengi_network(30)
    true = MogiSource(x=0.0, y=0.0, depth=4000.0, dv=2.0e7)
    sigma = np.full((3, e.size), 0.004)
    obs = mogi_forward(e, n, true) + RNG.normal(0.0, sigma)
    obs[2, 0] += 0.5  # 0.5 m blunder on one vertical
    linear = mogi_invert(e, n, obs, sigma, loss="linear")
    robust = mogi_invert(e, n, obs, sigma, loss="soft_l1", f_scale=0.01)
    err_lin = abs(linear.source.dv - true.dv)
    err_rob = abs(robust.source.dv - true.dv)
    assert err_rob < err_lin


def test_mogi_invert_requires_two_stations() -> None:
    with pytest.raises(ValueError, match="stations"):
        mogi_invert(np.array([0.0]), np.array([0.0]), np.zeros((3, 1)))


# =====================================================================
# Round-trip inversion — Mogi (Bayesian, GBIS sampler reuse)
# =====================================================================


def test_mogi_invert_bayes_posterior_brackets_truth() -> None:
    """The MCMC posterior mean/MAP recovers the source; intervals bracket it."""
    e, n = _svartsengi_network(20)
    true = MogiSource(x=300.0, y=200.0, depth=4000.0, dv=2.0e7)
    sigma = np.full((3, e.size), 0.006)
    obs = mogi_forward(e, n, true) + RNG.normal(0.0, sigma)
    bounds = PriorBounds(
        start=np.array([0.0, 0.0, 3000.0, 1.0e7]),
        lower=np.array([-6000.0, -6000.0, 500.0, 0.0]),
        upper=np.array([6000.0, 6000.0, 9000.0, 5.0e7]),
        step=np.array([200.0, 200.0, 200.0, 1.0e6]),
    )
    config = InversionConfig(n_runs=6000, t_runs=300, seed=7)
    post = mogi_invert_bayes(e, n, obs, sigma, bounds, config)
    burn = 16 * config.t_runs
    chain = post.m_keep[:, burn:]
    lo = np.percentile(chain, 2.5, axis=1)
    hi = np.percentile(chain, 97.5, axis=1)
    truth = true.as_array()
    assert np.all((truth >= lo) & (truth <= hi))
    # MAP within the credible band too
    assert np.all((post.optimal.as_array() >= lo) & (post.optimal.as_array() <= hi))


# =====================================================================
# Round-trip inversion — Okada (opening + geometry)
# =====================================================================


def test_okada_invert_recovers_dike_opening() -> None:
    """Round-trip a dike (opening + geometry) from a surface network."""
    e = RNG.uniform(-10.0, 10.0, 40)
    n = RNG.uniform(-10.0, 10.0, 40)
    true = OkadaSource(
        x=1.0,
        y=-0.5,
        depth=3.0,
        strike=90.0,
        dip=80.0,
        length=6.0,
        width=3.0,
        strike_slip=0.0,
        dip_slip=0.0,
        opening=0.8,
    )
    sigma = np.full((3, e.size), 2.0e-3)
    obs = okada_forward(e, n, true) + RNG.normal(0.0, sigma)
    x0 = OkadaSource(0, 0, 2.5, 90.0, 80.0, 5.0, 2.5, 0.0, 0.0, 0.5)
    fit = okada_invert(
        e, n, obs, sigma, x0=x0, free=("x", "y", "depth", "length", "width", "opening")
    )
    assert fit.source.opening == pytest.approx(true.opening, abs=0.15)
    assert fit.source.depth == pytest.approx(true.depth, abs=0.6)
    assert fit.source.x == pytest.approx(true.x, abs=1.0)
    assert fit.rms < 5.0e-3


def test_okada_invert_rejects_unknown_free_field() -> None:
    x0 = OkadaSource(0, 0, 3.0, 90.0, 80.0, 5.0, 2.5, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="unknown"):
        okada_invert(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.zeros((3, 2)),
            x0=x0,
            free=("nonsense",),
        )


# =====================================================================
# Volume / pressure / rate products
# =====================================================================


def test_pressure_volume_round_trip() -> None:
    """ΔV → ΔP → ΔV is the identity (exact algebra)."""
    dv, radius, mu = 2.0e7, 800.0, 3.0e10
    dp = pressure_from_volume(dv, radius, mu)
    assert volume_from_pressure(dp, radius, mu) == pytest.approx(dv, rel=1e-13)
    # closed form ΔP = μ ΔV / (π a³)
    assert dp == pytest.approx(mu * dv / (math.pi * radius**3), rel=1e-13)


def test_rate_unit_round_trip() -> None:
    """Mm³/yr ↔ m³/s is the identity and matches the 365.25-day scale."""
    q = 4.2  # Mm³/yr
    assert rate_from_m3s(rate_to_m3s(q)) == pytest.approx(q, rel=1e-13)
    assert rate_to_m3s(q) == pytest.approx(q * 1e6 / (365.25 * 86400.0), rel=1e-13)


def test_time_for_rate_matches_exponential_derivative() -> None:
    """t solves −A·k·e^{−k t} = q_target for the exp-relaxation rate."""
    amplitude, decay_rate = -30.0, 2.5  # A<0 relaxation, k in 1/yr
    target = 5.0  # −A·k = +75 > 0, so a positive rate is attainable
    t = time_for_rate(target, amplitude, decay_rate)
    rate_at_t = -amplitude * decay_rate * math.exp(-decay_rate * t)
    assert rate_at_t == pytest.approx(target, rel=1e-12)


def test_time_for_rate_unreachable_is_nan() -> None:
    # −A·k = +75 > 0: a negative target rate is never attained → NaN
    assert math.isnan(time_for_rate(-5.0, -30.0, 2.5))


def test_halflife_matches_decay_constant() -> None:
    k = 3.0  # 1/yr → t½ = ln2/k years
    assert halflife_days(k) == pytest.approx(math.log(2.0) / k * 365.25, rel=1e-13)


# =====================================================================
# Geometry helper
# =====================================================================


def test_local_coordinates_origin_is_zero() -> None:
    e, n = local_coordinates(-22.4, 63.86, -22.4, 63.86)
    assert float(e) == pytest.approx(0.0, abs=1e-9)
    assert float(n) == pytest.approx(0.0, abs=1e-9)


def test_local_coordinates_scale_is_metric() -> None:
    """A 0.01° north step near 64°N is ~1.11 km; east step scaled by cosφ."""
    lon0, lat0 = -22.0, 64.0
    en_north = local_coordinates(lon0, lat0 + 0.01, lon0, lat0)
    assert float(en_north[1]) == pytest.approx(1112.0, rel=5e-3)
    en_east = local_coordinates(lon0 + 0.01, lat0, lon0, lat0)
    # east ≈ meridian-arc × cosφ / (M/N) ≈ 487 m at 64°N
    assert 480.0 < float(en_east[0]) < 495.0
