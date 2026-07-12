"""Deformation-source forward models and GPS-only inversion (Mogi / Okada).

Analytical volcano-deformation sources in a homogeneous, isotropic, elastic
half-space, and the nonlinear inversions that fit them to GNSS displacement
fields. This is the plan §10.2 ``deformation`` surface, GPS-only slice
(plan §9b revival): our OWN forward + inverse machinery, validated against —
but independent of — Vincent's operational Svartsengi Mogi procedure
(``insar.vedur.is:/mnt/scratch/vincent/model/svartsengi``, see plan §11).

Derivation chain
----------------
1. Geometry — :func:`local_coordinates` maps geodetic (lon, lat) to a local
   tangent-plane east/north frame [m] so the sources live in a Cartesian
   half-space (z up, surface at z = 0).
2. Forward models — :func:`mogi_forward` (point pressure/volume source;
   Mogi 1958, Segall 2010 §7.1), :func:`mogi_mctigue` (finite-sphere
   second-order correction; McTigue 1987), :func:`okada_forward`
   (rectangular dislocation — dike/sill/fault; Okada 1985 eqs. 25–30).
3. Inversion — :func:`mogi_invert` / :func:`okada_invert` (weighted
   nonlinear least squares, scipy trust-region-reflective, optional robust
   loss; formal covariance from the Jacobian) and :func:`mogi_invert_bayes`
   (Metropolis MCMC with the GBIS annealing/adaptive-step scheme shared
   with the transient lane via :mod:`gps_analysis._mcmc`; Bagnardi &
   Hooper 2018 §3).
4. Distributed slip — :func:`discretize_fault` tiles a fixed fault plane
   into patches, :func:`okada_greens` assembles the unit-slip
   Green's-function matrix ``d = G·s``, :func:`patch_laplacian` the
   roughness operator ∇², and :func:`okada_invert_slip` solves the
   Laplacian-regularized (optionally non-negative) linear inversion for the
   slip/opening distribution, with :func:`slip_lcurve` /
   :func:`lcurve_corner` for the λ trade-off (Okada 1985; Harris & Segall
   1987; Jónsson et al. 2002; Aster, Borchers & Thurber 2018 ch. 4;
   Hansen 1992).
5. Physical products — :func:`pressure_from_volume` /
   :func:`volume_from_pressure` (ΔV ↔ ΔP through the spherical-cavity
   relation ΔV = π a³ ΔP / μ; Segall 2010 §7.2), plus the volume-rate unit
   helpers :func:`rate_to_m3s` / :func:`rate_from_m3s` /
   :func:`time_for_rate` / :func:`halflife_days` ported from
   ``svartsengi_model.fitting`` (docs/CONSOLIDATION_MAP.md).

Conventions (binding, see ``docs/MATH_STANDARDS.md``)
-----------------------------------------------------
- Local Cartesian frame: x = east, y = north, z = up; the free surface is
  z = 0 and source depths are **positive down** [L].
- Displacement fields are ``(3, N)`` float64 arrays with rows ordered
  **(east, north, up)** — the deformation-literature ENU order. (The
  time-series lane's ``(3, N)`` N/E/U order does NOT apply here; the row
  order is stated on every function.)
- Unit discipline: coordinates, depths and source dimensions share one
  length unit [L]; volumes are [L³]; displacements come out in [L]. IMO
  production uses **meters** — convert mm-lane GNSS displacements to m at
  the call site (Vincent's operational inputs are meters, too).
- All functions are pure: no I/O, inputs never mutated; the data source is
  always a parameter (live GNSS arrays come from the analysis lane).

Reference implementations consulted (reimplemented, never copied):
``uafgeotools/vmod`` (Angarita et al., doi:10.5281/zenodo.10070627),
dMODELS (Battaglia, Cervelli & Murray 2013, JVGR 254), and F. Beauducel's
``okada85.m`` (IPGP deformation-lib, BSD) whose Okada (1985) Table 2
checklist pins :func:`okada_forward` in ``tests/test_deformation.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares, lsq_linear

from gps_analysis._mcmc import InversionConfig, PriorBounds, metropolis
from gps_analysis.models import FloatArray

__all__ = [
    "MogiSource",
    "OkadaSource",
    "MogiFit",
    "OkadaFit",
    "MogiPosterior",
    "FaultPatches",
    "SlipDistribution",
    "local_coordinates",
    "mogi_forward",
    "mogi_mctigue",
    "okada_forward",
    "mogi_invert",
    "mogi_invert_bayes",
    "okada_invert",
    "discretize_fault",
    "okada_greens",
    "patch_laplacian",
    "okada_invert_slip",
    "slip_lcurve",
    "lcurve_corner",
    "pressure_from_volume",
    "volume_from_pressure",
    "rate_to_m3s",
    "rate_from_m3s",
    "time_for_rate",
    "halflife_days",
]

#: Poisson's ratio of an isotropic elastic half-space (λ = μ); the standard
#: crustal value used by dMODELS, vmod and Vincent's operational procedure.
DEFAULT_NU = 0.25

#: WGS-84 semi-major axis [m] and first eccentricity squared (NIMA TR8350.2).
_WGS84_A = 6378137.0
_WGS84_E2 = 0.00669437999014

#: Seconds per Julian year (365.25 d) — volume-rate unit conversions.
_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

#: cos(dip) threshold below which the vertical-fault (cos δ = 0) forms of the
#: Okada I-terms are used; float64 eps, cf. okada85.m note 2 (cos 90° ≈ 6e-17).
_COS_DIP_EPS = float(np.finfo(np.float64).eps)

#: Guard for the singular R + η = 0 configuration (Okada 1985, p. 1148,
#: "singular point" prescription): terms with (R + η) in a denominator are
#: zeroed and ln(R + η) is replaced by −ln(R − η).
_R_ETA_TINY = 1.0e-14


# =====================================================================
# Geometry
# =====================================================================


def local_coordinates(
    lon: ArrayLike,
    lat: ArrayLike,
    lon_origin: float,
    lat_origin: float,
) -> tuple[FloatArray, FloatArray]:
    """Map geodetic (λ, φ) to local tangent-plane east/north offsets (e, n).

    Equation (small-offset equirectangular mapping on the WGS-84 ellipsoid):
        ``e = (λ − λ₀)·N(φ₀)·cos φ₀``,  ``n = (φ − φ₀)·M(φ₀)`` with
        ``N(φ) = a / W``, ``M(φ) = a(1 − e²) / W³``, ``W = √(1 − e² sin²φ)``
        (prime-vertical and meridian radii of curvature), angles in radians.

    Symbols → args:
        - ``λ, φ`` → ``lon``, ``lat``: station longitude/latitude [degrees]
        - ``λ₀, φ₀`` → ``lon_origin``, ``lat_origin``: frame origin [degrees]
        - ``a, e²``: WGS-84 semi-major axis [m] and eccentricity² (module
          constants)

    Returns:
        ``(e, n)`` — east and north offsets from the origin [m], float64,
        same shape as the inputs.

    Reference:
        Torge & Müller 2012, *Geodesy* (4th ed.), §4.1.2 (radii of curvature
        M, N of the rotational ellipsoid). Kept in-module (numpy-only) so the
        leaf does not grow a projection dependency; ``geofunc`` remains the
        home for full ECEF/ENU transforms.

    Numerical notes:
        Planar approximation: curvature/convergence errors grow as
        O(Δ²/R) ≈ 8 cm at 30 km offset — far below the source-position
        resolution of a GNSS network inversion and below the difference to
        any particular map projection (e.g. Vincent's ISN93 Lambert grid) at
        network scale. For offsets ≳ 100 km use a proper projection instead.
    """
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam0 = math.radians(lon_origin)
    phi0 = math.radians(lat_origin)
    w2 = 1.0 - _WGS84_E2 * math.sin(phi0) ** 2
    n_radius = _WGS84_A / math.sqrt(w2)  # N(φ₀)
    m_radius = _WGS84_A * (1.0 - _WGS84_E2) / w2**1.5  # M(φ₀)
    east = (lam - lam0) * n_radius * math.cos(phi0)
    north = (phi - phi0) * m_radius
    return east, north


# =====================================================================
# Source containers
# =====================================================================


@dataclass(frozen=True)
class MogiSource:
    """Mogi point pressure/volume source (Mogi 1958; Segall 2010 §7.1).

    Attributes:
        x: Source east coordinate [L] (local frame of the observations).
        y: Source north coordinate [L].
        depth: Source depth below the free surface, positive down [L], > 0.
        dv: Source volume change ΔV [L³] — the strength appearing in the
            surface solution ``C = (1 − ν)·ΔV/π``. Positive = inflation.
            (This is the "Mogi ΔV"; the magma-chamber volume change differs
            for compressible magma — Segall 2010 §7.2, Rivalta & Segall
            2008.)
    """

    x: float
    y: float
    depth: float
    dv: float

    def as_array(self) -> FloatArray:
        """Parameter vector ``[x, y, depth, dv]`` (inversion order)."""
        return np.array([self.x, self.y, self.depth, self.dv], dtype=np.float64)

    @classmethod
    def from_array(cls, params: ArrayLike) -> MogiSource:
        """Build from a ``[x, y, depth, dv]`` vector (inverse of as_array)."""
        p = np.asarray(params, dtype=np.float64)
        if p.shape != (4,):
            raise ValueError(f"expected 4 parameters [x, y, depth, dv], got {p.shape}")
        return cls(x=float(p[0]), y=float(p[1]), depth=float(p[2]), dv=float(p[3]))


@dataclass(frozen=True)
class OkadaSource:
    """Rectangular uniform dislocation (Okada 1985) — fault, dike or sill.

    Geometry follows the centroid convention of Beauducel's ``okada85.m``
    (IPGP deformation-lib): the fault plane's centroid sits at
    ``(x, y, −depth)``; STRIKE is the trace azimuth (degrees clockwise from
    north) with the fault dipping to the **right** of the strike direction;
    DIP is measured down from horizontal (0 < dip ≤ 90).

    Attributes:
        x: Fault-centroid east coordinate [L].
        y: Fault-centroid north coordinate [L].
        depth: Fault-centroid depth, positive down [L], > 0 and large enough
            that the up-dip edge stays below the surface
            (``depth ≥ sin(dip)·width/2``).
        strike: Strike azimuth [deg], 0–360, dip direction 90° clockwise.
        dip: Dip angle [deg], 0 < dip ≤ 90 (Aki & Richards 1980 convention).
        length: Along-strike fault length L [L], > 0.
        width: Down-dip fault width W [L], > 0.
        strike_slip: U₁, slip parallel to strike [L] (left-lateral > 0).
        dip_slip: U₂, slip up-dip [L] (reverse faulting > 0).
        opening: U₃, tensile opening normal to the plane [L] (dike/sill > 0).
    """

    x: float
    y: float
    depth: float
    strike: float
    dip: float
    length: float
    width: float
    strike_slip: float
    dip_slip: float
    opening: float

    def as_array(self) -> FloatArray:
        """Parameter vector in field order (see class docstring)."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)


# =====================================================================
# Forward models
# =====================================================================


def mogi_forward(
    e: ArrayLike,
    n: ArrayLike,
    source: MogiSource,
    nu: float = DEFAULT_NU,
) -> FloatArray:
    """Surface ENU displacement of a Mogi point source in a half-space.

    Equation (point center of dilatation at depth d, volume change ΔV):
        ``u_e = C·Δx/R³``, ``u_n = C·Δy/R³``, ``u_z = C·d/R³`` with
        ``C = (1 − ν)·ΔV/π``, ``Δx = e − x_s``, ``Δy = n − y_s``,
        ``R = √(Δx² + Δy² + d²)``.

    Symbols → args:
        - ``e, n`` → ``e``, ``n``: observation east/north coordinates [L]
        - ``x_s, y_s, d, ΔV`` → ``source.x``, ``source.y``, ``source.depth``
          (> 0, positive down) [L], ``source.dv`` [L³]
        - ``ν`` → ``nu``: Poisson's ratio [-] (default 0.25)

    Returns:
        ``(3, N)`` float64 displacements [L], rows **(east, north, up)**.

    Reference:
        Mogi 1958, Bull. Earthquake Res. Inst. 36, 99–134; Segall 2010,
        *Earthquake and Volcano Deformation*, Princeton UP
        (doi:10.1515/9781400833856), §7.1 — point-source limit of the
        pressurized sphere, ΔV-parameterized as in dMODELS (Battaglia,
        Cervelli & Murray 2013, JVGR 254) and vmod (Angarita et al.).
        Reconciled offline against Vincent Drouin's operational Svartsengi
        Mogi output (day-359 inflation episode): fixing his source geometry,
        our volume-only fit recovers his ΔV to 0.2 % (session report).

    Numerical notes:
        R ≥ d > 0, so the expression is regular everywhere on the surface;
        d → 0 is rejected. Valid for sources whose radius a ≪ d (point
        approximation; use :func:`mogi_mctigue` when a/d ≳ 0.3). Float64
        throughout; scalar inputs broadcast to length-1 axes.
    """
    if not source.depth > 0.0:
        raise ValueError(f"source depth must be > 0, got {source.depth}")
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if ee.shape != nn.shape:
        raise ValueError(f"e shape {ee.shape} != n shape {nn.shape}")
    dx = ee - source.x
    dy = nn - source.y
    r3 = (dx * dx + dy * dy + source.depth * source.depth) ** 1.5
    c = (1.0 - nu) * source.dv / math.pi
    return np.stack((c * dx / r3, c * dy / r3, c * source.depth / r3))


def mogi_mctigue(
    e: ArrayLike,
    n: ArrayLike,
    x: float,
    y: float,
    depth: float,
    radius: float,
    dp_over_mu: float,
    nu: float = DEFAULT_NU,
) -> FloatArray:
    """Surface ENU displacement of a finite pressurized sphere (McTigue).

    Equation (second-order interaction correction to the Mogi point source;
    dimensionless ρ = R/d, ε = a/d, with R = √(Δx² + Δy² + d²)):
        ``u_z = (ΔP·d/μ)·ε³·[ (1 − ν)/ρ³ − ε³·(A/ρ³ − B/ρ⁵) ]``
        ``u_h = (ΔP·d/μ)·ε³·[ (1 − ν)·r/ρ³ − ε³·(A·r/ρ³ − B·r/ρ⁵) ]``
        with ``r = √(Δx² + Δy²)/d`` and coefficients
        ``A = (1 + ν)(1 − ν) / (2(7 − 5ν))``,
        ``B = 15(2 − ν)(1 − ν) / (4(7 − 5ν))``;
        the horizontal vector points radially away from the source.

    Symbols → args:
        - ``e, n``: observation east/north coordinates [L]
        - ``x, y, d`` → ``x``, ``y``, ``depth`` (> 0, positive down) [L]
        - ``a`` → ``radius``: sphere radius [L], 0 < a < d
        - ``ΔP/μ`` → ``dp_over_mu``: pressure change over shear modulus [-]
        - ``ν`` → ``nu``: Poisson's ratio [-]

    Returns:
        ``(3, N)`` float64 displacements [L], rows **(east, north, up)**.

    Reference:
        McTigue 1987, J. Geophys. Res. 92(B12), 12931–12940 (higher-order
        spherical-source solution, surface form of his eqs. 52–53);
        coefficient forms cross-checked against ``vmod/source/mctigue.py``
        (Angarita et al., doi:10.5281/zenodo.10070627) and dMODELS
        ``mctigue.m`` (Battaglia et al. 2013). First-order term ≡
        :func:`mogi_forward` with ΔV = π a³ ΔP/μ (Segall 2010 §7.2).

    Numerical notes:
        The ε⁶ correction is O((a/d)³) relative — sub-percent for a/d < 0.2,
        ~13 % at the source axis for a/d = 0.5 (ν = 0.25). The expansion
        degrades as a/d → 1 (source approaching the surface); inputs with
        a ≥ d are rejected. Regular everywhere (ρ ≥ 1).
    """
    if not depth > 0.0:
        raise ValueError(f"depth must be > 0, got {depth}")
    if not 0.0 < radius < depth:
        raise ValueError(f"radius must satisfy 0 < radius < depth, got {radius}")
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if ee.shape != nn.shape:
        raise ValueError(f"e shape {ee.shape} != n shape {nn.shape}")
    dx = (ee - x) / depth
    dy = (nn - y) / depth
    rho2 = dx * dx + dy * dy + 1.0
    rho3 = rho2**1.5
    rho5 = rho2**2.5
    eps3 = (radius / depth) ** 3
    a_coef = ((1.0 + nu) * (1.0 - nu)) / (2.0 * (7.0 - 5.0 * nu))
    b_coef = (15.0 * (2.0 - nu) * (1.0 - nu)) / (4.0 * (7.0 - 5.0 * nu))
    scale = dp_over_mu * depth * eps3
    first = (1.0 - nu) / rho3
    second = eps3 * (a_coef / rho3 - b_coef / rho5)
    shape = first - second
    return np.stack((scale * dx * shape, scale * dy * shape, scale * shape))


# ---------------------------------------------------------------------
# Okada 1985 — private kernel (Chinnery-differenced fault-plane integrals)
# ---------------------------------------------------------------------
# Notation per Okada 1985, BSSA 75(4), 1135-1154 (doi:10.1785/BSSA0750041135):
# fault-plane coordinates (ξ, η), observation substitution p = y·cosδ + d·sinδ,
# q = y·sinδ − d·cosδ; each displacement is the Chinnery difference
# f(ξ,η)‖ = f(x,p) − f(x,p−W) − f(x−L,p) + f(x−L,p−W)   [eq. (24)].
# Reimplemented from the paper with F. Beauducel's okada85.m (IPGP
# deformation-lib, BSD) as the consulted reference; singular-case handling
# follows Okada's p. 1148 prescription. All kernels take float64 arrays
# (ξ, η, q) and scalar (δ, ν); μ/(λ+μ) = 1 − 2ν for the isotropic λ = μ case.

_KernelFunc = Callable[[FloatArray, FloatArray, FloatArray, float, float], FloatArray]


def _safe_atan(
    xi: FloatArray, eta: FloatArray, q: FloatArray, r: FloatArray
) -> FloatArray:
    """arctan(ξη/(qR)) with the q = 0 limit set to 0 (Okada 1985, p. 1148)."""
    q_safe = np.where(q == 0.0, 1.0, q)
    return np.where(q == 0.0, 0.0, np.arctan(xi * eta / (q_safe * r)))


def _r_eta_terms(r: FloatArray, eta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return (1/(R+η) with singular points zeroed, ln(R+η) with −ln(R−η) swap).

    Okada 1985 p. 1148: when R + η = 0 (observation aligned with the fault
    edge extension), terms with R + η in the denominator vanish and
    ln(R + η) is replaced by −ln(R − η).
    """
    re = r + eta
    singular = re < _R_ETA_TINY
    re_safe = np.where(singular, 1.0, re)
    inv = np.where(singular, 0.0, 1.0 / re_safe)
    log = np.where(singular, -np.log(r - eta), np.log(re_safe))
    return inv, log


def _i5(
    xi: FloatArray,
    eta: FloatArray,
    q: FloatArray,
    dip: float,
    nu: float,
    r: FloatArray,
    db: FloatArray,
) -> FloatArray:
    """Okada 1985 I₅ [eq. (28)/(29)]; I₅ = 0 on ξ = 0 (removable point)."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    if cd > _COS_DIP_EPS:
        x_big = np.sqrt(xi * xi + q * q)
        xi_safe = np.where(xi == 0.0, 1.0, xi)
        arg = (eta * (x_big + q * cd) + x_big * (r + x_big) * sd) / (
            xi_safe * (r + x_big) * cd
        )
        return np.where(xi == 0.0, 0.0, (1.0 - 2.0 * nu) * 2.0 / cd * np.arctan(arg))
    rdb = np.where(r + db == 0.0, 1.0, r + db)
    return -(1.0 - 2.0 * nu) * xi * sd / rdb


def _i4(
    eta: FloatArray,
    q: FloatArray,
    dip: float,
    nu: float,
    r: FloatArray,
    db: FloatArray,
) -> FloatArray:
    """Okada 1985 I₄ [eq. (28)/(29)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    _, log_re = _r_eta_terms(r, eta)
    if cd > _COS_DIP_EPS:
        return (1.0 - 2.0 * nu) / cd * (np.log(r + db) - sd * log_re)
    return -(1.0 - 2.0 * nu) * q / (r + db)


def _i3(
    eta: FloatArray, q: FloatArray, dip: float, nu: float, r: FloatArray
) -> FloatArray:
    """Okada 1985 I₃ [eq. (28)/(29)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    yb = eta * cd + q * sd
    db = eta * sd - q * cd
    _, log_re = _r_eta_terms(r, eta)
    if cd > _COS_DIP_EPS:
        return (1.0 - 2.0 * nu) * (yb / (cd * (r + db)) - log_re) + sd / cd * _i4(
            eta, q, dip, nu, r, db
        )
    rdb = r + db
    return (1.0 - 2.0 * nu) / 2.0 * (eta / rdb + yb * q / rdb**2 - log_re)


def _i2(
    eta: FloatArray, q: FloatArray, dip: float, nu: float, r: FloatArray
) -> FloatArray:
    """Okada 1985 I₂ [eq. (28)]."""
    _, log_re = _r_eta_terms(r, eta)
    return (1.0 - 2.0 * nu) * (-log_re) - _i3(eta, q, dip, nu, r)


def _i1(
    xi: FloatArray,
    eta: FloatArray,
    q: FloatArray,
    dip: float,
    nu: float,
    r: FloatArray,
) -> FloatArray:
    """Okada 1985 I₁ [eq. (28)/(29)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    db = eta * sd - q * cd
    if cd > _COS_DIP_EPS:
        return (1.0 - 2.0 * nu) * (-xi / (cd * (r + db))) - sd / cd * _i5(
            xi, eta, q, dip, nu, r, db
        )
    return -(1.0 - 2.0 * nu) / 2.0 * xi * q / (r + db) ** 2


def _ux_ss(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Strike-slip x-displacement kernel [Okada 1985 eq. (25)]."""
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    return (
        xi * q * inv_re / r
        + _i1(xi, eta, q, dip, nu, r) * math.sin(dip)
        + _safe_atan(xi, eta, q, r)
    )


def _uy_ss(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Strike-slip y-displacement kernel [Okada 1985 eq. (25)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    yb = eta * cd + q * sd
    return yb * q * inv_re / r + q * cd * inv_re + _i2(eta, q, dip, nu, r) * sd


def _uz_ss(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Strike-slip z-displacement kernel [Okada 1985 eq. (25)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    db = eta * sd - q * cd
    return db * q * inv_re / r + q * sd * inv_re + _i4(eta, q, dip, nu, r, db) * sd


def _ux_ds(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Dip-slip x-displacement kernel [Okada 1985 eq. (26)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    return q / r - _i3(eta, q, dip, nu, r) * sd * cd


def _uy_ds(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Dip-slip y-displacement kernel [Okada 1985 eq. (26)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    yb = eta * cd + q * sd
    return (
        yb * q / (r * (r + xi))
        + cd * _safe_atan(xi, eta, q, r)
        - _i1(xi, eta, q, dip, nu, r) * sd * cd
    )


def _uz_ds(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Dip-slip z-displacement kernel [Okada 1985 eq. (26)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    db = eta * sd - q * cd
    return (
        db * q / (r * (r + xi))
        + sd * _safe_atan(xi, eta, q, r)
        - _i5(xi, eta, q, dip, nu, r, db) * sd * cd
    )


def _ux_tf(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Tensile x-displacement kernel [Okada 1985 eq. (27)]."""
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    return q * q * inv_re / r - _i3(eta, q, dip, nu, r) * sd * sd


def _uy_tf(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Tensile y-displacement kernel [Okada 1985 eq. (27)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    db = eta * sd - q * cd
    return (
        -db * q / (r * (r + xi))
        - sd * xi * q * inv_re / r
        + sd * _safe_atan(xi, eta, q, r)
        - _i1(xi, eta, q, dip, nu, r) * sd * sd
    )


def _uz_tf(
    xi: FloatArray, eta: FloatArray, q: FloatArray, dip: float, nu: float
) -> FloatArray:
    """Tensile z-displacement kernel [Okada 1985 eq. (27)]."""
    cd = math.cos(dip)
    sd = math.sin(dip)
    r = np.sqrt(xi * xi + eta * eta + q * q)
    inv_re, _ = _r_eta_terms(r, eta)
    db = eta * sd - q * cd
    yb = eta * cd + q * sd
    return (
        yb * q / (r * (r + xi))
        + cd * xi * q * inv_re / r
        - cd * _safe_atan(xi, eta, q, r)
        - _i5(xi, eta, q, dip, nu, r, db) * sd * sd
    )


def _chinnery(
    f: _KernelFunc,
    x: FloatArray,
    p: FloatArray,
    length: float,
    width: float,
    q: FloatArray,
    dip: float,
    nu: float,
) -> FloatArray:
    """Chinnery finite-fault difference f(ξ,η)‖ [Okada 1985 eq. (24)].

    ``f‖ = f(x, p) − f(x, p − W) − f(x − L, p) + f(x − L, p − W)``.
    """
    return (
        f(x, p, q, dip, nu)
        - f(x, p - width, q, dip, nu)
        - f(x - length, p, q, dip, nu)
        + f(x - length, p - width, q, dip, nu)
    )


def okada_forward(
    e: ArrayLike,
    n: ArrayLike,
    source: OkadaSource,
    nu: float = DEFAULT_NU,
) -> FloatArray:
    """Surface ENU displacement of a rectangular dislocation (Okada 1985).

    Equation (fault-frame displacements, then rotation to geographic):
        ``u_i = −U₁/(2π)·f_i^ss‖ − U₂/(2π)·f_i^ds‖ + U₃/(2π)·f_i^tf‖``
        for i ∈ {x, y, z}, with the Chinnery differences ‖ of the kernel
        integrals of Okada 1985 eqs. (25)–(27) over the fault plane
        (ξ ∈ [0, L], η ∈ [0, W]), substitutions ``p = y·cosδ + d·sinδ``,
        ``q = y·sinδ − d·cosδ`` [eq. (30)], and I-terms of eqs. (28)–(29)
        with μ/(λ + μ) = 1 − 2ν. Geographic rotation:
        ``u_e = sinφ·u_x − cosφ·u_y``, ``u_n = cosφ·u_x + sinφ·u_y``
        (φ = strike; the Okada x-axis points along strike).

    Symbols → args:
        - ``e, n``: observation east/north coordinates [L]
        - ``U₁, U₂, U₃`` → ``source.strike_slip``, ``source.dip_slip``,
          ``source.opening`` [L]
        - ``L, W, δ, φ`` → ``source.length``, ``source.width`` [L],
          ``source.dip``, ``source.strike`` [deg]
        - centroid ``(source.x, source.y, source.depth)`` [L] — converted
          internally to Okada's bottom-edge frame:
          ``d = depth + sinδ·W/2`` plus the horizontal centroid shift
        - ``ν`` → ``nu``: Poisson's ratio [-]

    Returns:
        ``(3, N)`` float64 displacements [L], rows **(east, north, up)**.

    Reference:
        Okada 1985, *Surface deformation due to shear and tensile faults in
        a half-space*, BSSA 75(4), 1135–1154 (doi:10.1785/BSSA0750041135),
        eqs. (24)–(30). Reimplemented from the paper; conventions and the
        centroid transform follow the consulted reference implementation
        ``okada85.m`` by F. Beauducel (IPGP deformation-lib, BSD, 1997–2025).
        Pinned to Okada's Table 2 numerical checklist (cases 2–4) in
        ``tests/test_deformation.py``.

    Numerical notes:
        - Vertical faults: cos δ = 0 kernels switch to Okada's eq. (29)
          forms when cos δ < float64 eps (cos 90° ≈ 6.1e-17 ≠ 0!).
        - Singular points (q = 0 arctan limit; R + η = 0 edge alignment)
          follow Okada's p. 1148 prescription (:func:`_safe_atan`,
          :func:`_r_eta_terms`).
        - The fault must not breach the surface: requires
          ``depth ≥ sinδ·W/2`` (up-dip edge at or below z = 0).
    """
    if not 0.0 < source.dip <= 90.0:
        raise ValueError(f"dip must be in (0, 90] degrees, got {source.dip}")
    if source.length <= 0.0 or source.width <= 0.0:
        raise ValueError("fault length and width must be > 0")
    dip = math.radians(source.dip)
    if source.depth - math.sin(dip) * source.width / 2.0 < -1.0e-9:
        raise ValueError(
            "fault breaches the surface: depth must be >= sin(dip)*width/2 "
            f"(depth={source.depth}, sin(dip)*W/2={math.sin(dip) * source.width / 2.0})"
        )
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if ee.shape != nn.shape:
        raise ValueError(f"e shape {ee.shape} != n shape {nn.shape}")

    strike = math.radians(source.strike)
    cs, ss = math.cos(strike), math.sin(strike)
    cd, sd = math.cos(dip), math.sin(dip)
    length, width = source.length, source.width

    # Centroid (E, N, depth) → Okada bottom-edge frame (x, y, d) [okada85.m]
    d = source.depth + sd * width / 2.0
    ec = (ee - source.x) + cs * cd * width / 2.0
    nc = (nn - source.y) - ss * cd * width / 2.0
    x = cs * nc + ss * ec + length / 2.0
    y = ss * nc - cs * ec + cd * width

    p = y * cd + d * sd
    q = y * sd - d * cd

    u1, u2, u3 = source.strike_slip, source.dip_slip, source.opening
    two_pi = 2.0 * math.pi

    def combine(f_ss: _KernelFunc, f_ds: _KernelFunc, f_tf: _KernelFunc) -> FloatArray:
        out = np.zeros_like(x)
        if u1 != 0.0:
            out = out - u1 / two_pi * _chinnery(f_ss, x, p, length, width, q, dip, nu)
        if u2 != 0.0:
            out = out - u2 / two_pi * _chinnery(f_ds, x, p, length, width, q, dip, nu)
        if u3 != 0.0:
            out = out + u3 / two_pi * _chinnery(f_tf, x, p, length, width, q, dip, nu)
        return out

    # Okada's kernels carry removable singularities (q = 0, R + η = 0) handled
    # by the where-masks in the I-terms; a masked branch may still evaluate a
    # log(0)/÷0 that IEEE turns to ∞ before np.where discards it. Suppress those
    # non-finite intermediates locally (okada85.m relies on the same behaviour).
    # An observation ON a fault edge extension yields non-finite output by
    # design — keep stations off the fault trace.
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = combine(_ux_ss, _ux_ds, _ux_tf)
        uy = combine(_uy_ss, _uy_ds, _uy_tf)
        uz = combine(_uz_ss, _uz_ds, _uz_tf)

    ue = ss * ux - cs * uy
    un = cs * ux + ss * uy
    return np.stack((ue, un, uz))


# =====================================================================
# Volume / pressure / rate products
# =====================================================================


def pressure_from_volume(dv: float, radius: float, mu: float) -> float:
    """Cavity pressure change ΔP equivalent to a Mogi volume change ΔV.

    Equation (pressurized sphere, point-source limit):
        ``ΔP = μ·ΔV / (π·a³)``

    Symbols → args:
        - ``ΔV`` → ``dv``: Mogi source volume change [L³]
        - ``a`` → ``radius``: spherical-cavity radius [L], > 0
        - ``μ`` → ``mu``: shear modulus [Pa (or any pressure unit)]

    Returns:
        ΔP in the unit of ``mu``.

    Reference:
        Segall 2010, §7.2 (ΔV = π a³ ΔP/μ for the point spherical source);
        same relation in McTigue 1987 (leading order) and dMODELS
        (Battaglia et al. 2013). For the chamber volume change of
        compressible magma the ratio ΔV_chamber/ΔV_mogi departs from 1
        (Segall 2010 §7.2; Rivalta & Segall 2008, Geology 36) — this
        function is the incompressible/elastic bookkeeping step only.

    Numerical notes:
        Exact algebra; only invalid for radius ≤ 0 (rejected).
    """
    if not radius > 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    return mu * dv / (math.pi * radius**3)


def volume_from_pressure(dp: float, radius: float, mu: float) -> float:
    """Mogi volume change ΔV equivalent to a cavity pressure change ΔP.

    Equation:
        ``ΔV = π·a³·ΔP / μ`` — exact inverse of :func:`pressure_from_volume`.

    Symbols → args:
        - ``ΔP`` → ``dp``: cavity pressure change [same unit as ``mu``]
        - ``a`` → ``radius``: spherical-cavity radius [L], > 0
        - ``μ`` → ``mu``: shear modulus [pressure unit], > 0

    Returns:
        ΔV [L³].

    Reference:
        Segall 2010, §7.2; see :func:`pressure_from_volume` for caveats.

    Numerical notes:
        Exact algebra; invalid for radius ≤ 0 or mu ≤ 0 (rejected).
    """
    if not radius > 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not mu > 0.0:
        raise ValueError(f"mu must be > 0, got {mu}")
    return math.pi * radius**3 * dp / mu


def rate_to_m3s(rate_mm3_yr: float) -> float:
    """Convert a volume rate from Mm³/yr to m³/s.

    Equation:
        ``q[m³/s] = q[Mm³/yr] · 10⁶ / (365.25·86400)``

    Symbols → args:
        - ``q`` → ``rate_mm3_yr``: volume rate [Mm³/yr = 10⁶ m³/yr]

    Returns:
        Rate [m³/s].

    Reference:
        Port of ``svartsengi_model.fitting.rate_to_m3s``
        (docs/CONSOLIDATION_MAP.md); Julian year of 365.25 d
        (31 557 600 s) as upstream.

    Numerical notes:
        Exact scale factor; float64.
    """
    return rate_mm3_yr * 1.0e6 / _SECONDS_PER_YEAR


def rate_from_m3s(rate_m3s: float) -> float:
    """Convert a volume rate from m³/s to Mm³/yr.

    Equation:
        ``q[Mm³/yr] = q[m³/s] · (365.25·86400) / 10⁶`` — exact inverse of
        :func:`rate_to_m3s`.

    Symbols → args:
        - ``q`` → ``rate_m3s``: volume rate [m³/s]

    Returns:
        Rate [Mm³/yr].

    Reference:
        Port of ``svartsengi_model.fitting.rate_from_m3s``
        (docs/CONSOLIDATION_MAP.md).

    Numerical notes:
        Exact scale factor; float64.
    """
    return rate_m3s * _SECONDS_PER_YEAR / 1.0e6


def time_for_rate(target_rate: float, amplitude: float, decay_rate: float) -> float:
    """Time at which an exponential-relaxation rate reaches a target value.

    Equation (rate of the ``A·exp(−k·t)`` transient of
    :func:`gps_analysis.models.exp_linear` with v = 0):
        ``dx/dt = −A·k·e^{−k·t} = q_target  ⇒  t = −ln(q_target / (−A·k))/k``

    Symbols → args:
        - ``q_target`` → ``target_rate``: target rate [Q/yr] (same quantity
          unit Q as ``amplitude``, e.g. Mm³ for volume curves)
        - ``A`` → ``amplitude``: transient amplitude at t = 0 [Q]
        - ``k`` → ``decay_rate``: decay constant [1/yr], ≠ 0

    Returns:
        t [yr from the reference epoch], or NaN when the target rate is
        never attained (ratio ≤ 0, i.e. wrong sign relative to −A·k).

    Reference:
        Port of ``svartsengi_model.fitting.time_for_rate``
        (docs/CONSOLIDATION_MAP.md); the Svartsengi inflow-forecast helper
        on :func:`gps_analysis.models.exp_linear` fits (Segall 2010 §7.10
        conduit-recharge interpretation, cf. Sigmundsson et al. 2020,
        Nat. Commun. 11, doi:10.1038/s41467-020-16054-6).

    Numerical notes:
        NaN (not an exception) for unattainable targets — upstream contract;
        exact otherwise. ``decay_rate = 0`` raises (constant rate).
    """
    if decay_rate == 0.0:
        raise ValueError("decay_rate must be nonzero (constant-rate model)")
    ratio = target_rate / (-amplitude * decay_rate)
    if ratio <= 0.0:
        return float("nan")
    return -math.log(ratio) / decay_rate


def halflife_days(decay_rate: float, year_days: float = 365.25) -> float:
    """Half-life in days of an exponential decay constant k [1/yr].

    Equation:
        ``t½ = ln 2 / k · D`` with D days per year.

    Symbols → args:
        - ``k`` → ``decay_rate``: decay constant [1/yr], > 0
        - ``D`` → ``year_days``: days per year (default 365.25)

    Returns:
        Half-life [days].

    Reference:
        Port of ``svartsengi_model.fitting.halflife_days``
        (docs/CONSOLIDATION_MAP.md).

    Numerical notes:
        Exact; decay_rate ≤ 0 raises.
    """
    if not decay_rate > 0.0:
        raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
    return math.log(2.0) / decay_rate * year_days


# =====================================================================
# Inversion — weighted nonlinear least squares
# =====================================================================


@dataclass(frozen=True)
class MogiFit:
    """Mogi least-squares inversion result (:func:`mogi_invert`).

    Attributes:
        source: Best-fit :class:`MogiSource` (x, y, depth [L], ΔV [L³]).
        covariance: ``(4, 4)`` formal parameter covariance in the order
            ``[x, y, depth, dv]`` (see :func:`_lsq_covariance`).
        sigma: ``(4,)`` formal 1-σ uncertainties, ``√diag(covariance)``.
        chi2_reduced: Reduced chi-square ``χ²/(N_obs − 4)`` of the weighted
            fit (≈ 1 when σ are honest and the model adequate).
        rms: Unweighted RMS of the residuals [L].
        n_obs: Number of scalar observations (3 × stations used).
    """

    source: MogiSource
    covariance: FloatArray
    sigma: FloatArray
    chi2_reduced: float
    rms: float
    n_obs: int


@dataclass(frozen=True)
class OkadaFit:
    """Okada least-squares inversion result (:func:`okada_invert`).

    Attributes:
        source: Best-fit :class:`OkadaSource` (fixed fields kept from x0).
        free: Names of the fitted parameters, in covariance order.
        covariance: ``(p, p)`` formal covariance of the free parameters.
        sigma: ``(p,)`` formal 1-σ uncertainties of the free parameters.
        chi2_reduced: Reduced chi-square ``χ²/(N_obs − p)``.
        rms: Unweighted RMS of the residuals [L].
        n_obs: Number of scalar observations (3 × stations used).
    """

    source: OkadaSource
    free: tuple[str, ...]
    covariance: FloatArray
    sigma: FloatArray
    chi2_reduced: float
    rms: float
    n_obs: int


def _as_obs_arrays(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Coerce/validate inversion inputs to float64 (e, n, obs(3,N), σ(3,N)).

    obs rows are (east, north, up); ``sigma=None`` means unit weights.
    Rejects NaNs (mask upstream), shape mismatches and σ ≤ 0.
    """
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    dd = np.asarray(obs, dtype=np.float64)
    if ee.ndim != 1 or ee.shape != nn.shape:
        raise ValueError(f"e/n must be equal-length 1-D, got {ee.shape}/{nn.shape}")
    if dd.shape != (3, ee.size):
        raise ValueError(f"obs must have shape (3, {ee.size}), got {dd.shape}")
    if sigma is None:
        ss = np.ones_like(dd)
    else:
        ss = np.asarray(sigma, dtype=np.float64)
        if ss.shape != dd.shape:
            raise ValueError(f"sigma must have shape {dd.shape}, got {ss.shape}")
        if not bool(np.all(ss > 0.0)):
            raise ValueError("sigma must be strictly positive")
    if not (
        bool(np.all(np.isfinite(ee)))
        and bool(np.all(np.isfinite(nn)))
        and bool(np.all(np.isfinite(dd)))
    ):
        raise ValueError("e, n and obs must be finite (mask NaNs upstream)")
    return ee, nn, dd, ss


def _lsq_covariance(
    jac: FloatArray, cost: float, n_obs: int, n_params: int
) -> FloatArray:
    """Formal parameter covariance of a weighted nonlinear LSQ solution.

    Equation (linearization at the optimum; J is the Jacobian of the
    σ-weighted residual vector r = (d − G(m))/σ):
        ``C_m = s²·(JᵀJ)⁻¹``, ``s² = 2·cost/(N − p) = χ²/(N − p)``
        (scipy's ``cost = ½·Σ rᵢ²``), computed through the SVD
        ``J = U·S·Vᵀ ⇒ (JᵀJ)⁻¹ = V·S⁻²·Vᵀ`` with rank-deficient singular
        values (S < ε·S_max) truncated.

    Symbols → args:
        - ``J`` → ``jac``: ``(N, p)`` weighted-residual Jacobian at optimum
        - ``cost`` → ``cost``: scipy ``least_squares`` cost ½Σr²
        - ``N, p`` → ``n_obs``, ``n_params``

    Returns:
        ``(p, p)`` covariance, scaled by the reduced chi-square (a-posteriori
        variance factor).

    Reference:
        Aster, Borchers & Thurber 2018, *Parameter Estimation and Inverse
        Problems* (3rd ed.), §9.2 (covariance of nonlinear LSQ estimates by
        linearization about m*).

    Numerical notes:
        SVD, never an explicit normal-equation inverse: (JᵀJ) squares the
        condition number, the SVD does not. Truncation threshold
        ε = N·float64-eps on relative singular values; truncated directions
        get infinite variance omitted (pseudo-inverse convention) — inspect
        ``sigma`` for absurdly small values as a rank-deficiency symptom.
        With a robust loss the residuals are re-weighted and this formal
        covariance is only approximate (document at the call site).
    """
    dof = n_obs - n_params
    if dof <= 0:
        raise ValueError(f"underdetermined: {n_obs} observations, {n_params} params")
    s2 = 2.0 * cost / dof
    _, sing, vt = np.linalg.svd(jac, full_matrices=False)
    tol = max(jac.shape) * np.finfo(np.float64).eps * float(sing[0])
    keep = sing > tol
    inv_s2 = np.zeros_like(sing)
    inv_s2[keep] = 1.0 / sing[keep] ** 2
    cov: FloatArray = (vt.T * inv_s2) @ vt * s2
    return cov


def _mogi_start(e: FloatArray, n: FloatArray, obs: FloatArray, nu: float) -> FloatArray:
    """Data-driven Mogi starting model [x₀, y₀, d₀, ΔV₀].

    Equations (heuristics, not physics):
        ``(x₀, y₀) = Σ|u_z|·(e, n) / Σ|u_z|`` (uplift-weighted centroid),
        ``d₀ = median(√((e−x₀)² + (n−y₀)²))`` (the network samples the
        anomaly at r ≈ d where the pattern carries depth information),
        ``ΔV₀ = u_z(peak)·π·d₀²/(1 − ν)`` (inverts the peak-uplift
        relation u_z(0) = (1−ν)ΔV/(π d²)).

    Symbols → args: as in :func:`mogi_invert` (validated arrays).

    Returns:
        ``[x₀, y₀, d₀, ΔV₀]`` float64.

    Reference:
        Standard practice (cf. vmod's default initial models); the peak-
        uplift inversion is Segall 2010 §7.1.

    Numerical notes:
        Purely a trust-region seed — the optimum must not depend on it for
        well-posed data (round-trip tested). Degenerate all-zero u_z falls
        back to the unweighted centroid.
    """
    w = np.abs(obs[2])
    wsum = float(w.sum())
    if wsum > 0.0:
        x0 = float((w * e).sum() / wsum)
        y0 = float((w * n).sum() / wsum)
    else:
        x0 = float(e.mean())
        y0 = float(n.mean())
    d0 = float(np.median(np.hypot(e - x0, n - y0)))
    d0 = max(d0, 1.0e-3 * (float(np.ptp(e)) + float(np.ptp(n)) + 1.0))
    iz = int(np.argmax(np.abs(obs[2])))
    dv0 = float(obs[2][iz]) * math.pi * d0 * d0 / (1.0 - nu)
    if dv0 == 0.0:
        dv0 = 1.0
    return np.array([x0, y0, d0, dv0], dtype=np.float64)


def _mogi_jacobian(
    params: FloatArray, e: FloatArray, n: FloatArray, nu: float
) -> FloatArray:
    """Analytic Jacobian ∂u/∂[x_s, y_s, d, ΔV] of :func:`mogi_forward`.

    Equations (C = (1−ν)ΔV/π, Δx = e−x_s, Δy = n−y_s, R² = Δx²+Δy²+d²):
        ``∂u_e/∂x_s = −C(R⁻³ − 3Δx²R⁻⁵)``   ``∂u_e/∂y_s = 3CΔxΔyR⁻⁵``
        ``∂u_e/∂d = −3CΔx·d·R⁻⁵``           ``∂u_e/∂ΔV = u_e/ΔV``
        (and cyclically for u_n; for u_z the numerator is d:
        ``∂u_z/∂x_s = 3CΔx·d·R⁻⁵``, ``∂u_z/∂d = C(R⁻³ − 3d²R⁻⁵)``.)

    Symbols → args:
        - ``params``: ``[x_s, y_s, d, ΔV]``; ``e, n``: station coords [L]
        - ``ν`` → ``nu``

    Returns:
        ``(3N, 4)`` float64 — rows stacked (all-east, all-north, all-up),
        matching ``mogi_forward(...).ravel()`` ordering.

    Reference:
        Analytic partials of the Mogi solution (Segall 2010 §7.1); standard
        calculus, no external source.

    Numerical notes:
        Shares R⁻³/R⁻⁵ subexpressions; regular for d > 0. Exact derivatives
        make the trust-region step and the formal covariance independent of
        finite-difference step choices.
    """
    xs, ys, d, dv = (float(v) for v in params)
    dx = e - xs
    dy = n - ys
    r2 = dx * dx + dy * dy + d * d
    r3 = r2**1.5
    r5 = r2**2.5
    c = (1.0 - nu) * dv / math.pi
    c_dv = (1.0 - nu) / math.pi  # ∂C/∂ΔV
    jac = np.empty((3 * e.size, 4), dtype=np.float64)
    m = e.size
    # ∂/∂x_s
    jac[0 * m : 1 * m, 0] = -c * (1.0 / r3 - 3.0 * dx * dx / r5)
    jac[1 * m : 2 * m, 0] = 3.0 * c * dx * dy / r5
    jac[2 * m : 3 * m, 0] = 3.0 * c * dx * d / r5
    # ∂/∂y_s
    jac[0 * m : 1 * m, 1] = 3.0 * c * dx * dy / r5
    jac[1 * m : 2 * m, 1] = -c * (1.0 / r3 - 3.0 * dy * dy / r5)
    jac[2 * m : 3 * m, 1] = 3.0 * c * dy * d / r5
    # ∂/∂d
    jac[0 * m : 1 * m, 2] = -3.0 * c * dx * d / r5
    jac[1 * m : 2 * m, 2] = -3.0 * c * dy * d / r5
    jac[2 * m : 3 * m, 2] = c * (1.0 / r3 - 3.0 * d * d / r5)
    # ∂/∂ΔV
    jac[0 * m : 1 * m, 3] = c_dv * dx / r3
    jac[1 * m : 2 * m, 3] = c_dv * dy / r3
    jac[2 * m : 3 * m, 3] = c_dv * d / r3
    return jac


def mogi_invert(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    x0: MogiSource | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    nu: float = DEFAULT_NU,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> MogiFit:
    """GPS-only Mogi source inversion by weighted nonlinear least squares.

    Estimator:
        ``m* = argmin_m Σᵢ ρ( ((dᵢ − Gᵢ(m))/σᵢ)² )`` over
        ``m = [x_s, y_s, d, ΔV]`` with G = :func:`mogi_forward`, solved by
        scipy ``least_squares`` (Trust Region Reflective, analytic Jacobian
        :func:`_mogi_jacobian`); ρ = identity for ``loss="linear"``, or a
        robust loss (``"soft_l1"``, ``"huber"``, ``"cauchy"``, ``"arctan"``)
        with scale ``f_scale``. Formal covariance from
        :func:`_lsq_covariance` at the optimum.

    Symbols → args:
        - ``dᵢ`` → ``obs``: ``(3, N)`` station displacements [L], rows
          **(east, north, up)** — live-GNSS arrays from the analysis lane
          (convert mm → coordinate unit at the call site)
        - ``σᵢ`` → ``sigma``: ``(3, N)`` 1-σ uncertainties [L], or ``None``
          for unit weights
        - ``e, n``: station coordinates [L] (e.g. :func:`local_coordinates`)
        - ``x0``: optional starting :class:`MogiSource`; default from
          :func:`_mogi_start`
        - ``bounds``: optional ``(lower, upper)`` 4-vectors in
          ``[x, y, depth, dv]`` order; default keeps x, y within the network
          footprint ± one span, depth ∈ [10⁻³·span, 20·span], ΔV free
        - ``ν`` → ``nu``; ``loss``/``f_scale``: scipy robust-loss controls

    Returns:
        :class:`MogiFit` — source, ``(4, 4)`` covariance, 1-σ, reduced χ²,
        RMS [L], and the observation count.

    Reference:
        Forward model: Mogi 1958; Segall 2010 §7.1. Estimator: standard
        weighted NLLS (Aster, Borchers & Thurber 2018 §9.2); robust-loss
        machinery as wrapped by scipy ``least_squares``. Deterministic
        daily-driver counterpart of :func:`mogi_invert_bayes` (Bagnardi &
        Hooper 2018 discuss why full posteriors matter when the formal
        Gaussian is optimistic).

    Numerical notes:
        - Parameters are internally scaled (``x_scale`` = network span for
          coordinates/depth, |ΔV₀| for volume) so the trust region treats
          m-scale positions and 10⁷ m³ volumes evenhandedly.
        - The 4-parameter Mogi problem is mildly non-convex (depth–ΔV
          trade-off); the data-driven start makes the basin reliable for
          network geometries that bracket the source. Round-trip recovery
          is test-pinned.
        - With a robust loss the covariance is the linearized-Gaussian
          approximation at the optimum — treat as indicative only.
        - Requires N ≥ 2 stations (3N ≥ 6 > 4 parameters).
    """
    ee, nn, dd, ss = _as_obs_arrays(e, n, obs, sigma)
    if ee.size < 2:
        raise ValueError(f"need >= 2 stations for 4 parameters, got {ee.size}")
    start = x0.as_array() if x0 is not None else _mogi_start(ee, nn, dd, nu)
    span = max(float(np.ptp(ee)), float(np.ptp(nn)), 1.0)
    if bounds is not None:
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        if lower.shape != (4,) or upper.shape != (4,):
            raise ValueError("bounds must be a pair of length-4 vectors")
    else:
        lower = np.array(
            [ee.min() - span, nn.min() - span, 1.0e-3 * span, -np.inf],
            dtype=np.float64,
        )
        upper = np.array(
            [ee.max() + span, nn.max() + span, 20.0 * span, np.inf],
            dtype=np.float64,
        )
    start = np.clip(start, lower, upper)
    sflat = ss.ravel()
    dflat = dd.ravel()

    def residual(mvec: FloatArray) -> FloatArray:
        model = mogi_forward(ee, nn, MogiSource.from_array(mvec), nu)
        out: FloatArray = (dflat - model.ravel()) / sflat
        return out

    def jacobian(mvec: FloatArray) -> FloatArray:
        out: FloatArray = -_mogi_jacobian(mvec, ee, nn, nu) / sflat[:, None]
        return out

    x_scale = np.array(
        [span, span, span, max(abs(float(start[3])), 1.0)], dtype=np.float64
    )
    res: Any = least_squares(
        residual,
        start,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale=x_scale,
        loss=loss,
        f_scale=f_scale,
    )
    if not res.success:
        raise RuntimeError(f"mogi_invert did not converge: {res.message}")
    n_obs = dflat.size
    cov = _lsq_covariance(
        np.asarray(res.jac, dtype=np.float64), float(res.cost), n_obs, 4
    )
    source = MogiSource.from_array(res.x)
    model_opt = mogi_forward(ee, nn, source, nu)
    rms = float(np.sqrt(np.mean((dd - model_opt) ** 2)))
    chi2 = float(np.sum(((dd - model_opt) / ss) ** 2)) / (n_obs - 4)
    return MogiFit(
        source=source,
        covariance=cov,
        sigma=np.sqrt(np.diag(cov)),
        chi2_reduced=chi2,
        rms=rms,
        n_obs=n_obs,
    )


_OKADA_FIELDS = tuple(f.name for f in fields(OkadaSource))


def okada_invert(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    x0: OkadaSource,
    free: tuple[str, ...] = ("x", "y", "depth", "length", "width", "opening"),
    bounds: dict[str, tuple[float, float]] | None = None,
    nu: float = DEFAULT_NU,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> OkadaFit:
    """GPS-only rectangular-dislocation inversion (nonlinear least squares).

    Estimator:
        ``m* = argmin Σᵢ ρ(((dᵢ − Gᵢ(m))/σᵢ)²)`` over the ``free`` subset of
        :class:`OkadaSource` fields, G = :func:`okada_forward`, scipy
        ``least_squares`` (TRF, 2-point finite-difference Jacobian —
        analytic Okada partials are not worth their complexity here).
        Formal covariance from :func:`_lsq_covariance`.

    Symbols → args:
        - ``dᵢ`` → ``obs``: ``(3, N)`` displacements [L], rows (east, north,
          up); ``σᵢ`` → ``sigma`` as in :func:`mogi_invert`
        - ``x0``: full starting :class:`OkadaSource` — fields not in
          ``free`` stay **fixed** at their x0 values
        - ``free``: fitted field names (default: dike/sill geometry +
          opening; add ``"strike"``/``"dip"``/slips as the data warrant)
        - ``bounds``: per-field ``(lower, upper)``; unbounded when absent.
          Positivity of depth/length/width and the surface-breach guard of
          :func:`okada_forward` are enforced with defaults when not given.

    Returns:
        :class:`OkadaFit` — best source (fixed fields carried through),
        free-parameter covariance/σ, reduced χ², RMS, observation count.

    Reference:
        Forward: Okada 1985 (see :func:`okada_forward`). Estimator: as
        :func:`mogi_invert` (Aster et al. 2018 §9.2). The 10-parameter
        problem is notoriously multi-modal (e.g. Bagnardi & Hooper 2018
        §2); fixing geometry via ``free`` and bounding via ``bounds`` is
        the intended use — a global search belongs in the Bayesian lane.

    Numerical notes:
        - Requires ``3N > len(free)``.
        - x_scale from bounds where finite, else from |x0| (min 1) —
          balances meter-scale slips against kilometer-scale geometry.
        - Angles are NOT wrapped: bound ``strike`` within one branch (e.g.
          [0, 360)) to avoid equivalent-minimum hopping.
        - depth's default lower bound keeps the up-dip edge below the
          surface for the CURRENT width; when both ``depth`` and ``width``
          are free the guard is re-imposed inside the residual by clipping,
          so trial faults never breach the surface.
    """
    ee, nn, dd, ss = _as_obs_arrays(e, n, obs, sigma)
    if not free:
        raise ValueError("free must name at least one OkadaSource field")
    unknown = set(free) - set(_OKADA_FIELDS)
    if unknown:
        raise ValueError(f"unknown OkadaSource fields: {sorted(unknown)}")
    n_free = len(free)
    if 3 * ee.size <= n_free:
        raise ValueError(
            f"underdetermined: {3 * ee.size} observations for {n_free} parameters"
        )
    bounds = dict(bounds or {})
    dip0 = math.radians(x0.dip)
    bounds.setdefault("depth", (math.sin(dip0) * x0.width / 2.0, math.inf))
    bounds.setdefault("length", (1.0e-6, math.inf))
    bounds.setdefault("width", (1.0e-6, math.inf))
    bounds.setdefault("dip", (1.0e-3, 90.0))
    lower = np.array([bounds.get(f, (-math.inf, math.inf))[0] for f in free])
    upper = np.array([bounds.get(f, (-math.inf, math.inf))[1] for f in free])
    start = np.array([getattr(x0, f) for f in free], dtype=np.float64)
    start = np.clip(start, lower, upper)
    sflat = ss.ravel()
    dflat = dd.ravel()

    def build(mvec: FloatArray) -> OkadaSource:
        src = replace(
            x0, **{name: float(v) for name, v in zip(free, mvec, strict=True)}
        )
        min_depth = math.sin(math.radians(src.dip)) * src.width / 2.0
        if src.depth < min_depth:
            src = replace(src, depth=min_depth)
        return src

    def residual(mvec: FloatArray) -> FloatArray:
        model = okada_forward(ee, nn, build(mvec), nu)
        out: FloatArray = (dflat - model.ravel()) / sflat
        return out

    # Per-parameter trust-region scale: a quarter of a finite prior range,
    # else the parameter's own magnitude (min 1) — balances meter-scale slips
    # against kilometre-scale geometry.
    x_scale = np.empty(n_free, dtype=np.float64)
    for i in range(n_free):
        width_i = upper[i] - lower[i]
        if math.isfinite(width_i) and width_i > 0.0:
            x_scale[i] = width_i / 4.0
        else:
            x_scale[i] = max(abs(float(start[i])), 1.0)
    res: Any = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        method="trf",
        x_scale=x_scale,
        loss=loss,
        f_scale=f_scale,
        diff_step=1.0e-6,
    )
    if not res.success:
        raise RuntimeError(f"okada_invert did not converge: {res.message}")
    n_obs = dflat.size
    cov = _lsq_covariance(
        np.asarray(res.jac, dtype=np.float64), float(res.cost), n_obs, n_free
    )
    source = build(np.asarray(res.x, dtype=np.float64))
    model_opt = okada_forward(ee, nn, source, nu)
    rms = float(np.sqrt(np.mean((dd - model_opt) ** 2)))
    chi2 = float(np.sum(((dd - model_opt) / ss) ** 2)) / (n_obs - n_free)
    return OkadaFit(
        source=source,
        free=tuple(free),
        covariance=cov,
        sigma=np.sqrt(np.diag(cov)),
        chi2_reduced=chi2,
        rms=rms,
        n_obs=n_obs,
    )


# =====================================================================
# Distributed slip — fault discretization, Green's functions, and the
# regularized linear inversion (Harris & Segall 1987; Jónsson et al. 2002)
# =====================================================================

#: Slip components an Okada patch can carry (OkadaSource field names).
_SLIP_COMPONENTS = ("strike_slip", "dip_slip", "opening")


@dataclass(frozen=True)
class FaultPatches:
    """A fault plane discretized into an ``n_down × n_along`` grid of patches.

    Produced by :func:`discretize_fault`; consumed by :func:`okada_greens`,
    :func:`patch_laplacian` and :func:`okada_invert_slip`. Patch ``k`` sits
    at grid position ``(i, j) = (k % n_along, k // n_along)`` — row-major,
    ``i`` along strike (in the strike direction), ``j`` down dip (j = 0 is
    the shallowest row).

    Attributes:
        plane: The parent :class:`OkadaSource` geometry (its slip fields are
            ignored — patches carry their own slip in the inversion).
        n_along: Number of patches along strike, ≥ 1.
        n_down: Number of patches down dip, ≥ 1.
        patch_length: Along-strike patch size ``L/n_along`` [L].
        patch_width: Down-dip patch size ``W/n_down`` [L].
        centers: ``(n_patches, 3)`` float64 patch-centroid coordinates
            ``(east, north, depth)`` [L], depth positive down, row-major as
            above.
    """

    plane: OkadaSource
    n_along: int
    n_down: int
    patch_length: float
    patch_width: float
    centers: FloatArray

    @property
    def n_patches(self) -> int:
        """Total number of patches ``n_along · n_down``."""
        return self.n_along * self.n_down

    @property
    def patch_area(self) -> float:
        """Area of one patch ``patch_length · patch_width`` [L²]."""
        return self.patch_length * self.patch_width

    def patch_source(
        self,
        k: int,
        *,
        strike_slip: float = 0.0,
        dip_slip: float = 0.0,
        opening: float = 0.0,
    ) -> OkadaSource:
        """The k-th patch as an :class:`OkadaSource` with the given slip.

        Geometry (patch centroid, strike/dip inherited from the plane):
            ``(x_k, y_k, d_k) = centers[k]``, ``L = patch_length``,
            ``W = patch_width``.

        Symbols → args: ``k``: row-major patch index (0 ≤ k < n_patches);
        slips ``U₁, U₂, U₃`` → ``strike_slip``, ``dip_slip``, ``opening`` [L].

        Reference:
            Okada 1985 (the rectangular element); the tiling itself is the
            standard planar-fault discretization of distributed-slip
            inversion (Harris & Segall 1987, JGR 92(B8), §"Inversion
            procedure"; Jónsson et al. 2002, BSSA 92(4), p. 1382).

        Numerical notes:
            Superposition over the tiles reproduces the uniform-slip parent
            plane to machine precision (linear elasticity; test-pinned at
            rtol 1e-12) — the patch grid introduces no geometric
            approximation, only a piecewise-constant slip representation.
        """
        if not 0 <= k < self.n_patches:
            raise ValueError(f"patch index {k} outside [0, {self.n_patches})")
        cx, cy, cd = (float(v) for v in self.centers[k])
        return OkadaSource(
            x=cx,
            y=cy,
            depth=cd,
            strike=self.plane.strike,
            dip=self.plane.dip,
            length=self.patch_length,
            width=self.patch_width,
            strike_slip=strike_slip,
            dip_slip=dip_slip,
            opening=opening,
        )


def discretize_fault(plane: OkadaSource, n_along: int, n_down: int) -> FaultPatches:
    """Tile a rectangular fault plane into an ``n_down × n_along`` patch grid.

    Equation (patch centroids in the plane's local frame; centroid
    convention of :class:`OkadaSource`):
        ``c_k = c₀ + u_i·ŝ + v_j·d̂`` with in-plane offsets
        ``u_i = (i + ½)·L/n_along − L/2``, ``v_j = (j + ½)·W/n_down − W/2``
        and the unit vectors (φ = strike, δ = dip; e, n, depth-down frame)
        ``ŝ = (sin φ, cos φ, 0)`` (along strike),
        ``d̂ = (cos φ·cos δ, −sin φ·cos δ, sin δ)`` (down dip — the plane
        dips to the right of the strike direction).

    Symbols → args:
        - ``c₀, L, W, φ, δ`` → ``plane``: parent geometry (slip ignored)
        - ``n_along``, ``n_down``: grid dimensions, ≥ 1

    Returns:
        :class:`FaultPatches` (row-major: ``k = j·n_along + i``, j = 0 the
        shallowest row, i increasing along strike).

    Reference:
        Okada 1985 (element geometry); standard fault discretization for
        distributed-slip inversion: Harris & Segall 1987, JGR 92(B8),
        7945–7962 (Parkfield, rectangular sub-fault grid); Jónsson et al.
        2002, BSSA 92(4), 1377–1389 (Hector Mine, 2 × 2.5 km patches).

    Numerical notes:
        ``d̂`` is derived from the same centroid → bottom-edge transform as
        :func:`okada_forward` (``okada85.m`` convention), so tiled patches
        with uniform slip superpose to the parent plane's field to machine
        precision (test-pinned ≤ 1e-12 relative). The parent's
        surface-breach guard ``depth ≥ sin δ·W/2`` guarantees every patch
        satisfies its own guard (equality at the top row's up-dip edge).
    """
    if n_along < 1 or n_down < 1:
        raise ValueError(f"grid must be >= 1x1, got {n_along}x{n_down}")
    if not 0.0 < plane.dip <= 90.0:
        raise ValueError(f"dip must be in (0, 90] degrees, got {plane.dip}")
    if plane.length <= 0.0 or plane.width <= 0.0:
        raise ValueError("fault length and width must be > 0")
    dip = math.radians(plane.dip)
    if plane.depth - math.sin(dip) * plane.width / 2.0 < -1.0e-9:
        raise ValueError(
            "fault breaches the surface: depth must be >= sin(dip)*width/2"
        )
    strike = math.radians(plane.strike)
    cs, ss = math.cos(strike), math.sin(strike)
    cd, sd = math.cos(dip), math.sin(dip)
    dl = plane.length / n_along
    dw = plane.width / n_down
    u = (np.arange(n_along, dtype=np.float64) + 0.5) * dl - plane.length / 2.0
    v = (np.arange(n_down, dtype=np.float64) + 0.5) * dw - plane.width / 2.0
    vv, uu = np.meshgrid(v, u, indexing="ij")  # row-major (j, i)
    centers = np.column_stack(
        (
            (plane.x + uu * ss + vv * cs * cd).ravel(),
            (plane.y + uu * cs - vv * ss * cd).ravel(),
            (plane.depth + vv * sd).ravel(),
        )
    )
    return FaultPatches(
        plane=plane,
        n_along=n_along,
        n_down=n_down,
        patch_length=dl,
        patch_width=dw,
        centers=centers,
    )


def okada_greens(
    e: ArrayLike,
    n: ArrayLike,
    patches: FaultPatches,
    components: tuple[str, ...] = ("opening",),
    nu: float = DEFAULT_NU,
) -> FloatArray:
    """Green's-function matrix G of unit patch slip → surface ENU displacement.

    Equation (columns are unit-slip Okada solutions; linear superposition):
        ``d = G·s``, ``G[:, c·n_p + k] = vec(u(e, n; patch_k, U_c = 1))``
    where ``u`` is :func:`okada_forward` of patch ``k`` carrying unit slip in
    component ``c`` (all other components zero), ``vec`` stacks the rows
    (all-east, all-north, all-up) — matching ``obs.ravel()`` ordering — and
    ``s`` is the slip vector in the same component-major layout.

    Symbols → args:
        - ``e, n``: observation east/north coordinates [L]
        - ``patches`` → :class:`FaultPatches` (n_p = ``patches.n_patches``)
        - ``components``: subset of ``("strike_slip", "dip_slip",
          "opening")`` — the slip directions to be estimated
        - ``ν`` → ``nu``: Poisson's ratio [-]

    Returns:
        G, shape ``(3·N, len(components)·n_p)``, float64. Unit: displacement
        [L] per unit slip [L] (dimensionless kernel).

    Reference:
        Okada 1985 (the kernel); G-matrix assembly for distributed slip:
        Harris & Segall 1987, JGR 92(B8), eq. (1) (``d = G·s + ε``);
        Jónsson et al. 2002, BSSA 92(4), eq. (1).

    Numerical notes:
        One :func:`okada_forward` call per column — O(N) each, n_comp·n_p
        total. Columns of distant/deep patches have small norms; the
        regularized inversion (:func:`okada_invert_slip`) handles the
        resulting ill-conditioning — do NOT invert G unregularized
        (Aster, Borchers & Thurber 2018, ch. 4: discrete ill-posed problem).
    """
    _validate_components(components)
    ee = np.atleast_1d(np.asarray(e, dtype=np.float64))
    nn = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if ee.ndim != 1 or ee.shape != nn.shape:
        raise ValueError(f"e/n must be equal-length 1-D, got {ee.shape}/{nn.shape}")
    cols: list[FloatArray] = []
    for comp in components:
        for k in range(patches.n_patches):
            src = patches.patch_source(k, **{comp: 1.0})
            cols.append(okada_forward(ee, nn, src, nu).ravel())
    return np.column_stack(cols)


def patch_laplacian(patches: FaultPatches, edge: str = "zero") -> FloatArray:
    """Discrete Laplacian ∇² over the patch grid (slip-roughness operator).

    Equation (5-point stencil on the fault plane, grid spacings
    ``h_l = patch_length`` along strike and ``h_w = patch_width`` down dip):
        ``(∇²s)_{ij} = (s_{i−1,j} − 2s_{ij} + s_{i+1,j})/h_l²
                     + (s_{i,j−1} − 2s_{ij} + s_{i,j+1})/h_w²``
    assembled as the matrix L with ``(∇²s) = L·s`` over the row-major patch
    vector. Edge treatment:
        - ``edge="zero"``: phantom zero-slip cells outside the fault
          (Dirichlet) — slip is penalized toward zero at the fault edges,
          the choice of Jónsson et al. 2002 (slip tapers to the perimeter).
        - ``edge="free"``: missing neighbors are dropped from the stencil
          (the center coefficient shrinks accordingly), so a constant slip
          field has exactly zero roughness (Neumann-like).

    Symbols → args:
        - ``s`` → the slip vector the operator will act on [L]
        - ``patches`` → :class:`FaultPatches` (grid shape + spacings)
        - ``edge``: ``"zero"`` | ``"free"`` boundary treatment

    Returns:
        L, shape ``(n_p, n_p)``, float64, symmetric, unit [1/L²].

    Reference:
        Harris & Segall 1987, JGR 92(B8) (smoothing operator D on the slip
        grid, their eq. (4)-(5) region); Jónsson et al. 2002, BSSA 92(4),
        p. 1382 (Laplacian smoothing with zero-slip edge condition);
        Aster, Borchers & Thurber 2018, ch. 4 (second-order Tikhonov: L as
        the roughening operator).

    Numerical notes:
        Symmetric by construction (equal spacing per axis); negative
        semi-definite. With ``edge="free"`` the constant vector spans the
        null space — pair it with data that constrain the mean slip, or use
        ``edge="zero"`` (strictly negative definite, unique minimizer).
        Dense (n_p²) assembly — fine for the ≤ O(10³) patches of GNSS-scale
        problems.
    """
    if edge not in ("zero", "free"):
        raise ValueError(f'edge must be "zero" or "free", got {edge!r}')
    na, nd = patches.n_along, patches.n_down
    inv_l2 = 1.0 / (patches.patch_length * patches.patch_length)
    inv_w2 = 1.0 / (patches.patch_width * patches.patch_width)
    n_p = na * nd
    lap = np.zeros((n_p, n_p), dtype=np.float64)
    for j in range(nd):
        for i in range(na):
            k = j * na + i
            diag = 0.0
            for ii, jj, w in (
                (i - 1, j, inv_l2),
                (i + 1, j, inv_l2),
                (i, j - 1, inv_w2),
                (i, j + 1, inv_w2),
            ):
                if 0 <= ii < na and 0 <= jj < nd:
                    lap[k, jj * na + ii] = w
                    diag -= w
                elif edge == "zero":
                    diag -= w  # phantom zero-slip neighbor keeps the stencil
            lap[k, k] = diag
    return lap


@dataclass(frozen=True)
class SlipDistribution:
    """Distributed-slip inversion result (:func:`okada_invert_slip`).

    Attributes:
        patches: The :class:`FaultPatches` grid the slip lives on.
        components: Estimated slip components, matching ``slip``'s first axis.
        slip: ``(n_comp, n_down, n_along)`` slip/opening values [L] on the
            grid (row j = 0 is the shallowest; i increases along strike).
        smoothing: Regularization weight λ used [L³ — see
            :func:`okada_invert_slip`].
        nonnegative: Whether the positivity constraint was imposed.
        predicted: ``(3, N)`` model displacements at the stations [L], rows
            (east, north, up).
        residual_norm: ``‖(d − G·s)/σ‖₂`` — σ-weighted misfit norm
            (dimensionless).
        roughness_norm: ``‖L_∇·s‖₂`` — unscaled Laplacian roughness of the
            solution [1/L]. (residual_norm, roughness_norm) is one point of
            the trade-off (L-)curve.
        rms: Unweighted RMS of the displacement residuals [L].
        n_obs: Number of scalar observations (3 × stations).
    """

    patches: FaultPatches
    components: tuple[str, ...]
    slip: FloatArray
    smoothing: float
    nonnegative: bool
    predicted: FloatArray
    residual_norm: float
    roughness_norm: float
    rms: float
    n_obs: int

    def potency(self) -> FloatArray:
        """Per-component geometric potency ``P_c = Σ_k s_ck·A_patch``.

        Equation:
            ``P_c = A_p · Σ_k s_{ck}``  [L³] — the surface integral of slip
            over the fault. The scalar seismic moment of a shear component
            is ``M₀ = μ·P`` (Aki & Richards 2002, *Quantitative Seismology*
            2nd ed., §3.2, eq. 3.16); for an opening component P is the
            cavity/dike volume change.

        Returns:
            ``(n_comp,)`` float64, ordered as ``components``.

        Numerical notes:
            Exact sum over the piecewise-constant slip representation.
        """
        return np.asarray(
            self.slip.reshape(len(self.components), -1).sum(axis=1)
            * self.patches.patch_area,
            dtype=np.float64,
        )


def _validate_components(components: tuple[str, ...]) -> None:
    """Reject empty/unknown/duplicated slip-component tuples."""
    if not components:
        raise ValueError("components must name at least one slip direction")
    unknown = set(components) - set(_SLIP_COMPONENTS)
    if unknown:
        raise ValueError(
            f"unknown slip components {sorted(unknown)}; choose from {_SLIP_COMPONENTS}"
        )
    if len(set(components)) != len(components):
        raise ValueError(f"duplicate slip components in {components}")


def _solve_regularized(
    g_w: FloatArray,
    d_w: FloatArray,
    reg: FloatArray,
    smoothing: float,
    nonnegative: bool,
) -> FloatArray:
    """Solve the damped LSQ system ``min ‖G_w·s − d_w‖² + λ²‖L_∇·s‖²`` (+ s ≥ 0).

    Equation (Tikhonov form via the augmented system):
        ``s* = argmin ‖ [G_w; λ·L_∇]·s − [d_w; 0] ‖²``  (Aster, Borchers &
        Thurber 2018, ch. 4, eq. 4.4-form with L = ∇²), optionally subject
        to ``s ≥ 0`` (bounded-variable least squares; the NNLS of Jónsson
        et al. 2002).

    Symbols → args:
        - ``G_w`` → ``g_w``: σ-weighted Green's matrix ``(m, p)``
        - ``d_w`` → ``d_w``: σ-weighted data ``(m,)``
        - ``L_∇`` → ``reg``: unscaled roughness operator ``(r, p)``
        - ``λ`` → ``smoothing``: regularization weight ≥ 0
        - ``nonnegative``: impose ``s ≥ 0`` elementwise

    Returns:
        ``s*`` ``(p,)`` float64.

    Reference:
        Aster, Borchers & Thurber 2018, *Parameter Estimation and Inverse
        Problems* (3rd ed.), ch. 4 (Tikhonov; augmented-matrix solution);
        Lawson & Hanson 1974 (NNLS, as wrapped by scipy ``lsq_linear``);
        Jónsson et al. 2002, BSSA 92(4) (NNLS + Laplacian for slip).

    Numerical notes:
        Unconstrained: LAPACK ``gelsd`` (SVD) via ``numpy.linalg.lstsq`` —
        rank-revealing, stable for the ill-conditioned G. Constrained:
        scipy ``lsq_linear`` (bounded-variable TRF), warm default tolerances;
        raises RuntimeError if it reports failure. The augmented system is
        dense ``(m + r, p)`` — fine at GNSS problem sizes.
    """
    a = np.vstack((g_w, smoothing * reg))
    b = np.concatenate((d_w, np.zeros(reg.shape[0], dtype=np.float64)))
    if nonnegative:
        res: Any = lsq_linear(a, b, bounds=(0.0, np.inf))
        if not res.success:
            raise RuntimeError(f"non-negative slip solve failed: {res.message}")
        return np.asarray(res.x, dtype=np.float64)
    sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(sol, dtype=np.float64)


def okada_invert_slip(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    patches: FaultPatches,
    components: tuple[str, ...] = ("opening",),
    smoothing: float,
    nonnegative: bool = False,
    edge: str = "zero",
    nu: float = DEFAULT_NU,
) -> SlipDistribution:
    """GPS-only distributed-slip inversion with Laplacian regularization.

    Estimator (second-order Tikhonov / smoothed slip):
        ``s* = argmin_s ‖(d − G·s)/σ‖² + λ²·‖L_∇·s‖²``, optionally s ≥ 0,
    with G = :func:`okada_greens` (unit-slip Okada patch responses),
    L_∇ = :func:`patch_laplacian` applied per component (block-diagonal for
    multi-component slip), λ = ``smoothing``. The fault **geometry is fixed**
    (given by ``patches``) — the problem is linear in s; geometry search
    belongs to :func:`okada_invert` / the Bayesian lane.

    Symbols → args:
        - ``d`` → ``obs``: ``(3, N)`` station displacements [L], rows
          **(east, north, up)**; ``σ`` → ``sigma``: ``(3, N)`` 1-σ [L] or
          ``None`` for unit weights
        - ``e, n``: station coordinates [L] (e.g. :func:`local_coordinates`)
        - ``patches``: fixed fault discretization (:func:`discretize_fault`)
        - ``components``: slip directions to estimate (default pure opening
          — dike/sill; use ``("strike_slip",)`` etc. for faulting)
        - ``λ`` → ``smoothing``: regularization weight ≥ 0. Units [L³]
          (G dimensionless kernel, L_∇ in 1/L²) — pick by
          :func:`slip_lcurve` (L-curve corner) or fix from experience;
          λ = 0 is the unregularized LSQ (requires ``3N ≥ n_params`` and is
          usually wildly oscillatory — Aster et al. 2018 ch. 4).
        - ``nonnegative``: impose s ≥ 0 on every component (physical
          one-signed slip/opening — Jónsson et al. 2002 use NNLS; flip the
          expected-negative component's sign convention at the call site
          if needed).
        - ``edge``: Laplacian boundary treatment (see
          :func:`patch_laplacian`; default ``"zero"`` tapers slip to the
          fault perimeter).
        - ``ν`` → ``nu``: Poisson's ratio [-].

    Returns:
        :class:`SlipDistribution` — slip grid, predicted field, misfit and
        roughness norms (one L-curve point), RMS.

    Reference:
        Okada 1985 (kernel); Harris & Segall 1987, JGR 92(B8), 7945–7962
        (regularized slip inversion from geodetic data — smoothing +
        positivity); Jónsson et al. 2002, BSSA 92(4), 1377–1389 (Laplacian
        smoothing + NNLS, distributed slip from InSAR/GPS); Aster, Borchers
        & Thurber 2018, ch. 4 (Tikhonov regularization, L-curve).

    Numerical notes:
        - SVD solve of the augmented system (see :func:`_solve_regularized`)
          — never normal equations (condition number would square).
        - The checkerboard/oscillation null-space of the pure LSQ is
          suppressed by λ‖L_∇s‖: resolution is *smoothing-limited*, so
          report recovered patterns together with λ (or the L-curve).
        - Requires ``3N > 0`` finite observations; λ = 0 additionally
          requires ``3N ≥ len(components)·n_patches``.
    """
    ee, nn, dd, ss = _as_obs_arrays(e, n, obs, sigma)
    _validate_components(components)
    if smoothing < 0.0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    n_comp = len(components)
    n_params = n_comp * patches.n_patches
    n_obs = 3 * ee.size
    if smoothing == 0.0 and n_obs < n_params:
        raise ValueError(
            f"underdetermined without smoothing: {n_obs} observations for "
            f"{n_params} slip parameters — set smoothing > 0"
        )
    g = okada_greens(ee, nn, patches, components, nu)
    w = 1.0 / ss.ravel()
    g_w = g * w[:, None]
    d_w = dd.ravel() * w
    lap = patch_laplacian(patches, edge)
    reg = (
        np.asarray(np.kron(np.eye(n_comp), lap), dtype=np.float64)
        if n_comp > 1
        else lap
    )
    sol = _solve_regularized(g_w, d_w, reg, smoothing, nonnegative)
    predicted = np.asarray(g @ sol, dtype=np.float64).reshape(3, ee.size)
    residual_norm = float(np.linalg.norm(d_w - g_w @ sol))
    roughness_norm = float(np.linalg.norm(reg @ sol))
    rms = float(np.sqrt(np.mean((dd - predicted) ** 2)))
    return SlipDistribution(
        patches=patches,
        components=tuple(components),
        slip=sol.reshape(n_comp, patches.n_down, patches.n_along),
        smoothing=float(smoothing),
        nonnegative=nonnegative,
        predicted=predicted,
        residual_norm=residual_norm,
        roughness_norm=roughness_norm,
        rms=rms,
        n_obs=n_obs,
    )


def lcurve_corner(residual_norms: ArrayLike, roughness_norms: ArrayLike) -> int:
    """Index of the L-curve corner (maximum curvature in log-log space).

    Equation (Menger three-point curvature on the log-log trade-off curve):
        for consecutive points ``P_i = (ln ρ_i, ln η_i)`` (ρ misfit norm,
        η roughness norm),
        ``κ_i = 4·Area(P_{i−1}, P_i, P_{i+1}) /
        (|P_{i−1}P_i|·|P_iP_{i+1}|·|P_{i−1}P_{i+1}|)``
        and the corner is ``argmax_i |κ_i|`` over the interior points — the
        point of maximum bending between the steep (under-smoothed) and flat
        (over-smoothed) branches.

    Symbols → args:
        - ``ρ_i`` → ``residual_norms``: weighted misfit per λ, ascending-λ order
        - ``η_i`` → ``roughness_norms``: solution roughness per λ, same order

    Returns:
        Corner index into the input arrays (1 ≤ index ≤ len − 2).

    Reference:
        Hansen 1992, SIAM Review 34(4), 561–580 (the L-curve criterion);
        Aster, Borchers & Thurber 2018, ch. 4 (L-curve for Tikhonov λ
        selection). Menger curvature: standard circumradius identity
        (used e.g. by Cultrera & Callegaro 2020, IOP SciNotes 1).

    Numerical notes:
        Sign-free (area-based) curvature — robust to the curve's traversal
        direction. Zero norms are floored at the float64 tiny value before
        the log. Requires ≥ 3 points; degenerate collinear segments give
        κ = 0 and cannot win unless all are collinear (then index 1).
    """
    rho = np.asarray(residual_norms, dtype=np.float64)
    eta = np.asarray(roughness_norms, dtype=np.float64)
    if rho.ndim != 1 or rho.shape != eta.shape:
        raise ValueError("residual_norms and roughness_norms must be equal 1-D")
    if rho.size < 3:
        raise ValueError(f"need >= 3 L-curve points, got {rho.size}")
    tiny = np.finfo(np.float64).tiny
    x = np.log(np.maximum(rho, tiny))
    y = np.log(np.maximum(eta, tiny))
    best_idx = 1
    best_kappa = -1.0
    for i in range(1, rho.size - 1):
        ax, ay = x[i] - x[i - 1], y[i] - y[i - 1]
        bx, by = x[i + 1] - x[i], y[i + 1] - y[i]
        cx_, cy_ = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
        area2 = abs(ax * by - ay * bx)  # 2·triangle area
        denom = math.hypot(ax, ay) * math.hypot(bx, by) * math.hypot(cx_, cy_)
        kappa = 2.0 * area2 / denom if denom > 0.0 else 0.0
        if kappa > best_kappa:
            best_kappa = kappa
            best_idx = i
    return best_idx


def slip_lcurve(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    patches: FaultPatches,
    smoothings: ArrayLike,
    components: tuple[str, ...] = ("opening",),
    nonnegative: bool = False,
    edge: str = "zero",
    nu: float = DEFAULT_NU,
) -> tuple[FloatArray, FloatArray, int]:
    """Trade-off (L-)curve of the distributed-slip inversion over λ values.

    Computes one :func:`okada_invert_slip` solve per λ (Green's matrix and
    Laplacian built once) and returns the misfit/roughness norms plus the
    corner index of :func:`lcurve_corner` — the standard Tikhonov
    λ-selection diagnostic:

        ``(ρ(λ), η(λ)) = (‖(d − G·s_λ)/σ‖, ‖L_∇·s_λ‖)``

    Symbols → args:
        - ``λ`` → ``smoothings``: 1-D array of trial weights, > 0, sorted
          ascending (a log-spaced decade scan is the usual choice)
        - others as :func:`okada_invert_slip`

    Returns:
        ``(residual_norms, roughness_norms, corner_index)`` — norms per λ
        (same order), and the recommended λ's index.

    Reference:
        Hansen 1992, SIAM Review 34(4) (L-curve criterion); Aster, Borchers
        & Thurber 2018, ch. 4; applied to slip inversion: Jónsson et al.
        2002, BSSA 92(4) (smoothing chosen from the misfit/roughness
        trade-off, their fig. 6).

    Numerical notes:
        With ``nonnegative=True`` the curve is that of the *constrained*
        estimator (norms need not be monotone in λ near active constraints);
        the corner heuristic still applies. Cost: one regularized solve per
        λ on the shared G — the G build (n_comp·n_p Okada evaluations)
        dominates and is done once.
    """
    ee, nn, dd, ss = _as_obs_arrays(e, n, obs, sigma)
    _validate_components(components)
    lams = np.asarray(smoothings, dtype=np.float64)
    if lams.ndim != 1 or lams.size < 3:
        raise ValueError("smoothings must be 1-D with >= 3 values")
    if not bool(np.all(lams > 0.0)):
        raise ValueError("smoothings must be strictly positive")
    if bool(np.any(np.diff(lams) <= 0.0)):
        raise ValueError("smoothings must be sorted strictly ascending")
    n_comp = len(components)
    g = okada_greens(ee, nn, patches, components, nu)
    w = 1.0 / ss.ravel()
    g_w = g * w[:, None]
    d_w = dd.ravel() * w
    lap = patch_laplacian(patches, edge)
    reg = (
        np.asarray(np.kron(np.eye(n_comp), lap), dtype=np.float64)
        if n_comp > 1
        else lap
    )
    residual_norms = np.empty(lams.size, dtype=np.float64)
    roughness_norms = np.empty(lams.size, dtype=np.float64)
    for idx, lam in enumerate(lams):
        sol = _solve_regularized(g_w, d_w, reg, float(lam), nonnegative)
        residual_norms[idx] = float(np.linalg.norm(d_w - g_w @ sol))
        roughness_norms[idx] = float(np.linalg.norm(reg @ sol))
    corner = lcurve_corner(residual_norms, roughness_norms)
    return residual_norms, roughness_norms, corner


# =====================================================================
# Inversion — Bayesian (GBIS Metropolis, reused from transient)
# =====================================================================


@dataclass(frozen=True)
class MogiPosterior:
    """Posterior samples of a Bayesian Mogi inversion.

    Attributes:
        m_keep: ``(4, n_runs)`` kept chain in ``[x, y, depth, dv]`` order
            (rejected iterations repeat the previous column — GBIS
            convention, as :class:`gps_analysis.transient.InversionResult`).
        p_keep: ``(n_runs,)`` log-posterior (log-likelihood, flat priors)
            per kept column.
        optimal: Maximum-a-posteriori :class:`MogiSource`.
        p_opt: Log-posterior at the optimum.
    """

    m_keep: FloatArray
    p_keep: FloatArray
    optimal: MogiSource
    p_opt: float


def mogi_invert_bayes(
    e: ArrayLike,
    n: ArrayLike,
    obs: ArrayLike,
    sigma: ArrayLike,
    bounds: PriorBounds,
    config: InversionConfig,
    nu: float = DEFAULT_NU,
) -> MogiPosterior:
    """Bayesian Mogi inversion — posterior sampling with the GBIS sampler.

    Model:
        Gaussian independent-error likelihood under uniform priors,
        ``ln P(m | d) = −½ Σᵢ ((dᵢ − Gᵢ(m))/σᵢ)² + const`` (constant
        dropped — only differences enter Metropolis ratios), with
        G = :func:`mogi_forward` over ``m = [x_s, y_s, d, ΔV]``, sampled by
        the shared GBIS sampler :func:`gps_analysis._mcmc.metropolis`
        (annealed adaptive Metropolis; Bagnardi & Hooper 2018 §3) — the same
        engine as the transient lane, hook-free here (no hyperparameter
        slot, no break-point step floor, no ordering guard).

    Symbols → args:
        - ``dᵢ`` → ``obs``: ``(3, N)`` displacements [L], rows (east, north,
          up); ``σᵢ`` → ``sigma``: ``(3, N)`` 1-σ [L] (REQUIRED — a Bayesian
          result without honest errors is meaningless)
        - ``e, n``: station coordinates [L]
        - ``bounds``: :class:`~gps_analysis.transient.PriorBounds` over
          ``[x, y, depth, dv]`` — start, uniform-prior limits, initial steps
        - ``config``: :class:`~gps_analysis.transient.InversionConfig`
          (``n_runs`` kept samples; seed for reproducibility)
        - ``ν`` → ``nu``

    Returns:
        :class:`MogiPosterior` — kept chain, log-posteriors, MAP source.
        Posterior intervals: discard the annealed burn-in
        (``16·config.t_runs`` samples) before taking percentiles, as in the
        GBIS4TS workflow.

    Reference:
        Bagnardi & Hooper 2018, G³ 19 (doi:10.1029/2018GC007585) — the GBIS
        methodology this reuses; forward model Mogi 1958 / Segall 2010 §7.1.
        Sampler infrastructure shared with
        :func:`gps_analysis.transient.run_inversion` (Yang et al. 2023
        adaptation), NOT reinvented.

    Numerical notes:
        Uncorrelated (diagonal) data covariance: appropriate for daily GNSS
        displacement snapshots; correlated-noise likelihoods belong to the
        time-series lane (:mod:`gps_analysis.transient`). Chain cost is one
        Mogi forward per iteration — O(N) — so 10⁵ samples on a 25-station
        network run in seconds.
    """
    ee, nn, dd, ss = _as_obs_arrays(e, n, obs, sigma)
    if bounds.start.size != 4:
        raise ValueError(
            f"bounds must cover the 4 Mogi parameters [x, y, depth, dv], "
            f"got {bounds.start.size}"
        )
    if not float(bounds.lower[2]) > 0.0:
        raise ValueError("depth prior lower bound must be > 0")
    dflat = dd.ravel()
    sflat = ss.ravel()

    def log_post(m: FloatArray) -> float:
        # The GBIS single-parameter sensitivity perturbation only reflects at
        # the UPPER bound (runInversion_ts.m l.235, reproduced in the sampler),
        # so a downward kick can transiently propose depth ≤ 0 — physically a
        # zero-prior region. Return −∞ (rejected) rather than evaluate the
        # forward model there; the uniform prior [lower, upper] is otherwise
        # enforced by the reflecting random-walk proposals.
        if m[2] <= 0.0:
            return -math.inf
        model = mogi_forward(ee, nn, MogiSource.from_array(m), nu)
        r = (dflat - model.ravel()) / sflat
        return -0.5 * float(r @ r)

    result = metropolis(log_post, bounds, config)
    return MogiPosterior(
        m_keep=result.m_keep,
        p_keep=result.p_keep,
        optimal=MogiSource.from_array(result.optimal),
        p_opt=result.p_opt,
    )
