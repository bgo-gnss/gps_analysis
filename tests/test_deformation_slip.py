"""Tests for the distributed-slip lane of gps_analysis.deformation.

Coverage (MATH_STANDARDS §4):
- **Superposition parity**: tiled patches with uniform slip reproduce the
  parent plane's :func:`okada_forward` field to machine precision (linear
  elasticity — the discretization is geometry-exact), across strike/dip
  orientations including the vertical-fault branch.
- **Green's matrix**: columns are the unit-slip patch responses; layout is
  component-major.
- **Laplacian operator**: 5-point stencil identities — symmetric, constant
  slip annihilated for ``edge="free"``, penalized at edges for
  ``edge="zero"``, interior rows identical between the two.
- **Round-trip recovery** (the plan's validation gate): impose a Gaussian
  slip patch → forward → add noise → invert with Laplacian smoothing →
  recover the pattern (correlation, relative L2, peak cell, potency), with
  and without the non-negativity constraint; L-curve corner within the
  scanned decades and usable.
- Validation errors (components, smoothing, underdetermined λ = 0, grid).

Tolerances: superposition at rtol 1e-12 (a few ulp of the analytic Okada
field); recovery thresholds are Monte-Carlo margins for the fixed seed
(corr > 0.95 / relL2 < 0.3 at the tuned λ — smoothing-limited resolution,
see okada_invert_slip numerical notes).
"""

import numpy as np
import pytest

from gps_analysis.deformation import (
    FaultPatches,
    OkadaSource,
    discretize_fault,
    lcurve_corner,
    okada_forward,
    okada_greens,
    okada_invert_slip,
    patch_laplacian,
    slip_lcurve,
)


def _plane(
    strike: float = 20.0,
    dip: float = 70.0,
    depth: float = 3600.0,
    length: float = 12000.0,
    width: float = 6000.0,
) -> OkadaSource:
    return OkadaSource(
        x=0.0,
        y=0.0,
        depth=depth,
        strike=strike,
        dip=dip,
        length=length,
        width=width,
        strike_slip=0.0,
        dip_slip=0.0,
        opening=0.0,
    )


def _stations(
    half_span: float = 15000.0, n_side: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.linspace(-half_span, half_span, n_side)
    ee, nn = np.meshgrid(gx, gx)
    return ee.ravel(), nn.ravel()


# =====================================================================
# Discretization geometry
# =====================================================================


def test_discretize_fault_grid_and_centroid() -> None:
    plane = _plane()
    patches = discretize_fault(plane, n_along=6, n_down=4)
    assert patches.n_patches == 24
    assert patches.patch_length == pytest.approx(2000.0)
    assert patches.patch_width == pytest.approx(1500.0)
    assert patches.patch_area == pytest.approx(3.0e6)
    assert patches.centers.shape == (24, 3)
    # symmetric tiling: patch centroids average to the plane centroid
    np.testing.assert_allclose(
        patches.centers.mean(axis=0), [plane.x, plane.y, plane.depth], atol=1e-9
    )
    # row-major: first row is the shallowest, depth increases with j
    depths = patches.centers[:, 2].reshape(4, 6)
    assert bool(np.all(np.diff(depths, axis=0) > 0.0))
    assert bool(np.all(depths[0] < plane.depth))


def test_discretize_fault_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=">= 1x1"):
        discretize_fault(_plane(), n_along=0, n_down=2)
    with pytest.raises(ValueError, match="breaches the surface"):
        discretize_fault(_plane(depth=1000.0, dip=90.0, width=4000.0), 2, 2)


@pytest.mark.parametrize(
    ("strike", "dip"),
    [(30.0, 60.0), (0.0, 45.0), (120.0, 90.0), (275.0, 30.0)],
)
def test_patch_superposition_matches_parent_plane(strike: float, dip: float) -> None:
    """Uniform slip over the tiles == the parent plane's field (exact)."""
    plane = OkadaSource(
        x=1000.0,
        y=-2000.0,
        depth=5000.0,
        strike=strike,
        dip=dip,
        length=8000.0,
        width=4000.0,
        strike_slip=1.0,
        dip_slip=0.7,
        opening=0.5,
    )
    patches = discretize_fault(plane, n_along=4, n_down=3)
    rng = np.random.default_rng(0)
    e = rng.uniform(-15000.0, 15000.0, 25)
    n = rng.uniform(-15000.0, 15000.0, 25)
    u_full = okada_forward(e, n, plane)
    u_sum = np.zeros_like(u_full)
    for k in range(patches.n_patches):
        u_sum += okada_forward(
            e,
            n,
            patches.patch_source(k, strike_slip=1.0, dip_slip=0.7, opening=0.5),
        )
    scale = float(np.abs(u_full).max())
    np.testing.assert_allclose(u_sum, u_full, atol=1e-12 * scale)


def test_patch_source_index_bounds() -> None:
    patches = discretize_fault(_plane(), 3, 2)
    with pytest.raises(ValueError, match="patch index"):
        patches.patch_source(6)


# =====================================================================
# Green's matrix
# =====================================================================


def test_okada_greens_columns_are_unit_slip_responses() -> None:
    patches = discretize_fault(_plane(), 4, 2)
    e, n = _stations(n_side=4)
    g = okada_greens(e, n, patches, ("strike_slip", "opening"))
    assert g.shape == (3 * e.size, 2 * patches.n_patches)
    # column 3 = patch 3 with unit strike-slip
    direct = okada_forward(e, n, patches.patch_source(3, strike_slip=1.0))
    np.testing.assert_allclose(g[:, 3], direct.ravel(), rtol=1e-14)
    # opening block starts at n_patches: column n_p + 5 = patch 5, unit opening
    direct2 = okada_forward(e, n, patches.patch_source(5, opening=1.0))
    np.testing.assert_allclose(g[:, patches.n_patches + 5], direct2.ravel(), rtol=1e-14)


def test_okada_greens_predicts_superposed_field() -> None:
    """G·s equals the summed forward fields of the slipping patches."""
    patches = discretize_fault(_plane(), 5, 3)
    e, n = _stations(n_side=5)
    rng = np.random.default_rng(1)
    s = rng.uniform(0.0, 2.0, patches.n_patches)
    g = okada_greens(e, n, patches, ("opening",))
    d_matrix = (g @ s).reshape(3, e.size)
    d_direct = np.zeros_like(d_matrix)
    for k in range(patches.n_patches):
        d_direct += okada_forward(e, n, patches.patch_source(k, opening=float(s[k])))
    np.testing.assert_allclose(d_matrix, d_direct, rtol=1e-12, atol=1e-18)


def test_okada_greens_rejects_bad_components() -> None:
    patches = discretize_fault(_plane(), 2, 2)
    e, n = _stations(n_side=3)
    with pytest.raises(ValueError, match="unknown slip components"):
        okada_greens(e, n, patches, ("slip",))
    with pytest.raises(ValueError, match="at least one"):
        okada_greens(e, n, patches, ())
    with pytest.raises(ValueError, match="duplicate"):
        okada_greens(e, n, patches, ("opening", "opening"))


# =====================================================================
# Laplacian roughness operator
# =====================================================================


def test_patch_laplacian_is_symmetric() -> None:
    patches = discretize_fault(_plane(), 5, 4)
    for edge in ("zero", "free"):
        lap = patch_laplacian(patches, edge)
        np.testing.assert_array_equal(lap, lap.T)


def test_patch_laplacian_free_edge_annihilates_constant() -> None:
    patches = discretize_fault(_plane(), 5, 4)
    lap = patch_laplacian(patches, "free")
    const = np.full(patches.n_patches, 3.7)
    np.testing.assert_allclose(lap @ const, 0.0, atol=1e-12)


def test_patch_laplacian_zero_edge_penalizes_boundary_slip() -> None:
    """Dirichlet phantom cells: constant slip is penalized at edges only."""
    patches = discretize_fault(_plane(), 5, 4)
    lap = patch_laplacian(patches, "zero")
    const = np.ones(patches.n_patches)
    r = (lap @ const).reshape(4, 5)
    assert bool(np.all(r[1:-1, 1:-1] == 0.0))  # interior stencils cancel
    assert bool(np.all(r[0] < 0.0)) and bool(np.all(r[-1] < 0.0))
    assert bool(np.all(r[:, 0] < 0.0)) and bool(np.all(r[:, -1] < 0.0))


def test_patch_laplacian_interior_stencil_weights() -> None:
    """Interior row: 1/h_l² along strike, 1/h_w² down dip, −2(Σ) center."""
    patches = discretize_fault(_plane(), 5, 4)  # dl = 2400, dw = 1500
    lap = patch_laplacian(patches, "zero")
    na = patches.n_along
    k = 2 * na + 2  # interior cell (i=2, j=2)
    inv_l2 = 1.0 / patches.patch_length**2
    inv_w2 = 1.0 / patches.patch_width**2
    assert lap[k, k - 1] == pytest.approx(inv_l2)
    assert lap[k, k + 1] == pytest.approx(inv_l2)
    assert lap[k, k - na] == pytest.approx(inv_w2)
    assert lap[k, k + na] == pytest.approx(inv_w2)
    assert lap[k, k] == pytest.approx(-2.0 * (inv_l2 + inv_w2))


def test_patch_laplacian_rejects_unknown_edge() -> None:
    patches = discretize_fault(_plane(), 2, 2)
    with pytest.raises(ValueError, match="edge"):
        patch_laplacian(patches, "periodic")


# =====================================================================
# Round-trip recovery (synthetic slip distribution)
# =====================================================================


def _roundtrip_setup() -> tuple[
    FaultPatches, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Gaussian strike-slip patch on a 12×6 grid; 81 stations; 2 mm noise."""
    patches = discretize_fault(_plane(), n_along=12, n_down=6)
    ii, jj = np.meshgrid(np.arange(12), np.arange(6), indexing="xy")
    true = 2.0 * np.exp(-(((ii - 7.0) / 2.2) ** 2 + ((jj - 2.0) / 1.4) ** 2))
    true_flat = true.ravel()
    e, n = _stations()
    g = okada_greens(e, n, patches, ("strike_slip",))
    clean = (g @ true_flat).reshape(3, e.size)
    rng = np.random.default_rng(3)
    sigma = np.full((3, e.size), 0.002)
    obs = clean + rng.normal(0.0, sigma)
    return patches, e, n, obs, sigma, true_flat


def test_okada_invert_slip_roundtrip_recovers_pattern() -> None:
    """Impose → forward → invert: pattern recovered under smoothing."""
    patches, e, n, obs, sigma, true_flat = _roundtrip_setup()
    fit = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        components=("strike_slip",),
        smoothing=1.0e6,
    )
    assert fit.slip.shape == (1, 6, 12)
    rec = fit.slip[0].ravel()
    corr = float(np.corrcoef(rec, true_flat)[0, 1])
    rel = float(np.linalg.norm(rec - true_flat) / np.linalg.norm(true_flat))
    assert corr > 0.95
    assert rel < 0.3
    # peak recovered in the true peak cell (j=2, i=7)
    peak_j, peak_i = np.unravel_index(int(np.argmax(fit.slip[0])), (6, 12))
    assert (int(peak_j), int(peak_i)) == (2, 7)
    # potency (slip integral) within 15 %
    pot_true = float(true_flat.sum()) * patches.patch_area
    assert fit.potency()[0] == pytest.approx(pot_true, rel=0.15)
    # fit statistics: data fit near the noise level
    assert fit.rms < 0.004
    assert fit.n_obs == 3 * e.size
    assert fit.predicted.shape == (3, e.size)


def test_okada_invert_slip_nonnegative_constraint() -> None:
    """s ≥ 0 (NNLS, Jónsson et al. 2002) holds and preserves recovery."""
    patches, e, n, obs, sigma, true_flat = _roundtrip_setup()
    fit = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        components=("strike_slip",),
        smoothing=1.0e6,
        nonnegative=True,
    )
    rec = fit.slip[0].ravel()
    assert bool(np.all(rec >= 0.0))
    assert float(np.corrcoef(rec, true_flat)[0, 1]) > 0.95
    assert fit.nonnegative


def test_okada_invert_slip_oversmoothed_flattens() -> None:
    """λ ≫ corner drives roughness → 0 and inflates the misfit (Tikhonov)."""
    patches, e, n, obs, sigma, _ = _roundtrip_setup()
    mild = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        smoothing=1.0e6,
        components=("strike_slip",),
    )
    hard = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        smoothing=1.0e10,
        components=("strike_slip",),
    )
    assert hard.roughness_norm < 1.0e-3 * mild.roughness_norm
    assert hard.residual_norm > 2.0 * mild.residual_norm


def test_slip_lcurve_corner_selects_usable_smoothing() -> None:
    """L-curve: monotone trade-off, interior corner, decent recovery there."""
    patches, e, n, obs, sigma, true_flat = _roundtrip_setup()
    lams = np.geomspace(1.0e4, 1.0e10, 13)
    rho, eta, corner = slip_lcurve(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        smoothings=lams,
        components=("strike_slip",),
    )
    assert rho.shape == eta.shape == (13,)
    # trade-off monotonicity: misfit grows, roughness shrinks with λ
    assert bool(np.all(np.diff(rho) > 0.0))
    assert bool(np.all(np.diff(eta) < 0.0))
    assert 0 < corner < 12
    fit = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        components=("strike_slip",),
        smoothing=float(lams[corner]),
    )
    corr = float(np.corrcoef(fit.slip[0].ravel(), true_flat)[0, 1])
    assert corr > 0.8  # corner is smoothing-limited, not optimal — usable


def test_lcurve_corner_finds_synthetic_kink() -> None:
    """Analytic L: two straight log-log branches meeting at index 3."""
    rho = np.array([1.0, 1.1, 1.2, 1.3, 10.0, 100.0, 1000.0])
    eta = np.array([1000.0, 100.0, 10.0, 1.0, 0.9, 0.8, 0.7])
    assert lcurve_corner(rho, eta) == 3


def test_lcurve_corner_validation() -> None:
    with pytest.raises(ValueError, match=">= 3"):
        lcurve_corner([1.0, 2.0], [2.0, 1.0])
    with pytest.raises(ValueError, match="equal 1-D"):
        lcurve_corner([1.0, 2.0, 3.0], [1.0, 2.0])


def test_okada_invert_slip_validation() -> None:
    patches, e, n, obs, sigma, _ = _roundtrip_setup()
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        okada_invert_slip(
            e,
            n,
            obs,
            sigma,
            patches=patches,
            smoothing=-1.0,
            components=("strike_slip",),
        )
    # λ = 0 with fewer observations than parameters is rejected
    few_e, few_n = e[:10], n[:10]
    with pytest.raises(ValueError, match="underdetermined"):
        okada_invert_slip(
            few_e,
            few_n,
            obs[:, :10],
            sigma[:, :10],
            patches=patches,
            smoothing=0.0,
            components=("strike_slip",),
        )
    with pytest.raises(ValueError, match="unknown slip components"):
        okada_invert_slip(
            e,
            n,
            obs,
            sigma,
            patches=patches,
            smoothing=1.0,
            components=("rake",),
        )


def test_slip_lcurve_validation() -> None:
    patches, e, n, obs, sigma, _ = _roundtrip_setup()
    with pytest.raises(ValueError, match=">= 3 values"):
        slip_lcurve(
            e,
            n,
            obs,
            sigma,
            patches=patches,
            smoothings=[1.0, 2.0],
            components=("strike_slip",),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        slip_lcurve(
            e,
            n,
            obs,
            sigma,
            patches=patches,
            smoothings=[0.0, 1.0, 2.0],
            components=("strike_slip",),
        )
    with pytest.raises(ValueError, match="ascending"):
        slip_lcurve(
            e,
            n,
            obs,
            sigma,
            patches=patches,
            smoothings=[2.0, 1.0, 3.0],
            components=("strike_slip",),
        )


def test_okada_invert_slip_multicomponent_opening() -> None:
    """Two-component inversion separates opening from strike slip.

    Uses the non-negativity constraint: the inter-component trade-off of the
    unconstrained joint problem is its dominant error source, and positivity
    suppresses it (exactly the observation motivating NNLS in Jónsson et al.
    2002) — corr rises from ~0.77/0.86 to ~0.93/0.95 at the same λ.
    """
    patches = discretize_fault(_plane(), n_along=8, n_down=4)
    ii, jj = np.meshgrid(np.arange(8), np.arange(4), indexing="xy")
    true_op = 1.5 * np.exp(-(((ii - 3.0) / 1.6) ** 2 + ((jj - 1.0) / 1.0) ** 2))
    true_ss = 1.0 * np.exp(-(((ii - 5.0) / 1.6) ** 2 + ((jj - 2.0) / 1.0) ** 2))
    s_true = np.concatenate([true_ss.ravel(), true_op.ravel()])
    e, n = _stations()
    g = okada_greens(e, n, patches, ("strike_slip", "opening"))
    rng = np.random.default_rng(11)
    sigma = np.full((3, e.size), 0.002)
    obs = (g @ s_true).reshape(3, e.size) + rng.normal(0.0, sigma)
    fit = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        components=("strike_slip", "opening"),
        smoothing=1.0e6,
        nonnegative=True,
    )
    assert fit.slip.shape == (2, 4, 8)
    corr_ss = float(np.corrcoef(fit.slip[0].ravel(), true_ss.ravel())[0, 1])
    corr_op = float(np.corrcoef(fit.slip[1].ravel(), true_op.ravel())[0, 1])
    assert corr_ss > 0.9
    assert corr_op > 0.9
    # potency per component, ordered as `components`
    pot = fit.potency()
    assert pot.shape == (2,)
    assert pot[1] == pytest.approx(true_op.sum() * patches.patch_area, rel=0.25)
