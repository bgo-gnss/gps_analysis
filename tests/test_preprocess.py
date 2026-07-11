"""Tests for gps_analysis.preprocess (MATH_STANDARDS §4).

Covers legacy parity — the ``vshift``/``iprep`` chain of
``geo_dataread.gps_read`` reproduced inline (uncertainty screen via
boolean-index+reshape, 1/sigma-weighted reference over the first
``Period`` kept samples or a time window, per-row subtraction) and
required to match **bit-exactly** (``assert_array_equal``, float64) —
plus analytic checks, the offset-reuse contract, purity (no input
mutation), and the explicit error branches that replace the legacy
broken-branch crashes.
"""

import numpy as np
import pytest

from gps_analysis.preprocess import (
    prep_neu_series,
    prep_plot_series,
    screen_uncertainty,
)

RNG = np.random.default_rng(20260711)


def _series(
    n: int = 200, nan_sigma: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A (t, y, sigma) triple shaped like an openGlobkTimes read (3, N)."""
    t = np.sort(RNG.uniform(2015.0, 2026.0, size=n))
    y = RNG.normal(0.0, 0.01, size=(3, n)) + np.array([[0.4], [-0.2], [0.05]])
    sigma = RNG.uniform(0.001, 2.0, size=(3, n))
    if nan_sigma:
        sigma[1, 7] = np.nan
        sigma[2, 13] = np.nan
    return t, y, sigma


def _legacy_vshift(
    yearf: np.ndarray,
    data: np.ndarray,
    Ddata: np.ndarray,
    uncert: float = 20.0,
    window: tuple[float, float] | None = None,
    Period: int = 5,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """geo_dataread.gps_read.vshift + estimate_offset, verbatim inline.

    ``window`` stands in for the legacy refdate/Period pair already
    converted to fractional years (that conversion is caller policy).
    """
    with np.errstate(invalid="ignore"):
        filt = Ddata < uncert
    filt = np.logical_and(np.logical_and(filt[0, :], filt[1, :]), filt[2, :])

    yearf = yearf[filt]
    data = np.reshape(data[np.array([filt, filt, filt])], (3, -1))
    Ddata = np.reshape(Ddata[np.array([filt, filt, filt])], (3, -1))

    if data.any():
        if offset is None:
            if window is not None:
                start, end = window
                index = np.where(yearf <= start - 0.001)
                tmpyearf = np.delete(yearf, index)
                tmpdata = np.delete(data, index, 1)
                tmpDdata = np.delete(Ddata, index, 1)
                index = np.where(tmpyearf >= end + 0.001)
                tmpdata = np.delete(tmpdata, index, 1)
                tmpDdata = np.delete(tmpDdata, index, 1)
                offset = np.average(tmpdata[0:3, :], 1, weights=1 / tmpDdata[0:3, :])
            else:
                offset = np.average(
                    data[0:3, 0:Period], 1, weights=1 / Ddata[0:3, 0:Period]
                )

    assert offset is not None
    data = np.array([data[i, :] - offset[i] for i in range(3)])
    return yearf, data, Ddata, offset


class TestScreenUncertainty:
    def test_analytic_mask(self) -> None:
        sigma = np.array(
            [
                [0.5, 3.0, 0.1, np.nan, 0.2],
                [0.5, 0.5, 0.1, 0.1, 0.2],
                [0.5, 0.5, 0.1, 0.1, 2.5],
            ]
        )
        np.testing.assert_array_equal(
            screen_uncertainty(sigma, 2.0), [True, False, True, False, False]
        )

    def test_one_dimensional(self) -> None:
        sigma = np.array([0.1, 5.0, np.nan, 1.0])
        np.testing.assert_array_equal(
            screen_uncertainty(sigma, 2.0), [True, False, False, True]
        )

    def test_strict_inequality(self) -> None:
        assert not screen_uncertainty(np.array([[2.0], [1.0], [1.0]]), 2.0)[0]

    def test_bad_ndim(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            screen_uncertainty(np.zeros((2, 2, 2)), 1.0)


class TestLegacyParityCountMode:
    """Bit-parity with vshift(refdate=None) — the live .NEU + plot path."""

    @pytest.mark.parametrize("uncert", [1.1, 15.0, 20.0])
    def test_neu_profile_matches_legacy(self, uncert: float) -> None:
        t, y, sigma = _series()
        ref = _legacy_vshift(t.copy(), y.copy(), sigma.copy(), uncert=uncert)
        got = prep_neu_series(t, y, sigma, max_sigma=uncert)
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_array_equal(g, r)

    def test_plot_profile_matches_legacy(self) -> None:
        # iprep minus its in-place m→mm scaling: mm-scaled inputs, uncert=15.
        t, y, sigma = _series()
        y_mm = y * 1000.0
        # realistic plot-profile uncertainties: a few to a few tens of mm,
        # so the 15 mm screen keeps a real subset
        s_mm = RNG.uniform(1.0, 30.0, size=sigma.shape)
        s_mm[1, 7] = np.nan
        ref = _legacy_vshift(t.copy(), y_mm.copy(), s_mm.copy(), uncert=15.0)
        got = prep_plot_series(t, y_mm, s_mm, max_sigma=15.0)
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_array_equal(g, r)

    def test_ref_samples_none_averages_everything(self) -> None:
        # legacy Period=None slices 0:None — the whole screened series.
        # No live caller uses this configuration; the full-series reduction
        # is only reproducible to ~1 ULP because numpy's pairwise/SIMD sum
        # grouping depends on the (allocation-route-dependent) array layout —
        # the live 5-sample count mode is bit-stable and asserted exactly
        # elsewhere in this class.
        t, y, sigma = _series(nan_sigma=False)
        ref = _legacy_vshift(t.copy(), y.copy(), sigma.copy(), Period=None)  # type: ignore[arg-type]
        got = prep_neu_series(t, y, sigma, max_sigma=20.0, ref_samples=None)
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_allclose(g, r, rtol=1e-12, atol=1e-15)


class TestLegacyParityWindowMode:
    """Bit-parity with vshift(refdate=...) — the window-referenced path."""

    def test_window_matches_legacy(self) -> None:
        t, y, sigma = _series()
        window = (2018.0, 2018.75)
        ref = _legacy_vshift(t.copy(), y.copy(), sigma.copy(), window=window)
        got = prep_neu_series(
            t, y, sigma, max_sigma=20.0, ref_start=window[0], ref_end=window[1]
        )
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_array_equal(g, r)

    def test_empty_window_is_explicit(self) -> None:
        # legacy: broken extrapolation branch (NameError on undefined `j`);
        # here: explicit ValueError from baseline.estimate_offset.
        t, y, sigma = _series()
        with pytest.raises(ValueError, match="no samples"):
            prep_neu_series(
                t, y, sigma, max_sigma=20.0, ref_start=2030.0, ref_end=2031.0
            )


class TestOffsetContract:
    def test_supplied_offset_bypasses_estimation(self) -> None:
        t, y, sigma = _series()
        offset = np.array([0.35, -0.15, 0.02])
        ref = _legacy_vshift(
            t.copy(), y.copy(), sigma.copy(), uncert=1.1, offset=offset
        )
        got = prep_neu_series(t, y, sigma, max_sigma=1.1, offset=offset)
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_array_equal(g, r)

    def test_offset_reuse_reproduces_shift(self) -> None:
        # getData's contract: the returned level, fed back in, pins a second
        # call to the same zero.
        t, y, sigma = _series()
        t1, y1, s1, level = prep_plot_series(t, y, sigma, max_sigma=15.0)
        t2, y2, s2, level2 = prep_plot_series(t, y, sigma, max_sigma=15.0, offset=level)
        np.testing.assert_array_equal(y2, y1)
        np.testing.assert_array_equal(level2, level)

    def test_returned_offset_is_first_samples_weighted_mean(self) -> None:
        t, y, sigma = _series(nan_sigma=False)
        sigma = np.minimum(sigma, 0.5)  # nothing screened out
        _, _, _, level = prep_neu_series(t, y, sigma, max_sigma=1.0, ref_samples=5)
        expected = np.average(y[:, :5], axis=1, weights=1.0 / sigma[:, :5])
        np.testing.assert_array_equal(level, expected)

    def test_supplied_offset_with_empty_screen_returns_empty(self) -> None:
        # legacy: subtraction on the empty (3, 0) arrays still runs.
        t, y, sigma = _series()
        offset = np.array([1.0, 2.0, 3.0])
        t2, y2, s2, level = prep_neu_series(t, y, sigma, max_sigma=0.0, offset=offset)
        assert t2.size == 0 and y2.shape == (3, 0) and s2.shape == (3, 0)
        np.testing.assert_array_equal(level, offset)


class TestAnalytic:
    def test_zeroes_the_reference_window_mean(self) -> None:
        # constant series → shifted series is exactly zero, offset the constant
        t = np.linspace(2020.0, 2021.0, 50)
        y = np.tile(np.array([[1.5], [-2.0], [0.25]]), (1, 50))
        sigma = np.full((3, 50), 0.1)
        t2, y2, s2, level = prep_neu_series(t, y, sigma, max_sigma=1.0)
        np.testing.assert_array_equal(level, [1.5, -2.0, 0.25])
        np.testing.assert_array_equal(y2, np.zeros((3, 50)))

    def test_one_dimensional_series(self) -> None:
        t = np.linspace(2020.0, 2021.0, 20)
        y = np.full(20, 3.0)
        sigma = np.full(20, 0.1)
        t2, y2, s2, level = prep_plot_series(t, y, sigma, max_sigma=1.0)
        assert float(level) == 3.0
        np.testing.assert_array_equal(y2, np.zeros(20))

    def test_no_data_no_offset_is_explicit(self) -> None:
        # legacy: offset stays None → TypeError; here: explicit ValueError.
        t, y, sigma = _series()
        with pytest.raises(ValueError, match="no nonzero data"):
            prep_neu_series(t, y, sigma, max_sigma=0.0)

    def test_shape_mismatch(self) -> None:
        t, y, sigma = _series()
        with pytest.raises(ValueError, match="does not match"):
            prep_neu_series(t, y, sigma[:, :-1], max_sigma=1.0)
        with pytest.raises(ValueError, match="t must be 1-D"):
            prep_neu_series(t[:-1], y, sigma, max_sigma=1.0)


class TestPurity:
    def test_inputs_never_mutated(self) -> None:
        t, y, sigma = _series()
        t0, y0, s0 = t.copy(), y.copy(), sigma.copy()
        prep_neu_series(t, y, sigma, max_sigma=1.1)
        prep_plot_series(t, y, sigma, max_sigma=15.0)
        np.testing.assert_array_equal(t, t0)
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(sigma, s0)

    def test_outputs_are_new_arrays(self) -> None:
        t, y, sigma = _series()
        t2, y2, s2, _ = prep_neu_series(t, y, sigma, max_sigma=20.0)
        assert t2 is not t and y2 is not y and s2 is not sigma
