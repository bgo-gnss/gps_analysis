"""Tests for gps_analysis.detrend (design §7.1 leaf test plan).

Follows ``docs/DESIGN_live_detrending.md`` §7.1: parameter recovery on
synthetics (white noise + injected outliers; colored noise + known steps
via ``transient.noise_covariance`` Cholesky, Williams 2003), the §2.2
validity gates, raw-preserving invertibility, the live-apply property
(future epochs, restriction consistency), term selection, bit-exact
serialization round-trips (including the ±inf covariance and
hand-pinned records), the absolute-``yearf`` phase-convention guard,
borrowed-record application (§0.6) and the ``detrend_method`` tag /
outlier-abort degrade (§0.2/§0.4).

Tolerances (recorded per MATH_STANDARDS §4):

- White-noise recovery: |p̂ − p_true| ≤ 4·σ̂ elementwise — 18
  simultaneous comparisons at the Gaussian 4σ level have < 0.2 %
  family-wise failure probability; seeds are fixed, so the tests are
  deterministic anyway.
- Colored-noise (flicker κ = −1) recovery uses ABSOLUTE tolerances
  (rate 2 mm/yr, seasonal 1.5 mm, steps 5 mm): the WLS formal σ is
  white-noise-optimistic under temporal correlation (Williams 2003), so
  σ̂-relative bounds would be dishonest; the absolute bounds are ~3× the
  empirically observed seed-fixed errors.
- Invertibility/evaluation identities hold to one float64 rounding
  (``a − b + b``): rtol 1e-12/atol 1e-9 on mm-scale values.
"""

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from gps_analysis.detrend import (
    DETREND_METHOD_PLAIN,
    DETREND_METHOD_ROBUST,
    RECORD_VERSION,
    DetrendEstimate,
    apply_detrend,
    estimate_detrend,
    evaluate_record,
    select_terms,
    trajectory_from_record,
)
from gps_analysis.models import (
    TrajectoryParams,
    exp_linear,
    heaviside_steps,
    linear,
    lineperiodic,
)
from gps_analysis.outliers import OutlierParams
from gps_analysis.transient import noise_covariance

FloatArr = NDArray[np.float64]

DAY = 1.0 / 365.25
WN = 2.0
"""White-noise sigma of the standard synthetic [mm]."""

TRUE_P = np.array(
    [
        [12.0, -3.5, 4.0, -2.0, 1.0, 0.5],
        [-5.0, 10.2, -2.5, 1.5, -0.8, 0.3],
        [30.0, 1.2, 6.0, 3.0, -1.0, 0.7],
    ]
)
"""Per-component lineperiodic truth [offset, rate, a, b, c, d] (N/E/U)."""


def _daily_t(n: int, start: float = 2015.0) -> FloatArr:
    return start + np.arange(n, dtype=np.float64) * DAY


def _truth(t: FloatArr) -> FloatArr:
    return np.stack([lineperiodic(t, *p) for p in TRUE_P])


def _white_series(
    n: int, seed: int, wn: float = WN
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Daily lineperiodic truth + seeded white noise + sigma array."""
    rng = np.random.default_rng(seed)
    t = _daily_t(n)
    y = _truth(t) + rng.normal(0.0, wn, size=(3, n))
    sigma = np.full((3, n), wn)
    return t, y, sigma


def _inject_spikes(
    y: FloatArr, amp: float, spacing: int = 37, one_sided: bool = False
) -> tuple[FloatArr, NDArray[np.intp]]:
    """Isolated single-epoch spikes on a copy; returns (y', indices)."""
    out = y.copy()
    idx = np.arange(25, y.shape[1] - 25, spacing)
    for k, i in enumerate(idx):
        sign = 1.0 if one_sided else (1.0 if k % 2 == 0 else -1.0)
        out[:, i] += sign * amp
    return out, idx


# ---------------------------------------------------------------- recovery


def test_recovery_white_noise_with_outliers() -> None:
    """§7.1-1: recover truth within 4σ̂; outliers removed before the fit."""
    n = 1096  # 3 yr daily
    t, y_clean, sigma = _white_series(n, seed=11)
    y, idx = _inject_spikes(y_clean, amp=10.0 * WN)

    est = estimate_detrend(lineperiodic, t, y, sigma)

    assert est.detrend_method == DETREND_METHOD_ROBUST
    assert not est.outlier_abort
    assert est.n_epochs == n
    assert est.model == "lineperiodic"
    assert [f.component for f in est.fits] == ["north", "east", "up"]

    inliers = np.asarray(est.inliers)
    for c in range(3):
        # >= 95 % of injected spikes rejected, <= 1 % clean epochs lost.
        caught = np.count_nonzero(~inliers[c, idx])
        assert caught >= 0.95 * idx.size, f"component {c}: caught {caught}/{idx.size}"
        clean = np.setdiff1d(np.arange(n), idx)
        false_pos = np.count_nonzero(~inliers[c, clean])
        assert false_pos <= 0.01 * clean.size
        assert est.n_rejected[c] > 0
        # parameter recovery within 4 sigma-hat (tolerance note above).
        err = np.abs(est.fits[c].params - TRUE_P[c])
        assert np.all(
            err <= 4.0 * est.fits[c].uncertainties
        ), f"component {c}: err {err} vs sigma {est.fits[c].uncertainties}"
        assert est.rms[c] == pytest.approx(WN, rel=0.15)
    assert est.span_used[0] >= t[0] and est.span_used[1] <= t[-1]


def test_outlier_removal_beats_plain_wls() -> None:
    """One-sided outliers bias the plain fit; the robust path resists."""
    n = 1096
    t, y_clean, sigma = _white_series(n, seed=3)
    y, _ = _inject_spikes(y_clean, amp=12.0 * WN, one_sided=True)

    robust = estimate_detrend(lineperiodic, t, y, sigma)
    plain = estimate_detrend(lineperiodic, t, y, sigma, detect=False)

    assert plain.detrend_method == DETREND_METHOD_PLAIN
    assert plain.n_rejected == (0, 0, 0)
    assert plain.detection is None
    truth = _truth(t)
    for c in range(3):
        # one-sided contamination (+12 sigma at ~2.6 % of epochs) shifts
        # the plain fit's LEVEL by ~ +0.026 * 12 sigma ~ +0.6 mm; the level
        # bias is the mean model-minus-truth over the window (the intercept
        # at absolute t = 0 is 2015 yr of rate extrapolation - too noisy a
        # metric). The robust path must remove essentially all of it.
        bias_robust = float(np.mean(lineperiodic(t, *robust.fits[c].params) - truth[c]))
        bias_plain = float(np.mean(lineperiodic(t, *plain.fits[c].params) - truth[c]))
        assert bias_plain > 0.4  # the contamination really bites
        assert abs(bias_robust) < 0.25 * bias_plain


def test_recovery_colored_noise_with_steps() -> None:
    """§7.1-1 colored variant: flicker noise + known steps (with_steps)."""
    n = 1461  # 4 yr daily
    rng = np.random.default_rng(7)
    t = _daily_t(n)
    cov = noise_covariance(n, WN, -1.0, 2.0)
    chol = np.linalg.cholesky(cov)
    step_epochs = np.array([2016.30, 2017.55])
    step_amps = np.array([[15.0, -20.0], [10.0, 25.0], [-30.0, 12.0]])
    y = _truth(t).copy()
    for c in range(3):
        y[c] += chol @ rng.standard_normal(n)
        y[c] += heaviside_steps(t, step_epochs, step_amps[c])
    sigma = np.full((3, n), WN)

    est = estimate_detrend(
        lineperiodic, t, y, sigma, step_epochs=[2010.0, *step_epochs, 2030.0]
    )

    # steps outside the window are dropped; inside ones augment the model
    assert np.array_equal(est.step_epochs, step_epochs)
    for c in range(3):
        params = est.fits[c].params
        assert params.size == 8  # 6 trajectory + 2 step amplitudes
        assert abs(float(params[1]) - TRUE_P[c, 1]) < 2.0  # rate [mm/yr]
        assert np.all(np.abs(params[2:6] - TRUE_P[c, 2:6]) < 1.5)  # seasonal
        assert np.all(np.abs(params[6:] - step_amps[c]) < 5.0)  # steps [mm]


# ------------------------------------------------------------------ gates


def test_validity_gates_raise_with_gate_name() -> None:
    """§7.1-2: each failed gate raises a ValueError naming itself."""
    # span: 1.5 yr < 2.0 yr
    t, y, sigma = _white_series(548, seed=1)
    with pytest.raises(ValueError, match="min_span_years"):
        estimate_detrend(lineperiodic, t, y, sigma)

    # epochs: 2.2 yr but only every 3rd day -> 268 < 365
    t3, y3, s3 = _white_series(804, seed=1)
    sl = slice(None, None, 3)
    with pytest.raises(ValueError, match="min_epochs"):
        estimate_detrend(lineperiodic, t3[sl], y3[:, sl], s3[:, sl])

    # gap: 3 yr daily with a 201-day hole -> largest gap 0.55 yr > 0.5 yr
    t4, y4, s4 = _white_series(1096, seed=1)
    keep = np.ones(1096, dtype=np.bool_)
    keep[400:601] = False
    with pytest.raises(ValueError, match="max_gap_years"):
        estimate_detrend(lineperiodic, t4[keep], y4[:, keep], s4[:, keep])

    # empty window
    with pytest.raises(ValueError, match="no epochs"):
        estimate_detrend(lineperiodic, t4, y4, s4, window=(2030.0, 2031.0))


def test_input_validation() -> None:
    t, y, sigma = _white_series(1096, seed=1)
    with pytest.raises(ValueError, match="sorted"):
        estimate_detrend(lineperiodic, t[::-1], y[:, ::-1], detect=False)
    with pytest.raises(ValueError, match="unknown model"):
        estimate_detrend("expsplinperiodic", t, y, sigma, detect=False)
    with pytest.raises(ValueError, match="finite"):
        bad = y.copy()
        bad[0, 5] = np.nan
        estimate_detrend(lineperiodic, t, bad, detect=False)
    with pytest.raises(ValueError, match="names"):
        estimate_detrend(lineperiodic, t, y, names=("a", "b"), detect=False)


# --------------------------------------------------- apply / invertibility


def test_invertibility_and_no_mutation() -> None:
    """§7.1-3: raw preserved; detrended view exactly invertible."""
    t, y, sigma = _white_series(1096, seed=5)
    t_orig, y_orig = t.copy(), y.copy()
    est = estimate_detrend(lineperiodic, t, y, sigma, detect=False)
    record = est.to_record()

    detrended = apply_detrend(record, t, y)
    back = detrended + evaluate_record(record, t)
    np.testing.assert_allclose(back, y, rtol=1e-12, atol=1e-9)
    assert detrended.shape == y.shape
    # inputs never mutated (regression vs the legacy in-place detrend)
    np.testing.assert_array_equal(t, t_orig)
    np.testing.assert_array_equal(y, y_orig)


def test_live_apply_future_epochs_and_restriction() -> None:
    """§7.1-4: stored params evaluate at epochs after the fit window."""
    n = 1461
    t, y, sigma = _white_series(n, seed=8)
    est = estimate_detrend(
        lineperiodic, t, y, sigma, window=(None, 2017.5), detect=False
    )
    record = est.to_record()
    assert est.window == (float(t[0]), 2017.5)
    assert est.span_used[1] <= 2017.5 + 1e-3

    future = t > 2018.0  # epochs the fit never saw
    t_new, y_new = t[future], y[:, future]
    detrended_new = apply_detrend(record, t_new, y_new)
    expected = np.stack(
        [y_new[c] - lineperiodic(t_new, *est.fits[c].params) for c in range(3)]
    )
    np.testing.assert_array_equal(detrended_new, expected)

    # restriction consistency: applying to a sub-window == slicing
    full = apply_detrend(record, t, y)
    np.testing.assert_array_equal(full[:, future], detrended_new)


def test_borrowed_record_applies_to_other_station() -> None:
    """§0.6: a self-contained record applies to ANOTHER station's epochs."""
    t_a, y_a, sigma_a = _white_series(1096, seed=13)
    est = estimate_detrend(lineperiodic, t_a, y_a, sigma_a, detect=False)
    borrowed = {"from": "DYNG", "terms": "all", "donor_fitted_at": None}
    record = est.to_record(fitted_at="2026-07-14T00:00:00Z", borrowed=borrowed)

    # station B: different grid (offset half-day), different truth offset,
    # epochs extending beyond the donor's fit span
    rng = np.random.default_rng(14)
    t_b = _daily_t(1200, start=2016.0) + 0.5 * DAY
    y_b = _truth(t_b) + 40.0 + rng.normal(0.0, WN, size=(3, 1200))

    detrended = apply_detrend(record, t_b, y_b)
    expected = y_b - evaluate_record(record, t_b)
    np.testing.assert_array_equal(detrended, expected)
    # borrowing provenance survives a JSON round trip verbatim
    round_tripped = json.loads(json.dumps(record))
    assert round_tripped["borrowed"] == borrowed


def test_frame_guard() -> None:
    """§2.5: cross-frame application is refused, matching frames pass."""
    t, y, sigma = _white_series(1096, seed=2)
    est = estimate_detrend(
        lineperiodic, t, y, sigma, detect=False, frame="plate_itrf2014"
    )
    record = est.to_record()
    assert record["frame"] == "plate_itrf2014"
    apply_detrend(record, t, y, frame="plate_itrf2014")  # ok
    with pytest.raises(ValueError, match="plate_itrf2014.*itrf2008"):
        apply_detrend(record, t, y, frame="itrf2008")
    # a frame-less record applies under any declared series frame
    frameless = est.to_record()
    frameless["frame"] = None
    apply_detrend(frameless, t, y, frame="itrf2008")


# ------------------------------------------------------------ select_terms


def test_select_terms_sum_and_legacy_parity() -> None:
    """§7.1-5: secular + periodic = all; secular == manual p[0:2] line."""
    t, y, sigma = _white_series(1096, seed=4)
    est = estimate_detrend(lineperiodic, t, y, sigma, detect=False)
    fits = list(est.fits)

    sec = select_terms(lineperiodic, fits, "secular")
    per = select_terms(lineperiodic, fits, "periodic")
    for c in range(3):
        f_all = lineperiodic(t, *fits[c].params)
        f_sec = lineperiodic(t, *sec[c].params)
        f_per = lineperiodic(t, *per[c].params)
        np.testing.assert_allclose(f_sec + f_per, f_all, rtol=1e-12, atol=1e-9)
        # legacy detrend_line parity: subtracting the secular view equals
        # subtracting the p[0:2] straight line
        p = fits[c].params
        np.testing.assert_allclose(
            f_sec, linear(t, float(p[0]), float(p[1])), rtol=0, atol=1e-9
        )
        # covariance rows/cols of zeroed params are zeroed; kept block intact
        cov = fits[c].covariance
        assert np.all(sec[c].covariance[2:, :] == 0.0)
        assert np.all(sec[c].covariance[:, 2:] == 0.0)
        np.testing.assert_array_equal(sec[c].covariance[:2, :2], cov[:2, :2])

    # "all" is the identity; single-fit input yields a one-element list
    assert select_terms(lineperiodic, fits[0], "all")[0] is fits[0]
    single = select_terms(lineperiodic, fits[0], "periodic")
    assert len(single) == 1 and float(single[0].params[1]) == 0.0


def test_select_terms_errors() -> None:
    fit = TrajectoryParams(params=np.zeros(6), covariance=np.zeros((6, 6)))
    with pytest.raises(ValueError, match="terms"):
        select_terms(lineperiodic, fit, "seasonal")
    with pytest.raises(ValueError, match="linear-in-parameters"):
        select_terms(exp_linear, fit, "secular")
    short = TrajectoryParams(params=np.zeros(2), covariance=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="parameters"):
        select_terms(lineperiodic, short, "secular")


def test_select_terms_steps_stay_with_secular() -> None:
    """Step amplitudes belong to the secular (background) group."""
    n = 1096
    rng = np.random.default_rng(21)
    t = _daily_t(n)
    step_epoch = 2016.5
    y = _truth(t) + rng.normal(0.0, WN, size=(3, n))
    y += heaviside_steps(t, [step_epoch], [25.0])[np.newaxis, :]
    sigma = np.full((3, n), WN)
    est = estimate_detrend(
        lineperiodic, t, y, sigma, step_epochs=[step_epoch], detect=False
    )
    record = est.to_record()

    sec = evaluate_record(
        record, np.array([step_epoch - DAY, step_epoch]), terms="secular"
    )
    per = evaluate_record(
        record, np.array([step_epoch - DAY, step_epoch]), terms="periodic"
    )
    for c in range(3):
        jump_sec = float(sec[c, 1] - sec[c, 0])
        step_amp = float(est.fits[c].params[6])
        rate_part = TRUE_P[c, 1] * DAY
        assert jump_sec == pytest.approx(step_amp + rate_part, abs=0.5)
        # the periodic view carries no step
        assert abs(per[c, 1] - per[c, 0]) < 0.5


# ---------------------------------------------------------- serialization


def test_record_roundtrip_bit_exact_including_intercept() -> None:
    """§7.1-6: store -> JSON -> load -> apply == fit -> apply, bit-exact."""
    t, y, sigma = _white_series(1096, seed=17)
    est = estimate_detrend(lineperiodic, t, y, sigma)
    record = est.to_record(fitted_at="2026-07-14T12:00:00Z", refs={"source": "test"})

    assert record["record_version"] == RECORD_VERSION
    assert record["detrend_method"] == DETREND_METHOD_ROBUST
    assert record["param_names"] == [
        "offset",
        "rate",
        "cos_annual",
        "sin_annual",
        "cos_semiannual",
        "sin_semiannual",
    ]
    loaded = json.loads(json.dumps(record))
    assert loaded == record  # float repr round-trip is lossless

    model_func, fits = trajectory_from_record(loaded)
    for c in range(3):
        # FULL vector, intercept included (the CSV 5-of-6 defect is gone)
        assert fits[c].params.size == 6
        assert float(fits[c].params[0]) != 0.0
        np.testing.assert_array_equal(fits[c].params, est.fits[c].params)
        iu = np.triu_indices(6)
        np.testing.assert_array_equal(
            fits[c].covariance[iu], est.fits[c].covariance[iu]
        )
        assert fits[c].component == est.fits[c].component
        # applying the loaded record needs no vshift: the intercept is live
        np.testing.assert_array_equal(
            model_func(t, *fits[c].params),
            lineperiodic(t, *est.fits[c].params),
        )
    # per-component record idempotence
    for fit in fits:
        rec = fit.to_record()
        assert TrajectoryParams.from_record(rec).to_record() == rec


def test_record_covariance_none_roundtrip() -> None:
    """cov_upper: None <-> inf-filled covariance (could-not-estimate)."""
    fit = TrajectoryParams(
        params=np.array([1.0, 2.0]), covariance=np.full((2, 2), np.inf)
    )
    rec = fit.to_record()
    assert rec["cov_upper"] is None
    loaded = TrajectoryParams.from_record(json.loads(json.dumps(rec)))
    assert np.all(np.isinf(loaded.covariance))
    np.testing.assert_array_equal(loaded.params, fit.params)


def test_hand_pinned_record_survives_unchanged() -> None:
    """§0.7: operator-pinned parameters load and apply verbatim."""
    t, y, sigma = _white_series(1096, seed=19)
    est = estimate_detrend(lineperiodic, t, y, sigma, detect=False)
    record = json.loads(json.dumps(est.to_record()))

    pinned_rate = -7.53125  # operator fixes the rate by hand
    record["components"][0]["params"][1] = pinned_rate
    _, fits = trajectory_from_record(record)
    assert float(fits[0].params[1]) == pinned_rate  # bit-exact, no refit

    prediction = evaluate_record(record, t)
    pinned_params = np.asarray(record["components"][0]["params"])
    np.testing.assert_array_equal(prediction[0], lineperiodic(t, *pinned_params))


def test_record_validation_rejects_bad_documents() -> None:
    """§7.1 schema rules: unknown version/model/names/NaN/shape all raise."""
    t, y, sigma = _white_series(1096, seed=23)
    est = estimate_detrend(lineperiodic, t, y, sigma, detect=False)
    good = est.to_record()

    bad = dict(good, record_version=99)
    with pytest.raises(ValueError, match="record_version"):
        trajectory_from_record(bad)

    bad = dict(good, model="exp_linear")
    with pytest.raises(ValueError, match="registry"):
        trajectory_from_record(bad)

    bad = dict(good, param_names=list(reversed(good["param_names"])))
    with pytest.raises(ValueError, match="param_names"):
        trajectory_from_record(bad)

    bad = dict(good, components=[])
    with pytest.raises(ValueError, match="components"):
        trajectory_from_record(bad)

    nan_comp = json.loads(json.dumps(good["components"][0]))
    nan_comp["params"][2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trajectory_from_record(dict(good, components=[nan_comp]))

    legacy_5of6 = json.loads(json.dumps(good["components"][0]))
    legacy_5of6["params"] = legacy_5of6["params"][1:]  # the CSV defect
    legacy_5of6["cov_upper"] = None
    with pytest.raises(ValueError, match="parameters"):
        trajectory_from_record(dict(good, components=[legacy_5of6]))

    short_cov = json.loads(json.dumps(good["components"][0]))
    short_cov["cov_upper"] = short_cov["cov_upper"][:-1]
    with pytest.raises(ValueError, match="cov_upper"):
        trajectory_from_record(dict(good, components=[short_cov]))

    with pytest.raises(ValueError, match="registry"):
        DetrendEstimate(
            fits=est.fits,
            inliers=est.inliers,
            span_used=est.span_used,
            n_epochs=est.n_epochs,
            n_rejected=est.n_rejected,
            rms=est.rms,
            window=est.window,
            model="my_custom_model",
            step_epochs=est.step_epochs,
            detrend_method=est.detrend_method,
        ).to_record()


# ------------------------------------------------- conventions / method tag


def test_integer_year_shift_phase_convention() -> None:
    """§7.1-7: absolute-yearf phase contract under an integer-year shift."""
    n = 1096
    t = _daily_t(n)
    y = _truth(t)  # noise-free: closed-form WLS is exact
    est1 = estimate_detrend(lineperiodic, t, y, detect=False)
    est2 = estimate_detrend(lineperiodic, t + 1.0, y, detect=False)
    for c in range(3):
        p1, p2 = est1.fits[c].params, est2.fits[c].params
        # rate and all four seasonal coefficients are shift-invariant
        np.testing.assert_allclose(p2[1:], p1[1:], rtol=1e-8, atol=1e-8)
        # only the intercept moves, by exactly -rate * (1 yr)
        assert float(p2[0] - p1[0]) == pytest.approx(-float(p1[1]), rel=1e-8)
        # stored params reproduce the original prediction on their own axis
        np.testing.assert_allclose(
            lineperiodic(t + 1.0, *p2), lineperiodic(t, *p1), rtol=0, atol=1e-6
        )


def test_method_tag_and_abort_degrade() -> None:
    """§0.2/§0.4: tag correctness; abort is loud and falls back to WLS."""
    t, y_clean, sigma = _white_series(1096, seed=29)
    y, _ = _inject_spikes(y_clean, amp=10.0 * WN)

    plain = estimate_detrend(lineperiodic, t, y, sigma, detect=False)
    assert plain.detrend_method == DETREND_METHOD_PLAIN
    assert plain.to_record()["detrend_method"] == DETREND_METHOD_PLAIN

    # a paranoid abort threshold turns the ~2.6 % spikes into an abort
    with pytest.warns(RuntimeWarning, match="outlier stage aborted"):
        degraded = estimate_detrend(
            lineperiodic,
            t,
            y,
            sigma,
            outlier_params=OutlierParams(max_flag_fraction=0.001),
        )
    assert degraded.outlier_abort
    assert degraded.detrend_method == DETREND_METHOD_PLAIN
    assert degraded.n_rejected == (0, 0, 0)
    assert degraded.detection is not None
    assert degraded.detection.excess_flag_abort
    assert bool(np.all(np.asarray(degraded.inliers)))


def test_one_component_input() -> None:
    """1-D y: single unlabeled fit, 1-D inlier mask, applicable record."""
    t, y, sigma = _white_series(1096, seed=31)
    est = estimate_detrend(lineperiodic, t, y[0], sigma[0])
    assert len(est.fits) == 1
    assert est.fits[0].component is None  # default N/E/U labels skipped
    assert np.asarray(est.inliers).ndim == 1
    record = est.to_record()
    detrended = apply_detrend(record, t, y[0])
    assert detrended.shape == (t.size,)
    np.testing.assert_allclose(
        detrended + evaluate_record(record, t)[0], y[0], rtol=1e-12, atol=1e-9
    )
