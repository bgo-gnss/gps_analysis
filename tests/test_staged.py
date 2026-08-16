"""Tests for gps_analysis.staged (MATH_STANDARDS §4).

Two claims carry this module and both are checked against something
independent rather than against themselves:

- the HELD FIXED POINT — hold a term at the value a joint fit gave it, refit
  the rest on the same data, and the free parameters must be unchanged. This
  is a mathematical identity (at the joint optimum the joint normal equations
  ARE the reduced ones), so it fails loudly if the partition or the
  whitening is wrong. Written first, before the module existed.
- the PROPAGATED COVARIANCE — ``C_cond + K C_v Kᵀ`` compared to exact linear
  propagation of the data covariance through the composed estimator, built
  independently in the test.

Tolerances (per MATH_STANDARDS §4): the fixed point is asserted at 1e-6
RELATIVE, not 1e-12 absolute, because these designs use absolute ``yearf``
(t ≈ 2×10³) where the raw [1, t] columns have condition number ~10⁷ — a
float64 eps of 2e-16 leaves ~1e-9 of headroom, and the assertion is set two
decades above the observed 1.2e-8 so it pins the identity without pinning
BLAS. Covariance identities hold to 1e-9 relative.
"""

import numpy as np
import pytest

from gps_analysis.fitting import _wls_solve
from gps_analysis.staged import (
    HeldExplicit,
    HeldFromStage,
    Stage,
    compose_held,
    estimate_staged,
    fit_held_partition,
)

TRUTH = np.array([3.0, 12.5, 2.0, 1.5, 0.4, -0.3])
"""offset, rate, cos_annual, sin_annual, cos_semiannual, sin_semiannual."""

SECULAR = np.array([True, True, False, False, False, False])
PERIODIC = ~SECULAR


def _design(t: np.ndarray) -> np.ndarray:
    """lineperiodic's design at absolute yearf — the production convention."""
    return np.column_stack(
        [
            np.ones_like(t),
            t,
            np.cos(2 * np.pi * t),
            np.sin(2 * np.pi * t),
            np.cos(4 * np.pi * t),
            np.sin(4 * np.pi * t),
        ]
    )


def _series(n: int = 2500, seed: int = 4, start: float = 2001.6):
    rng = np.random.default_rng(seed)
    t = start + np.arange(n) / 365.25
    sigma = 1.0 + 0.5 * rng.random(n)
    a = _design(t)
    y = a @ TRUTH + rng.normal(0.0, 1.0, n) * sigma
    return t, a, y, sigma


class TestHeldFixedPoint:
    """The identity that catches a wrong partition."""

    def test_holding_at_the_joint_optimum_leaves_the_rest_unchanged(self) -> None:
        """At the joint optimum the reduced normal equations are the joint ones.

        Written before the implementation. If the held columns are dropped
        without subtracting their contribution, or the subtraction is done
        on unwhitened data, or the mask is inverted, this fails by orders
        of magnitude rather than subtly.
        """
        _t, a, y, sigma = _series()
        p_joint, _ = _wls_solve(a, y, sigma, absolute_sigma=True)

        p_staged, _ = fit_held_partition(
            a,
            y,
            sigma,
            held_mask=PERIODIC,
            held_values=p_joint,  # full-length; only the held entries are read
            absolute_sigma=True,
        )
        assert np.allclose(p_staged[SECULAR], p_joint[SECULAR], rtol=1e-6, atol=0.0), (
            f"free params moved: {p_staged[SECULAR] - p_joint[SECULAR]}"
        )

    def test_the_held_values_pass_through_untouched(self) -> None:
        _t, a, y, sigma = _series()
        v = np.zeros(6)
        v[PERIODIC] = [9.0, -9.0, 9.0, -9.0]
        p, _ = fit_held_partition(
            a, y, sigma, held_mask=PERIODIC, held_values=v, absolute_sigma=True
        )
        np.testing.assert_array_equal(p[PERIODIC], v[PERIODIC])

    def test_holding_a_wrong_value_moves_the_free_params(self) -> None:
        """Negative control: the test above must not pass vacuously."""
        _t, a, y, sigma = _series()
        p_joint, _ = _wls_solve(a, y, sigma, absolute_sigma=True)
        wrong = p_joint.copy()
        wrong[PERIODIC] += 5.0
        p_bad, _ = fit_held_partition(
            a, y, sigma, held_mask=PERIODIC, held_values=wrong, absolute_sigma=True
        )
        assert not np.allclose(p_bad[SECULAR], p_joint[SECULAR], rtol=1e-6)


class TestPropagatedCovariance:
    """``C_cond + K C_v Kᵀ`` against exact linear propagation."""

    @staticmethod
    def _nested():
        """Stage 1 on a strict SUBSET of stage 2's window — the real geometry."""
        t, a, y, sigma = _series()
        w1 = t < 2005.0
        return t, a, y, sigma, w1

    def test_matches_exact_propagation(self) -> None:
        """The formula is finite-sample exact, not asymptotic.

        The reference is built independently here: express β̂ as ONE linear
        functional ``L`` of the full data vector (stage 2's solve minus the
        held contribution, whose values are themselves a linear functional
        of the window-1 rows) and propagate Σ = diag(σ²) through it.
        """
        _t, a, y, sigma, w1 = self._nested()
        a1, y1, s1 = a[w1], y[w1], sigma[w1]
        _p1, c1 = _wls_solve(a1, y1, s1, absolute_sigma=True)
        c_v = c1[np.ix_(PERIODIC, PERIODIC)]  # held block of the JOINT stage-1 cov

        p1_full, _ = _wls_solve(a1, y1, s1, absolute_sigma=True)
        _p, cov = fit_held_partition(
            a,
            y,
            sigma,
            held_mask=PERIODIC,
            held_values=p1_full,
            held_cov=c_v,
            absolute_sigma=True,
        )

        # --- independent reference ---
        aw_f = a[:, SECULAR] / sigma[:, None]
        m = np.linalg.inv(aw_f.T @ aw_f) @ aw_f.T / sigma  # (2, n)
        aw1 = a1 / s1[:, None]
        g1 = np.linalg.inv(aw1.T @ aw1) @ aw1.T / s1  # (6, n1)
        vh = np.zeros((int(PERIODIC.sum()), y.size))
        vh[:, w1] = g1[PERIODIC]
        lin = m - (m @ a[:, PERIODIC]) @ vh  # β̂ = lin @ y
        c_exact = (lin * sigma**2) @ lin.T

        got = cov[np.ix_(SECULAR, SECULAR)]
        assert np.allclose(got, c_exact, rtol=1e-9, atol=0.0), (
            f"max rel err {np.max(np.abs(got - c_exact)) / np.max(np.abs(c_exact)):.2e}"
        )

    def test_the_conditional_form_always_understates(self) -> None:
        """``K C_v Kᵀ ⪰ 0``, so omitting it can only be optimistic."""
        _t, a, y, sigma, w1 = self._nested()
        a1, y1, s1 = a[w1], y[w1], sigma[w1]
        p1, c1 = _wls_solve(a1, y1, s1, absolute_sigma=True)
        c_v = c1[np.ix_(PERIODIC, PERIODIC)]

        _p, cond = fit_held_partition(
            a, y, sigma, held_mask=PERIODIC, held_values=p1, absolute_sigma=True
        )
        _p, prop = fit_held_partition(
            a,
            y,
            sigma,
            held_mask=PERIODIC,
            held_values=p1,
            held_cov=c_v,
            absolute_sigma=True,
        )
        i_rate = 1
        assert prop[i_rate, i_rate] > cond[i_rate, i_rate]
        # the correction is real but modest in this geometry -- record it
        ratio = prop[i_rate, i_rate] / cond[i_rate, i_rate] - 1.0
        assert 0.0 < ratio < 0.5, f"understatement {ratio * 100:.2f} % out of range"

    def test_the_composed_covariance_is_psd(self) -> None:
        """The −K C_v off-blocks are what make it a genuine covariance.

        Zeroing the held rows/columns instead would return something that
        LOOKS like a joint covariance and is not one.
        """
        _t, a, y, sigma, w1 = self._nested()
        a1, y1, s1 = a[w1], y[w1], sigma[w1]
        p1, c1 = _wls_solve(a1, y1, s1, absolute_sigma=True)
        _p, cov = fit_held_partition(
            a,
            y,
            sigma,
            held_mask=PERIODIC,
            held_values=p1,
            held_cov=c1[np.ix_(PERIODIC, PERIODIC)],
            absolute_sigma=True,
        )
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)
        eig = np.linalg.eigvalsh(cov)
        assert eig.min() > -1e-12 * max(1.0, eig.max()), f"not PSD: min eig {eig.min()}"
        # the held block comes back as itself
        np.testing.assert_allclose(
            cov[np.ix_(PERIODIC, PERIODIC)], c1[np.ix_(PERIODIC, PERIODIC)], rtol=1e-12
        )

    def test_omitting_held_cov_gives_the_conditional_form(self) -> None:
        _t, a, y, sigma, _w1 = self._nested()
        p_joint, _ = _wls_solve(a, y, sigma, absolute_sigma=True)
        _p, cov = fit_held_partition(
            a, y, sigma, held_mask=PERIODIC, held_values=p_joint, absolute_sigma=True
        )
        assert not cov[np.ix_(PERIODIC, PERIODIC)].any()
        assert not cov[np.ix_(SECULAR, PERIODIC)].any()


class TestValidation:
    def test_all_held_and_all_free_are_refused(self) -> None:
        _t, a, y, sigma = _series(n=400)
        with pytest.raises(ValueError, match="holds every column"):
            fit_held_partition(
                a, y, sigma, held_mask=np.ones(6, bool), held_values=np.zeros(6)
            )
        with pytest.raises(ValueError, match="holds nothing"):
            fit_held_partition(
                a, y, sigma, held_mask=np.zeros(6, bool), held_values=np.zeros(6)
            )

    def test_shape_mismatches_are_named(self) -> None:
        _t, a, y, sigma = _series(n=400)
        with pytest.raises(ValueError, match="held_values must be full length"):
            fit_held_partition(a, y, sigma, held_mask=PERIODIC, held_values=np.zeros(4))
        with pytest.raises(ValueError, match="held_cov must be"):
            fit_held_partition(
                a,
                y,
                sigma,
                held_mask=PERIODIC,
                held_values=np.zeros(6),
                held_cov=np.eye(3),
            )

    def test_asymmetric_held_cov_is_refused(self) -> None:
        _t, a, y, sigma = _series(n=400)
        bad = np.eye(4)
        bad[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            fit_held_partition(
                a,
                y,
                sigma,
                held_mask=PERIODIC,
                held_values=np.zeros(6),
                held_cov=bad,
            )


class TestComposeHeld:
    GROUPS = {"secular": SECULAR, "periodic": PERIODIC}

    def test_assembles_a_full_length_mask_and_vector(self) -> None:
        held = {"periodic": HeldExplicit(np.array([1.0, 2.0, 3.0, 4.0]), "DYNG")}
        mask, values = compose_held(6, self.GROUPS, held)
        np.testing.assert_array_equal(mask, PERIODIC)
        np.testing.assert_array_equal(values[PERIODIC], [1.0, 2.0, 3.0, 4.0])
        assert not values[SECULAR].any()

    def test_unknown_group_and_bad_width_are_refused(self) -> None:
        with pytest.raises(KeyError, match="nonesuch"):
            compose_held(6, self.GROUPS, {"nonesuch": HeldExplicit(np.zeros(2), "x")})
        with pytest.raises(ValueError, match="needs 4 values"):
            compose_held(6, self.GROUPS, {"periodic": HeldExplicit(np.zeros(2), "x")})


class TestStagePlanShapes:
    """The plan structures are data; pin the contract, not behaviour."""

    def test_borrowing_and_staging_share_one_mechanism(self) -> None:
        """Both produce a `held` entry — only the provenance differs."""
        staged = Stage(
            name="rate_long",
            free=("secular",),
            held={"periodic": HeldFromStage("seasonal_clean")},
        )
        borrowed = Stage(
            name="borrowed",
            free=("secular",),
            held={"periodic": HeldExplicit(np.zeros(4), source="OLAC")},
        )
        assert set(staged.held) == set(borrowed.held) == {"periodic"}
        assert staged.held["periodic"].stage == "seasonal_clean"
        assert borrowed.held["periodic"].source == "OLAC"

    def test_a_stage_defaults_to_no_holds_and_the_inherited_domain(self) -> None:
        s = Stage(name="clean", free=("secular", "periodic"))
        assert s.held == {}
        assert s.segments is None


class TestStagingEarnsItsPlace:
    """When does staging actually beat a joint fit — and when does it lose?

    The operator recipe's premise is UNMODELED SIGNAL on the flanks. A
    synthetic without it proves nothing, so three contamination regimes are
    measured and the losing case is asserted as firmly as the winning one.
    200 seeded realizations, RMSE compared (one draw can order either way).

    Measured, and the middle row is why the contamination has to be built
    with care:

    ======================  ==================  ====================
    flank contamination     rate                seasonal
    ======================  ==================  ====================
    none                    staged 0.0073 WIN   full-span 0.0255 win
    raw random walk         clean-win 0.0221 W  staged 0.0380 WIN
    trend-free wander       staged 0.0073 WIN   staged 0.0380 WIN
    ======================  ==================  ====================

    A RAW random walk carries a net slope, so it biases the rate and a
    longer baseline propagates that bias — staging then LOSES on rate by
    25x. Removing the walk's mean and slope over the dirty flank leaves
    low-frequency power that corrupts a seasonal fit while leaving the rate
    unbiased, which is the actual geophysical situation (post-unrest
    deformation is not a trend the background model should absorb).
    """

    N_REAL = 200

    @staticmethod
    def _realization(seed: int, mode: str):
        """``mode``: 'clean' | 'raw' | 'detrended' (see the class docstring)."""
        rng = np.random.default_rng(seed)
        n = 3000
        t = 2001.6 + np.arange(n) / 365.25  # ~8.2 yr
        clean = t < 2005.6  # pre-unrest window: the first half
        sigma = np.ones(n)
        a = _design(t)
        y = a @ TRUTH + rng.normal(0.0, 1.0, n)
        if mode != "clean":
            walk = np.cumsum(rng.normal(0.0, 0.06, n))
            walk[clean] = 0.0
            if mode == "detrended":
                dirty = ~clean
                td = t[dirty]
                g = np.column_stack([np.ones(td.size), td])
                walk[dirty] -= g @ np.linalg.lstsq(g, walk[dirty], rcond=None)[0]
            y = y + walk
        return t, a, y, sigma, clean

    def _rmse(self, mode: str):
        """(staged rate, clean-window rate, staged seasonal, full-span seasonal)."""
        e_sr, e_cr, e_ss, e_fs = [], [], [], []
        for s in range(self.N_REAL):
            _t, a, y, sigma, clean = self._realization(1000 + s, mode)
            p1, _ = _wls_solve(a[clean], y[clean], sigma[clean], absolute_sigma=True)
            p2, _ = fit_held_partition(
                a, y, sigma, held_mask=PERIODIC, held_values=p1, absolute_sigma=True
            )
            p_full, _ = _wls_solve(a, y, sigma, absolute_sigma=True)
            e_sr.append(p2[1] - TRUTH[1])
            e_cr.append(p1[1] - TRUTH[1])
            e_ss.append(p1[PERIODIC] - TRUTH[PERIODIC])
            e_fs.append(p_full[PERIODIC] - TRUTH[PERIODIC])

        def rms(v):
            return float(np.sqrt(np.mean(np.asarray(v) ** 2)))

        return rms(e_sr), rms(e_cr), rms(e_ss), rms(e_fs)

    def test_staging_wins_both_under_trend_free_flank_wander(self) -> None:
        """The premise, made falsifiable."""
        sr, cr, ss, fs = self._rmse("detrended")
        assert sr < cr, f"rate: staged {sr:.4f} should beat clean-window {cr:.4f}"
        assert ss < fs, f"seasonal: staged {ss:.4f} should beat full-span {fs:.4f}"

    def test_staging_loses_when_the_contamination_carries_a_trend(self) -> None:
        """The case the operator must not apply staging to.

        A contaminating signal with a net slope biases any rate fitted over
        it, and the long baseline that staging exists to exploit propagates
        that bias instead of averaging it away. Measured 25x worse than
        simply fitting the clean window. This is the honest boundary of the
        recipe and it belongs in the test suite, not in a footnote.
        """
        sr, cr, _ss, _fs = self._rmse("raw")
        assert sr > 5.0 * cr, (
            f"expected staging to lose badly here; staged {sr:.4f} vs "
            f"clean-window {cr:.4f}"
        )

    def test_on_clean_data_staging_is_a_trade_not_a_free_lunch(self) -> None:
        """No contamination: the rate gains from baseline, the seasonal pays.

        If staging won on both here, the premise test above would be
        measuring a rigged synthetic rather than the mechanism.
        """
        sr, cr, ss, fs = self._rmse("clean")
        assert sr < cr, "the long baseline should still help the rate"
        assert ss > fs, (
            f"seasonal from the half-length clean window ({ss:.4f}) must NOT "
            f"beat the full span ({fs:.4f}) when nothing is contaminated"
        )

    def test_the_propagated_covariance_covers(self) -> None:
        """Monte Carlo as the GATE TEST of the formula, not as the error bar.

        Honest limit, stated rather than papered over: in this geometry the
        correction is ~1.1 % of the variance, while a variance estimated
        from 200 draws carries ~10 % noise. So "the conditional form
        under-covers" is NOT resolvable here and is deliberately not
        asserted — that claim is proved analytically instead
        (``K C_v Kᵀ ⪰ 0``, see
        ``TestPropagatedCovariance::test_the_conditional_form_always_understates``).
        What IS resolvable is that the propagated variance tracks the
        empirical one to within MC noise.
        """
        errs, var_prop = [], []
        for s in range(self.N_REAL):
            _t, a, y, sigma, clean = self._realization(2000 + s, "clean")
            p1, c1 = _wls_solve(a[clean], y[clean], sigma[clean], absolute_sigma=True)
            p2, prop = fit_held_partition(
                a,
                y,
                sigma,
                held_mask=PERIODIC,
                held_values=p1,
                held_cov=c1[np.ix_(PERIODIC, PERIODIC)],
                absolute_sigma=True,
            )
            errs.append(p2[1] - TRUTH[1])
            var_prop.append(prop[1, 1])
        empirical = float(np.var(errs))
        modelled = float(np.mean(var_prop))
        assert 0.8 < modelled / empirical < 1.25, (
            f"propagated/empirical = {modelled / empirical:.3f}, expected ~1 "
            f"within MC noise at n={self.N_REAL}"
        )


class TestEstimateStaged:
    """The orchestrator: katlafitlong as a plan, plus its refusals."""

    @staticmethod
    def _series3(seed: int = 1, n: int = 3000):
        rng = np.random.default_rng(seed)
        t = 2001.6 + np.arange(n) / 365.25
        a = _design(t)
        y = np.vstack([a @ TRUTH + rng.normal(0.0, 1.0, n) for _ in range(3)])
        return t, y

    def test_one_stage_reproduces_a_plain_joint_fit(self) -> None:
        """Parity gate: a single free-everything stage IS the joint fit.

        Required before `estimate_detrend` could ever delegate here — if
        one stage is not the same fit, nothing built on top can be trusted.
        """
        t, y = self._series3()
        est = estimate_staged(
            "lineperiodic",
            t,
            y[0],
            plan=[Stage("all", ("secular", "periodic"))],
        )
        a = _design(t)
        p_joint, _ = _wls_solve(a, y[0], None, absolute_sigma=False)
        np.testing.assert_allclose(est.fits[0].params, p_joint, rtol=1e-9, atol=0.0)

    def test_katlafitlong_is_two_stages(self) -> None:
        """The operator recipe, expressed rather than hand-coded."""
        t, y = self._series3()
        est = estimate_staged(
            "lineperiodic",
            t,
            y,
            plan=[
                Stage("clean", ("secular", "periodic"), segments=[(None, 2005.6)]),
                Stage(
                    "long",
                    ("secular",),
                    held={"periodic": HeldFromStage("clean")},
                ),
            ],
            names=["north", "east", "up"],
        )
        assert [s.name for s in est.stages] == ["clean", "long"]
        assert est.stages[0].n_epochs < est.stages[1].n_epochs, (
            "the clean window must be a strict subset of the long span"
        )
        # the rate comes from the LAST stage that freed it; the seasonal from
        # the only stage that freed it
        assert est.fits[0].params[1] == pytest.approx(TRUTH[1], abs=0.1)
        np.testing.assert_allclose(
            est.fits[0].params[PERIODIC], TRUTH[PERIODIC], atol=0.25
        )
        assert [f.component for f in est.fits] == ["north", "east", "up"]

    def test_holding_from_an_earlier_stage_propagates_its_covariance(self) -> None:
        t, y = self._series3()
        est = estimate_staged(
            "lineperiodic",
            t,
            y[0],
            plan=[
                Stage("clean", ("secular", "periodic"), segments=[(None, 2005.6)]),
                Stage("long", ("secular",), held={"periodic": HeldFromStage("clean")}),
            ],
        )
        assert est.stages[0].held_covariance == "n/a"
        assert est.stages[1].held_covariance == "propagated"

    def test_an_explicit_borrow_without_covariance_is_flagged_conditional(
        self,
    ) -> None:
        """A donor's coefficients with no covariance CAN be used — but the
        result is conditional on them and must say so."""
        t, y = self._series3()
        est = estimate_staged(
            "lineperiodic",
            t,
            y[0],
            plan=[
                Stage(
                    "borrowed",
                    ("secular",),
                    held={"periodic": HeldExplicit(TRUTH[PERIODIC], source="OLAC")},
                )
            ],
        )
        assert est.stages[0].held_covariance == "conditional"
        assert est.stages[0].held_sources == {"periodic": "explicit:OLAC"}
        np.testing.assert_array_equal(est.fits[0].params[PERIODIC], TRUTH[PERIODIC])

    def test_the_record_fragment_is_additive_and_names_provenance(self) -> None:
        t, y = self._series3()
        est = estimate_staged(
            "lineperiodic",
            t,
            y[0],
            plan=[
                Stage("clean", ("secular", "periodic"), segments=[(None, 2005.6)]),
                Stage("long", ("secular",), held={"periodic": HeldFromStage("clean")}),
            ],
        )
        frag = est.to_record_fragment()
        assert set(frag) == {"stage_plan", "stages"}
        assert frag["stage_plan"][1]["held"] == {"periodic": "stage:clean"}
        assert frag["stages"][1]["held_covariance"] == "propagated"
        import json

        assert json.loads(json.dumps(frag)) == frag, "must be JSON-round-trippable"

    @pytest.mark.parametrize(
        "plan, match",
        [
            ([], "at least one stage"),
            (
                [Stage("a", ("secular",)), Stage("a", ("periodic",))],
                "duplicate stage name",
            ),
            (
                [Stage("a", ("secular",), held={"periodic": HeldFromStage("later")})],
                "has not run",
            ),
            (
                [
                    Stage(
                        "a",
                        ("secular", "periodic"),  # everything owned, so the
                        held={"secular": HeldExplicit(np.zeros(2), "x")},
                    )  # orphan check passes and the overlap check is what fires
                ],
                "both frees and holds",
            ),
            ([Stage("a", ("secular",))], "never estimated and not held"),
            (
                [Stage("a", ("secular", "periodic"), segments=[(2050.0, 2060.0)])],
                "contains no epochs",
            ),
        ],
    )
    def test_malformed_plans_are_refused(self, plan, match) -> None:
        """Each refusal prevents a silently wrong composed record."""
        t, y = self._series3(n=1200)
        with pytest.raises(ValueError, match=match):
            estimate_staged("lineperiodic", t, y[0], plan=plan)

    def test_a_nonlinear_model_is_refused(self) -> None:
        """Holding a term is column arithmetic — it only removes that term
        when the model is linear in its parameters."""
        from gps_analysis.models import exp_linear

        t, y = self._series3(n=1200)
        with pytest.raises(ValueError, match="linear-in-parameters"):
            estimate_staged(
                exp_linear, t, y[0], plan=[Stage("a", ("secular", "periodic"))]
            )


class TestGroupVocabulary:
    """Staged estimation and select_terms classify groups DIFFERENTLY, on purpose.

    ``detrend._term_keep_mask`` is an apply-time selector (design §5.3) and
    folds step amplitudes into "secular", because a Heaviside jump is
    background rather than seasonal. 37 deployed records and the workbench's
    ``--terms`` depend on that meaning.

    Staged estimation asks which parameters a STAGE estimates, and there
    ``step`` must be separable from ``rate``: a stage whose window excludes a
    step epoch cannot estimate its amplitude, and folding them together made
    that design rank-deficient (measured on SELF).
    """

    @staticmethod
    def _stepped():
        import numpy as np

        from gps_analysis import with_steps
        from gps_analysis.detrend import _resolve_model

        base, _ = _resolve_model("lineperiodic")
        return with_steps(base, np.array([2008.4085]))

    def test_staged_separates_step_from_secular(self) -> None:
        from gps_analysis.staged import group_parameter_mask

        m = self._stepped()
        assert list(group_parameter_mask(m, "secular")) == [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        assert list(group_parameter_mask(m, "step")) == [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ]

    def test_select_terms_still_folds_step_into_secular(self) -> None:
        # The compatibility pin. Changing this changes what
        # apply_detrend(terms="secular") REMOVES from 37 deployed records.
        from gps_analysis.detrend import _term_keep_mask

        assert list(_term_keep_mask(self._stepped(), "secular")) == [
            True,
            True,
            False,
            False,
            False,
            False,
            True,
        ]

    def test_the_two_classifiers_disagree_only_about_steps(self) -> None:
        import numpy as np

        from gps_analysis.detrend import _term_keep_mask
        from gps_analysis.staged import group_parameter_mask

        m = self._stepped()
        staged_secular = group_parameter_mask(m, "secular")
        apply_secular = _term_keep_mask(m, "secular")
        diff = staged_secular != apply_secular
        assert np.array_equal(diff, group_parameter_mask(m, "step"))

    def test_every_group_is_addressable(self) -> None:
        from gps_analysis import GROUP_ORDER
        from gps_analysis.staged import group_parameter_mask

        for g in GROUP_ORDER:
            group_parameter_mask(self._stepped(), g)  # must not raise

    def test_unknown_group_raises(self) -> None:
        import pytest

        from gps_analysis.staged import group_parameter_mask

        with pytest.raises(ValueError, match="unknown term group"):
            group_parameter_mask(self._stepped(), "bogus")

    def test_unclassifiable_parameter_raises(self) -> None:
        # Closed-world by design: a new term kind must fail loudly rather than
        # be silently dropped from every partition.
        import pytest

        from gps_analysis.staged import _staged_group_of

        with pytest.raises(ValueError, match="cannot classify"):
            _staged_group_of("mystery_param")

    def test_matches_trajectory_model_group_mask(self) -> None:
        # The name classifier and terms.py's structural one must agree, or the
        # picker would offer groups the estimator cannot address.
        import numpy as np

        from gps_analysis import (
            GROUP_ORDER,
            ExpTransient,
            LogTransient,
            Polynomial,
            Seasonal,
            Step,
            TrajectoryModel,
        )
        from gps_analysis.staged import group_parameter_mask

        tm = TrajectoryModel(
            (
                Polynomial(degree=1),
                Seasonal(n_harmonics=2),
                Step(epoch=2008.4),
                LogTransient(epoch=2008.4, tau=1.0),
                ExpTransient(epoch=2010.0, tau=0.5),
            )
        )
        mf = tm.as_modelfunc()
        for g in GROUP_ORDER:
            assert np.array_equal(tm.group_mask(g), group_parameter_mask(mf, g)), g


class TestComposedCovarianceOwnership:
    """One composed matrix must not be assembled from several estimators.

    The composed covariance used to be written block by block as the stages
    ran, so a cross-block between two coefficients owned by DIFFERENT stages
    kept whatever an earlier stage had written there — even after both of its
    diagonal blocks were overwritten. The result was an off-diagonal from a
    fit whose parameters had been discarded, in a matrix whose variances came
    from somewhere else. Diagonals looked right, which is why it survived.
    """

    @staticmethod
    def _series(n: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        t = 2020.0 + np.arange(n) / 365.25
        y = -8.0 + 4.0 * (t - 2020.0) + 2.0 * np.cos(2 * np.pi * t)
        y = y + rng.normal(0.0, 0.5, t.size)
        return t, y, np.full_like(y, 0.5)

    def test_cross_block_comes_from_the_owning_stage_not_a_discarded_one(
        self,
    ) -> None:
        t, y, s = self._series()
        plan = (
            Stage(name="A", free=("secular", "periodic"), held={}, segments=None),
            Stage(
                name="B",
                free=("periodic",),
                held={"secular": HeldFromStage("A")},
                segments=((2020.5, None),),
            ),
        )
        est = estimate_staged("lineperiodic", t, y, s, plan=plan, absolute_sigma=True)
        cov = est.fits[0].covariance
        a_cov, b_cov = est.stages[0].covariance[0], est.stages[1].covariance[0]
        sec, per = [0, 1], [2, 3, 4, 5]

        # B owns everything here: it freed the periodic terms and holds the
        # secular ones, so `fit_held_partition`'s full P x P covariance IS the
        # joint one for the composed values.
        assert np.array_equal(cov[np.ix_(sec, per)], b_cov[np.ix_(sec, per)])
        assert not np.allclose(
            cov[np.ix_(sec, per)], a_cov[np.ix_(sec, per)], atol=1e-4
        )
        # and the two really do disagree — this is not a distinction without
        # a difference (offset x sin_annual: A -0.0031 vs B +0.356, a factor
        # 100 and a sign apart)
        assert a_cov[0, 3] * b_cov[0, 3] < 0
        assert abs(b_cov[0, 3]) > 50 * abs(a_cov[0, 3])

    def test_held_free_cross_terms_survive_composition(self) -> None:
        """The Askja plan's cross-block used to be dropped, not merely stale.

        With A freeing secular and B freeing periodic while holding it, the
        old block-by-block write touched only [secular, secular] and
        [periodic, periodic], so the secular x periodic terms stayed at their
        initialized zero — discarding exactly the coupling
        ``fit_held_partition`` had propagated. Taking B's whole joint block
        keeps it (offset x sin_semiannual: 0 -> -0.75).
        """
        t, y, s = self._series()
        plan = (
            Stage(name="A", free=("secular",), held={}, segments=((2020.0, 2021.5),)),
            Stage(
                name="B",
                free=("periodic",),
                held={"secular": HeldFromStage("A")},
                segments=None,
            ),
        )
        est = estimate_staged("lineperiodic", t, y, s, plan=plan, absolute_sigma=True)
        cov = est.fits[0].covariance
        sec, per = [0, 1], [2, 3, 4, 5]
        assert np.any(cov[np.ix_(sec, per)] != 0.0)
        assert np.array_equal(
            cov[np.ix_(sec, per)], est.stages[1].covariance[0][np.ix_(sec, per)]
        )

    def test_disjoint_owners_get_zero_rather_than_a_borrowed_number(self) -> None:
        """No estimator formed that covariance, so nothing may be asserted.

        Zero understates (it claims independence across stages), the same
        direction as the documented conditional-covariance caveat — but it
        never states a number that no fit computed.
        """
        t, y, s = self._series()
        plan = (
            Stage(name="A", free=("secular",), held={}, segments=((2020.0, 2021.5),)),
            Stage(name="B", free=("periodic",), held={}, segments=None),
        )
        est = estimate_staged("lineperiodic", t, y, s, plan=plan, absolute_sigma=True)
        cov = est.fits[0].covariance
        sec, per = [0, 1], [2, 3, 4, 5]
        assert np.all(cov[np.ix_(sec, per)] == 0.0)
        # the diagonal blocks still come from their own owners
        assert np.array_equal(
            cov[np.ix_(sec, sec)], est.stages[0].covariance[0][np.ix_(sec, sec)]
        )
        assert np.array_equal(
            cov[np.ix_(per, per)], est.stages[1].covariance[0][np.ix_(per, per)]
        )

    def test_values_are_untouched_by_the_covariance_fix(self) -> None:
        """Ownership was already right for the PARAMETERS; only cov was not.

        Pinned because the rewrite derives both from one ``owner`` array: if
        that array were wrong, the stored science would move, and this is the
        assertion that would say so.
        """
        t, y, s = self._series()
        plan = (
            Stage(name="A", free=("secular",), held={}, segments=((2020.0, 2021.5),)),
            Stage(
                name="B",
                free=("periodic",),
                held={"secular": HeldFromStage("A")},
                segments=None,
            ),
        )
        est = estimate_staged("lineperiodic", t, y, s, plan=plan, absolute_sigma=True)
        a, b = est.stages[0].params[0], est.stages[1].params[0]
        composed = est.fits[0].params
        # secular from the stage that freed it, periodic from the one that did
        assert np.array_equal(composed[[0, 1]], b[[0, 1]])  # held -> final stage
        assert np.allclose(composed[[0, 1]], a[[0, 1]])  # == what A fitted
        assert np.array_equal(composed[2:], b[2:])
