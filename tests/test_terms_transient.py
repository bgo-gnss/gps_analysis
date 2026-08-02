"""Tests for the transient terms, the identifiability guard and profiled τ.

MATH_STANDARDS §4 for the Phase 3 additions to :mod:`gps_analysis.terms`:

- **Analytic checks** — noise-free exact recovery of the amplitudes at
  fixed τ; the step-only omitted-transient bias pinned against its CLOSED
  FORM ``b·[ln(1+s) − 1 + ln(1+s)/s]`` (log — logarithmically divergent)
  and ``b·[1 − (1−e^(−s))/s]`` (exp — saturates at b), asserting the
  MECHANISM, not just the symptom.
- **The refutation pin** — across ~2 decades of ``T_post/τ`` the delivered
  quantity ``σ_b/|b̂|`` crosses its gate monotonically while
  ``|corr(â, b̂)|`` never reaches the ρ gate: exactly why the
  smallest-eigenvalue proposal was replaced by delivered-quantity gates.
- **Property tests** — τ lives in a dataclass field where ``select_terms``'
  coefficient-zeroing cannot reach it; canonical ordering keeps
  ``param_names[1] == "rate"``; ``to_spec``/``from_spec`` round-trips.

Tolerances: closed-form bias compared at 2e-3 relative (midpoint-sum
discretization of the continuous-sampling closed form at N = 2·10⁵);
noise-free recovery at 1e-7 (measured ~6e-9 relative — the SVD
closed-form path with the centered trend column, dominated by the
absolute-yearf back-substitution of the offset).
"""

import dataclasses
import json

import numpy as np
import pytest

from gps_analysis.fitting import fit_components
from gps_analysis.terms import (
    BkwDependency,
    ExpTransient,
    LogTransient,
    Polynomial,
    Seasonal,
    Step,
    TrajectoryModel,
    bkw_dependencies,
    check_transient_identifiability,
    profile_transient_tau,
    term_from_spec,
)

DAY = 1.0 / 365.25


def _series(pre: float, post: float, epoch: float = 2010.0) -> np.ndarray:
    """Daily epochs spanning [epoch − pre, epoch + post)."""
    return np.arange(epoch - pre, epoch + post, DAY)


class TestColumns:
    """The accumulated amplitude convention — zero at the event epoch."""

    @pytest.mark.parametrize("cls", [LogTransient, ExpTransient])
    def test_zero_before_and_at_the_epoch(self, cls: type) -> None:
        term = cls(epoch=2010.0, tau=0.5)
        t = np.array([2005.0, 2009.999, 2010.0])
        np.testing.assert_array_equal(term.columns(t), np.zeros((3, 1)))

    def test_log_value_and_exp_asymptote(self) -> None:
        log = LogTransient(epoch=2010.0, tau=0.5)
        exp = ExpTransient(epoch=2010.0, tau=0.5)
        t = np.array([2010.5, 2060.0])  # Δt = τ and Δt = 100τ
        np.testing.assert_allclose(log.columns(t)[0, 0], np.log(2.0), rtol=1e-15)
        np.testing.assert_allclose(exp.columns(t)[0, 0], 1.0 - np.exp(-1.0))
        # exp reaches its asymptote 1 (·Φ); log keeps rising past it
        np.testing.assert_allclose(exp.columns(t)[1, 0], 1.0, rtol=1e-12)
        assert log.columns(t)[1, 0] > 4.0

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="tau must be finite and > 0"):
            LogTransient(epoch=2010.0, tau=0.0)
        with pytest.raises(ValueError, match="tau must be finite and > 0"):
            ExpTransient(epoch=2010.0, tau=-1.0)
        with pytest.raises(ValueError, match="epoch must be finite"):
            LogTransient(epoch=np.inf, tau=1.0)


class TestExactRecovery:
    """Noise-free recovery of the amplitudes at known FIXED τ (both forms)."""

    @pytest.mark.parametrize(
        "transient",
        [LogTransient(epoch=2010.0, tau=0.7), ExpTransient(epoch=2010.0, tau=0.3)],
    )
    def test_amplitude_recovered_exactly(self, transient) -> None:
        model = TrajectoryModel([Polynomial(1), Seasonal(2), Step(2010.0), transient])
        t = _series(4.0, 6.0)
        truth = np.array([3.0, 12.5, 2.0, 1.5, 0.4, -0.3, -8.0, 25.0])
        y = model.design(t) @ truth
        fit = fit_components(model.as_modelfunc(), t, y)[0]
        np.testing.assert_allclose(fit.params, truth, rtol=1e-7, atol=1e-7)


class TestStepOnlyBias:
    """The plan's closed forms, asserted as MEASURED = PREDICTED.

    Design {intercept, step} fitted to truth ``a·H + b·f_transient`` on
    dense uniform sampling: the LS step amplitude is
    ``â = a + b·mean_post(f)``, and the post-event mean of each transient
    column is the closed form. Pinning the equality pins the mechanism —
    the step absorbs the transient column's projection — not merely that
    "some bias exists".
    """

    A, B, TAU = 10.0, 5.0, 0.5
    N = 200_001

    def _measured_bias(self, kind: str, s: float) -> float:
        t = np.linspace(-2.0, s * self.TAU, self.N)
        transient = (
            LogTransient(epoch=0.0, tau=self.TAU)
            if kind == "log"
            else ExpTransient(epoch=0.0, tau=self.TAU)
        )
        y = self.A * (t >= 0.0) + self.B * transient.columns(t)[:, 0]
        design = np.column_stack([np.ones_like(t), (t >= 0.0).astype(float)])
        p, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(p[1]) - self.A

    @pytest.mark.parametrize("s", [0.5, 2.0, 10.0, 40.0])
    def test_log_bias_matches_the_closed_form(self, s: float) -> None:
        predicted = self.B * (np.log1p(s) - 1.0 + np.log1p(s) / s)
        np.testing.assert_allclose(self._measured_bias("log", s), predicted, rtol=2e-3)

    @pytest.mark.parametrize("s", [0.5, 2.0, 10.0, 40.0])
    def test_exp_bias_matches_the_closed_form(self, s: float) -> None:
        predicted = self.B * (1.0 - (1.0 - np.exp(-s)) / s)
        np.testing.assert_allclose(self._measured_bias("exp", s), predicted, rtol=2e-3)

    def test_log_diverges_where_exp_saturates(self) -> None:
        """The contrast the plan pins: the exp bias saturates at b (the step
        absorbs the finite asymptote); the log bias, lacking an asymptote,
        walks past b and tracks b·(ln s − 1)."""
        s = 40.0
        exp_bias = self._measured_bias("exp", s)
        log_bias = self._measured_bias("log", s)
        assert exp_bias < self.B  # saturating from below
        np.testing.assert_allclose(exp_bias, self.B * (1.0 - 1.0 / s), rtol=1e-2)
        assert log_bias > 2.0 * self.B  # already far beyond the saturation level
        np.testing.assert_allclose(log_bias, self.B * (np.log(s) - 1.0), rtol=0.1)


class TestDeliveredQuantitySweep:
    """~2 decades of T_post/τ: what gates, what was refuted."""

    TAU, B, SIG = 0.5, 5.0, 2.0

    def _fit(self, s: float):
        model = TrajectoryModel(
            [Polynomial(1), Step(2010.0), LogTransient(epoch=2010.0, tau=self.TAU)]
        )
        t = _series(3.0, s * self.TAU)
        truth = np.array([3.0, 12.5, -8.0, self.B])
        y = model.design(t) @ truth
        fit = fit_components(
            model.as_modelfunc(),
            t,
            y,
            sigma=np.full(t.size, self.SIG),
            absolute_sigma=True,
        )[0]
        return model, t, fit

    def test_rel_sigma_gates_while_rho_stays_benign(self) -> None:
        """The refutation pin. σ_b/|b̂| decreases monotonically through the
        0.2 gate near s ≈ 1 (= the operator condition T_post ≥ τ for fixed
        τ); |corr(â, b̂)| never reaches the 0.95 ρ gate anywhere in the
        sweep — a collinearity measure alone can neither fire here nor
        rank these designs correctly."""
        s_values = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        rels, rhos = [], []
        for s in s_values:
            _model, _t, fit = self._fit(s)
            c = fit.covariance
            rels.append(float(np.sqrt(c[3, 3])) / abs(float(fit.params[3])))
            rhos.append(abs(float(c[2, 3]) / np.sqrt(c[2, 2] * c[3, 3])))
        assert all(a > b for a, b in zip(rels[:-1], rels[1:], strict=True))
        assert rels[0] > 0.2 and rels[1] > 0.2  # s < 1: gated
        assert all(r <= 0.2 for r in rels[2:])  # s >= 1: delivered
        assert all(r < 0.95 for r in rhos)

    @pytest.mark.parametrize("s,fires", [(0.3, True), (0.5, True), (5.0, False)])
    def test_guard_follows_the_rel_sigma_gate(self, s: float, fires: bool) -> None:
        model, t, fit = self._fit(s)
        check = lambda: check_transient_identifiability(  # noqa: E731
            model, t, fit.params, fit.covariance
        )
        if fires:
            with pytest.raises(ValueError, match="relative sigma"):
                check()
        else:
            check()


class TestSelectTermsCannotPerturbTau:
    """The reason τ is a dataclass field, not a parameter slot."""

    def test_tau_and_epoch_are_not_in_the_parameter_vector(self) -> None:
        model = TrajectoryModel(
            [
                Polynomial(1),
                Step(2010.0),
                LogTransient(epoch=2010.0, tau=0.5),
                ExpTransient(epoch=2015.0, tau=0.2),
            ]
        )
        assert model.param_names == (
            "offset",
            "rate",
            "step_amp_1",
            "log_amp_1",
            "exp_amp_1",
        )
        assert "tau" not in " ".join(model.param_names)
        assert "epoch" not in " ".join(model.param_names)

    def test_zeroing_coefficients_removes_the_term_and_leaves_tau_intact(
        self,
    ) -> None:
        """Coefficient zeroing (what ``select_terms`` does) evaluates the
        remaining terms exactly — and the transient's τ is untouched
        because it never sat in the vector."""
        transient = LogTransient(epoch=2010.0, tau=0.5)
        model = TrajectoryModel([Polynomial(1), transient])
        t = _series(3.0, 3.0)
        p = np.array([3.0, 12.5, 25.0])
        zeroed = np.where(model.group_mask("transient"), 0.0, p)
        reduced = TrajectoryModel([Polynomial(1)])
        np.testing.assert_array_equal(
            model.design(t) @ zeroed, reduced.design(t) @ p[:2]
        )
        assert model.terms[1] is transient and transient.tau == 0.5

    def test_tau_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            LogTransient(epoch=2010.0, tau=0.5).tau = 1.0  # type: ignore[misc]


class TestCanonicalOrdering:
    def test_rate_stays_at_index_1_with_transients(self) -> None:
        for terms in (
            [LogTransient(epoch=2010.0, tau=0.5), Polynomial(1)],
            [
                ExpTransient(epoch=2010.0, tau=0.2),
                Step(2010.0),
                Seasonal(2),
                Polynomial(1),
            ],
            [
                LogTransient(epoch=2010.0, tau=1.0),
                Seasonal(1),
                Step(2005.0),
                Polynomial(2),
            ],
        ):
            assert TrajectoryModel(terms).param_names[1] == "rate"

    def test_transients_sort_after_steps(self) -> None:
        m = TrajectoryModel(
            [
                LogTransient(epoch=2010.0, tau=0.5),
                Step(2010.0),
                Seasonal(2),
                Polynomial(1),
                ExpTransient(epoch=2015.0, tau=0.2),
            ]
        )
        assert [t.group for t in m.terms] == [
            "secular",
            "periodic",
            "step",
            "transient",
            "transient",
        ]
        # stable: the two transients keep their INPUT order
        assert [t.kind for t in m.terms[-2:]] == ["log_transient", "exp_transient"]

    def test_amp_families_number_independently(self) -> None:
        m = TrajectoryModel(
            [
                Polynomial(1),
                Step(2005.0),
                Step(2010.0),
                LogTransient(epoch=2005.0, tau=1.0),
                LogTransient(epoch=2010.0, tau=0.5),
                ExpTransient(epoch=2012.0, tau=0.2),
            ]
        )
        assert m.param_names == (
            "offset",
            "rate",
            "step_amp_1",
            "step_amp_2",
            "log_amp_1",
            "log_amp_2",
            "exp_amp_1",
        )


class TestSerialization:
    def test_round_trip_including_transients(self) -> None:
        m = TrajectoryModel(
            [
                Polynomial(1),
                Seasonal(2),
                Step(2008.4085),
                LogTransient(epoch=2008.4085, tau=1.0),
                ExpTransient(epoch=2015.75, tau=0.33),
            ]
        )
        spec = m.to_spec()
        assert json.loads(json.dumps(spec)) == spec
        back = TrajectoryModel.from_spec(spec)
        assert back.param_names == m.param_names
        t = _series(3.0, 3.0, epoch=2008.4085)
        np.testing.assert_array_equal(back.design(t), m.design(t))

    def test_spec_carries_epoch_and_tau(self) -> None:
        spec = LogTransient(epoch=2008.4085, tau=1.0).to_spec()
        assert spec == {"kind": "log_transient", "epoch": 2008.4085, "tau": 1.0}
        assert term_from_spec(spec) == LogTransient(epoch=2008.4085, tau=1.0)


class TestGuard:
    """Separable when the data allow it; loud, localized and actionable when not."""

    TAU = 0.5

    def _fit(self, t: np.ndarray, model: TrajectoryModel, truth: np.ndarray):
        y = model.design(t) @ truth
        return fit_components(
            model.as_modelfunc(),
            t,
            y,
            sigma=np.full(t.size, 2.0),
            absolute_sigma=True,
        )[0]

    def _model(self) -> TrajectoryModel:
        return TrajectoryModel(
            [Polynomial(1), Step(2010.0), LogTransient(epoch=2010.0, tau=self.TAU)]
        )

    def test_step_plus_log_at_one_epoch_is_separable_with_pre_event_data(
        self,
    ) -> None:
        """Different shapes ⇒ separable: constant-after vs zero-at-event
        rising. Noise-free recovery of BOTH amplitudes, and the guard
        passes."""
        model = self._model()
        t = _series(3.0, 5.0)
        truth = np.array([3.0, 12.5, -8.0, 5.0])
        fit = self._fit(t, model, truth)
        np.testing.assert_allclose(fit.params, truth, rtol=1e-7, atol=1e-7)
        check_transient_identifiability(model, t, fit.params, fit.covariance)

    def test_missing_pre_event_data_fires_and_names_the_failure(self) -> None:
        model = self._model()
        t = _series(0.3, 5.0)
        fit = self._fit(t, model, np.array([3.0, 12.5, -8.0, 5.0]))
        with pytest.raises(ValueError) as err:
            check_transient_identifiability(model, t, fit.params, fit.covariance)
        msg = str(err.value)
        assert "log_transient at 2010.00000" in msg
        assert "pre-event epochs span" in msg
        assert "the step IS the intercept" in msg
        assert "Ways out:" in msg

    def test_post_event_gap_fires_the_rho_gate_and_bkw_names_the_terms(
        self,
    ) -> None:
        """First post-event epoch 6τ after the event: the sampled log column
        is a scaled near-step (ρ ≈ −0.97), and no epoch sits in the
        curvature window. The message names the ρ, and the BKW localizer
        names the participating terms without being told which."""
        model = self._model()
        t = np.concatenate(
            [_series(3.0, 0.0), np.arange(2010.0 + 6 * self.TAU, 2020.0, DAY)]
        )
        fit = self._fit(t, model, np.array([3.0, 12.5, -8.0, 5.0]))
        with pytest.raises(ValueError) as err:
            check_transient_identifiability(model, t, fit.params, fit.covariance)
        msg = str(err.value)
        assert "corr(step_amp_1, log_amp_1)" in msg
        assert "kept epochs in [t_k, t_k + min(3*tau, T_post)]" in msg
        assert "condition index" in msg
        assert "step_amp_1, log_amp_1" in msg  # BKW participants, in order
        assert "never excise the event with segments" in msg

    def test_short_record_fires_on_t_post_and_snr(self) -> None:
        model = self._model()
        t = _series(3.0, 0.15)  # T_post = 0.3 tau
        fit = self._fit(t, model, np.array([3.0, 12.5, -8.0, 5.0]))
        with pytest.raises(ValueError) as err:
            check_transient_identifiability(model, t, fit.params, fit.covariance)
        msg = str(err.value)
        assert "the record must reach one time constant" in msg
        assert "relative sigma" in msg

    def test_unknown_transient_kind_is_refused(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class Custom:
            kind: str = "custom_transient"
            group: str = "transient"

            @property
            def names(self) -> tuple[str, ...]:
                return ("custom_amp",)

            def columns(self, t, *, t_ref: float = 0.0):
                return np.ones_like(np.asarray(t, float))[:, np.newaxis]

            def uncentering(self, t_ref: float):
                return None

            def to_spec(self) -> dict:
                return {"kind": self.kind}

        model = TrajectoryModel([Polynomial(1), Custom()])
        t = _series(3.0, 3.0)
        with pytest.raises(ValueError, match="house transient"):
            check_transient_identifiability(model, t, np.zeros(3), np.eye(3))

    def test_shape_mismatch_is_refused(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="params/covariance"):
            check_transient_identifiability(
                model, _series(1.0, 1.0), np.zeros(3), np.eye(3)
            )


class TestBkw:
    """Isolated BKW checks run at a t-origin near 0: in absolute yearf
    (t ≈ 2010) the equilibrated columns 1 and t are near-parallel, so the
    UNCENTERED diagnosis honestly reports an (offset, rate) dependency at
    η ≈ 2·10³ for every model with a polynomial — that is the Belsley 1984
    point, not a bug, and in guard messages it simply precedes the
    dependency that matters (see the gap-case guard test)."""

    def test_two_near_identical_steps_are_localized_exactly(self) -> None:
        """Textbook near-dependency: the two step columns differ by one
        epoch. BKW must name exactly those two, at a large η."""
        t = _series(3.0, 3.0, epoch=0.0)
        model = TrajectoryModel([Polynomial(1), Step(0.0), Step(DAY)])
        deps = bkw_dependencies(model.design(t), model.param_names)
        assert deps and isinstance(deps[0], BkwDependency)
        assert deps[0].participants == ("step_amp_1", "step_amp_2")
        assert deps[0].condition_index > 30.0

    def test_well_conditioned_design_yields_no_dependency(self) -> None:
        t = _series(3.0, 3.0, epoch=0.0)
        model = TrajectoryModel([Polynomial(1), Seasonal(2)])
        assert bkw_dependencies(model.design(t), model.param_names) == ()

    def test_zero_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="all-zero column"):
            bkw_dependencies(np.column_stack([np.ones(10), np.zeros(10)]), ["a", "b"])


class TestProfiledTau:
    """The opt-in path — estimate_varpro is the solver, this is its caller."""

    TAU_TRUE = 0.4

    def _make(self, *, post: float, noise: float = 0.0, seed: int = 0):
        model = TrajectoryModel(
            [Polynomial(1), Step(2010.0), LogTransient(epoch=2010.0, tau=self.TAU_TRUE)]
        )
        t = _series(4.0, post)
        truth = np.array([3.0, 12.5, -8.0, 25.0])
        y = model.design(t) @ truth
        if noise:
            y = y + np.random.default_rng(seed).normal(0.0, noise, t.size)
        return model, t, y, truth

    def test_noise_free_recovery_of_tau_and_amplitudes(self) -> None:
        model, t, y, truth = self._make(post=5.0)
        fit = profile_transient_tau(
            model, t, y, sigma=np.full(t.size, 1.0), absolute_sigma=True
        )
        np.testing.assert_allclose(fit.tau, self.TAU_TRUE, rtol=1e-6)
        np.testing.assert_allclose(fit.params, truth, rtol=1e-6)
        # the refined tau lives in the FIELD of the returned model's term
        term = fit.model.terms[fit.term_index]
        assert isinstance(term, LogTransient)
        np.testing.assert_allclose(term.tau, fit.tau)
        assert "tau" not in " ".join(fit.model.param_names)

    def test_noisy_recovery_with_closed_interval(self) -> None:
        model, t, y, _truth = self._make(post=8.0, noise=2.0)
        fit = profile_transient_tau(
            model, t, y, sigma=np.full(t.size, 2.0), absolute_sigma=True
        )
        lo, hi = fit.tau_interval
        assert not fit.interval_open_lower and not fit.interval_open_upper
        assert lo < self.TAU_TRUE < hi
        assert fit.tau_sigma > 0.0
        # theta is LAST in the joint covariance; delta method consistency
        theta_sigma = float(np.sqrt(fit.covariance[-1, -1]))
        np.testing.assert_allclose(fit.tau_sigma, fit.tau * theta_sigma, rtol=1e-12)

    def test_marginal_identification_warns_a_bound(self) -> None:
        """T_post ≈ 2τ < 5τ: the term-specific condition this caller owns."""
        model, t, y, _truth = self._make(post=2.0 * self.TAU_TRUE, noise=2.0)
        with pytest.warns(UserWarning, match="publish a tau bound"):
            profile_transient_tau(
                model, t, y, sigma=np.full(t.size, 2.0), absolute_sigma=True
            )

    def test_guard_accepts_a_profiled_fit(self) -> None:
        model, t, y, _truth = self._make(post=8.0, noise=2.0)
        fit = profile_transient_tau(
            model, t, y, sigma=np.full(t.size, 2.0), absolute_sigma=True
        )
        check_transient_identifiability(
            fit.model, t, fit.params, fit.covariance[:-1, :-1], tau_fit=fit
        )

    def test_errors_are_named(self) -> None:
        t = _series(2.0, 2.0)
        y = np.zeros(t.size)
        no_transient = TrajectoryModel([Polynomial(1)])
        with pytest.raises(ValueError, match="0 transient terms"):
            profile_transient_tau(no_transient, t, y)
        two = TrajectoryModel(
            [
                Polynomial(1),
                LogTransient(epoch=2010.0, tau=0.5),
                ExpTransient(epoch=2011.0, tau=0.2),
            ]
        )
        with pytest.raises(ValueError, match="2 transient terms"):
            profile_transient_tau(two, t, y)
        with pytest.raises(ValueError, match="profiling needs"):
            profile_transient_tau(two, t, y, term_index=0)
        one = TrajectoryModel([Polynomial(1), LogTransient(epoch=2010.0, tau=0.5)])
        with pytest.raises(ValueError, match="tau_bounds"):
            profile_transient_tau(one, t, y, tau_bounds=(-1.0, 2.0))
