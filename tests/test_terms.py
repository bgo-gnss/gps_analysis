"""Tests for gps_analysis.terms (MATH_STANDARDS §4).

The gate for this module is EXACT design-matrix parity with the deployed
hand-written designs. Not `allclose` — `assert_array_equal`. If a term's
columns differ from `_design_linear` / `_design_periodic` /
`_design_lineperiodic` / the `with_steps` hstack by even one ulp, then 37
stored station records change meaning the day anything switches over, and a
tolerance-based test would let that through.
"""

import inspect

import numpy as np
import pytest

from gps_analysis.fitting import (
    _design_linear,
    _design_lineperiodic,
    _design_periodic,
    _resolve_linear_design,
    fit_components,
    with_steps,
)
from gps_analysis.models import linear, lineperiodic, periodic
from gps_analysis.terms import (
    Polynomial,
    Seasonal,
    Step,
    TrajectoryModel,
    term_from_spec,
)

T = 2001.6 + np.arange(3000) / 365.25


class TestDesignParity:
    """Exact equality with the designs already in production."""

    def test_polynomial_matches_design_linear(self) -> None:
        m = TrajectoryModel([Polynomial(1)])
        np.testing.assert_array_equal(m.design(T), _design_linear(T))
        assert m.param_names == tuple(inspect.signature(linear).parameters)[1:]

    def test_seasonal_matches_design_periodic(self) -> None:
        m = TrajectoryModel([Seasonal(2)])
        np.testing.assert_array_equal(m.design(T), _design_periodic(T))
        assert m.param_names == tuple(inspect.signature(periodic).parameters)[1:]

    def test_polynomial_plus_seasonal_matches_lineperiodic(self) -> None:
        m = TrajectoryModel([Polynomial(1), Seasonal(2)])
        np.testing.assert_array_equal(m.design(T), _design_lineperiodic(T))
        assert m.param_names == tuple(inspect.signature(lineperiodic).parameters)[1:]

    def test_steps_match_the_with_steps_hstack(self) -> None:
        epochs = [2005.0, 2010.5]
        m = TrajectoryModel([Polynomial(1), Seasonal(2), Step(2005.0), Step(2010.5)])
        stepped = with_steps(lineperiodic, epochs)
        ref = _resolve_linear_design(stepped)
        assert ref is not None
        np.testing.assert_array_equal(m.design(T), ref.build(T))

    def test_step_amplitude_names_are_byte_identical_to_with_steps(self) -> None:
        """The stored records and trajectory_from_record's cross-check
        compare these names as strings."""
        m = TrajectoryModel([Polynomial(1), Seasonal(2), Step(2005.0), Step(2010.5)])
        stepped = with_steps(lineperiodic, [2005.0, 2010.5])
        assert m.param_names == tuple(inspect.signature(stepped).parameters)[1:]


class TestConventions:
    """The two that are silent when wrong."""

    def test_seasonal_ignores_t_ref(self) -> None:
        """The phase convention is ABSOLUTE yearf.

        Centering the trig columns would rotate every coefficient and
        invalidate all 37 deployed seasonal sets — with no error anywhere.
        """
        s = Seasonal(2)
        np.testing.assert_array_equal(
            s.columns(T, t_ref=1000.0), s.columns(T, t_ref=0.0)
        )
        assert s.uncentering(1000.0) is None

    def test_polynomial_uncentering_reproduces_the_back_substitution(self) -> None:
        """Degree 1 must be exactly [[1, −t_ref], [0, 1]] — the map
        `_fit_linear_design` applies explicitly."""
        t_ref = float(np.mean(T))
        m = Polynomial(1).uncentering(t_ref)
        assert m is not None
        np.testing.assert_allclose(m, [[1.0, -t_ref], [0.0, 1.0]], rtol=0, atol=0)

    def test_centered_design_times_uncentering_recovers_absolute(self) -> None:
        """A′·(M⁻¹p) == A·p, i.e. the reparameterization is exact."""
        model = TrajectoryModel([Polynomial(1), Seasonal(2)])
        t_ref = float(np.mean(T))
        p = np.array([3.0, 12.5, 2.0, 1.5, 0.4, -0.3])
        a_abs = model.design(T)
        a_cen = model.design(T, t_ref=t_ref)
        m = model.uncentering(t_ref)
        np.testing.assert_allclose(a_cen @ np.linalg.solve(m, p), a_abs @ p, rtol=1e-9)

    def test_shape_parameters_are_not_in_the_vector(self) -> None:
        """A Step's epoch is DATA. In the vector, select_terms would zero it."""
        m = TrajectoryModel([Polynomial(1), Step(2005.0)])
        assert m.param_names == ("offset", "rate", "step_amp_1")
        assert "epoch" not in " ".join(m.param_names)


class TestCanonicalOrder:
    def test_rate_stays_at_index_1(self) -> None:
        """velocity._RATE_INDEX = 1 reads params[1] at six sites."""
        for terms in (
            [Polynomial(1)],
            [Seasonal(2), Polynomial(1)],  # deliberately out of order
            [Step(2005.0), Seasonal(2), Polynomial(1)],
            [Polynomial(2), Seasonal(1), Step(2005.0), Step(2007.0)],
        ):
            assert TrajectoryModel(terms).param_names[1] == "rate"

    def test_terms_are_stable_sorted_into_group_order(self) -> None:
        m = TrajectoryModel([Step(2010.0), Step(2005.0), Seasonal(2), Polynomial(1)])
        assert [t.kind for t in m.terms] == [
            "polynomial",
            "seasonal",
            "step",
            "step",
        ]
        # stable: the two steps keep their INPUT order, not epoch order
        assert [t.epoch for t in m.terms if t.kind == "step"] == [2010.0, 2005.0]


class TestAsModelFunc:
    def test_the_generated_callable_keeps_the_closed_form_path(self) -> None:
        m = TrajectoryModel([Polynomial(1), Seasonal(2)])
        f = m.as_modelfunc()
        assert _resolve_linear_design(f) is not None

    def test_it_fits_and_recovers_truth(self) -> None:
        truth = np.array([3.0, 12.5, 2.0, 1.5, 0.4, -0.3, -8.0])
        m = TrajectoryModel([Polynomial(1), Seasonal(2), Step(2005.0)])
        f = m.as_modelfunc()
        y = m.design(T) @ truth
        fit = fit_components(f, T, y)[0]
        np.testing.assert_allclose(fit.params, truth, rtol=1e-6)

    def test_evaluation_equals_lineperiodic(self) -> None:
        p = [3.0, 12.5, 2.0, 1.5, 0.4, -0.3]
        f = TrajectoryModel([Polynomial(1), Seasonal(2)]).as_modelfunc()
        np.testing.assert_allclose(f(T, *p), lineperiodic(T, *p), rtol=1e-12)

    def test_wrong_parameter_count_is_refused(self) -> None:
        f = TrajectoryModel([Polynomial(1)]).as_modelfunc()
        with pytest.raises(ValueError, match="takes 2 parameters"):
            f(T, 1.0)


class TestGroupMask:
    def test_masks_replace_the_name_classifier(self) -> None:
        m = TrajectoryModel([Polynomial(1), Seasonal(2), Step(2005.0)])
        np.testing.assert_array_equal(
            m.group_mask("secular"), [True, True, False, False, False, False, False]
        )
        np.testing.assert_array_equal(
            m.group_mask("periodic"), [False, False, True, True, True, True, False]
        )
        np.testing.assert_array_equal(
            m.group_mask(["secular", "step"]),
            [True, True, False, False, False, False, True],
        )

    def test_unknown_group_is_named(self) -> None:
        with pytest.raises(ValueError, match="nonesuch"):
            TrajectoryModel([Polynomial(1)]).group_mask("nonesuch")


class TestSerialization:
    def test_round_trip(self) -> None:
        import json

        m = TrajectoryModel([Polynomial(2), Seasonal(1), Step(2008.4085)])
        spec = m.to_spec()
        assert json.loads(json.dumps(spec)) == spec
        back = TrajectoryModel.from_spec(spec)
        assert back.param_names == m.param_names
        np.testing.assert_array_equal(back.design(T), m.design(T))

    def test_unknown_kind_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="known:"):
            term_from_spec({"kind": "wormhole"})

    def test_transient_kinds_are_absent_until_the_citation_is_verified(self) -> None:
        """MATH_STANDARDS §2.4 needs a specific equation number, and Bevis &
        Brown 2014 is not in reference/ yet. Deliberate, not an oversight."""
        for kind in ("exponential", "log", "logarithmic"):
            with pytest.raises(ValueError, match="unknown term kind"):
                term_from_spec({"kind": kind, "epoch": 2008.4, "tau": 0.1})


class TestValidation:
    def test_empty_model_and_bad_degrees(self) -> None:
        with pytest.raises(ValueError, match="at least one term"):
            TrajectoryModel([])
        with pytest.raises(ValueError, match="degree must be"):
            Polynomial(-1)
        with pytest.raises(ValueError, match="n_harmonics must be"):
            Seasonal(0)

    def test_duplicate_names_are_refused(self) -> None:
        """Two seasonals would collide; the message names the collision
        rather than surfacing from deep inside Signature.replace."""
        with pytest.raises(ValueError, match="duplicate parameter names"):
            _ = TrajectoryModel([Seasonal(2), Seasonal(2)]).param_names
