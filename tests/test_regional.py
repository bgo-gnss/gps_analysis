"""Regional seasonal interpolation (gps_analysis.regional)."""

import numpy as np
import pytest

from gps_analysis.regional import (
    great_circle_km,
    interpolate_seasonal,
    leave_one_out_sigma,
)


def _c(v: float) -> np.ndarray:
    return np.full((3, 4), v)


def test_great_circle_one_degree_of_latitude() -> None:
    assert great_circle_km(-22.0, 64.0, [-22.0], [65.0])[0] == pytest.approx(
        111.19, abs=0.01
    )


def test_identical_donors_give_their_seasonal_and_weights_sum_to_one() -> None:
    donors = {"A": (-22.4, 63.9, _c(2.0)), "B": (-22.3, 63.95, _c(2.0))}
    got = interpolate_seasonal((-22.35, 63.92), donors)
    assert got.refused is None and np.allclose(got.coeffs, 2.0)
    assert sum(got.weights.values()) == pytest.approx(1.0)


def test_the_nearer_donor_weighs_more() -> None:
    donors = {"NEAR": (-22.30, 63.90, _c(0.0)), "FAR": (-21.9, 63.90, _c(10.0))}
    got = interpolate_seasonal((-22.29, 63.90), donors)
    assert list(got.weights) == ["NEAR", "FAR"] and 0.0 < got.coeffs[0, 0] < 5.0


def test_too_few_near_donors_is_refused_not_a_far_field_mean() -> None:
    donors = {"A": (-22.4, 63.9, _c(2.0)), "FAR": (-18.0, 65.0, _c(9.0))}
    got = interpolate_seasonal((-22.35, 63.92), donors)
    assert got.refused is not None and "donor(s) within 100 km" in got.refused
    assert np.isnan(got.coeffs).all() and "FAR" not in got.distances_km


def test_leave_one_out_on_a_smooth_field_is_small() -> None:
    rng = np.random.default_rng(1)
    donors = {}
    for i in range(12):
        lon, lat = -22.6 + 0.6 * rng.random(), 63.8 + 0.3 * rng.random()
        donors[f"S{i:02d}"] = (lon, lat, _c(3.0 + (lon + 22.3)))  # a gentle gradient
    sigma, n = leave_one_out_sigma(donors, sorted(donors))
    assert n == 12 and np.all(sigma < 0.5)


def test_bad_shapes_and_lengths_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        interpolate_seasonal((0.0, 0.0), {}, length_km=0.0)
    with pytest.raises(ValueError, match=r"\(3, 4\)"):
        interpolate_seasonal((0.0, 0.0), {"A": (0.0, 0.0, np.zeros((3, 2)))})
