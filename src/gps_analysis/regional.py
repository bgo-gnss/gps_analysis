"""Regional seasonal: a station's seasonal terms interpolated from its neighbours'.

A station installed in an unrest has no quiet span to estimate a seasonal on;
its rate over weeks–months then aliases the seasonal. Nearby stations' stored
seasonal terms predict it (BGÓ 2026-10-17; leave-one-out on Reykjanes closes
70–85 % of the U aliasing at 2–4 months, 20–70 % horizontally).

Equation (per component and periodic coefficient):
    ``w_i = exp(−(d_i / L)²)``,  ``c(x) = Σ w_i c_i / Σ w_i``

over donors i within ``R`` of the target x, ``d_i`` the great-circle distance.
The coefficients are linear in the seasonal, so interpolating them is
interpolating the seasonal curve. Refused — never a silent far-field mean —
when fewer than ``min_near`` donors lie within ``2L`` or ``Σ w_i <
min_weight``.

Pure numpy; the caller chooses the donors (stations whose own seasonal was
estimated, never a regional one, so predictions cannot chain).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: mean Earth radius [km] (great-circle distances to ~0.3 %, ample for weights)
EARTH_RADIUS_KM = 6371.0
#: Gaussian length and donor radius validated by leave-one-out (2026-10-17)
DEFAULT_LENGTH_KM = 50.0
DEFAULT_RADIUS_KM = 150.0
MIN_NEAR_DONORS = 2
MIN_WEIGHT = 0.1


@dataclasses.dataclass(frozen=True)
class RegionalSeasonal:
    """One target's interpolated seasonal, or why there is none.

    Attributes:
        coeffs: (3, 4) periodic coefficients per N/E/U [mm]; NaN when refused.
        weights: Donor code → normalised weight (sums to 1), largest first.
        distances_km: Donor code → distance to the target [km].
        refused: None, or the reason no seasonal was predicted.
        length_km: The Gaussian length L used.
        radius_km: The donor radius R used.
    """

    coeffs: FloatArray
    weights: dict[str, float]
    distances_km: dict[str, float]
    refused: str | None
    length_km: float
    radius_km: float


def great_circle_km(
    lon0: float, lat0: float, lon: Sequence[float], lat: Sequence[float]
) -> FloatArray:
    """Haversine distance [km] from (lon0, lat0) to each (lon, lat) [deg]."""
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    h = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin((lam - l0) / 2) ** 2
    out: FloatArray = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return out


def interpolate_seasonal(
    target: tuple[float, float],
    donors: Mapping[str, tuple[float, float, FloatArray]],
    *,
    length_km: float = DEFAULT_LENGTH_KM,
    radius_km: float = DEFAULT_RADIUS_KM,
    min_near: int = MIN_NEAR_DONORS,
    min_weight: float = MIN_WEIGHT,
    exclude: str | None = None,
) -> RegionalSeasonal:
    """The target's seasonal from its donors' (see the module equation).

    Args:
        target: (lon, lat) of the station [deg].
        donors: Code → (lon, lat, coeffs (3, 4)): the donors' own estimated
            periodic coefficients (cos/sin annual, cos/sin semiannual) per
            N/E/U [mm].
        length_km: Gaussian length L [km].
        radius_km: Donors farther than R are not used [km].
        min_near: Fewest donors within 2L for a prediction.
        min_weight: Smallest Σ w for a prediction.
        exclude: A donor code to leave out (the target itself in a
            leave-one-out).

    Returns:
        A :class:`RegionalSeasonal`; ``refused`` says why when there is none.

    Raises:
        ValueError: A donor's coefficients are not (3, 4), or L/R are not
            positive.
    """
    if not (length_km > 0 and radius_km > 0):
        raise ValueError("length_km and radius_km must be positive")
    codes = sorted(c for c in donors if c != exclude)
    nan = np.full((3, 4), np.nan)
    empty = RegionalSeasonal(nan, {}, {}, "no donors", length_km, radius_km)
    if not codes:
        return empty
    lon = [donors[c][0] for c in codes]
    lat = [donors[c][1] for c in codes]
    coeffs = np.stack([np.asarray(donors[c][2], dtype=float) for c in codes])
    if coeffs.shape[1:] != (3, 4):
        raise ValueError(f"donor coefficients must be (3, 4), got {coeffs.shape[1:]}")
    d = great_circle_km(target[0], target[1], lon, lat)
    use = d <= radius_km
    w = np.exp(-((d / length_km) ** 2)) * use
    n_near = int(np.sum(d[use] <= 2 * length_km))
    dist = {c: round(float(x), 3) for c, x, u in zip(codes, d, use, strict=True) if u}
    reason = None
    if n_near < min_near:
        reason = f"{n_near} donor(s) within {2 * length_km:g} km (< {min_near})"
    elif w.sum() < min_weight:
        reason = f"donor weight {w.sum():.3f} < {min_weight:g}"
    if reason is not None:
        return dataclasses.replace(empty, distances_km=dist, refused=reason)
    wn = w / w.sum()
    pred: FloatArray = np.tensordot(wn, coeffs, axes=1)
    order = np.argsort(-wn, kind="stable")
    weights = {codes[i]: round(float(wn[i]), 6) for i in order if wn[i] > 0}
    return RegionalSeasonal(pred, weights, dist, None, length_km, radius_km)


def leave_one_out_sigma(
    donors: Mapping[str, tuple[float, float, FloatArray]],
    targets: Sequence[str],
    **kw: Any,
) -> tuple[FloatArray, int]:
    """RMS prediction error of the annual amplitude vector per component.

    Each target donor is predicted from the others (:func:`interpolate_seasonal`
    with ``exclude``); the error is |Δ(cos_annual, sin_annual)| [mm].

    Returns:
        (σ per N/E/U [mm], number of targets predicted). NaN when none.
    """
    errs = []
    for code in targets:
        if code not in donors:
            continue
        lon, lat, _ = donors[code]
        got = interpolate_seasonal((lon, lat), donors, exclude=code, **kw)
        if got.refused is None:
            own = np.asarray(donors[code][2], dtype=float)
            errs.append(np.hypot(*(got.coeffs[:, :2] - own[:, :2]).T))
    if not errs:
        return np.full(3, np.nan), 0
    e = np.asarray(errs)
    return np.sqrt(np.mean(e**2, axis=0)), len(errs)
