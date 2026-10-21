"""Network days: whole-solution bad days, from every station's running verdicts.

A bad solution day (reference-frame realisation, orbits, a failed network
adjustment) moves every station at once. On one station it is a blunder like
any other, or — when modest — nothing at all; across the network it is
unmistakable. Measured 2026-10-21 over 190 stations (itrf14 24 h): the share of
stations with a dominant two-sided blunder on a day is bimodal — 30 days at
≥ 60 % (most at 80–100 %), none at 50–60 %, 8 at 30–50 %, the rest below. Real
deformation stays low because the two-sided verdict is blind to steps (the
2014 Bárðarbunga dike scores 0 %, the 2021-03-19 eruption onset 15–20 %).

Equation (per day d, over stations s with a two-sided verdict that day):
    ``v_d = #{s : any component two-sided}``
    ``b_d = #{s : max_c score_{s,c,d} > z_dom}``
    ``share_d = b_d / v_d``
    ``decided ⇔ v_d ≥ n_min ∧ share_d ≥ q``,
    ``partial ⇔ v_d ≥ n_min ∧ q_p ≤ share_d < q``

with ``score`` from :func:`day_scores`: |z| of a two-sided candidate, 0 for any
other two-sided epoch, NaN without a two-sided verdict (no vote).

Pure numpy; the caller aligns the stations on a common day axis (no dates
here) and owns what is masked before the scan — the scan is defined on
detection WITHOUT the network mask.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from gps_analysis.outliers import PROTECT_STEP, OutlierDetection

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "NETWORK_DOMINANT_SIGMA",
    "NETWORK_MIN_STATIONS",
    "NETWORK_PARTIAL_SHARE",
    "NETWORK_SHARE",
    "NetworkDays",
    "day_scores",
    "network_days",
]

NETWORK_DOMINANT_SIGMA = 6.0
"""A station's vote is "bad" when a two-sided candidate exceeds this many noise
units in any component. Its own constant — not ``RUNNING_DOMINANT_SIGMA`` —
so tuning the per-station run rule never moves the network list."""

NETWORK_SHARE = 0.5
"""Share of voting stations for a ``decided`` network day: the empty 50–60 %
gap of the 2026-10-21 scan (BGÓ 2026-10-21: 50 %)."""

NETWORK_PARTIAL_SHARE = 0.3
"""Share for a ``partial`` day — listed for the operator, never masked
(30–50 %: likely a partial solution, a subnet or a stabilisation subset)."""

NETWORK_MIN_STATIONS = 20
"""Voting stations a day needs for any verdict: pre-2010 the network had 15–60
stations, where 10 voters let 4 blunders read as 40 %."""


@dataclasses.dataclass(frozen=True)
class NetworkDays:
    """Per-day network verdicts on the caller's day axis (length D).

    Attributes:
        n_voting: Stations with a two-sided verdict that day, shape (D,).
        n_bad: Voting stations with a dominant blunder, shape (D,).
        n_bad_component: The same per component, shape (C, D) — which
            component carries a day's signature.
        share: ``n_bad / n_voting`` (NaN without voters), shape (D,).
        decided: Network days to set aside, shape (D,) bool.
        partial: Days listed for review only, shape (D,) bool.
    """

    n_voting: IntArray
    n_bad: IntArray
    n_bad_component: IntArray
    share: FloatArray
    decided: NDArray[np.bool_]
    partial: NDArray[np.bool_]


def day_scores(detection: OutlierDetection) -> FloatArray:
    """One station's per-epoch votes from a running detection.

    Returns, shape of ``detection.z`` (C, N) or (N,): |z| where the epoch is a
    two-sided candidate (off both side lines, one sign, beyond k), 0 where it
    has a two-sided verdict and is not, NaN where it has none (series start,
    gaps, the newest days). Valid for ``k ≤`` the dominant level, the
    detector default (4.5) included: a two-sided same-sign epoch beyond 6
    noise units is then always a candidate.

    Raises:
        ValueError: The detection did not come from
            :func:`gps_analysis.detect_outliers_running` (no ``two_sided``).
    """
    if detection.two_sided is None:
        raise ValueError("day_scores needs a running-method detection (two_sided)")
    two_sided = np.asarray(detection.two_sided, dtype=np.bool_)
    candidate = np.asarray(detection.candidates, dtype=np.bool_) & (
        (np.asarray(detection.protected) & PROTECT_STEP) == 0
    )
    z = np.abs(np.asarray(detection.z, dtype=np.float64))
    return np.where(two_sided, np.where(candidate, z, 0.0), np.nan)


def network_days(
    scores: FloatArray,
    *,
    dominant: float = NETWORK_DOMINANT_SIGMA,
    share: float = NETWORK_SHARE,
    partial_share: float = NETWORK_PARTIAL_SHARE,
    min_stations: int = NETWORK_MIN_STATIONS,
) -> NetworkDays:
    """Network days from every station's :func:`day_scores` on one day axis.

    Args:
        scores: Shape (S, C, D) — stations × components × days; NaN = no
            vote (also: no data that day).
        dominant: Noise units for a bad vote (``z_dom``).
        share: ``q``, the decided level.
        partial_share: ``q_p``, the partial level (``≤ share``).
        min_stations: ``n_min``, voters a day needs.

    Returns:
        :class:`NetworkDays` on the D days.

    Raises:
        ValueError: Not 3-D, or levels out of order / out of (0, 1].
    """
    sc = np.asarray(scores, dtype=np.float64)
    if sc.ndim != 3:
        raise ValueError(f"scores must be (stations, components, days), got {sc.shape}")
    if not 0.0 < partial_share <= share <= 1.0:
        raise ValueError(
            f"need 0 < partial_share <= share <= 1, got {partial_share}, {share}"
        )
    if min_stations < 1:
        raise ValueError(f"min_stations must be >= 1, got {min_stations}")
    vote = np.isfinite(sc)  # (S, C, D)
    with np.errstate(invalid="ignore"):
        bad_c = vote & (sc > dominant)
    voting = vote.any(axis=1)  # (S, D)
    bad = bad_c.any(axis=1)
    n_voting = voting.sum(axis=0).astype(np.int64)
    n_bad = bad.sum(axis=0).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_voting > 0, n_bad / np.maximum(n_voting, 1), np.nan)
    enough = n_voting >= min_stations
    with np.errstate(invalid="ignore"):
        decided = enough & (frac >= share)
        partial = enough & (frac >= partial_share) & ~decided
    return NetworkDays(
        n_voting=n_voting,
        n_bad=n_bad,
        n_bad_component=bad_c.sum(axis=0).astype(np.int64),
        share=frac.astype(np.float64),
        decided=decided,
        partial=partial,
    )
