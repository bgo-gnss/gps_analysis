"""Reference-offset and time-window utilities for GNSS time series.

Consolidated from ``~/work/projects/gps_data_analyses`` and the legacy
``geo_dataread.gps_read`` helpers (see ``docs/CONSOLIDATION_MAP.md``):
``dPeriod`` (window slicing), ``estimate_offset``/``vshift`` (reference-
window level + shift), and the pure core of
``find_line_offsets``/``find_exp_offsets`` in ``detrend_rnes.py`` (step
offset between adjacent fitted segments, differenced at a common epoch).

Derivation chain
----------------
Given epochs t ∈ ℝᴺ (fractional years, ``yearf``) and per-component
observations y ([L], caller's unit):

1. :func:`slice_window` selects the samples inside a time window
   [t_start, t_end] (boolean mask; the legacy ``dPeriod`` semantics with
   its ±0.001 yr tolerance made explicit).
2. :func:`estimate_offset` estimates the reference level ŷ₀ of a series
   over such a window as a weighted mean — the quantity ``vshift``
   subtracted to zero a series at its reference epoch.
3. :func:`remove_offset` subtracts a level: y − ŷ₀ (pure counterpart of
   the in-place shift inside ``vshift``).
4. :func:`estimate_step_offset` computes the discontinuity Δ between two
   *independently fitted* trajectory segments (before/after an event)
   evaluated at a common epoch — the midpoint-difference core of the
   legacy ``find_*_offsets`` (which mixed it with file I/O and plotting;
   only the math lives here). The segment fits themselves come from
   :func:`gps_analysis.fitting.fit_components`.

Naming caveat (decision 2, ``docs/CONSOLIDATION_MAP.md``): "baseline"
here means the reference-offset/window sense only — not RTK rover–base
vectors nor inter-station baselines.

All functions are pure array utilities: float64, no I/O, no unit
assumptions, inputs never mutated.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import ModelFunc
from .models import FloatArray, TrajectoryParams

__all__ = [
    "estimate_offset",
    "estimate_step_offset",
    "remove_offset",
    "slice_window",
]

#: Legacy window tolerance of ``dPeriod``: 0.001 yr ≈ 8.77 h.
_DEFAULT_TOL = 1e-3


def slice_window(
    t: ArrayLike,
    start: float | None = None,
    end: float | None = None,
    *,
    tol: float = _DEFAULT_TOL,
) -> NDArray[np.bool_]:
    """Compute the sample mask of a time window [t_start, t_end].

    Equation:
        ``mᵢ = (tᵢ > t_start − δ) ∧ (tᵢ < t_end + δ)``

    (either bound may be absent, in which case that condition is
    dropped).

    Symbols → args:
        - ``tᵢ``      → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``t_start`` → ``start``: window start [yr]; ``None`` = open
        - ``t_end``   → ``end``: window end [yr]; ``None`` = open
        - ``δ``       → ``tol``: boundary tolerance [yr]

    Args:
        t: Epochs, shape (N,) [yr].
        start: Window start [yr], inclusive to within ``tol``.
        end: Window end [yr], inclusive to within ``tol``.
        tol: Boundary tolerance δ [yr]; the legacy default 0.001 yr
            (≈ 8.77 h) keeps a daily solution stamped exactly at the
            bound.

    Returns:
        Boolean mask, shape (N,) — True inside the window. Apply as
        ``t[m]``, ``y[..., m]`` (replaces the legacy triple-return of
        ``dPeriod(yearf, data, Ddata, …)``).

    Reference:
        Legacy source: ``dPeriod`` in ``geo_dataread.gps_read`` (and its
        ``detrend-*`` copies) — its delete conditions
        ``t ≤ start − 0.001`` / ``t ≥ end + 0.001`` are reproduced
        exactly (strict inequalities on the keep side).

    Numerical notes:
        Pure comparison; ``t`` need not be sorted. NaN epochs compare
        False and are excluded.
    """
    tt = np.asarray(t, dtype=np.float64)
    mask = np.ones(tt.shape, dtype=np.bool_)
    if start is not None:
        mask &= tt > start - tol
    if end is not None:
        mask &= tt < end + tol
    return mask


def estimate_offset(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    start: float | None = None,
    end: float | None = None,
    tol: float = _DEFAULT_TOL,
    weighting: str = "inverse_sigma",
) -> FloatArray:
    """Estimate the reference level ŷ₀ of a series over a time window.

    Equation (per component c, over window samples i):
        ``ŷ₀_c = Σᵢ wᵢ·y_cᵢ / Σᵢ wᵢ``,
        ``wᵢ = 1/σ_cᵢ`` (``weighting="inverse_sigma"``, legacy) or
        ``wᵢ = 1/σ_cᵢ²`` (``weighting="inverse_variance"``) or
        ``wᵢ = 1`` (``sigma=None``).

    Symbols → args:
        - ``tᵢ``   → ``t``: epochs [yr]
        - ``y_cᵢ`` → ``y``: observations [L]
        - ``σ_cᵢ`` → ``sigma``: 1-σ uncertainties [L]
        - window   → ``start``/``end``/``tol`` (see
          :func:`slice_window`) [yr]

    Args:
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        sigma: 1-σ uncertainties, same shape as ``y`` [L]; ``None`` gives
            the unweighted mean.
        start: Reference-window start [yr]; ``None`` = open.
        end: Reference-window end [yr]; ``None`` = open (with both open
            the whole series is averaged — the caller selects the window,
            replacing the legacy count-based ``Period`` argument).
        tol: Window boundary tolerance [yr] (see :func:`slice_window`).
        weighting: ``"inverse_sigma"`` (default — exact legacy parity
            with ``np.average(…, weights=1/Ddata)`` in
            ``geo_dataread.gps_read.estimate_offset``) or
            ``"inverse_variance"`` (wᵢ = σᵢ⁻², the minimum-variance /
            Gauss–Markov weighting — preferred for new work).

    Returns:
        Reference level ŷ₀ [L], float64 — shape (C,) for 2-D ``y``, 0-d
        for 1-D ``y``.

    Raises:
        ValueError: If the window contains no samples (the legacy code
            fell into a broken extrapolation branch referencing an
            undefined variable — made explicit here), on shape mismatch,
            or on an unknown ``weighting``.

    Reference:
        Weighted mean; inverse-variance optimality: Aitken 1936 /
        Strang & Borre 1997 ch. 9. Legacy sources: ``estimate_offset`` +
        ``vshift`` in ``geo_dataread.gps_read``.

    Numerical notes:
        The legacy 1/σ weighting is *not* the minimum-variance estimator;
        it is kept as the default for golden parity and the choice is
        surfaced in ``weighting``. Zero σ produces infinite weight
        (``inf``/NaN propagation) — screen uncertainties first, as the
        legacy ``vshift`` did with its ``Ddata < uncert`` filter.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if tt.ndim != 1 or yy.shape[-1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[-1]}, got shape {tt.shape}"
        )
    if yy.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D (C, N), got shape {yy.shape}")

    if start is None and end is None:
        # Open window: skip the all-True boolean-mask copy. Not only a fast
        # path — a masked copy can shift ``np.average``'s pairwise/SIMD
        # reduction grouping by 1 ULP, and the legacy count-mode callers
        # (``vshift`` Period slicing) averaged the arrays directly; operating
        # on the inputs as-is keeps bit-parity with them.
        if tt.size == 0:
            raise ValueError(f"reference window [{start}, {end}] contains no samples")
        ywin = yy
    else:
        mask = slice_window(tt, start, end, tol=tol)
        if not bool(np.any(mask)):
            raise ValueError(f"reference window [{start}, {end}] contains no samples")
        ywin = yy[..., mask]
    if sigma is None:
        weights = None
    else:
        ss = np.asarray(sigma, dtype=np.float64)
        if ss.shape != yy.shape:
            raise ValueError(
                f"sigma shape {ss.shape} does not match y shape {yy.shape}"
            )
        swin = ss if (start is None and end is None) else ss[..., mask]
        if weighting == "inverse_sigma":
            weights = 1.0 / swin
        elif weighting == "inverse_variance":
            weights = 1.0 / swin**2
        else:
            raise ValueError(
                "weighting must be 'inverse_sigma' or 'inverse_variance', "
                f"got {weighting!r}"
            )
    level = np.average(ywin, axis=-1, weights=weights)
    return np.asarray(level, dtype=np.float64)


def remove_offset(y: ArrayLike, offset: ArrayLike) -> FloatArray:
    """Subtract a reference level from a series.

    Equation (per component c):
        ``y'_cᵢ = y_cᵢ − ŷ₀_c``

    Symbols → args:
        - ``y_cᵢ``  → ``y``: observations [L]
        - ``ŷ₀_c`` → ``offset``: reference level(s), e.g. from
          :func:`estimate_offset` [L]

    Args:
        y: Observations, shape (N,) or (C, N) [L].
        offset: Scalar, shape (C,) (one level per component row of a 2-D
            ``y``), or any shape broadcastable against ``y`` [L].

    Returns:
        Shifted series y′ [L], float64, new array with the shape of ``y``
        (input is not mutated — unlike the in-place shift inside the
        legacy ``vshift``).

    Raises:
        ValueError: If ``offset`` cannot be aligned with ``y``.

    Reference:
        Legacy source: the subtraction step of ``vshift`` in
        ``geo_dataread.gps_read``
        (``data[i, :] - offset[i]`` per component).

    Numerical notes:
        A (C,)-shaped ``offset`` against a (C, N) ``y`` is reshaped to
        (C, 1) so it shifts per *component*, never per epoch; all other
        shapes follow NumPy broadcasting.
    """
    yy = np.asarray(y, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    if yy.ndim == 2 and off.ndim == 1:
        if off.size != yy.shape[0]:
            raise ValueError(
                f"offset has {off.size} entries for {yy.shape[0]} components"
            )
        off = off[:, np.newaxis]
    return np.asarray(yy - off, dtype=np.float64)


def estimate_step_offset(
    model: ModelFunc,
    params_before: TrajectoryParams | Sequence[float] | FloatArray,
    params_after: TrajectoryParams | Sequence[float] | FloatArray,
    epoch: float,
) -> float:
    """Estimate the step Δ between two fitted segments at a common epoch.

    Equation:
        ``Δ = f(t*; p̂_after) − f(t*; p̂_before)``

    Symbols → args:
        - ``f``         → ``model``: trajectory model both segments were
          fitted with (``f(t, *p)``)
        - ``p̂_before`` → ``params_before``: fit of the segment *ending*
          before the step (units per model)
        - ``p̂_after``  → ``params_after``: fit of the segment *starting*
          after the step
        - ``t*``        → ``epoch``: evaluation epoch [yr]; the legacy
          convention is the midpoint between the end of the earlier
          window and the start of the later one

    Args:
        model: Model callable ``f(t, *params) -> ndarray``.
        params_before: Fitted parameters of the pre-step segment — a
            :class:`~gps_analysis.models.TrajectoryParams` or a bare
            parameter vector.
        params_after: Fitted parameters of the post-step segment.
        epoch: Evaluation epoch t* [yr], in the same time reference the
            segments were fitted in.

    Returns:
        Step offset Δ [L], sign convention **after − before** (a positive
        Δ means the series jumped up across the step).

    Reference:
        Step term of the extended trajectory model (Bevis & Brown 2014,
        eq. 2 — here estimated by segment differencing rather than a
        joint Heaviside fit). Legacy sources: the midpoint-difference
        cores of ``find_line_offsets`` / ``find_exp_offsets`` in
        ``detrend_rnes.py`` (their I/O and plotting are not ported); the
        segment fits come from :func:`gps_analysis.fitting.fit_components`.

    Numerical notes:
        Both parameter sets must share the model *and* the time
        reference of ``epoch`` (an event-referenced exponential fit
        cannot be differenced at an absolute ``yearf`` epoch). Δ inherits
        the extrapolation error of both segment fits at t* — keep the
        gap between the fitted windows short relative to their spans.
        One scalar per call; loop components (or use each component's
        ``TrajectoryParams``) for N/E/U.
    """
    p_before = (
        params_before.params
        if isinstance(params_before, TrajectoryParams)
        else np.asarray(params_before, dtype=np.float64)
    )
    p_after = (
        params_after.params
        if isinstance(params_after, TrajectoryParams)
        else np.asarray(params_after, dtype=np.float64)
    )
    before = float(np.asarray(model(epoch, *p_before), dtype=np.float64))
    after = float(np.asarray(model(epoch, *p_after), dtype=np.float64))
    return after - before
