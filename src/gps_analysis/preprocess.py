"""Preprocessing filters for GNSS time series — screen + reference-shift.

Consolidated from the legacy ``geo_dataread.gps_read`` preprocessing
filters (refactor-B slice 2; see ``docs/CONSOLIDATION_MAP.md``):
``vshift`` (uncertainty screen + zero at a reference level, the
**.NEU/gamittoNEU** profile) and ``iprep`` (its plot-pipeline wrapper,
the **plot/getData** profile; its mm unit conversion is caller policy
and stays with the caller).

Derivation chain
----------------
Given epochs t ∈ ℝᴺ (fractional years, ``yearf``), per-component
observations y ([L], caller's unit, shape (C, N) or (N,)) and 1-σ formal
uncertainties σ (same shape):

1. :func:`screen_uncertainty` computes the per-epoch quality mask
   ``mᵢ = ∧_c (σ_cᵢ < σ_max)`` — a *formal-uncertainty* screen, applied
   before any model is fitted. Contrast
   :func:`gps_analysis.fitting.reject_outliers`, which is a
   model-*residual* robust rejection; the two are different operations
   and both exist deliberately.
2. The reference level ŷ₀ of the screened series is estimated with
   :func:`gps_analysis.baseline.estimate_offset` (1/σ-weighted mean,
   legacy parity) — either over the first n kept samples (count mode,
   the legacy ``Period=5`` convention) or over a time window
   [t_start, t_end] (window mode, the legacy ``refdate`` path) — unless
   the caller supplies ŷ₀ directly (offset reuse across series).
3. :func:`gps_analysis.baseline.remove_offset` subtracts ŷ₀.

Two consumer profiles (decision D1, ``PLAN-geo_dataread-refactor-B.md``):
the legacy call sites use the same core with *different thresholds,
units and reference handling*, and are kept as two named entry points —
**not** silently unified:

- :func:`prep_plot_series` — the plot pipeline (legacy ``iprep`` →
  ``getData`` → ``gps_plot``): data in mm (caller-converted),
  ``σ_max = 15 mm`` at the ``getData`` call site, count-mode reference
  only, and the returned ŷ₀ is reused by the caller to pin several
  traces to a common zero.
- :func:`prep_neu_series` — the ``.NEU`` publication pipeline (legacy
  ``vshift`` → ``gamittoNEU``/``gamittooneuf`` → cdn.vedur.is): data in
  meters, ``σ_max = 1.1 m`` hardcoded at the ``gamittoNEU`` call site
  (pinned by the geo_dataread ``real_neu_*`` golden masters), optional
  window-mode reference.

All functions are pure: float64, no I/O, no unit conversions, inputs
never mutated. Thresholds are always passed in (``max_sigma`` has no
default — the legacy defaults live with the legacy shims).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baseline import _DEFAULT_TOL, estimate_offset, remove_offset
from .models import FloatArray

__all__ = [
    "prep_neu_series",
    "prep_plot_series",
    "screen_uncertainty",
]

PrepResult = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
"""Return bundle of the prep profiles: ``(t, y_shifted, sigma, offset)``,
all float64 — the screened epochs, the screened-and-shifted observations,
their screened uncertainties, and the reference level that was removed."""


def screen_uncertainty(sigma: ArrayLike, max_sigma: float) -> NDArray[np.bool_]:
    """Compute the per-epoch formal-uncertainty quality mask m.

    Equation (per epoch i, over components c):
        ``mᵢ = ∧_c (σ_cᵢ < σ_max)``

    (strict inequality; a NaN σ compares False, so an epoch with any
    NaN uncertainty is screened out — the legacy behavior under
    ``np.errstate(invalid="ignore")``).

    Symbols → args:
        - ``σ_cᵢ``  → ``sigma``: 1-σ formal uncertainties [L]
        - ``σ_max`` → ``max_sigma``: screening threshold [L], caller
          policy (legacy call sites: 15 mm plot profile, 1.1 m .NEU
          profile, 20 [L] legacy default)

    Args:
        sigma: Uncertainties, shape (N,) or (C, N) [L].
        max_sigma: Threshold σ_max [L]; an epoch survives only if *all*
            components are strictly below it.

    Returns:
        Boolean mask, shape (N,) — True where the epoch is kept. Apply
        as ``t[m]``, ``y[..., m]``.

    Raises:
        ValueError: If ``sigma`` is not 1-D or 2-D.

    Reference:
        Legacy source: the ``Ddata < uncert`` filter of ``vshift`` in
        ``geo_dataread.gps_read`` (component-wise AND over the three
        rows) — reproduced exactly for (3, N) input, generalized to
        (C, N).

    Numerical notes:
        Pure elementwise comparison; no model involved. This screens on
        *formal* uncertainty, not on residuals — for residual-based
        robust rejection use :func:`gps_analysis.fitting.reject_outliers`.
    """
    ss = np.asarray(sigma, dtype=np.float64)
    if ss.ndim not in (1, 2):
        raise ValueError(f"sigma must be 1-D or 2-D (C, N), got shape {ss.shape}")
    with np.errstate(invalid="ignore"):
        below = ss < max_sigma
    if ss.ndim == 1:
        return np.asarray(below, dtype=np.bool_)
    return np.asarray(np.logical_and.reduce(below, axis=0), dtype=np.bool_)


def _screen_and_reference(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    max_sigma: float,
    offset: ArrayLike | None,
    ref_start: float | None,
    ref_end: float | None,
    ref_samples: int | None,
    tol: float,
) -> PrepResult:
    """Shared core of the two prep profiles: screen → reference → shift.

    Orchestration only (MATH_STANDARDS §1): composes
    :func:`screen_uncertainty`,
    :func:`gps_analysis.baseline.estimate_offset` and
    :func:`gps_analysis.baseline.remove_offset`. Semantics preserved
    from the legacy ``vshift``: the reference level is estimated on the
    *screened* series; a supplied ``offset`` bypasses estimation
    entirely (even for an empty screened series); window mode is used
    whenever either window bound is given, count mode otherwise.

    Raises:
        ValueError: On shape mismatch; if no offset is supplied and the
            screened series has no nonzero sample (the legacy code fell
            through to indexing ``None`` — a ``TypeError`` — made
            explicit here); or, via ``estimate_offset``, if a requested
            reference window contains no samples (legacy: a broken
            extrapolation branch referencing an undefined variable).
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ss = np.asarray(sigma, dtype=np.float64)
    if tt.ndim != 1 or yy.shape[-1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[-1]}, got shape {tt.shape}"
        )
    if ss.shape != yy.shape:
        raise ValueError(f"sigma shape {ss.shape} does not match y shape {yy.shape}")

    mask = screen_uncertainty(ss, max_sigma)
    t_s = tt[mask]
    y_s = yy[..., mask]
    s_s = ss[..., mask]

    if offset is None:
        if not bool(y_s.any()):
            raise ValueError(
                "uncertainty screening left no nonzero data and no offset was given"
            )
        if ref_start is not None or ref_end is not None:
            level = estimate_offset(
                t_s, y_s, s_s, start=ref_start, end=ref_end, tol=tol
            )
        else:
            level = estimate_offset(
                t_s[:ref_samples], y_s[..., :ref_samples], s_s[..., :ref_samples]
            )
    else:
        level = np.asarray(offset, dtype=np.float64)

    return t_s, remove_offset(y_s, level), s_s, level


def prep_plot_series(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    max_sigma: float,
    offset: ArrayLike | None = None,
    ref_samples: int | None = 5,
) -> PrepResult:
    """Prepare a series for plotting: screen and zero at its start (ŷ₀).

    The **plot/getData profile** (decision D1) — legacy ``iprep`` in
    ``geo_dataread.gps_read``, minus its in-place m→mm conversion
    (units are caller policy; the geo_dataread shim keeps the scaling).

    Equation (composition; per component c):
        ``m = screen(σ; σ_max)``,
        ``ŷ₀_c = Σᵢ<n wᵢ·y_cᵢ[m] / Σᵢ<n wᵢ`` with ``wᵢ = 1/σ_cᵢ[m]``
        over the first n kept samples (or ŷ₀ supplied by the caller),
        ``y′_c = y_c[m] − ŷ₀_c``

    Symbols → args:
        - ``tᵢ``    → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``y_cᵢ``  → ``y``: observations [L] (mm in the plot pipeline)
        - ``σ_cᵢ``  → ``sigma``: 1-σ uncertainties [L]
        - ``σ_max`` → ``max_sigma``: screen threshold [L] (``getData``
          passes 15 mm; the legacy ``iprep`` default was 20)
        - ``ŷ₀_c``  → ``offset``: reference level to *reuse* [L];
          ``None`` estimates it
        - ``n``     → ``ref_samples``: reference sample count (legacy
          ``Period=5``); ``None`` averages the whole screened series

    Args:
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        sigma: 1-σ uncertainties, same shape as ``y`` [L].
        max_sigma: Screening threshold σ_max [L] (required — caller
            policy, no hidden default).
        offset: Reference level ŷ₀ [L] to apply instead of estimating —
            the **offset-reuse contract**: ``getData`` returns ŷ₀ so the
            caller can pin several traces to a common zero.
        ref_samples: Count n of leading kept samples averaged for ŷ₀.

    Returns:
        ``(t, y′, σ, ŷ₀)`` — screened epochs, shifted observations,
        screened uncertainties, and the reference level actually
        removed (estimated, or the supplied one as float64).

    Raises:
        ValueError: See :func:`_screen_and_reference`.

    Reference:
        Legacy source: ``iprep`` → ``vshift`` (count mode) in
        ``geo_dataread.gps_read``; weighted mean per
        :func:`gps_analysis.baseline.estimate_offset` (Aitken 1936 /
        Strang & Borre 1997 ch. 9 — legacy 1/σ weighting kept for
        golden parity).

    Numerical notes:
        Bit-parity with the legacy chain: the mask, the ``np.average``
        weighting and the subtraction reproduce the legacy float64
        operations exactly (pinned by the geo_dataread golden masters
        through ``getData``). Inputs are never mutated — the legacy
        ``iprep`` scaled its inputs in place; that side effect stays in
        the geo_dataread shim.
    """
    return _screen_and_reference(
        t,
        y,
        sigma,
        max_sigma=max_sigma,
        offset=offset,
        ref_start=None,
        ref_end=None,
        ref_samples=ref_samples,
        tol=_DEFAULT_TOL,
    )


def prep_neu_series(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    max_sigma: float,
    offset: ArrayLike | None = None,
    ref_start: float | None = None,
    ref_end: float | None = None,
    ref_samples: int | None = 5,
    tol: float = _DEFAULT_TOL,
) -> PrepResult:
    """Prepare a series for .NEU publication: screen and reference-shift.

    The **.NEU/gamittoNEU profile** (decision D1) — legacy ``vshift``
    in ``geo_dataread.gps_read``, the filter on the
    ``gamittoNEU``/``gamittooneuf`` → ``.NEU`` → cdn.vedur.is path.

    Equation (composition; per component c):
        ``m = screen(σ; σ_max)``,
        ``ŷ₀_c = Σᵢ wᵢ·y_cᵢ[m] / Σᵢ wᵢ``, ``wᵢ = 1/σ_cᵢ[m]``, over the
        reference window [t_start, t_end] (window mode) or the first n
        kept samples (count mode; ŷ₀ may instead be supplied),
        ``y′_c = y_c[m] − ŷ₀_c``

    Symbols → args:
        - ``tᵢ``      → ``t``: epochs, fractional years (``yearf``) [yr]
        - ``y_cᵢ``    → ``y``: observations [L] (meters in the .NEU
          pipeline)
        - ``σ_cᵢ``    → ``sigma``: 1-σ uncertainties [L]
        - ``σ_max``   → ``max_sigma``: screen threshold [L]
          (``gamittoNEU`` hardcodes 1.1 m — pinned by the geo_dataread
          ``real_neu_*`` golden masters; the legacy ``vshift`` default
          was 20)
        - ``t_start`` → ``ref_start``: reference-window start [yr];
          the legacy ``refdate``/``Period`` pair converts to this
          window at the caller (date → fractional-year is caller
          policy)
        - ``t_end``   → ``ref_end``: reference-window end [yr]
        - ``n``       → ``ref_samples``: count mode n (legacy
          ``Period=5``) when no window bound is given
        - ``ŷ₀_c``    → ``offset``: reference level to reuse [L]
        - ``δ``       → ``tol``: window boundary tolerance [yr] (see
          :func:`gps_analysis.baseline.slice_window`)

    Args:
        t: Epochs, shape (N,) [yr].
        y: Observations, shape (N,) or (C, N) [L].
        sigma: 1-σ uncertainties, same shape as ``y`` [L].
        max_sigma: Screening threshold σ_max [L] (required).
        offset: Reference level ŷ₀ [L] to apply instead of estimating.
        ref_start: Reference-window start [yr]; giving either bound
            selects window mode.
        ref_end: Reference-window end [yr].
        ref_samples: Count-mode sample count n (used only when both
            window bounds are ``None``); ``None`` averages the whole
            screened series.
        tol: Window boundary tolerance δ [yr].

    Returns:
        ``(t, y′, σ, ŷ₀)`` — screened epochs, shifted observations,
        screened uncertainties, and the reference level actually
        removed (estimated, or the supplied one as float64).

    Raises:
        ValueError: See :func:`_screen_and_reference`.

    Reference:
        Legacy source: ``vshift`` (+ its ``estimate_offset`` helper) in
        ``geo_dataread.gps_read``; weighted mean per
        :func:`gps_analysis.baseline.estimate_offset` (Aitken 1936 /
        Strang & Borre 1997 ch. 9 — legacy 1/σ weighting kept for
        golden parity).

    Numerical notes:
        Bit-parity with legacy ``vshift`` on its live paths (count mode
        and window mode) — pinned end-to-end by the geo_dataread
        ``real_neu_*``/``real_neufile_*`` golden masters. Divergences
        exist only in broken/degenerate legacy branches (see
        :func:`_screen_and_reference`: explicit ``ValueError`` instead
        of ``TypeError``/``NameError``). Inputs are never mutated.
    """
    return _screen_and_reference(
        t,
        y,
        sigma,
        max_sigma=max_sigma,
        offset=offset,
        ref_start=ref_start,
        ref_end=ref_end,
        ref_samples=ref_samples,
        tol=tol,
    )
