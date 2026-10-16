"""Honest σ and refusal gates for windowed GNSS velocities.

A fixed-window WLS rate (:func:`gps_analysis.velocity.estimate_velocity`)
carries a white-noise formal σ that is **optimistic** for daily GNSS
solutions: temporally correlated noise makes the true scatter of a rate
several times larger (Williams 2003, J. Geodesy 76). Measured on NYLA's
quiet 2012–2019 span, the scatter of sliding linear rates exceeds their
median formal σ by k ≈ 4–5 at 2 months and k ≈ 4–7 at one year.

This module measures that factor per station, component and window
length, the **empirical rate scale** k(T), and decides when a window is
not fit to give a rate at all:

1. :func:`empirical_rate_scale` — on a series with its background
   removed (so the true rate is zero), slide windows of length T, fit a
   line in each (:func:`~gps_analysis.velocity.sliding_velocity`,
   ``model="linear"``) and compare the robust spread of the rates
   (``1.4826 · MAD``) with their median formal σ:
   ``k = max(1, spread / median σ_formal)``. The honest σ of a rate over a
   window of length T on that station is then ``k(T) · σ_formal``.
2. :func:`window_gates` — the reasons a window gives no rate: too few
   epochs, too little of the window covered, a declared step inside it.
3. :func:`largest_gap` — the provenance of how gapped the window was.
4. :func:`line_fit` — the window's line in closed form, with what the
   gates of a short window need: the a-priori σ (no χ² scaling, which has
   no meaning on 2–3 epochs), χ²/ν, and the F statistic of a quadratic
   against the line (:data:`NONLINEAR_F`: a decaying pulse, not a rate).
5. :func:`signal_to_noise` — |v|/σ of the horizontal pair and of up, the
   key of the short tier (:data:`SHORT_SNR`, :data:`SHORT_SNR_PAIR`).

The leaf stays pure: which series is "quiet" (a station's curated
detrend window), and what to do with a refusal, are the caller's.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .models import FloatArray
from .velocity import sliding_velocity

#: robust spread of a normal sample from its median absolute deviation
MAD_TO_SIGMA = 1.4826
#: fewest valid windows behind a k (fewer: NaN, the caller's fallback)
MIN_WINDOWS = 8
#: the default gates of a windowed rate: a count floor small enough for
#: the week-long inter-episode windows of an unrest (``min_cover`` is what
#: keeps a sparse window out), and the share of the window spanned
MIN_OBS = 5
MIN_COVER = 0.8
#: the short tier (fewer epochs or less cover than the standard gates): a
#: part (horizontal pair, up) stands when |v|/σ reaches this. Measured
#: 2026-10-16 on 3 yr quiet spans of 20 stations, raw series, background
#: removed, every 1-8 day window (~20000 per n): false accepts per 10⁴ at
#: |v_h|/σ_h ≥ 5 are 2 (n=3), 1.6 (n=4); at n=2, with no χ² to inflate σ on
#: a blunder, 23 at 5 and 2 at 10, hence its own threshold. The fast
#: motion passes: Svartsengi 2023-11-12 (post-dike) 9-15 from n=2-5; the
#: dike itself ~200 at n=2; the slower 2024 inflation from ~10-14 days.
SHORT_SNR = 5.0
SHORT_SNR_PAIR = 10.0
#: a window's line is flagged nonlinear when the quadratic's F against it
#: reaches this, in a component. Quiet p99 is 23-42 at 7-61 d, 80 at 91 d
#: (correlated noise); Svartsengi 2023-11-12 + 14/30 d windows (the decay
#: after the dike) reach 30-310. A flag, not a refusal.
NONLINEAR_F = 50.0
#: a window is about this fraction full of epochs at most for daily data
_DAYS_PER_YEAR = 365.25


@dataclasses.dataclass(frozen=True)
class RateScale:
    """The empirical rate scale k(T) of one series, per component.

    Attributes:
        k: Scale factors, shape (C,); ``max(1, spread / median σ)``, NaN
            where fewer than ``min_windows`` windows were valid.
        spread: Robust spread of the sliding rates, shape (C,) [L/yr].
        median_sigma: Median formal σ of the sliding rates, shape (C,).
        n_windows: Valid windows per component, shape (C,).
        window_years: The window length T [yr].
        step_years: The centre step [yr].
        span: (first, last) epoch of the series used [yr].
    """

    k: FloatArray
    spread: FloatArray
    median_sigma: FloatArray
    n_windows: np.ndarray
    window_years: float
    step_years: float
    span: tuple[float, float]


def empirical_rate_scale(
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    *,
    window_years: float,
    step_years: float | None = None,
    min_obs: int | None = None,
    min_windows: int = MIN_WINDOWS,
) -> RateScale:
    """The factor k(T) that turns a formal WLS rate σ into an honest one.

    ``y`` must have its background removed (secular + seasonal: the
    detrended residual on a quiet span), so the true rate in every window
    is zero and the spread of the fitted rates is noise alone.

    Args:
        t: Epochs, shape (N,) [yr]; finite.
        y: Residuals, shape (C, N) or (N,) [L]; NaN epochs are dropped per
            component.
        sigma: 1-σ uncertainties, shaped as ``y`` [L].
        window_years: Window length T [yr] — the length of the rate the
            factor will scale.
        step_years: Centre step [yr]; default T/2 (windows half overlap:
            the spread is then of correlated samples, which leaves it
            unbiased but noisier — ``min_windows`` guards that).
        min_obs: Fewest epochs in a window; default half its daily count,
            and at least :data:`MIN_OBS`.
        min_windows: Fewest valid windows for a k.

    Returns:
        The :class:`RateScale`.

    Raises:
        ValueError: on a non-positive window or step, or mismatched shapes.
    """
    if window_years <= 0:
        raise ValueError(f"window_years must be > 0, got {window_years}")
    step = window_years / 2.0 if step_years is None else float(step_years)
    if step <= 0:
        raise ValueError(f"step_years must be > 0, got {step}")
    tt = np.asarray(t, dtype=np.float64)
    yy = np.atleast_2d(np.asarray(y, dtype=np.float64))
    ss = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
    if yy.shape != ss.shape or yy.shape[1] != tt.size:
        raise ValueError(f"shapes differ: t {tt.shape}, y {yy.shape}, sigma {ss.shape}")
    if min_obs is None:
        min_obs = max(MIN_OBS, int(0.5 * window_years * _DAYS_PER_YEAR))
    n_comp = yy.shape[0]
    k = np.full(n_comp, np.nan)
    spread = np.full(n_comp, np.nan)
    med = np.full(n_comp, np.nan)
    n_win = np.zeros(n_comp, dtype=np.int64)
    span = (float(np.nanmin(tt)), float(np.nanmax(tt))) if tt.size else (np.nan, np.nan)
    for c in range(n_comp):
        ok = np.isfinite(tt) & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
        if ok.sum() < min_obs or np.ptp(tt[ok]) < window_years:
            continue
        sv = sliding_velocity(
            tt[ok], yy[c, ok], ss[c, ok], window_years=window_years,
            step_years=step, model="linear", min_obs=min_obs,
        )  # fmt: skip
        rates = np.asarray(sv.rates, dtype=np.float64).ravel()
        sigmas = np.asarray(sv.sigmas, dtype=np.float64).ravel()
        valid = np.isfinite(rates) & np.isfinite(sigmas) & (sigmas > 0)
        n_win[c] = int(valid.sum())
        if n_win[c] < min_windows:
            continue
        r = rates[valid]
        spread[c] = MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r))))
        med[c] = float(np.median(sigmas[valid]))
        k[c] = max(1.0, spread[c] / med[c])
    return RateScale(k, spread, med, n_win, float(window_years), step, span)


def largest_gap(t: ArrayLike) -> float:
    """The longest interval between consecutive epochs [yr] (0 for < 2)."""
    tt = np.sort(
        np.asarray(t, dtype=np.float64)[np.isfinite(np.asarray(t, dtype=float))]
    )
    return float(np.max(np.diff(tt))) if tt.size > 1 else 0.0


def window_gates(
    t: ArrayLike,
    window: tuple[float, float],
    *,
    step_epochs: Sequence[float] = (),
    min_obs: int = MIN_OBS,
    min_cover: float = MIN_COVER,
    epoch_span: float = 0.0,
) -> tuple[str, ...]:
    """Why the epochs ``t`` inside ``window`` give no rate; empty when fit.

    Reasons, each a short code with its numbers:

    - ``n_obs=<n><min``: fewer than ``min_obs`` epochs;
    - ``cover=<f><min``: the epochs span less than ``min_cover`` of the
      window (a rate from a corner of it would be passed off as the
      window's);
    - ``step_in_window@<epoch>``: a declared step lies strictly inside the
      window, so a line through it measures the step.

    Args:
        t: The epochs inside the window [yr].
        window: (start, end) [yr], start < end.
        step_epochs: Declared step epochs of the station [yr].
        min_obs: Fewest epochs.
        min_cover: Least fraction of the window spanned by the epochs.
        epoch_span: The time one epoch stands for [yr] (a daily solution:
            its day, 1/365.25), added to the epochs' span: five daily
            epochs cover a five-day window. 0: the bare span.

    Raises:
        ValueError: on an empty or reversed window.
    """
    start, end = float(window[0]), float(window[1])
    if not end > start:
        raise ValueError(f"window end {end} is not after its start {start}")
    tt = np.asarray(t, dtype=np.float64)
    tt = tt[np.isfinite(tt)]
    reasons: list[str] = []
    if tt.size < min_obs:
        reasons.append(f"n_obs={tt.size}<{min_obs}")
    cover = (
        min(1.0, (float(np.ptp(tt)) + epoch_span) / (end - start)) if tt.size else 0.0
    )
    if cover < min_cover - 1e-9:  # fractional-year rounding: 0.79999… is 0.8
        reasons.append(f"cover={cover:.2f}<{min_cover:g}")
    for epoch in sorted(float(e) for e in step_epochs):
        if start < epoch < end:
            reasons.append(f"step_in_window@{epoch:.4f}")
    return tuple(reasons)


@dataclasses.dataclass(frozen=True)
class LineFit:
    """A weighted line through one component of a window.

    Attributes:
        rate: The slope [L/yr].
        sigma_apriori: Its σ from the data σ alone [L/yr].
        chi2nu: χ²/ν of the line (NaN for 2 epochs).
        f_quadratic: F of a quadratic against the line (NaN below 4
            epochs; :data:`NONLINEAR_F`).
        n: Epochs used.
    """

    rate: float
    sigma_apriori: float
    chi2nu: float
    f_quadratic: float
    n: int

    @property
    def sigma_scaled(self) -> float:
        """The WLS formal σ: a-priori × √(χ²/ν) (as
        :func:`~gps_analysis.velocity.estimate_velocity`), the a-priori σ
        for 2 epochs."""
        if not math.isfinite(self.chi2nu):
            return self.sigma_apriori
        return self.sigma_apriori * math.sqrt(self.chi2nu)

    @property
    def sigma_short(self) -> float:
        """The σ of the short tier: a-priori × √max(1, χ²/ν). On 2-4
        epochs χ²/ν is too noisy to shrink σ by, but a blunder must
        still inflate it."""
        if not math.isfinite(self.chi2nu):
            return self.sigma_apriori
        return self.sigma_apriori * math.sqrt(max(1.0, self.chi2nu))


def line_fit(t: ArrayLike, y: ArrayLike, sigma: ArrayLike) -> LineFit:
    """The weighted least-squares line ``y = a + v·(t − t̄)``, closed form.

    Equations (w = 1/σ², t̄ the weighted mean epoch):
        ``v = Σw(t−t̄)y / Σw(t−t̄)²``, ``σ_v = 1/√Σw(t−t̄)²``,
        ``χ²/ν = Σw r² / (n − 2)``;
        ``F = (χ²_line − χ²_quad) / (χ²_quad / (n − 3))`` (one more
        parameter; NaN below 4 epochs).

    Args:
        t: Epochs [yr], shape (N,), N ≥ 2, finite, not all equal.
        y: Values [L], shape (N,), finite.
        sigma: 1-σ [L], shape (N,), > 0.

    Raises:
        ValueError: fewer than 2 epochs, or all at one epoch.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    w = 1.0 / np.asarray(sigma, dtype=np.float64) ** 2
    n = tt.size
    if n < 2:
        raise ValueError(f"a line needs 2 epochs, got {n}")
    tm = float(np.sum(w * tt) / np.sum(w))
    dt = tt - tm
    stt = float(np.sum(w * dt * dt))
    if not stt > 0:
        raise ValueError("all epochs at one time")
    v = float(np.sum(w * dt * yy) / stt)
    a = float(np.sum(w * yy) / np.sum(w))
    chi_line = float(np.sum(w * (yy - a - v * dt) ** 2))
    chi2nu = chi_line / (n - 2) if n > 2 else math.nan
    f = math.nan
    if n > 3:
        sw = np.sqrt(w)
        scale = float(np.max(np.abs(dt))) or 1.0
        x = dt / scale
        design = np.column_stack([np.ones(n), x, x * x]) * sw[:, None]
        coef = np.linalg.lstsq(design, yy * sw, rcond=None)[0]
        chi_quad = float(np.sum((yy * sw - design @ coef) ** 2))
        f = (chi_line - chi_quad) / (chi_quad / (n - 3)) if chi_quad > 0 else math.inf
    return LineFit(v, 1.0 / math.sqrt(stt), chi2nu, f, n)


def signal_to_noise(rate: ArrayLike, sigma: ArrayLike) -> tuple[float, float]:
    """(|v_h|/σ_h, |v_u|/σ_u) of an N/E/U rate.

    σ_h is the σ of the horizontal speed,
    ``√((v_n σ_n)² + (v_e σ_e)²) / |v_h|``; NaN where a rate or σ is.
    """
    v = np.asarray(rate, dtype=np.float64).reshape(3)
    s = np.asarray(sigma, dtype=np.float64).reshape(3)
    vh = math.hypot(v[0], v[1])
    sh = math.hypot(v[0] * s[0], v[1] * s[1]) / vh if vh > 0 else math.nan
    snr_h = vh / sh if sh > 0 else math.nan
    snr_u = abs(v[2]) / s[2] if s[2] > 0 else math.nan
    return float(snr_h), float(snr_u)


def rate_scale_at(
    window_years: float, table_years: ArrayLike, table_k: ArrayLike
) -> FloatArray:
    """k at ``window_years`` from a table of k by window length.

    Linear in log T between the table's lengths, held at the end values
    outside them (k changes slowly with T; extrapolating a trend in it
    would invent one). NaN entries of a component are skipped.

    Args:
        window_years: The window length T [yr], > 0.
        table_years: Window lengths, shape (M,), increasing [yr].
        table_k: k per length and component, shape (M, C).

    Returns:
        k, shape (C,); NaN for a component with no finite entry.

    Raises:
        ValueError: on a non-positive T, or mismatched table shapes.
    """
    if window_years <= 0:
        raise ValueError(f"window_years must be > 0, got {window_years}")
    ty = np.asarray(table_years, dtype=np.float64)
    tk = np.asarray(table_k, dtype=np.float64)
    if tk.ndim == 1:
        tk = tk[:, None]
    if tk.shape[0] != ty.size or ty.size == 0:
        raise ValueError(f"table shapes differ: years {ty.shape}, k {tk.shape}")
    out = np.full(tk.shape[1], np.nan)
    x = math.log(window_years)
    for c in range(tk.shape[1]):
        ok = np.isfinite(tk[:, c]) & (ty > 0)
        if ok.any():
            out[c] = float(np.interp(x, np.log(ty[ok]), tk[ok, c]))
    return out
