"""Staged trajectory estimation: fit some terms here, hold others there.

Formalises an operator manoeuvre that was previously hand-written per
station (``gps_data_analyses/detrend-OLAC/detrend_test.py::katlafitlong``):
fit ``lineperiodic`` on a CLEAN sub-window, remove **only** the seasonal part
from the full series, then re-fit the secular rate over the LONGER span. The
reasoning is that different terms are best constrained on different domains —
the seasonal where the data is clean, the rate where the baseline is longest —
and that fitting both jointly on either domain alone is worse than either.

The same structure covers **borrowing** (design §2.6's ``UseSTA``): "fit my own
offset+rate with a donor station's seasonal held fixed" is one stage whose held
values came from another station instead of from an earlier stage. Borrowing
and staged refinement are one operation with different provenance for the held
vector, which is why they share a mechanism here rather than each getting one.

Derivation chain
----------------
Given the design A ∈ ℝ^{N×P} of a linear-in-parameters trajectory model,
observations y with 1-σ uncertainties σ, and a partition of the columns into
FREE and HELD:

1. **Hold** — move the known part to the left-hand side,
   ``y′ = y − A_h·v``, and drop its columns. Exact for terms linear in their
   parameters: moving a known constant across the equation changes nothing
   else (:func:`fit_held_partition`).
2. **Solve** — :func:`gps_analysis.fitting._wls_solve` on the reduced system,
   unchanged. Its reduced-χ² rescale then uses ``p_free``, which is the
   correct degrees of freedom for held-as-known — see the Numerical notes of
   :func:`fit_held_partition` before "fixing" it.
3. **Propagate** — the held values carry their own uncertainty into the free
   parameters through ``K``, the WLS regression of the held columns on the
   free ones. The conditional covariance (held rows/columns zeroed) always
   UNDERSTATES; the corrected one is closed-form and finite-sample exact.

Honesty note, because it decides the default elsewhere: staging is an operator
tool, not the statistically preferred route. Estimating all terms jointly under
a proper stochastic model is better when the model is adequate (Bos et al.
2013; Williams 2003). Staging earns its place when the flanks carry UNMODELED
signal — and the principled alternatives to it are already here: excise the
stretch (``estimate_detrend(segments=...)``), model it (transient terms), or
use an estimator that does not care (MIDAS).

Everything is pure, float64, unit-agnostic, inputs never mutated (leaf rules
R2/R6).
"""

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import _wls_solve
from .models import FloatArray

__all__ = [
    "HeldExplicit",
    "HeldFromStage",
    "Stage",
    "compose_held",
    "fit_held_partition",
]


@dataclasses.dataclass(frozen=True)
class HeldFromStage:
    """Hold a term group at the value an EARLIER stage fitted.

    Attributes:
        stage: Name of the stage whose fitted values are reused.
    """

    stage: str


@dataclasses.dataclass(frozen=True)
class HeldExplicit:
    """Hold a term group at values supplied by the caller.

    The borrowing case (design §0.6/§2.6): ``values`` are a donor station's
    coefficients and ``source`` names it, so the record's ``borrowed``
    provenance slot stops being decorative.

    Attributes:
        values: Coefficients in the model's absolute-t parameterization,
            one per parameter of the held group [model units].
        source: Free-form provenance — a station code, a catalog, a note.
        covariance: Optional (k, k) covariance of ``values``.  When given it
            propagates into the free parameters (see
            :func:`fit_held_partition`); when None the held values are
            treated as exactly known and the result is conditional on them.
    """

    values: FloatArray
    source: str
    covariance: FloatArray | None = None


Held = HeldFromStage | HeldExplicit


@dataclasses.dataclass(frozen=True)
class Stage:
    """One step of a staged estimation plan.

    Attributes:
        name: Stage label, referenced by :class:`HeldFromStage`.
        free: Term-group names estimated in this stage.
        held: Group name → where its value comes from.  Holds are
            TERM-GRANULAR by contract, never per-parameter within a term —
            see :func:`fit_held_partition`.
        segments: Fit domain for this stage, in the
            :func:`gps_analysis.baseline.slice_windows` union form; None
            inherits the caller's.
    """

    name: str
    free: tuple[str, ...]
    held: Mapping[str, Held] = dataclasses.field(default_factory=dict)
    segments: Sequence[tuple[float | None, float | None]] | None = None


def fit_held_partition(
    design: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    held_mask: ArrayLike,
    held_values: ArrayLike,
    held_cov: ArrayLike | None = None,
    absolute_sigma: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Fit the free columns of a design with the rest held at known values.

    Equation (columns partitioned into free f and held h):
        ``y = A_f·β + A_h·v + ε``  with v known
        ⟹ ``y′ = y − A_h·v = A_f·β + ε``,  solved by WLS for β̂.

        Uncertainty in v propagates through the WLS regression of the held
        columns on the free ones,

        ``K = (A_fᵀWA_f)⁻¹ A_fᵀW A_h``,
        ``Cov(β̂) = C_cond + K·C_v·Kᵀ``,   ``C_cond = (A_fᵀWA_f)⁻¹``

        and the full composed covariance of (β̂, v̂) is

        ``[[C_cond + K C_v Kᵀ,  −K C_v],
           [      −C_v Kᵀ,        C_v ]]``

        which is PSD by construction, being ``[K; −I]·C_v·[Kᵀ, −I]`` plus
        ``diag(C_cond, 0)``.

    Symbols → args:
        - ``A``   → ``design``: (N, P) design matrix, columns in the model's
          positional parameter order
        - ``y``   → ``y``: observations, (N,) [L]
        - ``σ``   → ``sigma``: 1-σ uncertainties, (N,) [L]; None ⇒ unweighted
        - ``h``   → ``held_mask``: (P,) bool, True where held
        - ``v``   → ``held_values``: (P,) [model units]; read only where
          ``held_mask``
        - ``C_v`` → ``held_cov``: (k, k) covariance of the held values,
          k = ``held_mask.sum()``; None ⇒ treat v as exactly known and
          return the CONDITIONAL covariance

    Args:
        design: The full (N, P) design.
        y: One component's observations, (N,).
        sigma: 1-σ uncertainties, (N,).
        held_mask: Which columns are held.
        held_values: Values for the held columns, in a full-length (P,)
            vector so callers need not re-index.
        held_cov: Covariance of the held values; see above.
        absolute_sigma: Passed through to
            :func:`~gps_analysis.fitting._wls_solve`.

    Returns:
        ``(p̂, C_p̂)`` — full length (P,) and (P, P), in the model's own
        parameter order, with the held entries carrying ``held_values`` and
        their covariance block ``C_v`` (or zeros when ``held_cov`` is None).

    Raises:
        ValueError: On shape mismatch, an all-held or all-free mask (nothing
            to fit / nothing to hold), or a non-symmetric ``held_cov``.

    Reference:
        The partition is the standard linear restriction (Seber & Lee,
        *Linear Regression Analysis* 2nd ed., ch. 3).  The propagation is the
        two-step estimator covariance; for two linear WLS stages on NESTED
        windows the Murphy & Topel 1985 (*JBES* 3(4), 370–379) cross-score
        terms vanish identically — see Numerical notes — so the general
        stacked-moment form (Newey & McFadden 1994, *Handbook of
        Econometrics* IV ch. 36 §6; Pagan 1984) collapses to the expression
        above.  Covariance transform per GUM (JCGM 100:2008 §5.1.2).

    Numerical notes:
        **Why the cross-term is absent.**  One might expect a
        ``Cov(A_fᵀWy, v̂)`` term, since a staged fit re-uses window-1 data.
        It is identically zero when stage 1 estimated the free parameters
        JOINTLY with the held ones: the stage-1 normal equations then make
        v̂'s error W-orthogonal to every column stage 2 re-uses, and the term
        reduces to the (free, held) block of the identity.  Verified to
        1.2e-11 against exact linear propagation on a 2500-epoch nested
        geometry.  This is why ``held_cov`` must be the held block of stage
        1's JOINT covariance and not a held-only fit's — passing the latter
        silently violates the condition that makes the formula exact.

        **Exactness conditions.**  (i) the held values come from a window
        contained in this one, with the same epochs and σ on shared rows;
        (ii) each free column, restricted to that window, is zero or a column
        of the earlier design; (iii) diagonal Σ with W = Σ⁻¹.  Under
        temporally correlated noise the cross-term is no longer exactly zero
        — the same caveat that already applies to every formal WLS covariance
        here (Williams 2003).

        ``K·C_v·Kᵀ ⪰ 0``, so the conditional covariance ALWAYS understates,
        never overstates, and vanishes iff ``A_fᵀW A_h = 0``.  Measured: 0.12 %
        on rate variance for a 25-yr record with an 18-yr clean window, 2.2 %
        when the two spans are closer in length.

        **The reduced-χ² rescale is already right.**  ``_wls_solve`` divides
        by ``n − p`` with p = the number of FREE columns, because that is
        literally the shape it is handed — and that is the correct degrees of
        freedom for held-as-known, since the held values consumed none in
        this stage.  It carries a second-order bias from the double use of
        window-1 data (``E[χ²] = (n₂−p_f) + tr(N₂,h|f C_v) − 2p_h``), small
        and in the deflating direction.  Do not "fix" the divisor.
    """
    a = np.asarray(design, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"design must be 2-D (N, P), got shape {a.shape}")
    if yy.ndim != 1 or yy.size != a.shape[0]:
        raise ValueError(f"y must be ({a.shape[0]},), got shape {yy.shape}")
    held = np.asarray(held_mask, dtype=np.bool_)
    if held.shape != (a.shape[1],):
        raise ValueError(f"held_mask must be ({a.shape[1]},), got shape {held.shape}")
    free = ~held
    if not held.any():
        raise ValueError("held_mask holds nothing; use fit_components directly")
    if not free.any():
        raise ValueError("held_mask holds every column; nothing left to fit")
    values = np.asarray(held_values, dtype=np.float64)
    if values.shape != (a.shape[1],):
        raise ValueError(
            f"held_values must be full length ({a.shape[1]},), got {values.shape}"
        )

    ss = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if ss is not None and ss.shape != yy.shape:
        raise ValueError(f"sigma shape {ss.shape} does not match y {yy.shape}")

    # (1) move the known part across: exact for linear-in-parameters terms
    y_reduced = yy - a[:, held] @ values[held]
    params_free, cov_free = _wls_solve(a[:, free], y_reduced, ss, absolute_sigma)

    p_full = values.copy()
    p_full[free] = params_free

    n_p = a.shape[1]
    cov_full = np.zeros((n_p, n_p), dtype=np.float64)
    idx_f = np.flatnonzero(free)
    idx_h = np.flatnonzero(held)
    cov_full[np.ix_(idx_f, idx_f)] = cov_free

    if held_cov is None:
        return p_full, cov_full

    c_v = np.asarray(held_cov, dtype=np.float64)
    k_held = int(held.sum())
    if c_v.shape != (k_held, k_held):
        raise ValueError(
            f"held_cov must be ({k_held}, {k_held}) for {k_held} held columns, "
            f"got shape {c_v.shape}"
        )
    if not np.allclose(c_v, c_v.T, rtol=0.0, atol=1e-12 * max(1.0, np.abs(c_v).max())):
        raise ValueError("held_cov must be symmetric")

    # (3) K = WLS regression of the held columns on the free ones. Solved as a
    # least-squares problem on the WHITENED columns rather than by forming
    # (A_f^T W A_f)^-1 explicitly -- same reason _wls_solve avoids the normal
    # equations: squaring the condition number of an absolute-yearf design.
    if ss is None:
        aw_f, aw_h = a[:, free], a[:, held]
    else:
        aw_f = a[:, free] / ss[:, np.newaxis]
        aw_h = a[:, held] / ss[:, np.newaxis]
    k_mat = np.linalg.lstsq(aw_f, aw_h, rcond=None)[0]

    cov_full[np.ix_(idx_f, idx_f)] = cov_free + k_mat @ c_v @ k_mat.T
    cross = -k_mat @ c_v
    cov_full[np.ix_(idx_f, idx_h)] = cross
    cov_full[np.ix_(idx_h, idx_f)] = cross.T
    cov_full[np.ix_(idx_h, idx_h)] = c_v
    return p_full, cov_full


def compose_held(
    n_params: int,
    groups: Mapping[str, NDArray[np.bool_]],
    held: Mapping[str, HeldExplicit],
) -> tuple[NDArray[np.bool_], FloatArray]:
    """Assemble a full-length held mask and value vector from group holds.

    Thin bookkeeping over :class:`Stage`'s ``held`` mapping — no math.

    Args:
        n_params: P, the model's parameter count.
        groups: Group name → (P,) boolean membership mask.
        held: Group name → an object exposing ``values`` for that group's
            parameters, in the group's own column order.

    Returns:
        ``(held_mask, held_values)`` sized (P,) and (P,), ready for
        :func:`fit_held_partition`.

    Raises:
        KeyError: For a held group name that has no membership mask.
        ValueError: When a group's supplied values do not match its width,
            or two held groups overlap (which would make the value of a
            shared column ambiguous).
    """
    mask = np.zeros(n_params, dtype=np.bool_)
    values = np.zeros(n_params, dtype=np.float64)
    for name, source in held.items():
        if name not in groups:
            raise KeyError(f"held group {name!r} is not a known term group")
        member = np.asarray(groups[name], dtype=np.bool_)
        if (mask & member).any():
            raise ValueError(f"held group {name!r} overlaps an earlier held group")
        supplied = np.asarray(source.values, dtype=np.float64)
        width = int(member.sum())
        if supplied.shape != (width,):
            raise ValueError(
                f"held group {name!r} needs {width} values, got {supplied.shape}"
            )
        mask |= member
        values[member] = supplied
    return mask, values
