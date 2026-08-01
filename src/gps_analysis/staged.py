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

from .fitting import ModelFunc, _wls_solve
from .models import FloatArray, TrajectoryParams

__all__ = [
    "HeldExplicit",
    "HeldFromStage",
    "Stage",
    "StageResult",
    "StagedEstimate",
    "compose_held",
    "estimate_staged",
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


@dataclasses.dataclass(frozen=True)
class StageResult:
    """Diagnostics of one executed stage.

    Attributes:
        name: The stage's label.
        free_mask: (P,) parameters this stage estimated.
        held_mask: (P,) parameters it held.
        held_values: (P,) the values held, zero off ``held_mask``.
        held_sources: Group name → provenance string (an earlier stage's
            name, or a donor station for a borrow).
        n_epochs: Epochs inside this stage's domain.
        params: Per component, the stage's own full-length solution.
        covariance: Per component, its (P, P) covariance — propagated when
            the held values carried one, conditional otherwise.
        held_covariance: ``"propagated"`` or ``"conditional"``, so a reader
            of a stored record can tell which was reported.
    """

    name: str
    free_mask: NDArray[np.bool_]
    held_mask: NDArray[np.bool_]
    held_values: FloatArray
    held_sources: Mapping[str, str]
    n_epochs: int
    params: tuple[FloatArray, ...]
    covariance: tuple[FloatArray, ...]
    held_covariance: str


@dataclasses.dataclass(frozen=True)
class StagedEstimate:
    """Result of :func:`estimate_staged` — one parameter set, many stages.

    Attributes:
        fits: One :class:`~gps_analysis.models.TrajectoryParams` per
            component, the COMPOSED solution.  Composition rule: the last
            stage in which a term was free owns its value.
        stages: Per-stage diagnostics in execution order.
        plan: The plan as executed.
        model: Model registry code.
        param_names: Positional parameter names of the model.
    """

    fits: tuple[TrajectoryParams, ...]
    stages: tuple[StageResult, ...]
    plan: tuple[Stage, ...]
    model: str
    param_names: tuple[str, ...]

    def to_record_fragment(self) -> dict[str, object]:
        """The additive record keys describing how this was estimated.

        Additive at ``RECORD_VERSION`` 1, following the ``segments``
        precedent: ``trajectory_from_record`` reads only
        version/model/step_epochs/param_names/components, and
        ``TrajectoryParams.from_record`` ignores unknown keys, so a staged
        record and a single-fit one coexist in one document.
        """
        return {
            "stage_plan": [
                {
                    "name": s.name,
                    "free": list(s.free),
                    "held": {k: _held_provenance(v) for k, v in s.held.items()},
                    "segments": (
                        None if s.segments is None else [[a, b] for a, b in s.segments]
                    ),
                }
                for s in self.plan
            ],
            "stages": [
                {
                    "name": r.name,
                    "n_epochs": int(r.n_epochs),
                    "free": [
                        n
                        for n, k in zip(self.param_names, r.free_mask, strict=True)
                        if k
                    ],
                    "held_sources": dict(r.held_sources),
                    "held_covariance": r.held_covariance,
                }
                for r in self.stages
            ],
        }


def _held_provenance(held: Held) -> str:
    """One string naming where a held value came from."""
    if isinstance(held, HeldFromStage):
        return f"stage:{held.stage}"
    return f"explicit:{held.source}"


def _group_masks(
    model: ModelFunc, groups: Sequence[str]
) -> dict[str, NDArray[np.bool_]]:
    """Term-group membership masks, via the existing classifier.

    Reuses ``detrend._term_keep_mask`` rather than restating which
    parameter names belong to which group — one definition of "secular",
    in one place, shared with :func:`gps_analysis.detrend.select_terms`.
    """
    from .detrend import _term_keep_mask

    return {g: _term_keep_mask(model, g) for g in groups}


def estimate_staged(
    model: str | ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    plan: Sequence[Stage],
    segments: Sequence[tuple[float | None, float | None]] | None = None,
    absolute_sigma: bool = False,
    names: Sequence[str] | None = None,
    tol: float = 1e-3,
) -> StagedEstimate:
    """Run a staged estimation plan and compose one parameter set.

    Thin orchestration (MATH_STANDARDS §1, no new math): per stage, resolve
    the free/held column partition from the term groups, slice the stage's
    domain with :func:`~gps_analysis.baseline.slice_windows`, and call
    :func:`fit_held_partition`.  All the mathematics lives there.

    Composition rule: **the last stage in which a term is free owns its
    value.**  A term that is never free and never held in the final stage
    would leave an unowned coefficient, so the plan is rejected up front
    rather than silently emitting a zero.

    Symbols → args:
        - ``f`` → ``model``: registry code or a registered callable
        - ``tᵢ`` → ``t``: epochs, fractional years [yr]
        - ``y_cᵢ`` → ``y``: observations, (N,) or (C, N) [L]
        - ``σ_cᵢ`` → ``sigma``: 1-σ uncertainties, shape of ``y`` [L]
        - stages → ``plan``: the :class:`Stage` sequence, executed in order
        - default domain → ``segments``: used by any stage whose own
          ``segments`` is None

    Args:
        model: Trajectory model.
        t: Epochs, sorted ascending.
        y: Observations.
        sigma: Uncertainties.
        plan: Stages, in execution order.
        segments: Default fit domain (union form); None = the whole series.
        absolute_sigma: Passed to the WLS solve.
        names: Per-component labels.
        tol: Window boundary tolerance [yr].

    Returns:
        A :class:`StagedEstimate`.

    Raises:
        ValueError: For an empty plan, a duplicate stage name, a reference
            to a stage that has not run yet, a term owned by no stage, or a
            stage whose domain holds no epochs.

    Reference:
        The manoeuvre is the operator recipe of
        ``detrend-OLAC/detrend_test.py::katlafitlong``; the held mechanism
        and its covariance are :func:`fit_held_partition`.

    Numerical notes:
        Each stage's held covariance is taken from the SOURCE stage's own
        joint covariance block, which is the condition that makes the
        propagation exact — see :func:`fit_held_partition`'s notes.  A
        :class:`HeldExplicit` without a ``covariance`` yields a conditional
        result for that stage, flagged as such in
        :attr:`StageResult.held_covariance` so it cannot be mistaken for a
        propagated one.
    """
    from .baseline import slice_windows
    from .detrend import _resolve_model
    from .fitting import _components_2d, _resolve_linear_design

    if not plan:
        raise ValueError("plan must contain at least one stage")
    seen: set[str] = set()
    for stage in plan:
        if stage.name in seen:
            raise ValueError(f"duplicate stage name {stage.name!r}")
        seen.add(stage.name)

    model_func, model_name = _resolve_model(model)
    design_spec = _resolve_linear_design(model_func)
    if design_spec is None:
        raise ValueError(
            "estimate_staged requires a linear-in-parameters model: holding a "
            "term is column arithmetic, which only removes that term when the "
            f"model is linear in its parameters; got {model_name!r}"
        )
    tt = np.asarray(t, dtype=np.float64)
    yy, was_1d = _components_2d(y, "y")
    ss = None if sigma is None else _components_2d(sigma, "sigma")[0]
    n_components = yy.shape[0]

    full_design = design_spec.build(tt)
    n_params = full_design.shape[1]
    from .detrend import _param_names

    param_names = tuple(_param_names(model_func))

    wanted = {g for s in plan for g in (*s.free, *s.held)}
    masks = _group_masks(model_func, sorted(wanted))

    # Composition ownership, validated BEFORE any fitting: a coefficient that
    # no stage ever frees and that the last stage does not hold would silently
    # be emitted as zero.
    owner = np.zeros(n_params, dtype=np.bool_)
    for stage in plan:
        for g in stage.free:
            owner |= masks[g]
    for g in plan[-1].held:
        owner |= masks[g]
    if not owner.all():
        orphan = [n for n, o in zip(param_names, owner, strict=True) if not o]
        raise ValueError(
            f"parameters {orphan} are never estimated and not held in the final "
            f"stage, so the composed record would carry them as zero"
        )

    composed = [np.zeros(n_params, dtype=np.float64) for _ in range(n_components)]
    composed_cov = [
        np.zeros((n_params, n_params), dtype=np.float64) for _ in range(n_components)
    ]
    by_stage: dict[str, StageResult] = {}
    results: list[StageResult] = []

    for stage in plan:
        segs = stage.segments if stage.segments is not None else segments
        mask = (
            np.ones(tt.size, dtype=np.bool_)
            if segs is None
            else slice_windows(tt, segs, tol=tol)
        )
        n_epochs = int(np.count_nonzero(mask))
        if n_epochs == 0:
            raise ValueError(f"stage {stage.name!r} domain {segs} contains no epochs")

        held_mask = np.zeros(n_params, dtype=np.bool_)
        sources: dict[str, str] = {}
        for g in stage.held:
            held_mask |= masks[g]
            sources[g] = _held_provenance(stage.held[g])

        free_mask = np.zeros(n_params, dtype=np.bool_)
        for g in stage.free:
            free_mask |= masks[g]
        if (free_mask & held_mask).any():
            overlap = [
                n for n, k in zip(param_names, free_mask & held_mask, strict=True) if k
            ]
            raise ValueError(f"stage {stage.name!r} both frees and holds {overlap}")

        stage_params: list[FloatArray] = []
        stage_cov: list[FloatArray] = []
        kind = "conditional"
        for c in range(n_components):
            values = np.zeros(n_params, dtype=np.float64)
            held_cov_block: FloatArray | None = None
            blocks: list[tuple[NDArray[np.bool_], FloatArray | None]] = []
            for g, src in stage.held.items():
                gm = masks[g]
                if isinstance(src, HeldFromStage):
                    if src.stage not in by_stage:
                        raise ValueError(
                            f"stage {stage.name!r} holds {g!r} from "
                            f"{src.stage!r}, which has not run"
                        )
                    prior = by_stage[src.stage]
                    values[gm] = prior.params[c][gm]
                    blocks.append((gm, prior.covariance[c][np.ix_(gm, gm)]))
                else:
                    values[gm] = np.asarray(src.values, dtype=np.float64)
                    blocks.append(
                        (
                            gm,
                            None
                            if src.covariance is None
                            else np.asarray(src.covariance, dtype=np.float64),
                        )
                    )
            if blocks and all(b is not None for _m, b in blocks):
                idx = np.flatnonzero(held_mask)
                held_cov_block = np.zeros((idx.size, idx.size), dtype=np.float64)
                for gm, blk in blocks:
                    pos = np.searchsorted(idx, np.flatnonzero(gm))
                    held_cov_block[np.ix_(pos, pos)] = blk
                kind = "propagated"

            if held_mask.any():
                p, cov = fit_held_partition(
                    full_design[mask],
                    yy[c][mask],
                    None if ss is None else ss[c][mask],
                    held_mask=held_mask,
                    held_values=values,
                    held_cov=held_cov_block,
                    absolute_sigma=absolute_sigma,
                )
            else:
                p, cov = _wls_solve(
                    full_design[mask][:, free_mask],
                    yy[c][mask],
                    None if ss is None else ss[c][mask],
                    absolute_sigma,
                )
                full_p = np.zeros(n_params, dtype=np.float64)
                full_p[free_mask] = p
                full_c = np.zeros((n_params, n_params), dtype=np.float64)
                fi = np.flatnonzero(free_mask)
                full_c[np.ix_(fi, fi)] = cov
                p, cov = full_p, full_c
            stage_params.append(p)
            stage_cov.append(cov)
            # the last stage that FREED a term owns its value
            composed[c][free_mask] = p[free_mask]
            fi = np.flatnonzero(free_mask)
            composed_cov[c][np.ix_(fi, fi)] = cov[np.ix_(fi, fi)]

        result = StageResult(
            name=stage.name,
            free_mask=free_mask,
            held_mask=held_mask,
            held_values=stage_params[0] * held_mask,
            held_sources=sources,
            n_epochs=n_epochs,
            params=tuple(stage_params),
            covariance=tuple(stage_cov),
            held_covariance=kind if held_mask.any() else "n/a",
        )
        by_stage[stage.name] = result
        results.append(result)

    # anything only ever HELD in the final stage keeps that stage's value
    final = results[-1]
    for c in range(n_components):
        composed[c][final.held_mask] = final.params[c][final.held_mask]
        hi = np.flatnonzero(final.held_mask)
        composed_cov[c][np.ix_(hi, hi)] = final.covariance[c][np.ix_(hi, hi)]

    label: list[str | None] = (
        list(names) if names is not None else [None] * n_components
    )
    fits = tuple(
        TrajectoryParams(
            params=composed[c], covariance=composed_cov[c], component=label[c]
        )
        for c in range(n_components)
    )
    if was_1d and len(fits) != 1:  # pragma: no cover - shape guard
        raise ValueError("1-D y produced multiple components")
    return StagedEstimate(
        fits=fits,
        stages=tuple(results),
        plan=tuple(plan),
        model=model_name,
        param_names=param_names,
    )
