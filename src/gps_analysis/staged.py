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
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import ModelFunc, _wls_solve
from .models import FloatArray, TrajectoryParams
from .terms import GROUP_ORDER

__all__ = [
    "HeldExplicit",
    "HeldFromStage",
    "Stage",
    "StageResult",
    "StagedEstimate",
    "compose_held",
    "estimate_staged",
    "evaluate_group_values",
    "fit_held_partition",
    "group_parameter_mask",
    "record_group_mask",
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
        free: Term-group names estimated in this stage.  MAY be empty for
            an apply-only stage that holds everything — the fully-borrowed
            station of design §2.6, where every term group comes from a
            donor and nothing is estimable locally.  Such a stage must hold
            at least one group; :func:`estimate_staged` takes an apply path
            for it (no fit, no invented covariance) rather than routing it
            through :func:`fit_held_partition`, which correctly refuses an
            all-held mask.
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


def evaluate_group_values(
    names: Sequence[str],
    values: ArrayLike,
    t: ArrayLike,
) -> FloatArray:
    """Evaluate named secular/periodic coefficients as a model value g(t).

    Equation:
        ``g(tᵢ) = Σ_j v_j·φ_j(tᵢ)`` — the partial trajectory carried by the
        named coefficients alone, where each basis function φ_j is denoted by
        the parameter NAME: ``offset``/``rate``/``curvature``/``poly_m`` are
        the absolute-t monomials ``t⁰, t¹, t², tᵐ`` of :class:`terms.Polynomial`,
        and ``cos_annual``/``sin_annual``/``cos_semiannual``/``sin_semiannual``/
        ``cos_harmonicH``/``sin_harmonicH`` are the absolute-``yearf``
        trigonometric columns ``cos(2πh·t), sin(2πh·t)`` of
        :class:`terms.Seasonal`.

    Symbols → args:
        - ``φ_j`` → ``names``: parameter names, each denoting one basis
          function (dimensionless labels)
        - ``v_j`` → ``values``: the coefficients, one per name [model units,
          typically mm and mm/yr on fractional-year epochs]
        - ``tᵢ`` → ``t``: evaluation epochs, absolute fractional years [yr]

    Returns:
        ``g(t)``, shape ``(N,)``, float64 [same units as the amplitudes].

    Raises:
        ValueError: on a length mismatch, or a name outside the secular and
            periodic groups.  Step and transient amplitudes are refused on
            purpose: their basis functions need per-station metadata (a step
            epoch, a transient τ) that a bare parameter name does not carry,
            so evaluating them "by name" would have to guess it.

    Reference:
        Bevis & Brown 2014, J. Geodesy 88 — the polynomial and seasonal
        blocks of the trajectory model (their eqs. 3 and the ``n_F = 2``
        seasonal series).

    Numerical notes:
        The columns come from the SAME :class:`terms.Polynomial` /
        :class:`terms.Seasonal` classes that generate every fitted design,
        indexed by name — never a locally restated ``t**m`` or phase
        convention, so the borrow path (which uses this to evaluate a
        donor's background on a borrower's epochs — ``geo_dataread``'s
        re-anchoring) cannot drift from what the estimator fitted.  Any
        SUBSET of a group's names is valid input: dropping ``offset``
        evaluates the donor's s(t) without its datum, which is exactly the
        re-anchoring use.  Trig columns are bounded; the monomials are the
        raw absolute-t columns and share the conditioning caveat of
        ``fitting._fit_linear_design`` — harmless here because nothing is
        inverted, it is a pure weighted sum.
    """
    from .terms import Polynomial, Seasonal

    vv = np.asarray(values, dtype=np.float64).ravel()
    if vv.size != len(names):
        raise ValueError(
            f"evaluate_group_values: {len(names)} names but {vv.size} values"
        )
    tt = np.asarray(t, dtype=np.float64)

    # Widest term that covers every requested name, so the name -> column
    # mapping is read off the term's OWN `names` tuple rather than restated.
    #
    # Degree 2 is the widest NAMED tier -- ("offset", "rate", "curvature") --
    # and every higher monomial is `poly_<m>`, parsed below. So this probe is
    # complete and, unlike the `max(max_degree, 1)` it replaces, INDEPENDENT
    # OF NAME ORDER: that probe only ever held ("offset", "rate"), so
    # `curvature` fell through to the "unrecognised" raise unless a `poly_m`
    # name happened to be seen FIRST and lifted max_degree to >= 2. A degree-2
    # donor yields exactly ("offset", "rate", "curvature") through
    # geo_dataread's `_entry_group_names`, so the borrow re-anchoring this
    # function exists to serve raised on its most ordinary input.
    secular_named = Polynomial(degree=2).names
    max_degree = 0
    max_harmonic = 0
    for name in names:
        group = _staged_group_of(name)
        if group == "secular":
            if name in secular_named:
                max_degree = max(max_degree, secular_named.index(name))
            elif name.startswith("poly_"):
                max_degree = max(max_degree, int(name.removeprefix("poly_")))
            else:  # pragma: no cover - _staged_group_of is the classifier
                raise ValueError(f"unrecognised secular parameter {name!r}")
        elif group == "periodic":
            tag = name.split("_", 1)[1]
            harmonic = {"annual": 1, "semiannual": 2}.get(tag)
            if harmonic is None:
                harmonic = int(tag.removeprefix("harmonic"))
            max_harmonic = max(max_harmonic, harmonic)
        else:
            raise ValueError(
                f"evaluate_group_values cannot evaluate {name!r}: group "
                f"{group!r} needs per-station metadata (step epoch, "
                f"transient tau) a parameter name does not carry"
            )

    columns: dict[str, FloatArray] = {}
    if max_degree or any(_staged_group_of(n) == "secular" for n in names):
        poly = Polynomial(degree=max_degree)
        cols = poly.columns(tt)
        columns.update({n: cols[:, j] for j, n in enumerate(poly.names)})
    if max_harmonic:
        seasonal = Seasonal(n_harmonics=max_harmonic)
        cols = seasonal.columns(tt)
        columns.update({n: cols[:, j] for j, n in enumerate(seasonal.names)})

    out = np.zeros(tt.shape, dtype=np.float64)
    for name, value in zip(names, vv, strict=True):
        out += value * columns[name]
    return out


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
            the held values carried one, conditional otherwise.  For an
            apply-only stage: zeros outside any supplied ``C_v`` block,
            because no fit ran and no conditional block exists to report.
        held_covariance: ``"propagated"``, ``"conditional"`` or
            ``"applied"``, so a reader of a stored record can tell which
            was reported.  ``"applied"`` is the apply-only marker: NOTHING
            was estimated in this stage — the parameter vector is exactly
            the composed held values.
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
        record and a single-fit one coexist in one document.  ``groups`` does
        not bump the version for the same reason ``terms`` DID: a reader that
        ignores ``terms`` mis-evaluates the trajectory (the spec is required
        to rebuild the model), while a reader that ignores ``groups`` loses
        only provenance — every number it evaluates is unchanged.

        ``groups`` is per-term-group provenance — for each group the model
        carries: which parameters are its (explicit ``indices`` into
        ``param_names``), which stage's fit owns the composed value, on what
        domain, and whether the value is this station's own (``"self"``) or a
        held-explicit source (a borrow's ``HeldExplicit.source``, e.g.
        ``donor:VMEY@<fitted_at>``).  An UNSTAGED record gets no ``groups``
        key at all: for a single fit the answer ("everything self, on the
        record's own domain") is fully derivable from keys the record already
        has, and writing it down again would be a second copy that can drift
        — absence already means "legacy single-window" for ``segments``, and
        it means "single-stage, all self" here.
        """
        return {
            # Explicit indices rather than a start/stop range, for two
            # reasons: a range is exactly the shape the `segments` comment in
            # detrend.py warns about (a 2-list a reader could index
            # positionally as a window), and contiguity per group is an
            # accident of today's term ordering (with_steps APPENDS step
            # amplitudes after transient ones, already breaking GROUP_ORDER
            # ordering) — indices stay correct whatever the order becomes.
            "groups": self._groups_block(),
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

    def _groups_block(self) -> dict[str, dict[str, object]]:
        """Per-group provenance of the composed parameter set.

        Membership comes from :func:`_staged_group_of` over ``param_names``
        — the SAME classifier :func:`group_parameter_mask` wraps — rather
        than from ``group_parameter_mask(self.model, ...)``, because
        ``self.model`` is a registry code only for registry models; a
        composed transient model carries a name like
        ``"polynomial+seasonal+log_transient"`` that no resolver accepts.
        Classifying the names we already hold keeps the one-definition rule
        without re-resolving anything.
        """
        members: dict[str, list[int]] = {}
        for j, name in enumerate(self.param_names):
            members.setdefault(_staged_group_of(name), []).append(j)
        out: dict[str, dict[str, object]] = {}
        for group in GROUP_ORDER:
            if group not in members:
                continue
            stage, provenance, segments = self._group_origin(group)
            out[group] = {
                "indices": members[group],
                "stage": stage,
                # The estimating stage's domain; None means the record's own
                # fit domain (its `segments`/`window` keys) — either because
                # the stage inherited it, or because the value was never
                # estimated on THIS station at all (a held-explicit borrow,
                # whose domain lives in the donor's record). `provenance`
                # disambiguates the two.
                "segments": (
                    None if segments is None else [[a, b] for a, b in segments]
                ),
                "provenance": provenance,
            }
        return out

    def _group_origin(
        self, group: str
    ) -> tuple[str, str, Sequence[tuple[float | None, float | None]] | None]:
        """``(stage, provenance, segments)`` of a group's composed value.

        Mirrors the composition rule of :func:`estimate_staged` (a group
        held in the FINAL stage is owned by that hold; otherwise the last
        stage that freed it owns it), then walks ``stage:`` pointers to
        where the value was actually MADE — so provenance says ``"self"``
        with the estimating stage, not the stage that merely re-held it.
        """
        by_name = {s.name: s for s in self.plan}
        stage = self.plan[-1]
        if group not in stage.held:
            for st in reversed(self.plan):
                if group in st.free:
                    return st.name, "self", st.segments
            raise ValueError(
                f"group {group!r} is never freed and not held in the final "
                f"stage; estimate_staged would have refused this plan"
            )
        while True:
            src = stage.held.get(group)
            if src is None:
                # A stage: chain that dead-ends in a stage which neither
                # freed nor held the group: estimate_staged composed a zero
                # there. Name the stage rather than claim "self".
                return stage.name, f"stage:{stage.name}", None
            if isinstance(src, HeldExplicit):
                return stage.name, src.source, None
            stage = by_name[src.stage]
            if group in stage.free:
                return stage.name, "self", stage.segments


def _held_provenance(held: Held) -> str:
    """One string naming where a held value came from."""
    if isinstance(held, HeldFromStage):
        return f"stage:{held.stage}"
    return f"explicit:{held.source}"


#: Parameter-name prefixes of the transient amplitudes (``terms.py``).
_TRANSIENT_AMP_PREFIXES = ("log_amp", "exp_amp")

#: Named secular parameters of the polynomial trend, by degree.
# The secular/periodic membership rules now live in `detrend`, shared with
# `_term_keep_mask` so the two classifiers -- which differ ON PURPOSE only
# in where STEP amplitudes go -- cannot silently drift anywhere else.


def _staged_group_of(name: str) -> str:
    """Classify one parameter name into a :data:`terms.GROUP_ORDER` group.

    Equation:
        A pure predicate ``name ↦ group``, no arithmetic.

    Symbols → args:
        - parameter name → ``name``: an entry of the model's
          ``param_names`` (dimensionless label)

    Returns:
        One of ``"secular"``, ``"periodic"``, ``"step"``, ``"transient"``.

    **This is deliberately NOT** ``detrend._term_keep_mask``.  That one is an
    APPLY-time selector (design §5.3: which terms to *remove* when
    detrending) and folds step amplitudes INTO ``"secular"``, because a
    Heaviside jump is background rather than seasonal.  Changing it would
    change what ``apply_detrend(terms="secular")`` removes, and 37 deployed
    records plus the workbench's ``--terms`` depend on that meaning.

    Staged estimation asks a different question — *which parameters does
    this stage estimate?* — and there ``step`` must be separable from
    ``rate``: a stage whose window excludes a step epoch cannot estimate its
    amplitude, and folding the two together made that design rank-deficient
    (measured on SELF, 2026-08-02).  So the two classifiers coexist on
    purpose, and this one matches
    :meth:`gps_analysis.terms.TrajectoryModel.group_mask`.

    Raises:
        ValueError: on a name it has not been taught.  Closed-world by
            design: a new term kind must fail loudly here rather than be
            silently dropped from every partition.
    """
    from .detrend import _STEP_AMP_PREFIX, _is_periodic_param, _is_secular_param

    if _is_secular_param(name):
        return "secular"
    if _is_periodic_param(name):
        return "periodic"
    if name.startswith(_STEP_AMP_PREFIX):
        return "step"
    if any(name.startswith(p) for p in _TRANSIENT_AMP_PREFIXES):
        return "transient"
    raise ValueError(
        f"cannot classify model parameter {name!r} into a term group; "
        f"known groups: {list(GROUP_ORDER)}"
    )


def _group_masks(
    model: ModelFunc, groups: Sequence[str]
) -> dict[str, NDArray[np.bool_]]:
    """Term-group membership masks over the four :data:`GROUP_ORDER` groups."""
    return {g: group_parameter_mask(model, g) for g in groups}


def group_parameter_mask(model: str | ModelFunc, group: str) -> NDArray[np.bool_]:
    """Which parameters of ``model`` belong to term group ``group``.

    Equation:
        ``mask[j] = [param_names[j] ∈ group]`` — a pure membership
        predicate over the model's parameter vector, no arithmetic.

    Symbols → args:
        - ``f`` → ``model``: registry code or registered callable
        - group name → ``group``: ``"secular"`` | ``"periodic"`` |
          ``"step"`` | … (dimensionless label)

    Returns:
        Boolean mask of shape ``(P,)``, aligned with the model's
        ``param_names`` and therefore with a stored record's per-component
        parameter vector.

    Public because a *caller outside this package* needs it to slice one
    group out of a donor station's stored record when resolving a borrow
    (``geo_dataread.stage_plan``).  Exposed here rather than duplicated
    there so "secular" keeps exactly one definition — the same classifier
    :func:`gps_analysis.detrend.select_terms` and :func:`estimate_staged`
    use.

    Numerical notes:
        Membership only; it does not check that the group is non-empty for
        this model.  An all-False mask means the model has no such term,
        which the caller should treat as an error rather than as an empty
        borrow.
    """
    from .detrend import _param_names, _resolve_model

    if group not in GROUP_ORDER:
        raise ValueError(f"unknown term group {group!r}; known: {list(GROUP_ORDER)}")
    model_func, _ = _resolve_model(model)
    names = _param_names(model_func)
    return np.array([_staged_group_of(n) == group for n in names], dtype=np.bool_)


def record_group_mask(
    record: Mapping[str, Any], group: str | Sequence[str]
) -> NDArray[np.bool_]:
    """Which of a stored RECORD's parameters belong to the named term groups.

    :func:`group_parameter_mask` answers this for a *model*, whose parameter
    vector is the unaugmented one.  A stored record's is longer:
    ``StationEstimate.to_record`` APPENDS one ``step_amp_k`` per declared
    step to the model's ``param_names``, so a station with a declared step
    stores 7 parameters against ``lineperiodic``'s 6.

    Callers that compared the two widths directly therefore REFUSED every
    record carrying a step — which took out borrowing ``secular`` or
    ``periodic`` from any station in ``steps.yaml`` (SELF, HOFN), and with it
    the whole "hold this station's own saved background and estimate only the
    events" workflow.  Measured 2026-08-23.

    Classification still comes from ``param_names``, never from a local list;
    the appended tail is the one thing decided by construction, because no
    classifier ever sees it.

    Args:
        record: A stored detrend record (needs ``param_names``, or ``model``
            plus a component to infer the width from).
        group: One term-group name, or several.

    Returns:
        Boolean mask aligned with the record's per-component parameter vector.

    Raises:
        ValueError: On an unknown group name, an unsupported
            ``record_version``, or ``param_names`` that disagree with the
            declared model's own positional names.

    Numerical notes:
        ``record_version`` is enforced here for the same reason
        :func:`detrend.trajectory_from_record` enforces it: this is the OTHER
        record reader, and it is the one the borrow path actually calls
        (``geo_dataread`` reaches a donor through
        :func:`record_group_mask` -> ``donor_group_values``; it has no
        ``trajectory_from_record`` call site at all).  Without the check the
        version claim was decorative on that path -- a ``record_version`` 0
        record whose ``param_names`` had been permuted to
        ``["rate", "offset", ...]`` returned a happy ``[T, T, F, ...]`` mask,
        and the positional ``params[mask]`` that follows then borrowed
        ``(12.5, 3.0)`` as ``(offset, rate)``.  The same record raises in
        ``trajectory_from_record``.  Two readers, one record format, one
        enforcement.
    """
    from .detrend import SUPPORTED_RECORD_VERSIONS, _param_names, _resolve_model

    wanted = {group} if isinstance(group, str) else set(group)
    unknown = wanted - set(GROUP_ORDER)
    if unknown:
        raise ValueError(
            f"unknown term group(s) {sorted(unknown)}; known: {list(GROUP_ORDER)}"
        )
    # Enforced WHEN PRESENT. `trajectory_from_record` additionally refuses a
    # record with no version at all; this reader cannot yet, because it also
    # serves the documented hand-written/pre-`param_names` records (see the
    # fallback below) and 19 such fixtures live in geo_dataread. So the gap
    # that remains is a record carrying NO `record_version` key -- malformed
    # in production, where `to_record` always writes one. Closing it is a
    # cross-package fixture change, deliberately not made here.
    if "record_version" in record:
        version = record.get("record_version")
        if version not in SUPPORTED_RECORD_VERSIONS:
            raise ValueError(
                f"unknown record_version {version!r}; this reader supports "
                f"{sorted(SUPPORTED_RECORD_VERSIONS)}"
            )
    names = list(record.get("param_names") or ())
    if names:
        # `param_names` drives the classification, so it must agree with the
        # model it claims to describe -- otherwise the mask is right about the
        # NAMES and wrong about the SLOTS the caller indexes with it. Only the
        # unaugmented head is checked: `to_record` appends one `step_amp_k`
        # per declared step, and that tail is decided by construction (see
        # above). A v2 record's `model` is a "+"-joined term-kind string
        # rather than a registry code, so it is left to the terms reader.
        model_code = record.get("model")
        if record.get("terms") is None and isinstance(model_code, str):
            model_func, _ = _resolve_model(model_code)
            expected = tuple(_param_names(model_func))
            head = tuple(names[: len(expected)])
            # Only when the record is at least as wide as its model. A SHORTER
            # `param_names` is a different malformation -- a record carrying
            # only `["step_amp_1"]` against lineperiodic has no background at
            # all -- and its callers diagnose that far better than a generic
            # mismatch would ("no background to save"). This check is about
            # PERMUTED or renamed slots, which needs a full head to compare.
            if len(names) >= len(expected) and head != expected:
                raise ValueError(
                    f"record param_names {head} disagree with model "
                    f"{model_code!r} whose parameters are {expected}; the mask "
                    f"would be indexed positionally against the wrong slots"
                )
        return np.array([_staged_group_of(n) in wanted for n in names], dtype=np.bool_)

    # No param_names (a hand-written or very old record): fall back to the
    # model's own vector and pad the step tail from the component width.
    model = record.get("model")
    if not isinstance(model, str):
        raise ValueError("record has neither param_names nor a model code")
    base = np.zeros(0, dtype=np.bool_)
    for name in wanted:
        one = group_parameter_mask(model, name)
        base = one if base.size == 0 else (base | one)
    components = record.get("components") or ()
    width = base.size
    if components:
        first = components[0]
        if isinstance(first, Mapping):
            width = len(first.get("params") or ())
    mask = np.zeros(max(width, base.size), dtype=np.bool_)
    mask[: base.size] = base
    if "step" in wanted:
        mask[base.size :] = True
    return mask


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

    **Apply-only stages.**  A stage that frees NOTHING and holds at least one
    group takes an apply path: its parameter vector is exactly the composed
    held values, no fit is attempted, and no covariance is invented — the
    stage's covariance is all-zero outside any supplied ``C_v`` block, which
    is precisely how :func:`fit_held_partition` already reports held blocks
    (``C_v`` where given, zeros when the values are treated as exactly
    known), so a reader of the composed matrix needs no second convention.
    The stage is flagged ``held_covariance="applied"`` so a stored record
    says unmistakably that nothing was estimated on this station.  This is
    the fully-borrowed station of design §2.6 (ELDC holding both secular and
    periodic from a donor): before this path existed the case was
    unrepresentable, because :func:`fit_held_partition` — correctly, it is a
    fitting primitive — refuses an all-held mask.  The apply path goes
    AROUND that refusal, never through it.

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
        ValueError: For an empty plan, a duplicate stage name, a stage that
            neither frees nor holds any group, a reference to a stage that
            has not run yet, a term owned by no stage, or a stage whose
            domain holds no epochs.

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
        if not stage.free and not stage.held:
            raise ValueError(
                f"stage {stage.name!r} frees nothing and holds nothing; a "
                f"stage either estimates term groups or applies held ones"
            )

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

    # A HeldFromStage can only relay what its source actually ESTIMATED. A
    # stage's parameter vector is zero outside what that stage freed or held,
    # so holding a group from a stage that did neither relays a ZERO -- and
    # because the source's covariance BLOCK for it is likewise zero rather
    # than None, `all(b is not None for ...)` passes and the composed record
    # labels that zero `held_covariance="propagated"`: an invented value,
    # documented as sourced. The docstring above has always promised such a
    # plan is "rejected up front rather than silently emitting a zero"; the
    # ownership check could not see it, because it counts a group as owned as
    # soon as ANY stage frees it, without asking whether the stage being
    # borrowed FROM is that stage.
    estimated_by: dict[str, set[str]] = {
        stage.name: {*stage.free, *stage.held} for stage in plan
    }
    ran: set[str] = set()
    for stage in plan:
        for g, src in stage.held.items():
            if not isinstance(src, HeldFromStage):
                continue
            if src.stage not in ran:
                raise ValueError(
                    f"stage {stage.name!r} holds {g!r} from {src.stage!r}, "
                    f"which has not run"
                )
            if g not in estimated_by[src.stage]:
                raise ValueError(
                    f"stage {stage.name!r} holds {g!r} from stage "
                    f"{src.stage!r}, which neither frees nor holds {g!r}; its "
                    f"value there is a composed zero, not an estimate"
                )
        ran.add(stage.name)

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
        # A freed group the model has no term for contributes no columns.
        # `group_parameter_mask` documents an all-False mask as something "the
        # caller should treat as an error", and until the apply-only path
        # existed it effectively was one: the stage fell through to
        # `fit_held_partition`, which refused an all-held mask. Now an empty
        # `free_mask` selects APPLY instead, so a plan asking to FIT a
        # transient on a model that has none returned a record saying nothing
        # was estimated -- indistinguishable from a deliberate fully-borrowed
        # station. Checked per group, not just on the union, so a partly
        # resolvable `free=("secular", "transient")` cannot quietly fit half
        # of what was asked.
        absent = [g for g in stage.free if not masks[g].any()]
        if absent:
            raise ValueError(
                f"stage {stage.name!r} frees {absent}, which model "
                f"{model_name!r} has no term for; nothing would be estimated "
                f"for {'them' if len(absent) > 1 else 'it'}"
            )
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
                            (
                                None
                                if src.covariance is None
                                else np.asarray(src.covariance, dtype=np.float64)
                            ),
                        )
                    )
            if blocks and all(b is not None for _m, b in blocks):
                idx = np.flatnonzero(held_mask)
                held_cov_block = np.zeros((idx.size, idx.size), dtype=np.float64)
                for gm, blk in blocks:
                    pos = np.searchsorted(idx, np.flatnonzero(gm))
                    held_cov_block[np.ix_(pos, pos)] = blk
                kind = "propagated"

            if not free_mask.any():
                # APPLY, don't fit (the all-held case): the stage's parameter
                # vector IS the composed held values, and detecting it BEFORE
                # fit_held_partition is deliberate — that primitive refuses an
                # all-held mask, and the refusal is correct for a fitting
                # primitive. No fit ran, so no covariance is invented: zeros
                # outside any supplied C_v block, matching how
                # fit_held_partition already reports held blocks. Blocks are
                # relayed PER GROUP (unlike the fit path's all-or-nothing
                # held_cov_block, whose propagation math needs the whole held
                # covariance): nothing is propagated here, so a group that
                # carried its C_v keeps it even beside one that did not.
                p = values.copy()
                cov = np.zeros((n_params, n_params), dtype=np.float64)
                for gm, blk in blocks:
                    if blk is not None:
                        gi = np.flatnonzero(gm)
                        cov[np.ix_(gi, gi)] = blk
                kind = "applied"
            elif held_mask.any():
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

    # Composition is by OWNERSHIP: the last stage that freed a coefficient owns
    # it, and anything only ever HELD in the final stage is owned by that stage
    # (which is what the `owner.all()` check above guarantees is exhaustive).
    #
    # Values were previously written stage-by-stage and the covariance with
    # them, block by block -- which quietly composed ONE matrix out of SEVERAL
    # estimators.  A cross-block between two coefficients owned by different
    # stages was left holding whatever an earlier stage wrote, even when both
    # of its own diagonal blocks had since been overwritten: an off-diagonal
    # from a fit whose parameters were discarded.  Measured on a two-stage
    # lineperiodic plan (A frees secular+periodic, B re-frees periodic holding
    # A's secular), composed cov[rate, cos_annual] was A's -0.0031 while the
    # cos_annual variance was B's -- B's own value is +0.356, a factor 100 and
    # a sign away.
    #
    # An off-diagonal is only meaningful when ONE estimator produced both
    # coefficients, so it survives only when the owners agree.  Where they do
    # -- the ordinary case, since `fit_held_partition` returns a full P x P
    # covariance spanning the final stage's free AND held blocks -- the whole
    # joint block is taken from that stage, cross terms included.  Where they
    # do not, no estimator ever formed the covariance and zero is the honest
    # answer: it understates (asserts independence across stages), the same
    # direction as the conditional-covariance caveat documented on
    # `fit_held_partition`, and it never asserts a number nothing computed.
    owner = np.full(n_params, -1, dtype=np.int64)
    for k, r in enumerate(results):
        owner[r.free_mask] = k
    owner[results[-1].held_mask] = len(results) - 1
    for c in range(n_components):
        for k, r in enumerate(results):
            sel = np.flatnonzero(owner == k)
            if sel.size:
                composed[c][sel] = r.params[c][sel]
                composed_cov[c][np.ix_(sel, sel)] = r.covariance[c][np.ix_(sel, sel)]

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
