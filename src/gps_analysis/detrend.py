"""Stored-parameter detrending: estimate once, apply anywhere (leaf slice).

Implements the leaf surface of ``docs/DESIGN_live_detrending.md`` (§0
locked decisions, §2 estimation, §5 signatures): the raw series is the
durable record; detrending is a *view* computed from separately stored,
versioned parameters. Estimation is a deliberate, occasional act
(:func:`estimate_detrend`); application is a cheap pure evaluation
``y − f(t; p̂)`` valid for ANY epoch — including live epochs that arrive
after the fit (:func:`apply_detrend` / :func:`evaluate_record`). The
legacy alternative — re-fitting the whole series on every read — makes
the "background" definition change with every new epoch and lets
transients contaminate the fit (design §0).

Derivation chain
----------------
Given epochs t ∈ ℝᴺ (fractional years, ``yearf``, sorted ascending),
per-component observations y ∈ ℝ^{C×N} with 1-σ uncertainties σ, a
trajectory model f(t; p) (:mod:`gps_analysis.models`, default
:func:`~gps_analysis.models.lineperiodic` — Bevis & Brown 2014 eq. 1)
and the station's known step epochs t_k:

1. **Window** — ``m = slice_window(t, t_start, t_end)``
   (:func:`gps_analysis.baseline.slice_window`), then the mechanical
   validity gates of design §2.2 rule 3 (span, epoch count, largest
   gap) — a silently bad background is worse than no background, so a
   failed gate raises, naming itself.
2. **Step augmentation** — steps inside the window extend the model,
   ``f(t; p, a) = f_traj(t; p) + Σ_k a_k·H(t − t_k)``
   (:func:`gps_analysis.fitting.with_steps`; H(0) = 1) — epochs fixed,
   amplitudes estimated, so long windows spanning known offsets stay
   usable (design §2.3 v2, pulled forward per §0.7 "as long as
   possible").
3. **Outlier removal BEFORE the fit** (BGÓ hard rule, §0.2) —
   :func:`gps_analysis.outliers.detect_outliers`: the model-aware,
   signal-protecting detector (Hampel + global identifiers on
   studentized residuals of a robust Huber step-augmented fit, with the
   §3.4 protection stage so genuine signal is never eaten).
4. **Final clean WLS on the inliers** —
   :func:`gps_analysis.fitting.fit_components` (closed-form SVD WLS for
   the linear-in-parameters house models), so the reported covariance
   is the standard Gauss–Markov ``C_p̂ = (AᵀWA)⁻¹`` with the reduced-χ²
   rescaling; uncertainties propagate linearly per GUM (JCGM 100:2008
   §5.1.2).
5. **Package** — :class:`DetrendEstimate`: per-component
   :class:`~gps_analysis.models.TrajectoryParams` (absolute-t
   parameterization, **intercept included** — design §0.1 #1), the
   inlier mask (raw-preserving, the outlier precedent), window/span
   diagnostics, the ``detrend_method`` provenance tag
   (``"step_augmented_robust"`` | ``"plain_wls"``, §0.2) and the
   caller's opaque reference-frame tag (§0.5 — detrend runs AFTER plate
   removal; the leaf stays frame-agnostic and only carries the tag).
6. **Serialize** — :meth:`DetrendEstimate.to_record` →
   plain JSON-ready dict (model code, param names, step epochs, full
   parameter vectors + upper-triangle covariances via
   :meth:`~gps_analysis.models.TrajectoryParams.to_record`, provenance:
   ``fitted_at``/``frame``/``record_version``/``borrowed``/``refs``).
   The leaf does NO file I/O — the caller owns the document (§3).
7. **Apply** — :func:`trajectory_from_record` validates and
   reconstructs (model, fits); :func:`evaluate_record` computes
   f(t; p̂) at any epochs; :func:`apply_detrend` subtracts via
   :func:`gps_analysis.fitting.remove_trend`. Raw is never mutated and
   the view is exactly invertible, ``raw = detrended + f(t; p̂)``.
   Records are self-contained, so a **borrowed** record estimated at a
   nearby station applies cleanly to another station's epochs (§0.6,
   the ``UseSTA`` mechanism made explicit).
8. **Term selection** — :func:`select_terms` zeroes the non-selected
   coefficients (and their covariance rows/columns) so
   ``remove_trend(model, t, y, select_terms(model, fits, "periodic"))``
   subtracts only the seasonal part — the house models are linear in
   parameters, so coefficient zeroing *is* term removal (§5.3); no new
   evaluator exists.

Everything is pure, array-first, unit-agnostic ([L] is the caller's
length unit; mm in IMO production), numpy/scipy only, no file reads, no
config, inputs never mutated (leaf rules R2/R6). Formal WLS σ are
white-noise-optimistic for GNSS (Williams 2003, J. Geodesy 76) — the
record's estimator provenance (``sigma_kind`` at document level, design
§2.4) leaves room for a colored-noise upgrade without a schema change.
"""

import dataclasses
import inspect
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import models
from .baseline import slice_window, slice_windows
from .fitting import (
    ModelFunc,
    _components_2d,
    _n_model_params,
    _per_component_p0,
    _per_component_sigma,
    _resolve_linear_design,
    fit_components,
    remove_trend,
    with_steps,
)
from .models import FloatArray, TrajectoryParams
from .outliers import OutlierDetection, OutlierParams, detect_outliers

__all__ = [
    "DETREND_METHOD_PLAIN",
    "DETREND_METHOD_ROBUST",
    "RECORD_VERSION",
    "SUPPORTED_RECORD_VERSIONS",
    "DetrendEstimate",
    "apply_detrend",
    "estimate_detrend",
    "evaluate_record",
    "select_terms",
    "trajectory_from_record",
]

RECORD_VERSION = 1
"""Version this writer EMITS for the leaf's station-record shape.

Distinct from what the reader accepts (:data:`SUPPORTED_RECORD_VERSIONS`).
Keeping the two separate is what lets a future shape ship without a flag day
for the deployed documents: the writer advances, the reader keeps the older
branch, and both live in one file (design §3.2 rules)."""

RECORD_VERSION_TERMS = 2
"""Version emitted when the model carries a TERM SPEC — i.e. terms a model
code plus ``step_epochs`` cannot express (transients).

Content-determined, not writer-determined: an ordinary model still emits
:data:`RECORD_VERSION` 1, so the 37 deployed records and every new plain
record stay byte-identical. Only a record that NEEDS the richer shape gets
it, which is what keeps v1 readers correct rather than merely tolerated."""

SUPPORTED_RECORD_VERSIONS: frozenset[int] = frozenset(
    {RECORD_VERSION, RECORD_VERSION_TERMS}
)
"""Record versions :func:`trajectory_from_record` can reconstruct.

An exact-equality check would make every shape change a coordinated
migration of every stored document at once.  Membership makes it additive:
add the new version here together with its reconstruction branch, and old
records keep reading BY CONSTRUCTION rather than by argument."""

DETREND_METHOD_ROBUST = "step_augmented_robust"
"""``detrend_method`` provenance tag (design §0.2): outlier-robust
step-augmented fit — the outlier stage ran and defined the inlier set."""

DETREND_METHOD_PLAIN = "plain_wls"
"""``detrend_method`` provenance tag (design §0.2): plain WLS on all
windowed epochs (legacy semantics) — outlier stage off or aborted."""

_MODEL_NAMES: dict[str, ModelFunc] = {
    "linear": models.linear,
    "periodic": models.periodic,
    "lineperiodic": models.lineperiodic,
}
"""Model registry-code ↔ callable map for the serializable records.
Only these house models can appear in a stored record — a record must
be re-evaluable from its ``model`` string alone (design §3.2)."""

_SECULAR_PARAM_NAMES = frozenset({"offset", "rate", "curvature"})
"""Model parameter names of the secular (non-seasonal) group.

Not exhaustive on its own — :func:`_is_secular_param` adds the ``poly_``
prefix, since :class:`~gps_analysis.terms.Polynomial` names only the first
three coefficients and falls back to ``poly_3``, ``poly_4``, ... above that."""

_PERIODIC_PARAM_NAMES = frozenset(
    {"cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"}
)
"""Model parameter names of the seasonal group at the production
``n_harmonics=2`` — see :func:`_is_periodic_param` for the general rule."""

_SECULAR_PARAM_PREFIX = "poly_"
_PERIODIC_PARAM_PREFIXES = ("cos_", "sin_")


def _is_secular_param(name: str) -> bool:
    """Whether a parameter name belongs to the polynomial (secular) group."""
    return name in _SECULAR_PARAM_NAMES or name.startswith(_SECULAR_PARAM_PREFIX)


def _is_periodic_param(name: str) -> bool:
    """Whether a parameter name belongs to the seasonal group.

    A membership test against the four production names was not enough:
    :class:`~gps_analysis.terms.Seasonal` labels only harmonics 1 and 2
    (``annual``/``semiannual``) and spells the rest ``cos_harmonic3``,
    ``sin_harmonic3``, ... so ``n_harmonics=3`` produced parameters that
    NEITHER classifier could place, and both raised.  ``cos_``/``sin_`` is
    the seasonal namespace -- no other term generates those prefixes -- so
    the rule generalises without widening what it claims.

    Shared by :func:`_term_keep_mask` and
    :func:`gps_analysis.staged._staged_group_of` so the two, which
    deliberately differ on where STEPS go, cannot also drift on this.
    """
    return name in _PERIODIC_PARAM_NAMES or name.startswith(_PERIODIC_PARAM_PREFIXES)


_TRANSIENT_AMP_PREFIXES = ("log_amp", "exp_amp")
"""Parameter-name prefixes of the transient amplitudes (``terms.py``) —
their own group, NOT folded into ``"secular"``: a postseismic decay is
signal, where a Heaviside jump is background."""

_STEP_AMP_PREFIX = "step_amp_"
"""Parameter-name prefix of :func:`~gps_analysis.fitting.with_steps`
amplitude parameters — classified with the secular group (a Heaviside
jump is background, not seasonal)."""

_TERMS = ("all", "secular", "periodic", "transient")
"""Valid ``terms`` selectors of :func:`select_terms`.

``"transient"`` joined the vocabulary on 2026-08-03, when transient terms
became fittable. The three original selectors keep their EXACT meanings —
``"secular"`` is still {offset, rate} ∪ {step amplitudes} (design §5.3: a
Heaviside jump is background, not seasonal) and does **not** reach a
transient. A postseismic decay is signal many analyses exist to study, so
removing it silently under an existing spelling would change what the
detrended view means for every future record; you ask for it by name, or
with ``"all"``.

:func:`select_terms` also accepts a TUPLE of these names (their union), so
a call site can say ``("secular", "transient")`` and decide for itself.
All 37 deployed records are byte-identical under this change: they carry no
transient terms, so no mask they produce can differ."""

_DEFAULT_TOL = 1e-3
"""Window boundary tolerance [yr] — the ``slice_window`` legacy default
(0.001 yr ≈ 8.77 h, keeps a daily solution stamped exactly at a bound)."""

_DEFAULT_NAMES: tuple[str, ...] = ("north", "east", "up")
"""Default component labels — applied only when ``y`` has exactly three
component rows (the N/E/U production shape); other shapes get unlabeled
fits unless ``names`` is passed explicitly (identity-checked sentinel)."""


@dataclasses.dataclass(frozen=True)
class DetrendEstimate:
    """Stored-detrend estimation result for one station (all components).

    Result of :func:`estimate_detrend` (design §5.1, extended per the §0
    locked decisions with the method tag, frame passthrough and the
    model/step identity a self-contained record needs). Serialize with
    :meth:`to_record`; apply with :func:`apply_detrend`.

    Attributes:
        fits: Per-component :class:`~gps_analysis.models.TrajectoryParams`
            in row order of the input ``y`` — **absolute-t
            parameterization** (the closed-form path re-centers
            internally and maps back exactly), full vector including the
            intercept (design §0.1 #1) plus one step amplitude per step
            epoch when the model was step-augmented.
        inliers: Boolean mask over the *windowed* epochs, shape
            (C, N_window) — or (N_window,) for 1-D input — True where
            the sample entered the final fit (raw-preserving: the mask
            annotates, nothing is deleted).
        span_used: (first, last) windowed epoch that is an inlier in at
            least one component [yr] — extrapolation-staleness baseline
            (design §6 T2).
        n_epochs: Number of windowed epochs offered to the fit.
        n_rejected: Outliers removed per component.
        rms: Inlier residual RMS per component [L] — drift-staleness
            baseline (design §6 T3); NaN for a component with no inliers.
        window: Requested fit window (start, end) [yr] — the HULL when
            the fit spans several segments; open bounds are resolved to
            the first/last windowed epoch.
        segments: The resolved fit segments, shape (J, 2) [yr], each
            open bound collapsed to the first/last kept epoch OF THAT
            SEGMENT.  ``(window,)`` when J = 1, a deliberate redundancy
            that keeps consumers reading one field rather than branching.
        segment_gaps: The REALIZED excised gaps, length J−1 [yr] — the
            distance from the last kept epoch of segment j to the first
            kept epoch of segment j+1.  Reported rather than gated: a
            segment boundary placed inside a genuine data outage hides
            that outage from ``max_gap_years`` (which is per segment),
            and this is what makes "you excised 0.35 yr" and "you excised
            1.02 yr" distinguishable without adding a threshold.
        window_mask: The union mask over the INPUT ``t`` that the fit
            actually used, shape (N,).  Returned so a caller lifting
            results back to its own index space uses the mask the fit
            used instead of re-deriving one that merely ought to agree.
            Not serialized.
        model: Model registry code (``"lineperiodic"`` | ``"linear"`` |
            ``"periodic"``) — or the callable's ``__name__`` for a
            custom model, in which case :meth:`to_record` refuses (a
            record must be re-evaluable from the code alone).
        step_epochs: Step epochs t_k inside the window actually used to
            augment the model, shape (K,) [yr]; empty when none.
        detrend_method: Provenance tag (design §0.2) —
            :data:`DETREND_METHOD_ROBUST` when the outlier stage
            defined the inlier set, :data:`DETREND_METHOD_PLAIN` when
            it was off or aborted.
        frame: Opaque reference-frame tag passed through from the
            caller (§0.5 — the plate-removed processing frame; the leaf
            never interprets it), or None.
        outlier_abort: True when the outlier stage aborted
            (excess-candidate rule §3.5 of the outlier design) and the
            fit fell back to plain WLS on all epochs — surfaced loudly
            (a ``RuntimeWarning`` was emitted; §0.4 graceful degrade).
        detection: Full :class:`~gps_analysis.outliers.OutlierDetection`
            diagnostics of the outlier stage, or None when ``detect``
            was False.
    """

    fits: tuple[TrajectoryParams, ...]
    inliers: NDArray[np.bool_]
    span_used: tuple[float, float]
    n_epochs: int
    n_rejected: tuple[int, ...]
    rms: tuple[float, ...]
    window: tuple[float, float]
    segments: tuple[tuple[float, float], ...]
    segment_gaps: tuple[float, ...]
    window_mask: NDArray[np.bool_]
    model: str
    step_epochs: FloatArray
    detrend_method: str
    frame: str | None = None
    outlier_abort: bool = False
    detection: OutlierDetection | None = None
    term_spec: list[dict[str, Any]] | None = None
    """Term spec when the model carries terms a registry code cannot express
    (transients). None for ordinary models, which keeps their records at
    :data:`RECORD_VERSION` 1 and byte-identical."""

    def __post_init__(self) -> None:
        n_components = len(self.fits)
        if n_components == 0:
            raise ValueError("fits must contain at least one component")
        for name in ("n_rejected", "rms"):
            if len(getattr(self, name)) != n_components:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries for "
                    f"{n_components} components"
                )
        object.__setattr__(
            self, "step_epochs", np.asarray(self.step_epochs, dtype=np.float64)
        )

    def to_record(
        self,
        *,
        fitted_at: str | None = None,
        borrowed: Mapping[str, Any] | None = None,
        refs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize the estimate to a JSON-ready station record.

        The record is **self-contained**: model code + fixed step epochs
        + full per-component parameter vectors (intercept included) +
        upper-triangle covariances + provenance — everything
        :func:`apply_detrend` needs to evaluate the stored trajectory at
        ANY epoch, on ANY station's series (borrowing, design §0.6/§2.6).
        Shape (leaf-owned; the multi-station document of design §3.2 is
        the caller's):

            ``{"record_version", "model", "param_names", "step_epochs",``
            `` "frame", "detrend_method", "fitted_at", "window",``
            `` "span_used", "n_epochs", "n_rejected", "rms",``
            `` "borrowed", "refs", "components": [per-component``
            :meth:`~gps_analysis.models.TrajectoryParams.to_record`]}``

        Symbols → args:
            - ``fitted_at`` → ``fitted_at``: estimation timestamp as an
              opaque caller-supplied string (ISO 8601 recommended) — the
              leaf is pure and reads no clock; None marks "unknown"
              (staleness trigger T1 fires immediately, design §6.1).
            - ``borrowed`` → ``borrowed``: borrowing provenance slot,
              e.g. ``{"from": "DYNG", "terms": "periodic",
              "donor_fitted_at": …}`` (§0.6); stored verbatim, None when
              the record is the station's own fit.
            - ``refs`` → ``refs``: free-form provenance passthrough
              (source series, software versions, …); stored verbatim.

        Returns:
            Plain dict of JSON-native values; floats keep full ``repr``
            precision so store → load → apply is bit-identical to
            fit → apply.

        Raises:
            ValueError: If the estimate's model is not a registry code
                (a custom callable cannot be re-resolved from a record).

        Reference:
            Design spec ``docs/DESIGN_live_detrending.md`` §3.2/§5.2 and
            MATH_STANDARDS §6 (derived-product provenance).

        Numerical notes:
            Pure — no I/O, no clock; see
            :meth:`~gps_analysis.models.TrajectoryParams.to_record` for
            the covariance round-trip contract.
        """
        term_spec = self.term_spec
        if term_spec is None and self.model not in _MODEL_NAMES:
            raise ValueError(
                f"model {self.model!r} is not a registry code "
                f"{sorted(_MODEL_NAMES)} - the record would not be re-evaluable"
            )
        if term_spec is not None:
            from .terms import TrajectoryModel

            # Steps are appended here for the same reason as on the v1 branch:
            # the spec describes the UNAUGMENTED model, and with_steps supplies
            # the rest. This mirrors trajectory_from_record exactly -- writer
            # and reader must agree on the order or every v2 record fails its
            # own param_names check. Pre-2026-08-09 records baked steps into
            # the spec and carry an empty step_epochs, so they append nothing
            # and keep the names they were written with.
            param_names = list(TrajectoryModel.from_spec(term_spec).param_names) + [
                f"{_STEP_AMP_PREFIX}{k + 1}" for k in range(self.step_epochs.size)
            ]
        else:
            param_names = _param_names(_MODEL_NAMES[self.model]) + [
                f"{_STEP_AMP_PREFIX}{k + 1}" for k in range(self.step_epochs.size)
            ]
        return {
            "record_version": (
                RECORD_VERSION if term_spec is None else RECORD_VERSION_TERMS
            ),
            "model": self.model,
            "param_names": param_names,
            # The terms list is what makes a transient record re-evaluable;
            # absent at v1, so a v1 reader never sees a key it must ignore.
            **({} if term_spec is None else {"terms": term_spec}),
            "step_epochs": [float(v) for v in self.step_epochs],
            "frame": self.frame,
            "detrend_method": self.detrend_method,
            "fitted_at": fitted_at,
            "window": [float(self.window[0]), float(self.window[1])],
            # Additive at RECORD_VERSION 1, deliberately. `window` keeps its
            # 2-tuple type and hull meaning because two readers index it
            # positionally (geo_dataread's StationResult detail line formats
            # window[0] with :.3f -> TypeError on a nested list; the workbench's
            # _add_window_edges guards len(window) != 2 and would silently stop
            # drawing). A NEW key cannot break either: trajectory_from_record
            # reads only record_version/model/step_epochs/param_names/components,
            # and TrajectoryParams.from_record ignores unknown keys -- so a
            # segmented record and the 37 deployed single-window ones coexist in
            # one document at the same version. Readers use .get("segments");
            # None means "legacy, the hull in `window` is the whole story".
            "segments": [[float(a), float(b)] for a, b in self.segments],
            "segment_gaps": [float(g) for g in self.segment_gaps],
            "span_used": [float(self.span_used[0]), float(self.span_used[1])],
            "n_epochs": int(self.n_epochs),
            "n_rejected": [int(v) for v in self.n_rejected],
            "rms": [float(v) for v in self.rms],
            "borrowed": None if borrowed is None else dict(borrowed),
            "refs": None if refs is None else dict(refs),
            "components": [fit.to_record() for fit in self.fits],
        }


def _param_names(model: ModelFunc) -> list[str]:
    """Positional parameter names of ``model(t, *params)`` (after t)."""
    return list(inspect.signature(model).parameters)[1:]


def _validate_segments(segs: Sequence[tuple[float | None, float | None]]) -> None:
    """Reject a segment sequence that cannot mean one fit domain.

    Three conditions, each preventing a distinct silent-wrong-science mode:

    - **interior bounds must be finite.**  ``[(2002, None), (2008.7, 2019)]``
      parses as an overlapping union, so the excision it was written to
      express does not exist and the coverage below is meaningless.
    - **strictly increasing and non-overlapping.**  Overlap makes the
      summed coverage of ``min_span_years`` DOUBLE-COUNT, i.e. gameable:
      ``[(2002, 2012), (2010, 2019)]`` would buy two years of span that no
      data supports.  This is why the check is here and not a matter of
      taste.
    - **each end after its start** — also caught by
      :func:`~gps_analysis.baseline.slice_windows`, deliberately duplicated
      at this trust boundary so the message names the caller's argument.
    """
    for j, (a, b) in enumerate(segs):
        if a is not None and b is not None and b <= a:
            raise ValueError(f"segment {j} has end {b} <= start {a}")
        if j > 0 and a is None:
            raise ValueError(
                f"segment {j} has an open start; only the FIRST segment may "
                f"be open on the left, else the segments overlap"
            )
        if j < len(segs) - 1 and b is None:
            raise ValueError(
                f"segment {j} has an open end; only the LAST segment may be "
                f"open on the right, else the segments overlap"
            )
        if j > 0:
            prev_end = segs[j - 1][1]
            if prev_end is not None and a is not None and a <= prev_end:
                raise ValueError(
                    f"segment {j} starts at {a} which is not after segment "
                    f"{j - 1}'s end {prev_end}; segments must be strictly "
                    f"increasing and non-overlapping"
                )


def _resolve_model(model: str | ModelFunc) -> tuple[ModelFunc, str]:
    """Resolve a model spec to ``(callable, registry code or __name__)``."""
    if callable(model):
        for name, func in _MODEL_NAMES.items():
            if func is model:
                return model, name
        return model, str(getattr(model, "__name__", "custom"))
    try:
        return _MODEL_NAMES[model], model
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; named models: {sorted(_MODEL_NAMES)}"
        ) from None


def _model_term_spec(model: ModelFunc) -> list[dict[str, Any]] | None:
    """Term spec of the fitted model, or None for an ordinary registry one."""
    from .terms import model_term_spec

    return model_term_spec(model)


def _term_keep_mask(model: ModelFunc, terms: str) -> NDArray[np.bool_]:
    """Build the parameter keep-mask of a term selection (design §5.3).

    Equation:
        ``keep_j = 1`` iff parameter j belongs to the selected group —
        secular = {offset, rate} ∪ {step amplitudes}, periodic =
        {cos/sin annual, semiannual} — classified by the model's
        positional parameter names.

    Raises:
        ValueError: For an unregistered model (no closed-form linear
            design — coefficient zeroing is only term removal for
            models linear in their parameters) or an unclassifiable
            parameter name.
    """
    if _resolve_linear_design(model) is None:
        raise ValueError(
            "select_terms is defined only for the registered "
            "linear-in-parameters models (and their with_steps "
            f"augmentations); got {getattr(model, '__name__', model)!r}"
        )
    wanted = {terms} if isinstance(terms, str) else set(terms)
    names = _param_names(model)
    keep = np.zeros(len(names), dtype=np.bool_)
    for j, name in enumerate(names):
        if _is_secular_param(name) or name.startswith(_STEP_AMP_PREFIX):
            group = "secular"
        elif _is_periodic_param(name):
            group = "periodic"
        elif any(name.startswith(pre) for pre in _TRANSIENT_AMP_PREFIXES):
            group = "transient"
        else:
            raise ValueError(
                f"cannot classify model parameter {name!r} as "
                f"secular/periodic/transient"
            )
        keep[j] = group in wanted
    return keep


def select_terms(
    model: ModelFunc,
    fits: TrajectoryParams | Sequence[TrajectoryParams],
    terms: str,
) -> list[TrajectoryParams]:
    """Zero the non-selected coefficients of a house-model fit.

    Equation (per component; keep-mask k from the parameter names):
        ``p̃_j = k_j·p̂_j``,   ``C̃_ij = k_i·k_j·Ĉ_ij``

    Because the house models are **linear in their parameters**,
    f(t; p) = A(t)·p, zeroing coefficients *is* term removal:
    ``f(t; p̃) = A(t)·p̃`` evaluates exactly the selected terms, and
    ``remove_trend(model, t, y, select_terms(model, fits, "periodic"))``
    subtracts only the seasonal part — no new evaluator (design §5.3).
    The zeroed covariance C̃ is the exact GUM covariance of the
    subtracted quantity A·p̃ (JCGM 100:2008 §5.1.2 with the selection
    matrix diag(k)).

    Symbols → args:
        - ``f, A`` → ``model``: a registered linear-in-parameters model
          (:func:`~gps_analysis.models.linear` /
          :func:`~gps_analysis.models.periodic` /
          :func:`~gps_analysis.models.lineperiodic`) or a
          :func:`~gps_analysis.fitting.with_steps` augmentation of one
        - ``p̂, Ĉ`` → ``fits``: fitted parameters per component [units
          per model]
        - ``k`` → ``terms``: ``"all"`` (identity) | ``"secular"``
          (offset + rate **+ step amplitudes** — a Heaviside jump is
          background, not seasonal) | ``"periodic"`` (the four trig
          coefficients)

    Args:
        model: Model callable the fits belong to.
        fits: One :class:`~gps_analysis.models.TrajectoryParams` or a
            sequence of them.
        terms: Term selector — ``"all"``, ``"secular"``, ``"periodic"``,
            ``"transient"``, or a tuple of those (their union).
            ``"secular"`` does NOT include a transient; see :data:`_TERMS`.

    Returns:
        New :class:`~gps_analysis.models.TrajectoryParams` list (inputs
        untouched), same order.

    Raises:
        ValueError: For an unknown ``terms`` value, an unregistered
            model callable, or a parameter-count mismatch.

    Reference:
        Design spec ``docs/DESIGN_live_detrending.md`` §5.3; the
        ``"secular"`` group keeps the intercept with the rate so the
        line-only view matches the legacy ``detrend_line`` behavior of
        subtracting ``p[0:2]``.

    Numerical notes:
        Exact — multiplication by 0/1 only. Covariance rows/columns of
        zeroed parameters are zeroed too (including ``inf`` entries of
        a could-not-estimate covariance: 0·inf would be NaN, so the
        mask is applied by assignment, not multiplication).
    """
    wanted = (terms,) if isinstance(terms, str) else tuple(terms)
    unknown = [t for t in wanted if t not in _TERMS]
    if unknown:
        raise ValueError(f"terms must be one of {_TERMS}, got {terms!r}")
    fit_list = [fits] if isinstance(fits, TrajectoryParams) else list(fits)
    if "all" in wanted:
        return fit_list
    keep = _term_keep_mask(model, terms)
    out: list[TrajectoryParams] = []
    for fit in fit_list:
        if fit.params.size != keep.size:
            raise ValueError(
                f"fit has {fit.params.size} parameters for a "
                f"{keep.size}-parameter model"
            )
        params = np.where(keep, fit.params, 0.0)
        covariance = fit.covariance.copy()
        covariance[~keep, :] = 0.0
        covariance[:, ~keep] = 0.0
        out.append(
            TrajectoryParams(
                params=params, covariance=covariance, component=fit.component
            )
        )
    return out


def estimate_detrend(
    model: str | ModelFunc,
    t: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    window: tuple[float | None, float | None] = (None, None),
    segments: Sequence[tuple[float | None, float | None]] | None = None,
    step_epochs: ArrayLike | None = None,
    min_span_years: float = 2.0,
    min_epochs: int = 365,
    max_gap_years: float = 0.5,
    detect: bool = True,
    outlier_params: OutlierParams | None = None,
    protect_windows: Sequence[tuple[float, float]] = (),
    min_outlier: ArrayLike | None = None,
    p0: ArrayLike | None = None,
    absolute_sigma: bool = False,
    names: Sequence[str] | None = _DEFAULT_NAMES,
    frame: str | None = None,
    tol: float = _DEFAULT_TOL,
) -> DetrendEstimate:
    """Estimate stored-detrend trajectory parameters for one station.

    Thin estimation orchestrator (MATH_STANDARDS §1, no new math) of the
    module derivation chain — per component c over the windowed inliers:

        ``p̂_c = argmin_p Σᵢ [ (y_cᵢ − f(tᵢ; p)) / σ_cᵢ ]²``,
        ``C_p̂ = (AᵀWA)⁻¹·s²``,  ``W = diag(1/σ²)``,
        ``f(t; p, a) = f_traj(t; p) + Σ_k a_k·H(t − t_k)``

    composed as :func:`~gps_analysis.baseline.slice_window` → validity
    gates → :func:`~gps_analysis.fitting.with_steps` (steps inside the
    window) → :func:`~gps_analysis.outliers.detect_outliers` (**outliers
    removed before the fit** — model-aware and signal-protecting, so
    transients/steps are not eaten into the background; BGÓ hard rule,
    design §0.2) → final clean WLS
    (:func:`~gps_analysis.fitting.fit_components`, closed form for the
    house models, reduced-χ² covariance unless ``absolute_sigma``).
    Mirrors the :func:`gps_analysis.velocity.estimate_velocity`
    precedent.

    Symbols → args:
        - ``tᵢ`` → ``t``: epochs, fractional years (``yearf``), sorted
          ascending [yr]
        - ``y_cᵢ`` → ``y``: observations, (N,) or (C, N), in the
          caller's (plate-removed, §0.5) processing frame [L]
        - ``σ_cᵢ`` → ``sigma``: formal 1-σ uncertainties, shape of ``y``
          [L]; None ⇒ unweighted
        - ``f_traj`` → ``model``: ``"lineperiodic"`` (production
          default) | ``"linear"`` | ``"periodic"`` | a compatible
          callable ``f(t, *p)``
        - ``[t_start, t_end]`` → ``window``/``tol``: requested fit
          window [yr] ± δ (:func:`~gps_analysis.baseline.slice_window`);
          open bounds use the data span — per §0.7 the window should be
          as long as the step-free/pre-unrest history allows
        - ``t_k`` → ``step_epochs``: known step epochs [yr] (caller's
          per-station table — TOS equipment changes, ``steps.csv``
          coseismic offsets; the leaf never reads config); only epochs
          strictly inside the window augment the model — earlier steps
          are absorbed by the intercept, later ones cannot be estimated
        - gates → ``min_span_years`` / ``min_epochs`` /
          ``max_gap_years``: design §2.2 rule 3 — span ≥ min (default
          2.0 yr per §0.7's "min 1–2 yr" floor; note Blewitt & Lavallée
          2002 show rate bias grows quickly below ~2.5 yr, the design
          §2.2 recommended policy default), epoch count ≥ min (365),
          largest gap ≤ max (0.5 yr, keeps the seasonal terms
          constrained around the year)
        - thresholds → ``outlier_params`` / ``protect_windows`` /
          ``min_outlier``: forwarded to
          :func:`~gps_analysis.outliers.detect_outliers` (None ⇒ spec
          defaults)

    Args:
        model: Trajectory model (name or callable, see above).
        t: Epochs, shape (N,) [yr]; finite, sorted ascending.
        y: Observations, shape (N,) or (C, N) [L]; finite; NEVER
            mutated.
        sigma: 1-σ uncertainties, shape of ``y`` [L]; optional.
        window: Requested fit window (start, end) [yr]; either bound
            may be None (open).  Sugar for ``segments=[(start, end)]``.
        segments: Fit the union of these ``(start, end)`` intervals [yr]
            instead of one window — the way to excise a post-seismic
            transient while keeping the flanks on both sides, which is
            what makes the coseismic step between them estimable.  Only
            the FIRST start and the LAST end may be None; the intervals
            must be strictly increasing and non-overlapping (overlap
            would double-count ``min_span_years``).  Mutually exclusive
            with a non-default ``window``.

            **Not to be confused with ``protect_windows``**, which has
            the same type and the opposite polarity: ``segments`` says
            *fit only here*, ``protect_windows`` says *do not FLAG here*.
        step_epochs: Known step epochs [yr]; None/empty ⇒ plain model.
        min_span_years: Span gate [yr] — the SUMMED coverage of the
            segments, not their hull (identical at J = 1, and uniformly
            stricter beyond it: two 18-day nubs 17 yr apart clear a
            2-yr hull while four seasonal terms fit 36 days of data).
        min_epochs: Total kept-epoch gate across all segments.
        max_gap_years: Largest-gap gate [yr], applied WITHIN each
            segment.  The excision between segments is deliberate and
            documented by the configuration, so it is not a gap in this
            sense; keeping the gate per segment preserves its actual
            meaning — no undocumented outage inside an interval the
            caller claimed to fit.
        detect: Run the outlier stage (True, production default). False
            = plain WLS on all windowed epochs (legacy semantics,
            tagged :data:`DETREND_METHOD_PLAIN`).
        outlier_params: :class:`~gps_analysis.outliers.OutlierParams`
            thresholds; None ⇒ defaults.
        protect_windows: Intervals [yr] where flagging is disabled.
        min_outlier: Outlier magnitude floor(s) [L], scalar or (C,).
        p0: Initial guess for the BASE model parameters, (P,) or
            (C, P); step amplitudes are seeded at 0. Only nonlinear
            custom models need it.
        absolute_sigma: If True, skip the reduced-χ² covariance
            rescaling in the final WLS (σ trusted as absolute).
        names: Per-component labels; the default ``("north", "east",
            "up")`` applies only to 3-component input (other shapes get
            None unless passed explicitly).
        frame: Opaque reference-frame tag stored on the result (§0.5);
            the leaf never interprets it.
        tol: Window boundary tolerance δ [yr].

    Returns:
        :class:`DetrendEstimate` — fits (+ covariances), inlier mask,
        window/span/RMS diagnostics, method tag, frame passthrough and
        the full outlier diagnostics.

    Raises:
        ValueError: On shape mismatches, non-finite/unsorted ``t``,
            non-finite ``y``, an unknown model name — or a failed
            validity gate, which names the gate
            (``min_span_years`` / ``min_epochs`` / ``max_gap_years``;
            hard errors by design: a silently bad background is worse
            than none, §2.2).

    Warns:
        RuntimeWarning: When the outlier stage aborts on the
            excess-candidate rule; the estimate falls back to plain WLS
            on all epochs with ``outlier_abort=True`` and the
            :data:`DETREND_METHOD_PLAIN` tag (loud graceful degrade,
            §0.4 — the caller decides whether to store such a record).

    Reference:
        Design spec ``docs/DESIGN_live_detrending.md`` §0/§2/§5.1;
        trajectory model: Bevis & Brown 2014, J. Geodesy 88, eq. (1);
        window floor / seasonal-rate aliasing: Blewitt & Lavallée 2002,
        JGR 107(B7); WLS covariance: Aitken 1936 / JCGM 100:2008 §5.1.2;
        formal-σ honesty caveat: Williams 2003, J. Geodesy 76.

    Numerical notes:
        The closed-form WLS path re-centers the trend column internally
        and maps back exactly, so the returned parameters are in the
        model's **absolute-t** parameterization — the phase convention
        (trig arguments 2πt, 4πt on absolute ``yearf``) matches
        :func:`gps_analysis.models.periodic` and the legacy CSV
        coefficients (migration contract, design §7.1-7). ``rms`` is
        the unwhitened inlier residual RMS [L]. ``span_used`` spans
        epochs kept in at least one component (per-component masks may
        differ under the default ``epoch_policy="per_component"``).
    """
    model_func, model_name = _resolve_model(model)

    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must be finite")
    if tt.size > 1 and np.any(np.diff(tt) < 0.0):
        raise ValueError("t must be sorted ascending")
    yy, was_1d = _components_2d(y, "y")
    if yy.shape[1] != tt.size:
        raise ValueError(
            f"t must be 1-D with y.shape[-1] = {yy.shape[1]}, got shape {tt.shape}"
        )
    if not np.all(np.isfinite(yy)):
        raise ValueError("y must be finite (no NaN/inf)")
    n_components = yy.shape[0]
    sigma_rows = _per_component_sigma(sigma, yy, was_1d)
    p0_rows = _per_component_p0(p0, n_components, was_1d)
    if names is _DEFAULT_NAMES and n_components != 3:
        names = None
    if names is not None and len(names) != n_components:
        raise ValueError(
            f"names has {len(names)} entries for {n_components} components"
        )

    # --- segments + validity gates (design §2.2 rule 3; hard errors) ---
    if segments is not None and window != (None, None):
        raise ValueError(
            "pass either window= or segments=, not both; two sources of truth "
            "for the fitted sample set is a silent-wrong-science hazard"
        )
    segs: tuple[tuple[float | None, float | None], ...] = (
        (tuple(window),) if segments is None else tuple(tuple(s) for s in segments)  # type: ignore[assignment, misc]
    )
    _validate_segments(segs)

    mask = slice_windows(tt, segs, tol=tol)
    n_epochs = int(np.count_nonzero(mask))
    if n_epochs == 0:
        raise ValueError(
            f"fit window {segs if len(segs) > 1 else window} contains no epochs"
        )
    t_win = tt[mask]

    # Per-segment accounting. Each segment's own mask comes from slice_window --
    # the same function slice_windows just OR-ed -- so the parts and the union
    # cannot disagree about a boundary epoch.
    seg_masks = [slice_window(tt, a, b, tol=tol) for a, b in segs]
    seg_epochs = [tt[m] for m in seg_masks]
    for j, ts in enumerate(seg_epochs):
        if ts.size == 0:
            raise ValueError(
                f"segment {j} {segs[j]} contains no epochs — it is a data gap "
                f"or a mistyped bound, not a fit window"
            )

    # min_span_years on SUMMED coverage, not the hull: Σ_j ≤ hull always, so
    # this rejects everything the hull rejects plus what it wrongly admits.
    span = float(sum(float(ts[-1] - ts[0]) for ts in seg_epochs))
    if span < min_span_years:
        raise ValueError(
            f"validity gate 'min_span_years' failed: "
            f"{'summed segment coverage' if len(segs) > 1 else 'window span'} "
            f"{span:.4f} yr < {min_span_years} yr"
        )
    if n_epochs < min_epochs:
        raise ValueError(
            f"validity gate 'min_epochs' failed: window has {n_epochs} "
            f"epochs < {min_epochs}"
        )
    # max_gap_years WITHIN each segment. On the hull the deliberate excision
    # would be the largest diff and reject every union; and the obvious
    # workaround -- raising the threshold past it -- would simultaneously
    # disable the gate inside every segment, so a genuine outage would sail
    # through unremarked. Per segment the gate keeps meaning what it says.
    for j, ts in enumerate(seg_epochs):
        if ts.size > 1:
            largest_gap = float(np.max(np.diff(ts)))
            if largest_gap > max_gap_years:
                where = f" in segment {j} {segs[j]}" if len(segs) > 1 else ""
                raise ValueError(
                    f"validity gate 'max_gap_years' failed: largest gap "
                    f"{largest_gap:.4f} yr > {max_gap_years} yr{where}"
                )
    # REALIZED excised gaps -- reported, never gated. A boundary placed inside a
    # real outage hides that outage from the per-segment gate above; surfacing
    # the realized distance is the honest answer, and it adds no knob.
    segment_gaps = tuple(
        float(seg_epochs[j + 1][0] - seg_epochs[j][-1]) for j in range(len(segs) - 1)
    )
    y_win = yy[:, mask]
    sigma_win = [None if s is None else s[mask] for s in sigma_rows]

    # --- step augmentation: only steps strictly inside the window ---
    # The bounds are the HULL of the kept epochs, and under segments that is
    # LOAD-BEARING rather than incidental: a coseismic epoch sitting inside an
    # excised transient still passes this filter, and its amplitude is estimable
    # precisely because the flanking segments constrain the level on both sides.
    # Narrowing this to per-segment membership would delete exactly the offset
    # the excision was performed to expose.
    if step_epochs is not None:
        all_steps = np.sort(np.asarray(step_epochs, dtype=np.float64).ravel())
        steps_in = all_steps[(all_steps > t_win[0]) & (all_steps <= t_win[-1])]
    else:
        steps_in = np.empty(0, dtype=np.float64)
    # Degeneracy (MATH_STANDARDS §3): two retained steps with NO kept epoch
    # between them build identical Heaviside columns over the fitted samples --
    # 0 on every earlier epoch, 1 on every later one -- so the design matrix is
    # rank-deficient and the amplitudes are individually meaningless. Reachable
    # today for two steps inside one outage; segments make it ordinary.
    if steps_in.size > 1:
        edges = np.searchsorted(t_win, steps_in, side="right")
        collide = np.flatnonzero(np.diff(edges) == 0)
        if collide.size:
            k = int(collide[0])
            raise ValueError(
                f"steps {steps_in[k]:.5f} and {steps_in[k + 1]:.5f} have no "
                f"fitted epoch between them, so their step columns are "
                f"identical and their amplitudes are not separable; declare "
                f"one of them, or widen the segments so data separates them"
            )
    fit_model = with_steps(model_func, steps_in) if steps_in.size else model_func
    n_steps = int(steps_in.size)
    guesses: list[FloatArray | None] = [
        None if g is None else np.concatenate([g, np.zeros(n_steps)]) for g in p0_rows
    ]

    # --- outlier removal BEFORE the fit (design §0.2 hard rule) ---
    detection: OutlierDetection | None = None
    outlier_abort = False
    if detect:
        sigma_2d = (
            None
            if sigma is None
            else np.vstack([s for s in sigma_win if s is not None])
        )
        p0_2d = None if p0 is None else np.vstack([g for g in p0_rows if g is not None])
        detection = detect_outliers(
            model_func,
            t_win,
            y_win,
            sigma_2d,
            step_epochs=steps_in if n_steps else None,
            protect_windows=protect_windows,
            min_outlier=min_outlier,
            p0=p0_2d,
            params=outlier_params,
            names=names,
        )
        if detection.excess_flag_abort:
            outlier_abort = True
            inliers = np.ones(y_win.shape, dtype=np.bool_)
            method = DETREND_METHOD_PLAIN
            warnings.warn(
                "outlier stage aborted (excess-candidate rule) - "
                "falling back to plain WLS on all windowed epochs; "
                "review before storing these parameters",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            inliers = ~np.atleast_2d(detection.flags)
            method = DETREND_METHOD_ROBUST
    else:
        inliers = np.ones(y_win.shape, dtype=np.bool_)
        method = DETREND_METHOD_PLAIN

    # --- final clean WLS on the inliers (Gauss-Markov covariance) ---
    fits: list[TrajectoryParams] = []
    rms: list[float] = []
    for c in range(n_components):
        keep = inliers[c]
        ss = sigma_win[c]
        fit = fit_components(
            fit_model,
            t_win[keep],
            y_win[c][keep],
            sigma=None if ss is None else ss[keep],
            p0=guesses[c],
            absolute_sigma=absolute_sigma,
            names=None if names is None else [names[c]],
        )[0]
        fits.append(fit)
        if bool(np.any(keep)):
            residual = y_win[c][keep] - np.asarray(
                fit_model(t_win[keep], *fit.params), dtype=np.float64
            )
            rms.append(float(np.sqrt(np.mean(residual**2))))
        else:
            rms.append(float("nan"))

    used_any = np.any(inliers, axis=0)
    t_used = t_win[used_any]
    span_used = (float(t_used[0]), float(t_used[-1]))
    # Each segment's open bounds collapse to ITS OWN first/last kept epoch; the
    # window is then the hull of the resolved segments, so J = 1 reproduces the
    # previous expression exactly.
    resolved_segments = tuple(
        (
            float(ts[0]) if a is None else float(a),
            float(ts[-1]) if b is None else float(b),
        )
        for (a, b), ts in zip(segs, seg_epochs, strict=True)
    )
    resolved_window = (resolved_segments[0][0], resolved_segments[-1][1])

    return DetrendEstimate(
        fits=tuple(fits),
        inliers=inliers[0] if was_1d else inliers,
        span_used=span_used,
        n_epochs=n_epochs,
        n_rejected=tuple(
            int(np.count_nonzero(~inliers[c])) for c in range(n_components)
        ),
        rms=tuple(rms),
        window=resolved_window,
        segments=resolved_segments,
        segment_gaps=segment_gaps,
        window_mask=mask,
        model=model_name,
        step_epochs=steps_in,
        detrend_method=method,
        frame=frame,
        outlier_abort=outlier_abort,
        term_spec=_model_term_spec(model_func),
        detection=detection,
    )


def trajectory_from_record(
    record: Mapping[str, Any],
) -> tuple[ModelFunc, list[TrajectoryParams]]:
    """Reconstruct (model, fits) from a stored station record.

    Validation + reconstruction (design §3.2 rules — a reader must
    raise, never fudge): known ``record_version``, model code in the
    registry, ``param_names`` (when present) verified against the
    model's positional signature, per-component parameter vectors of
    the right length with finite values
    (:meth:`~gps_analysis.models.TrajectoryParams.from_record`). Step
    epochs re-augment the model via
    :func:`~gps_analysis.fitting.with_steps`, so the returned callable
    evaluates the FULL stored trajectory
    ``f(t; p, a) = f_traj(t; p) + Σ_k a_k·H(t − t_k)`` exactly as
    fitted. The record is self-contained — no cross-record lookups
    (design §2.6: the apply path never chases donor references).

    Args:
        record: A :meth:`DetrendEstimate.to_record` dict (possibly
            hand-pinned: operator-edited parameters load unchanged, §0.7).

    Returns:
        ``(model, fits)`` — the (step-augmented) model callable and the
        per-component :class:`~gps_analysis.models.TrajectoryParams`,
        ready for :func:`~gps_analysis.fitting.remove_trend`.

    Raises:
        ValueError: On an unknown ``record_version``, an unregistered
            model code, a ``param_names`` mismatch, no components, a
            parameter-count mismatch, or non-finite parameters.

    Reference:
        Design spec ``docs/DESIGN_live_detrending.md`` §3.2/§4.1/§5.2.

    Numerical notes:
        Pure reconstruction — parameters pass through bit-exactly; no
        re-fitting, no renormalization.
    """
    version = record.get("record_version")
    if version not in SUPPORTED_RECORD_VERSIONS:
        raise ValueError(
            f"unknown record_version {version!r}; this reader supports "
            f"{sorted(SUPPORTED_RECORD_VERSIONS)}"
        )
    model_name = record.get("model")
    term_spec = record.get("terms")
    step_epochs = np.asarray(record.get("step_epochs", []), dtype=np.float64)

    if term_spec is not None:
        # v2: the record carries its own terms, because a registry code alone
        # cannot express a transient. Steps are NOT among them -- they live in
        # step_epochs and re-augment here exactly as in v1, because the writer
        # stores the UNAUGMENTED model's spec (`_model_term_spec(model_func)`)
        # beside `step_epochs=steps_in`. One augmentation site, one screen.
        #
        # Records written before 2026-08-09 baked their steps INTO the spec
        # and carry an empty step_epochs, so they reconstruct unchanged --
        # with_steps only applies when there is something to apply.
        from .terms import TrajectoryModel

        traj = TrajectoryModel.from_spec(term_spec)
        base_traj = traj.as_modelfunc()
        model_func = (
            with_steps(base_traj, step_epochs) if step_epochs.size else base_traj
        )
        expected_names = list(traj.param_names) + [
            f"{_STEP_AMP_PREFIX}{k + 1}" for k in range(step_epochs.size)
        ]
    else:
        # v1, byte-identical: registry code + with_steps, as before.
        if not isinstance(model_name, str) or model_name not in _MODEL_NAMES:
            raise ValueError(
                f"record model {model_name!r} is not a registry code "
                f"{sorted(_MODEL_NAMES)}"
            )
        base_model = _MODEL_NAMES[model_name]
        model_func = (
            with_steps(base_model, step_epochs) if step_epochs.size else base_model
        )
        expected_names = _param_names(base_model) + [
            f"{_STEP_AMP_PREFIX}{k + 1}" for k in range(step_epochs.size)
        ]
    stored_names = record.get("param_names")
    if stored_names is not None and list(stored_names) != expected_names:
        raise ValueError(
            f"record param_names {list(stored_names)} do not match the "
            f"model's positional order {expected_names}"
        )
    components = record.get("components")
    if not components:
        raise ValueError("record has no components")
    n_params = _n_model_params(model_func)
    fits: list[TrajectoryParams] = []
    for entry in components:
        fit = TrajectoryParams.from_record(entry)
        if fit.params.size != n_params:
            raise ValueError(
                f"component {fit.component!r} has {fit.params.size} parameters "
                f"for the {n_params}-parameter model {model_name!r} "
                f"(+{step_epochs.size} steps)"
            )
        fits.append(fit)
    return model_func, fits


def evaluate_record(
    record: Mapping[str, Any], t: ArrayLike, *, terms: str = "all"
) -> FloatArray:
    """Evaluate a stored trajectory record at arbitrary epochs.

    Equation (per stored component c):
        ``x_c(t) = f(t; p̂_c)``  — with ``f`` the record's
        (step-augmented) model and p̂ its stored parameters; for
        ``terms ≠ "all"`` the non-selected coefficients are zeroed
        first (:func:`select_terms`).

    Valid for ANY epochs — historical, today's, tomorrow's:
    extrapolation beyond the fit window is *by design* (that is what
    live detrending is; freshness is the caller's staleness policy,
    design §6, never a silent leaf concern).

    Symbols → args:
        - ``p̂`` → ``record``: stored station record
          (:meth:`DetrendEstimate.to_record` shape)
        - ``t`` → ``t``: evaluation epochs, fractional years [yr]
        - selection → ``terms``: ``"all"`` | ``"secular"`` |
          ``"periodic"`` (:func:`select_terms`)

    Returns:
        Model values, shape (C, N), float64 [L] — component rows in
        record order (index ``[0]`` for a single-component record).

    Raises:
        ValueError: Propagated from :func:`trajectory_from_record` /
            :func:`select_terms` validation.

    Reference:
        Design spec ``docs/DESIGN_live_detrending.md`` §4.1.

    Numerical notes:
        Pure evaluation — no state, no window: new epochs are simply
        new arguments.
    """
    model_func, fits = trajectory_from_record(record)
    if terms != "all":
        fits = select_terms(model_func, fits, terms)
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return np.stack(
        [np.asarray(model_func(tt, *fit.params), dtype=np.float64) for fit in fits]
    )


def apply_detrend(
    record: Mapping[str, Any],
    t: ArrayLike,
    y: ArrayLike,
    *,
    terms: str = "all",
    frame: str | None = None,
) -> FloatArray:
    """Subtract a stored trajectory record from observations (pure view).

    Equation (per component c):
        ``detrended_c(t) = y_c(t) − f(t; p̂_c)``
        (⇒ ``raw = detrended + f(t; p̂)`` — exactly invertible)

    via :func:`~gps_analysis.fitting.remove_trend` on the record's
    reconstructed (model, fits). Stateless pure evaluation at ANY
    epochs, including epochs newer than the fit window (live
    detrending) and epochs of a **different station** (borrowed
    records, design §0.6/§2.6 — the record is self-contained). The raw
    series is NEVER mutated: the result is a new array (design §4.1
    raw-preservation rules).

    Symbols → args:
        - ``p̂`` → ``record``: stored station record
        - ``t``, ``y`` → ``t``, ``y``: epochs [yr] and observations
          [L], shape (N,) or (C, N) with C matching the record's
          component count
        - selection → ``terms``: ``"all"`` | ``"secular"`` |
          ``"periodic"`` — partial detrend views from the SAME stored
          parameters (the legacy ``detrend_line``/``detrend_periodic``
          switches, design §4.2)
        - frame guard → ``frame``: the series' frame tag; when both it
          and the record's ``frame`` are non-None they must match —
          applying parameters across frames is an error to refuse, not
          fudge (design §2.5)

    Returns:
        Detrended series [L], float64, new array with the shape of
        ``y``.

    Raises:
        ValueError: On a frame mismatch (naming both frames), a
            component-count mismatch, or record validation failure.

    Reference:
        Design spec ``docs/DESIGN_live_detrending.md`` §4.1/§5.3.

    Numerical notes:
        Exact inverse: ``apply_detrend(...) + evaluate_record(record,
        t)`` reproduces ``y`` to one float64 rounding (a − b + b). The
        subtraction of an intercept-complete model needs no ``vshift``
        re-referencing pass (design §0.1 #1); callers wanting a
        different zero apply :func:`gps_analysis.baseline.remove_offset`
        afterwards, as an explicitly separate view step.
    """
    if frame is not None:
        record_frame = record.get("frame")
        if record_frame is not None and record_frame != frame:
            raise ValueError(
                f"frame mismatch: record frame {record_frame!r} != series "
                f"frame {frame!r} - refusing to apply (design §2.5)"
            )
    model_func, fits = trajectory_from_record(record)
    if terms != "all":
        fits = select_terms(model_func, fits, terms)
    return remove_trend(model_func, t, y, fits)
