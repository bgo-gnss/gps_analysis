"""Composable trajectory terms — the extended trajectory model as objects.

The package's models are hand-written callables (``linear``, ``periodic``,
``lineperiodic``) with their design matrices registered by callable identity,
plus one composition primitive (``fitting.with_steps``). That surface cannot
express "polynomial degree 2 plus one harmonic plus two steps plus a
postseismic decay" without a new hand-written callable per combination, and
its term bookkeeping is literal parameter-NAME matching that raises on any
name it has not been taught.

This module inverts the relationship: a **term list is primary** and the
``ModelFunc`` is generated from it (:meth:`TrajectoryModel.as_modelfunc`).
Each term owns its parameter names, its design columns and its centering
block, so composition is column concatenation and nothing downstream needs to
know which terms exist.

Derivation chain
----------------
Bevis & Brown 2014's extended trajectory model is a SUM of independent terms.
Given epochs t [yr] and a term list (T₁ … T_J):

1. Each term contributes ``T.columns(t, t_ref)`` ∈ ℝ^{N×k_j}, so
   ``f(t; p) = A(t)·p`` with ``A = [A₁ | A₂ | … | A_J]`` — one column per
   parameter, in the model's positional order (:meth:`TrajectoryModel.design`).
2. Centering is a linear reparameterization ``A′ = A·M``, ``p = M·p′``,
   ``C = M·C′·Mᵀ``. Each term owns its own block of M, so the assembled
   matrix is BLOCK-DIAGONAL and no term reaches across another's columns
   (:meth:`TrajectoryModel.uncentering`). This generalizes ``_LinearDesign``'s
   ``trend_column``/``intercept_column`` pair exactly — that pair worked
   because both indices belonged to the same (polynomial) term.
3. The generated callable carries a synthesized ``__signature__`` and a
   ``_LinearDesign`` on the attribute hook, so every existing consumer —
   ``fit_components``, ``remove_trend``, ``select_terms``,
   ``trajectory_from_record`` — keeps working unchanged.

Two conventions are pinned by test because getting either wrong is silent:

- **:class:`Seasonal` ignores ``t_ref``.** The phase convention is absolute
  ``yearf`` (``fitting._design_periodic``, and the stored documents record it
  as ``phase_convention``). Centering the trig columns would silently
  invalidate every deployed seasonal coefficient set.
- **Fixed/shape parameters are dataclass fields, never parameter slots.**
  ``Step(epoch)`` carries its epoch as data, ``LogTransient(epoch, tau)`` /
  ``ExpTransient(epoch, tau)`` their epoch and time scale. ``select_terms``
  removes a term by ZEROING its coefficients, so a shape parameter sitting
  in the vector would be zeroed too — silently changing the basis instead
  of removing the term. Keeping shape parameters off the vector makes that
  unreachable rather than guarded. τ is operator-fixed by default (Bevis &
  Brown's ELTM recipe); the opt-in nonlinear refinement is
  :func:`profile_transient_tau`, which profiles ``θ = ln τ`` through
  :func:`gps_analysis.varpro.estimate_varpro`, and
  :func:`check_transient_identifiability` gates what the fit delivered
  (with a Belsley–Kuh–Welsch localizer, :func:`bkw_dependencies`).

Everything is pure, float64, unit-agnostic, inputs never mutated (R2/R6).
"""

import dataclasses
import inspect
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import _LINEAR_DESIGN_ATTR, ModelFunc, _LinearDesign
from .models import FloatArray
from .varpro import VarproFit, estimate_varpro

__all__ = [
    "GROUP_ORDER",
    "TERM_SPEC_ATTR",
    "model_term_spec",
    "BkwDependency",
    "ExpTransient",
    "LogTransient",
    "Polynomial",
    "Seasonal",
    "Step",
    "Term",
    "TrajectoryModel",
    "TransientTauFit",
    "bkw_dependencies",
    "check_transient_identifiability",
    "profile_transient_tau",
    "term_from_spec",
]

GROUP_ORDER: tuple[str, ...] = ("secular", "periodic", "step", "transient")

TERM_SPEC_ATTR = "_gps_analysis_term_spec"
"""Attribute under which :meth:`TrajectoryModel.as_modelfunc` stashes its
term spec, so a stored record can reconstruct the model. Same hook pattern
as ``fitting._LINEAR_DESIGN_ATTR``."""
"""Canonical term ordering, and the reason it is not merely tidy.

:class:`TrajectoryModel` stable-sorts by this rank so a model carrying a
polynomial always has ``param_names[1] == "rate"``.  That is what keeps
``velocity._RATE_INDEX = 1`` correct — six sites read ``params[1]`` as the
secular rate, and a term algebra that reordered the vector would corrupt every
velocity product silently.
"""


@runtime_checkable
class Term(Protocol):
    """One additive term of a trajectory model.

    A Protocol rather than a base class so a caller can supply its own term
    without importing from here (leaf rule R2 — this package imposes no
    inheritance on its users).
    """

    kind: str
    group: str

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names, in this term's own positional order."""
        ...

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        """Design columns, shape (N, len(names))."""
        ...

    def uncentering(self, t_ref: float) -> FloatArray | None:
        """(k, k) map from centered to absolute-t parameters, or None."""
        ...

    def to_spec(self) -> dict[str, Any]:
        """JSON-ready description, round-tripped by :func:`term_from_spec`."""
        ...


@dataclasses.dataclass(frozen=True)
class Polynomial:
    """Secular polynomial in t — Bevis & Brown 2014's polynomial block.

    Equation:
        ``Σ_{m=0..M} p_m·(t − t_ref)^m``

    with ``M = degree``.  Degree 1 is ``models.linear``'s ``offset + rate·t``,
    and its parameter names are kept EXACTLY (``offset``, ``rate``) so records
    written before this module still validate against ``param_names``.

    Symbols → args:
        - ``M`` → ``degree``: polynomial degree (1 = offset + rate)
        - ``t_ref`` → centering epoch, supplied at ``columns`` time [yr]

    Reference:
        Bevis & Brown 2014, *J. Geodesy* 88 — the trajectory model's
        polynomial block; degree 1 matches ``models.linear`` and degree 2
        ``models.poly2`` (which is linear in its parameters and was merely
        unregistered).

    Numerical notes:
        This term OWNS the centering, and that is why the old
        ``trend_column``/``intercept_column`` pair worked: both indices
        belonged here.  Its :meth:`uncentering` is the binomial map, whose
        degree-1 case reproduces ``_fit_linear_design``'s back-substitution
        exactly.
    """

    degree: int = 1
    kind: str = dataclasses.field(default="polynomial", init=False)
    group: str = dataclasses.field(default="secular", init=False)

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError(f"degree must be >= 0, got {self.degree}")

    @property
    def names(self) -> tuple[str, ...]:
        base = ("offset", "rate", "curvature")
        return tuple(
            base[m] if m < len(base) else f"poly_{m}" for m in range(self.degree + 1)
        )

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64) - t_ref
        return np.column_stack([tt**m for m in range(self.degree + 1)])

    def uncentering(self, t_ref: float) -> FloatArray | None:
        """Binomial map so ``p = M·p′`` returns absolute-t parameters.

        For degree 1 this is ``[[1, −t_ref], [0, 1]]`` — identical to the
        explicit back-substitution in ``fitting._fit_linear_design``.
        """
        if t_ref == 0.0:
            return None
        k = self.degree + 1
        m = np.zeros((k, k), dtype=np.float64)
        from math import comb

        for i in range(k):
            for j in range(i, k):
                m[i, j] = comb(j, i) * (-t_ref) ** (j - i)
        return m

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "degree": int(self.degree)}


@dataclasses.dataclass(frozen=True)
class Seasonal:
    """Truncated Fourier seasonal series on ABSOLUTE ``yearf``.

    Equation:
        ``Σ_{h=1..H} [ a_h·cos(2πh·t) + b_h·sin(2πh·t) ]``

    with ``H = n_harmonics`` (2 = annual + semiannual, the production
    default).  ``t`` is absolute, NOT centered — see Numerical notes.

    Symbols → args:
        - ``H`` → ``n_harmonics``: number of harmonics
        - ``t`` → epochs [yr, fractional year]

    Reference:
        Bevis & Brown 2014, *J. Geodesy* 88, the seasonal block with
        ``n_F = 2``; matches ``models.periodic`` term for term.

    Numerical notes:
        **``t_ref`` is deliberately ignored.**  The fitted phase convention
        is absolute ``yearf`` (``fitting._design_periodic``; the stored
        documents record it as ``phase_convention``), so centering these
        columns would rotate every coefficient and silently invalidate the
        deployed seasonal sets.  :meth:`uncentering` returns None for the
        same reason.  Trig columns are bounded and need no conditioning fix.
    """

    n_harmonics: int = 2
    kind: str = dataclasses.field(default="seasonal", init=False)
    group: str = dataclasses.field(default="periodic", init=False)

    def __post_init__(self) -> None:
        if self.n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {self.n_harmonics}")

    @property
    def names(self) -> tuple[str, ...]:
        label = {1: "annual", 2: "semiannual"}
        out: list[str] = []
        for h in range(1, self.n_harmonics + 1):
            tag = label.get(h, f"harmonic{h}")
            out += [f"cos_{tag}", f"sin_{tag}"]
        return tuple(out)

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)  # absolute: see the docstring
        cols: list[FloatArray] = []
        for h in range(1, self.n_harmonics + 1):
            w = 2.0 * np.pi * h
            cols += [np.cos(w * tt), np.sin(w * tt)]
        return np.column_stack(cols)

    def uncentering(self, t_ref: float) -> FloatArray | None:
        return None

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_harmonics": int(self.n_harmonics)}


@dataclasses.dataclass(frozen=True)
class Step:
    """Heaviside offset at a FIXED epoch — amplitude estimated.

    Equation:
        ``a·H(t − t_k)``,  ``H(0) = 1``

    Symbols → args:
        - ``t_k`` → ``epoch``: step epoch [yr]; DATA, not a parameter
        - ``a`` → the single estimated amplitude [L]

    Reference:
        Bevis & Brown 2014, *J. Geodesy* 88(3), eq. (8) — the SLTM jump
        term; identical to ``models.heaviside_steps`` /
        ``fitting.with_steps`` (exact ``t >= t_k`` comparison, no
        tolerance).  **Documented divergence** (MATH_STANDARDS §2.2):
        their eq. (3) defines ``H(0) = 1/2``; this package pins
        ``H(0) = 1`` for byte-parity with the deployed ``with_steps``
        designs.  The two differ only when an observation falls exactly
        on the step epoch, where the amplitude estimate is undefined by
        half an epoch's weight either way.

    Numerical notes:
        The epoch is a dataclass field and not a parameter slot on purpose:
        ``select_terms`` removes a term by zeroing its coefficients, so an
        epoch in the vector would be zeroed to 1970 rather than removed.
        A step column with no observations on one side is rank-deficient —
        ``_wls_solve``'s inf-covariance + ``OptimizeWarning`` path applies.
    """

    epoch: float
    kind: str = dataclasses.field(default="step", init=False)
    group: str = dataclasses.field(default="step", init=False)

    @property
    def names(self) -> tuple[str, ...]:
        return ("step_amp",)  # renumbered globally by TrajectoryModel

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        return (tt >= self.epoch).astype(np.float64)[:, np.newaxis]

    def uncentering(self, t_ref: float) -> FloatArray | None:
        return None

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "epoch": float(self.epoch)}


@dataclasses.dataclass(frozen=True)
class LogTransient:
    """Logarithmic postseismic transient at a FIXED epoch — amplitude estimated.

    Equation:
        ``A·log(1 + Δt/T)``,  ``Δt = 0 for t < t_EQ, else t − t_EQ``

    — the ACCUMULATED form: exactly zero at the event epoch, rising
    without an asymptote (natural log, matching Bevis & Brown's usage).
    ``T`` is operator-fixed station metadata by default, which keeps the
    model linear in its parameters — their ELTM, default ``T = 1 yr`` —
    so ``select_terms``, the closed-form path and staged holds all keep
    working; profiled ``T`` is the opt-in
    :func:`profile_transient_tau`.

    Symbols → args:
        - ``t_EQ`` → ``epoch``: event epoch [yr]; DATA, not a parameter
        - ``T`` → ``tau``: logarithmic time scale [yr]; DATA, not a
          parameter — ``select_terms`` zeroes *coefficients*, so a ``T``
          in a slot would be zeroed and silently change the basis
        - ``A`` → the single estimated amplitude [L], named ``log_amp``
          (never ``amplitude`` — see :class:`ExpTransient` on why the
          two transient amplitude conventions must not share a name)

    Reference:
        Bevis & Brown 2014, *J. Geodesy* 88(3) 283–311,
        doi:10.1007/s00190-013-0685-5, eq. (9) (the transient form),
        composed into the ETM at eq. (10), where the ``Δt`` convention
        above is stated verbatim; Appendix 1 for the striking
        insensitivity of the composed fit to ``T`` (the SLTM
        coefficients, above all ``A``, absorb an erroneous ``T``).
        Equation numbers verified against the primary PDF
        (``reference/papers/``).

    Numerical notes:
        ``log1p(Δt/T)`` on ``Δt = max(t − t_EQ, 0)`` — no branch ever
        evaluates a negative argument, and log1p keeps precision for
        ``Δt ≪ T``.  Support uses ``t >= epoch`` like :class:`Step`,
        but at ``Δt = 0`` the column is 0 under any Heaviside
        convention, so the ``H(0)`` divergence documented on
        :class:`Step` cannot reach it.  Omitting this term from a
        step-only fit biases the step amplitude by
        ``b·[ln(1+s) − 1 + ln(1+s)/s]``, ``s = T_post/T`` — a
        logarithmically DIVERGENT bias in the post-event span
        (test-pinned closed form; relaxation is curvature, which
        piecewise rates cannot absorb).  Do not both excise the event
        with ``segments`` and model it with Step + LogTransient:
        excision removes the very data that constrains ``T``.
    """

    epoch: float
    tau: float
    kind: str = dataclasses.field(default="log_transient", init=False)
    group: str = dataclasses.field(default="transient", init=False)

    def __post_init__(self) -> None:
        _validate_transient_shape(self.epoch, self.tau)

    @property
    def names(self) -> tuple[str, ...]:
        return ("log_amp",)  # renumbered globally by TrajectoryModel

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        dt = np.maximum(np.asarray(t, dtype=np.float64) - self.epoch, 0.0)
        return np.asarray(np.log1p(dt / self.tau), dtype=np.float64)[:, np.newaxis]

    def uncentering(self, t_ref: float) -> FloatArray | None:
        return None

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "epoch": float(self.epoch), "tau": float(self.tau)}


@dataclasses.dataclass(frozen=True)
class ExpTransient:
    """Exponential inflation/relaxation transient at a FIXED epoch.

    Equation:
        ``Φ·(1 − e^(−Δt/τ))``,  ``Δt = 0 for t < t₀, else t − t₀``

    — the ACCUMULATED form: exactly zero at the event epoch, rising to
    the asymptote ``Φ``.  ``τ`` is operator-fixed by default (same
    linearity rationale as :class:`LogTransient`); profiled ``τ`` is the
    opt-in :func:`profile_transient_tau`.

    Symbols → args:
        - ``t₀`` → ``epoch``: onset epoch [yr]; DATA, not a parameter
        - ``τ`` → ``tau``: e-folding time [yr]; DATA, not a parameter
          (same ``select_terms`` reasoning as :class:`LogTransient`)
        - ``Φ`` → the single estimated amplitude [L], named ``exp_amp``

    **Amplitude convention** — this is NOT ``models.exp_linear``'s
    ``A·exp(−k·t)`` parameterization.  Same curve, reparameterized:
    ``A_expl = −Φ``, ``k = 1/τ``, with the asymptote absorbed into the
    intercept (``x₀ = C + Φ``).  A reader converting between the two
    must flip the amplitude's sign, which is why the parameter here is
    ``exp_amp`` and never a shared ``amplitude``.

    Reference:
        Reverso et al. 2014, *JGR Solid Earth* 119, 4666–4683,
        doi:10.1002/2013JB010569, eq. (20) — the fitted posteruptive
        inflation form ``Φ·(1 − e^(−t/τ)) + U̇_∞·t + C``.  Physics:
        their eqs. (11)–(12) (each reservoir overpressure is the sum of
        an exponential and a linear function; ``τ = 1/ξ`` from
        inter-reservoir pressure re-equilibration, the linear slope from
        constant basal inflow) and eq. (17) (surface displacement linear
        in overpressure).  Measured ``τ = 0.33 ± 0.08`` and
        ``0.13 ± 0.04 yr`` at Grímsvötn.  Equation numbers verified
        against the primary source.

    Numerical notes:
        ``1 − exp(−Δt/τ)`` on ``Δt = max(t − t₀, 0)`` — exact zero
        before and at the onset.  Omitting this term from a step-only
        fit biases the step amplitude by ``b·[1 − (1 − e^(−s))/s]``,
        ``s = T_post/τ``, which SATURATES at ``b`` (the step absorbs the
        finite asymptote) — the divergence of the log form is specific
        to its missing asymptote (both closed forms test-pinned).  The
        ``segments`` caveat on :class:`LogTransient` applies unchanged.
    """

    epoch: float
    tau: float
    kind: str = dataclasses.field(default="exp_transient", init=False)
    group: str = dataclasses.field(default="transient", init=False)

    def __post_init__(self) -> None:
        _validate_transient_shape(self.epoch, self.tau)

    @property
    def names(self) -> tuple[str, ...]:
        return ("exp_amp",)  # renumbered globally by TrajectoryModel

    def columns(self, t: FloatArray, *, t_ref: float = 0.0) -> FloatArray:
        dt = np.maximum(np.asarray(t, dtype=np.float64) - self.epoch, 0.0)
        return np.asarray(1.0 - np.exp(-dt / self.tau), dtype=np.float64)[:, np.newaxis]

    def uncentering(self, t_ref: float) -> FloatArray | None:
        return None

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "epoch": float(self.epoch), "tau": float(self.tau)}


def _validate_transient_shape(epoch: float, tau: float) -> None:
    """Shared shape-parameter validation of the two house transients."""
    if not math.isfinite(epoch):
        raise ValueError(f"epoch must be finite, got {epoch!r}")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be finite and > 0, got {tau!r}")


_HOUSE_TRANSIENTS = (LogTransient, ExpTransient)
"""The transient terms this module can profile and gate.

A caller-supplied :class:`Term` with ``group == "transient"`` composes and
fits like any other, but :func:`profile_transient_tau` and
:func:`check_transient_identifiability` need ``epoch``/``tau`` fields with
the house semantics, so both refuse unknown transient kinds loudly rather
than guessing."""


_TERM_KINDS: dict[str, Any] = {
    "polynomial": Polynomial,
    "seasonal": Seasonal,
    "step": Step,
    "log_transient": LogTransient,
    "exp_transient": ExpTransient,
}
"""Deserialization registry, ``kind`` → class.

The transient kinds were deliberately absent until their sources could be
cited by equation number (MATH_STANDARDS §2.4).  Both are now verified
against the primary PDFs: Bevis & Brown 2014 eq. (9)/(10) for
``log_transient``, Reverso et al. 2014 eq. (20) for ``exp_transient`` —
see ``reference/papers/README.md``.  (Bevis & Brown 2014 contains NO
exponential transient; the word does not occur in its 23 pages.)
"""


_NUMBERED_AMP_NAMES = ("step_amp", "log_amp", "exp_amp")
"""Per-term amplitude names that :attr:`TrajectoryModel.param_names`
renumbers globally, one counter per family (``step_amp_1``, ``log_amp_1``,
…).  ``step_amp`` numbering is byte-pinned against ``fitting.with_steps``."""


def model_term_spec(model: Any) -> list[dict[str, Any]] | None:
    """The term spec of a model callable, or None if it has none.

    None means "an ordinary registry model (+ ``with_steps``)", which a
    record expresses with a model code and ``step_epochs`` alone. A spec
    means the model carries terms that vocabulary cannot describe.
    """
    spec = getattr(model, TERM_SPEC_ATTR, None)
    return list(spec) if spec is not None else None


def term_from_spec(spec: Mapping[str, Any]) -> Term:
    """Rebuild a term from its :meth:`Term.to_spec` description."""
    kind = spec.get("kind")
    if kind not in _TERM_KINDS:
        raise ValueError(f"unknown term kind {kind!r}; known: {sorted(_TERM_KINDS)}")
    fields = {k: v for k, v in spec.items() if k != "kind"}
    return _TERM_KINDS[kind](**fields)  # type: ignore[no-any-return]


@dataclasses.dataclass(frozen=True)
class TrajectoryModel:
    """An additive trajectory model built from composable terms.

    A class rather than a bare sequence because it owns three things no
    individual term can: GLOBAL amplitude numbering (``step_amp_1 …``
    across all step terms), name-uniqueness validation, and the canonical
    :data:`GROUP_ORDER` sort that keeps ``param_names[1] == "rate"``.

    Attributes:
        terms: The terms, stable-sorted into canonical group order.
    """

    terms: tuple[Term, ...]

    def __init__(self, terms: Sequence[Term]) -> None:
        if not terms:
            raise ValueError("a model needs at least one term")
        ranked = sorted(
            enumerate(terms),
            key=lambda it: (GROUP_ORDER.index(it[1].group), it[0]),
        )
        object.__setattr__(self, "terms", tuple(term for _i, term in ranked))

    @property
    def param_names(self) -> tuple[str, ...]:
        """Positional parameter names, with amplitude families renumbered.

        ``step_amp`` numbering runs across ALL step terms in canonical
        order, which is what makes ``lineperiodic + K steps`` reproduce
        the names ``fitting.with_steps`` synthesizes byte-for-byte — the
        stored records and ``trajectory_from_record``'s cross-check
        depend on it.  ``log_amp`` / ``exp_amp`` are numbered the same
        way, each family on its own counter, so two transients at
        different epochs never collide.
        """
        out: list[str] = []
        counters: dict[str, int] = {}
        for term in self.terms:
            for name in term.names:
                if name in _NUMBERED_AMP_NAMES:
                    counters[name] = counters.get(name, 0) + 1
                    out.append(f"{name}_{counters[name]}")
                else:
                    out.append(name)
        dupes = {n for n in out if out.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate parameter names: {sorted(dupes)}")
        return tuple(out)

    @property
    def n_params(self) -> int:
        return sum(len(term.names) for term in self.terms)

    def term_slices(self) -> tuple[slice, ...]:
        """Column slice of each term, in :attr:`terms` order."""
        out: list[slice] = []
        start = 0
        for term in self.terms:
            width = len(term.names)
            out.append(slice(start, start + width))
            start += width
        return tuple(out)

    def design(self, t: ArrayLike, *, t_ref: float = 0.0) -> FloatArray:
        """Assemble ``A(t)`` by concatenating each term's columns."""
        tt = np.asarray(t, dtype=np.float64)
        return np.column_stack([term.columns(tt, t_ref=t_ref) for term in self.terms])

    def uncentering(self, t_ref: float) -> FloatArray:
        """Block-diagonal map from centered to absolute-t parameters.

        Each term contributes its own block (identity where it returns
        None), so centering never straddles a term boundary — which is the
        invariant that makes term-granular holds in
        :mod:`gps_analysis.staged` safe.
        """
        n = self.n_params
        m = np.eye(n, dtype=np.float64)
        for term, sl in zip(self.terms, self.term_slices(), strict=True):
            block = term.uncentering(t_ref)
            if block is not None:
                m[sl, sl] = block
        return m

    def group_mask(self, group: str | Sequence[str]) -> NDArray[np.bool_]:
        """(P,) membership mask of one or more term groups.

        Replaces the closed-world parameter-NAME classifier of
        ``detrend._term_keep_mask``, which raises on any name it has not
        been taught — the thing that would block a transient term.
        """
        wanted = {group} if isinstance(group, str) else set(group)
        unknown = wanted - set(GROUP_ORDER)
        if unknown:
            raise ValueError(
                f"unknown term group(s) {sorted(unknown)}; known: {list(GROUP_ORDER)}"
            )
        mask = np.zeros(self.n_params, dtype=np.bool_)
        for term, sl in zip(self.terms, self.term_slices(), strict=True):
            if term.group in wanted:
                mask[sl] = True
        return mask

    def as_modelfunc(self) -> ModelFunc:
        """Generate the ``f(t, *params)`` callable every consumer expects.

        The adapter that makes this module additive rather than a rewrite:
        the returned callable carries a synthesized ``__signature__`` (so
        ``_n_model_params`` and ``detrend._param_names`` work) and a
        ``_LinearDesign`` on the attribute hook (so ``fit_components``
        keeps the closed-form path), exactly as ``fitting.with_steps`` does.
        """
        names = self.param_names
        n_params = len(names)
        model = self

        def evaluate(t: ArrayLike, *params: float) -> FloatArray:
            if len(params) != n_params:
                raise ValueError(
                    f"model takes {n_params} parameters, got {len(params)}"
                )
            a = model.design(t)
            return np.asarray(a @ np.asarray(params, dtype=np.float64), np.float64)

        evaluate.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            + [
                inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for n in names
            ]
        )
        evaluate.__name__ = "+".join(term.kind for term in self.terms)

        poly = next(
            (
                (term, sl)
                for term, sl in zip(self.terms, self.term_slices(), strict=True)
                if term.kind == "polynomial" and term.degree >= 1  # type: ignore[attr-defined]
            ),
            None,
        )
        design = _LinearDesign(
            build=lambda tt: model.design(tt),
            trend_column=None if poly is None else poly[1].start + 1,
            intercept_column=None if poly is None else poly[1].start,
        )
        setattr(evaluate, _LINEAR_DESIGN_ATTR, design)
        # The term spec rides on the callable so a RECORD can carry it. A
        # registry code plus step epochs cannot express a transient, so
        # without this a transient model could be fitted and never stored
        # -- see detrend.RECORD_VERSION_TERMS.
        setattr(evaluate, TERM_SPEC_ATTR, self.to_spec())
        return evaluate

    def to_spec(self) -> list[dict[str, Any]]:
        """JSON-ready term list for a stored record."""
        return [term.to_spec() for term in self.terms]

    @classmethod
    def from_spec(cls, spec: Sequence[Mapping[str, Any]]) -> "TrajectoryModel":
        return cls([term_from_spec(s) for s in spec])


# ---------------------------------------------------------------------------
# Transient identifiability: BKW localizer + delivered-quantity gates
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BkwDependency:
    """One Belsley–Kuh–Welsch near-dependency of a design matrix.

    Attributes:
        condition_index: ``η_k = s_max/s_k`` of the participating singular
            value (dimensionless).
        participants: Parameter names whose variance-decomposition
            proportion on this singular value meets the π threshold —
            the columns entangled in this near-dependency.
        proportions: π of every parameter on this singular value, in
            model parameter order (each in [0, 1]).
    """

    condition_index: float
    participants: tuple[str, ...]
    proportions: tuple[float, ...]


def bkw_dependencies(
    design: ArrayLike,
    names: Sequence[str],
    *,
    eta_threshold: float = 30.0,
    pi_threshold: float = 0.5,
) -> tuple[BkwDependency, ...]:
    """Locate near-dependencies by BKW variance decomposition.

    What it computes: the Belsley–Kuh–Welsch collinearity diagnosis —
    which singular values of the column-equilibrated design are
    degenerate (condition index η) and which parameters participate in
    each degeneracy (variance-decomposition proportions π).

    Equation:
        ``X_e = X·diag(1/‖x_j‖₂)`` (UNCENTERED, column-equilibrated),
        ``X_e = U·S·Vᵀ``,  ``η_k = s_max/s_k``,
        ``φ_jk = v_jk²/s_k²``,  ``π_kj = φ_jk / Σ_k′ φ_jk′``
    a near-dependency is a singular value with ``η_k ≥ η*`` on which at
    least TWO parameters have ``π_kj ≥ π*`` (one large π alone names no
    dependency — a dependency needs partners).

    Symbols → args:
        - ``X`` → ``design``: design matrix, shape (N, P) [basis units;
          equilibration makes the diagnosis unit-free]
        - column names → ``names``: length P, used to name participants
        - ``η*`` → ``eta_threshold``: condition-index cutoff
          (dimensionless; BKW's "moderate to strong" onset is 30)
        - ``π*`` → ``pi_threshold``: proportion cutoff (dimensionless;
          BKW's suggested 0.5)

    Returns:
        Dependencies sorted by descending condition index (most severe
        first); empty tuple when the design is well-conditioned.

    Reference:
        Belsley, Kuh & Welsch 1980, *Regression Diagnostics* (Wiley),
        ch. 3 — condition indices, variance-decomposition proportions,
        and the η ≥ 30 / two-π ≥ 0.5 reading.  Belsley 1984,
        *Am. Stat.* 38(2) 73–77 — the diagnosis must run on the
        UNCENTERED equilibrated design; centering hides every
        dependency the intercept participates in.

    Numerical notes:
        SVD of the equilibrated design, never of ``XᵀX`` (squares the
        condition number — the quantity under diagnosis).  Columns of
        zero norm are refused: their "dependency" is emptiness, not
        collinearity, and upstream rank checks own that failure.
    """
    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(
            f"design must be 2-D with {len(names)} columns matching names, "
            f"got shape {x.shape}"
        )
    norms = np.linalg.norm(x, axis=0)
    if not bool(np.all(np.isfinite(x))) or bool(np.any(norms == 0.0)):
        raise ValueError(
            "design must be finite with no all-zero column - a zero column "
            "is a rank failure for the upstream solver, not a collinearity"
        )
    _, s, vt = np.linalg.svd(x / norms, full_matrices=False)
    eta = float(s[0]) / s
    phi = vt.T**2 / s**2  # (P, K): phi[j, k]
    pi = phi / phi.sum(axis=1, keepdims=True)
    out: list[BkwDependency] = []
    for k in range(s.size):
        if eta[k] < eta_threshold:
            continue
        involved = tuple(
            names[j] for j in range(len(names)) if pi[j, k] >= pi_threshold
        )
        if len(involved) >= 2:
            out.append(
                BkwDependency(
                    condition_index=float(eta[k]),
                    participants=involved,
                    proportions=tuple(float(p) for p in pi[:, k]),
                )
            )
    return tuple(sorted(out, key=lambda d: -d.condition_index))


def _amp_index(model: TrajectoryModel, term: Term) -> int:
    """Positional parameter index of a single-amplitude term."""
    for tm, sl in zip(model.terms, model.term_slices(), strict=True):
        if tm is term:
            return int(sl.start)
    raise ValueError("term is not part of this model")  # pragma: no cover


def check_transient_identifiability(
    model: TrajectoryModel,
    t: ArrayLike,
    params: ArrayLike,
    covariance: ArrayLike,
    *,
    min_pre_years: float = 1.0,
    min_curvature_epochs: int = 8,
    max_amp_rel_sigma: float = 0.2,
    max_step_corr: float = 0.95,
    tau_fit: "TransientTauFit | None" = None,
    max_tau_decades: float = 1.0,
    eta_threshold: float = 30.0,
    pi_threshold: float = 0.5,
) -> None:
    """Gate a fitted transient on the quantities the fit DELIVERED.

    What it checks, per transient term (raising :class:`ValueError` with
    every failure named and a BKW localizer attached):

    1. **Pre-event support** — kept epochs must precede the event by at
       least ``min_pre_years`` (without them the step IS the intercept),
       and the record must reach ``T_post ≥ τ`` when τ is fixed (one
       time constant of curvature).
    2. **Curvature support** — at least ``min_curvature_epochs`` kept
       epochs in ``[t_k, t_k + min(3τ, T_post)]``: the curvature
       information extends to ≈3τ, not τ.
    3. **Delivered quantities** — ``σ_b/|b̂| ≤ max_amp_rel_sigma``
       (default 0.2 = amplitude SNR ≥ 5); ``|corr(â, b̂)| ≤
       max_step_corr`` against a step at the same epoch, from the
       returned covariance; and, when τ was profiled (``tau_fit``), a
       CLOSED Δχ²=1 interval spanning at most ``max_tau_decades``
       decades — an open or decades-wide interval means publish a τ
       bound, not an estimate.

    A smallest-eigenvalue gate on the {rate, step, log} correlation was
    considered and REFUTED: for ``Δt ≪ τ`` the log column aliases a
    velocity break, not the global rate, whenever pre-event data exist —
    measured across ``T_post/τ ∈ [0.1, 160]``, λ_min decreases
    monotonically with a LONGER post-event span while σ_a and σ_b both
    improve, so that gate fires benign and stays silent in the
    motivating case.  Hence the gates above act on delivered quantities,
    and the Belsley–Kuh–Welsch decomposition
    (:func:`bkw_dependencies`, uncentered equilibrated design) is
    attached to the failure message as the LOCALIZER: it names the
    participating terms from the same SVD family ``_wls_solve``
    performs, without pre-guessing which triple is involved.

    Symbols → args:
        - ``t`` → kept fit epochs [yr, fractional year], the epochs the
          delivered ``params``/``covariance`` were estimated from
        - ``b̂, σ_b, â, corr`` → ``params``, ``covariance``: the fit's
          delivered parameter vector (P,) [L] and covariance (P, P)
          [L²], in ``model.param_names`` order (absolute-t
          parameterization)
        - thresholds as documented above (dimensionless / yr / count)

    Reference:
        Gate design: PLAN composable-terms Phase 3 (the Fable-review
        replacement of the eigenvalue gate); operator conditions τ fixed
        → ``T_post ≥ τ``, τ profiled → ``T_post ≳ 5τ`` and SNR ≳ 5:
        :mod:`gps_analysis.varpro` module docstring (identification).
        Localizer: Belsley, Kuh & Welsch 1980 ch. 3; Belsley 1984
        (uncentered design).

    Numerical notes:
        Non-finite delivered quantities (the ``inf`` covariance of a
        rank-deficient ``_wls_solve``) fail the corresponding gate
        rather than passing vacuously.  Message register follows the
        step-separability guard in ``detrend.estimate_detrend``: name
        the terms, name the ρ, say the ways out.
    """
    tt = np.asarray(t, dtype=np.float64)
    pp = np.asarray(params, dtype=np.float64)
    cc = np.asarray(covariance, dtype=np.float64)
    n_params = model.n_params
    if tt.ndim != 1 or tt.size == 0:
        raise ValueError(f"t must be a non-empty 1-D array, got shape {tt.shape}")
    if pp.shape != (n_params,) or cc.shape != (n_params, n_params):
        raise ValueError(
            f"params/covariance must have shapes ({n_params},) and "
            f"({n_params}, {n_params}) for this model, got {pp.shape} and {cc.shape}"
        )
    names = model.param_names
    t_min, t_max = float(np.min(tt)), float(np.max(tt))
    problems: list[str] = []
    for term in model.terms:
        if term.group != "transient":
            continue
        if not isinstance(term, _HOUSE_TRANSIENTS):
            raise ValueError(
                f"the identifiability gates understand the house transient "
                f"kinds {sorted(k for k, c in _TERM_KINDS.items() if c in _HOUSE_TRANSIENTS)}; "
                f"got {term.kind!r}"
            )
        epoch, tau = float(term.epoch), float(term.tau)
        j = _amp_index(model, term)
        label = f"{term.kind} at {epoch:.5f}"
        pre_span = epoch - t_min
        t_post = t_max - epoch
        if pre_span < min_pre_years:
            problems.append(
                f"{label}: pre-event epochs span {max(pre_span, 0.0):.2f} yr "
                f"< {min_pre_years} yr - without them the step IS the intercept"
            )
        tau_is_fixed = tau_fit is None or model.terms[tau_fit.term_index] is not term
        if tau_is_fixed and t_post < tau:
            problems.append(
                f"{label}: post-event span T_post = {max(t_post, 0.0):.2f} yr "
                f"< tau = {tau:.2f} yr - with tau fixed the record must reach "
                f"one time constant"
            )
        horizon = min(3.0 * tau, t_post)
        n_curv = int(np.count_nonzero((tt >= epoch) & (tt <= epoch + horizon)))
        if n_curv < min_curvature_epochs:
            problems.append(
                f"{label}: only {n_curv} kept epochs in "
                f"[t_k, t_k + min(3*tau, T_post)] = [{epoch:.4f}, "
                f"{epoch + horizon:.4f}] < {min_curvature_epochs} - the "
                f"curvature information extends to ~3*tau, not tau"
            )
        b = float(pp[j])
        sigma_b = float(np.sqrt(cc[j, j])) if cc[j, j] >= 0.0 else float("nan")
        rel = sigma_b / abs(b) if b != 0.0 else float("inf")
        if not math.isfinite(rel) or rel > max_amp_rel_sigma:
            problems.append(
                f"{names[j]} = {b:.3f} +/- {sigma_b:.3f}: relative sigma "
                f"{rel:.2f} > {max_amp_rel_sigma} (amplitude SNR < "
                f"{1.0 / max_amp_rel_sigma:.0f})"
            )
        for other in model.terms:
            if isinstance(other, Step) and math.isclose(
                other.epoch, epoch, rel_tol=0.0, abs_tol=1e-9
            ):
                i = _amp_index(model, other)
                denom = float(np.sqrt(cc[i, i] * cc[j, j]))
                rho = float(cc[i, j]) / denom if denom > 0.0 else float("nan")
                if not math.isfinite(rho) or abs(rho) > max_step_corr:
                    problems.append(
                        f"corr({names[i]}, {names[j]}) = {rho:+.3f} exceeds "
                        f"{max_step_corr} - the step absorbs the transient's "
                        f"projection"
                    )
        if tau_fit is not None and model.terms[tau_fit.term_index] is term:
            lo_yr, hi_yr = tau_fit.tau_interval
            if tau_fit.interval_open_lower or tau_fit.interval_open_upper:
                side = (
                    "both sides"
                    if tau_fit.interval_open_lower and tau_fit.interval_open_upper
                    else (
                        "the lower side"
                        if tau_fit.interval_open_lower
                        else "the upper side"
                    )
                )
                problems.append(
                    f"{label}: the profiled-tau Delta-chi-square = 1 interval "
                    f"did not close on {side} - only a tau bound is established"
                )
            else:
                decades = math.log10(hi_yr / lo_yr)
                if decades > max_tau_decades:
                    problems.append(
                        f"{label}: the profiled-tau interval "
                        f"[{lo_yr:.3g}, {hi_yr:.3g}] yr spans {decades:.1f} "
                        f"decades > {max_tau_decades} - publish a tau bound, "
                        f"not an estimate"
                    )
    if not problems:
        return
    deps = bkw_dependencies(
        model.design(tt), names, eta_threshold=eta_threshold, pi_threshold=pi_threshold
    )
    if deps:
        localizer = "; ".join(
            f"condition index {d.condition_index:.0f} among "
            f"({', '.join(d.participants)})"
            for d in deps
        )
    else:
        localizer = f"none at eta >= {eta_threshold:g}"
    raise ValueError(
        "transient identifiability gates failed: "
        + "; ".join(problems)
        + ". Near-dependencies (Belsley-Kuh-Welsch, uncentered equilibrated "
        "design): "
        + localizer
        + ". Ways out: lengthen the post-event record; fix tau from station "
        "metadata (donor or regional value) instead of profiling it, or "
        "publish a tau bound; or model the event with the step alone and "
        "accept the documented omitted-transient bias - and never excise the "
        "event with segments while also modeling it, excision removes the "
        "very data that constrains tau."
    )


# ---------------------------------------------------------------------------
# Profiled tau (opt-in): the term-aware caller of estimate_varpro
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TransientTauFit:
    """Result of profiling one transient's time scale by VARPRO.

    Attributes:
        tau: Profile optimum τ̂ [yr] (= ``exp(θ̂)``).
        tau_sigma: 1-σ of τ̂ [yr] by the delta method ``τ̂·σ_θ`` from the
            bordered covariance (θ = ln τ).
        tau_interval: Δχ²=1 profile-likelihood interval in τ [yr] — the
            exact monotone image of the θ interval (profile intervals
            are transformation-equivariant).
        interval_open_lower: The lower τ endpoint is pinned to
            ``tau_bounds[0]`` without a Δχ²=1 crossing.
        interval_open_upper: Mirror image.  Either flag ⇒ publish a τ
            bound, not an estimate.
        model: The input model with τ̂ substituted into the profiled
            term — evaluate/serialize THIS model.
        term_index: Index of the profiled term in ``model.terms``.
        params: Linear amplitudes ĉ(τ̂), shape (P,), in
            ``model.param_names`` order, absolute-t parameterization.
        covariance: Joint (ĉ, θ̂) covariance, shape (P+1, P+1),
            amplitudes first (absolute-t), **θ = ln τ last** [ln yr].
        chisq: Whitened residual sum of squares at τ̂.
        scale_sq: ŝ² applied to covariance and interval
            (:class:`~gps_analysis.varpro.VarproFit.scale_sq`).
        n_obs: Number of observations.
        varpro: The underlying :class:`~gps_analysis.varpro.VarproFit`
            (CENTERED parameterization — provenance, not for reuse).
    """

    tau: float
    tau_sigma: float
    tau_interval: tuple[float, float]
    interval_open_lower: bool
    interval_open_upper: bool
    model: TrajectoryModel
    term_index: int
    params: FloatArray
    covariance: FloatArray
    chisq: float
    scale_sq: float
    n_obs: int
    varpro: VarproFit


def profile_transient_tau(
    model: TrajectoryModel,
    t: ArrayLike,
    y: ArrayLike,
    *,
    tau_bounds: tuple[float, float] = (0.01, 10.0),
    term_index: int | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    n_grid: int = 33,
) -> TransientTauFit:
    """Profile one transient's time scale τ by variable projection.

    What it computes: the opt-in nonlinear refinement of an
    operator-fixed τ —

        ``θ̂ = argmin_θ ‖(I − U(θ)U(θ)ᵀ)·y_w‖₂²``,  ``θ = ln τ``,

    the concentrated VARPRO profile over the FULL trajectory design with
    the profiled term's τ set to ``e^θ``, delegated entirely to
    :func:`gps_analysis.varpro.estimate_varpro` (amplitude solve,
    eq.-(8) Jacobian, bordered covariance, Δχ²=1 interval — nothing is
    re-implemented here).  This function is the **term-aware caller**
    that varpro's contract assigns the identification conditions to:
    after the fit it checks ``T_post ≳ 5τ̂`` and amplitude SNR ≥ 5 and
    warns when either fails — publish a τ **bound** then, and carry the
    caveat into the amplitude (at ``T_post ≈ 3τ`` the interval spans
    decades or is one-sided regardless of SNR).

    Symbols → args:
        - model, with exactly one profiled transient → ``model`` +
          ``term_index`` (None ⇒ the model's single transient term)
        - ``t`` → kept epochs [yr]; ``y`` → observations, shape (N,) [L]
        - τ search box → ``tau_bounds``: (lo, hi) [yr], 0 < lo < hi;
          searched in θ = ln τ (near-symmetric profile when well
          identified — varpro module docstring)
        - ``σᵢ`` → ``sigma`` [L], ``absolute_sigma``, ``n_grid``: passed
          through to :func:`~gps_analysis.varpro.estimate_varpro`

    Returns:
        :class:`TransientTauFit` — τ̂ with delta-method σ and the exact
        τ-space profile interval, the model with τ̂ substituted, and the
        absolute-t amplitudes + joint covariance (θ last).

    Raises:
        ValueError: If the model has no transient term (or, without
            ``term_index``, more than one), ``term_index`` names a
            non-transient or non-house term, ``tau_bounds`` is not
            ``0 < lo < hi`` finite, or shapes mismatch; plus everything
            :func:`~gps_analysis.varpro.estimate_varpro` raises.

    Warns:
        UserWarning: When ``T_post < 5·τ̂`` or amplitude SNR < 5 (the
            term-specific identification conditions this caller owns),
            in addition to varpro's own open-interval warning.

    Reference:
        Golub & Pereyra 1973 / O'Leary & Rust 2013 via
        :mod:`gps_analysis.varpro`; Bevis & Brown 2014 §5.2 — τ as
        station metadata refined per station by a 1-D nonlinear search,
        exactly this shape of deferral ("estimate say 100 non-linear
        parameters one or two at a time, in separate inversions").

    Numerical notes:
        The design is built CENTERED (``t_ref = mean(t)``) to satisfy
        varpro's pre-conditioning contract for polynomial-in-t columns;
        the returned amplitudes and covariance are mapped back to the
        absolute-t parameterization through
        :meth:`TrajectoryModel.uncentering` (θ row/column unchanged —
        the package invariant that parameter vectors crossing a module
        boundary are absolute-t).  ``∂Φ/∂θ`` uses varpro's documented
        central finite difference; θ-independent columns difference to
        exactly zero.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if tt.ndim != 1 or tt.shape != yy.shape:
        raise ValueError(
            f"t and y must be 1-D with equal shapes, got {tt.shape} and {yy.shape}"
        )
    if not bool(np.all(np.isfinite(tt))):
        raise ValueError("t must be finite")
    transient_indices = [
        i for i, tm in enumerate(model.terms) if tm.group == "transient"
    ]
    if term_index is None:
        if len(transient_indices) != 1:
            raise ValueError(
                f"model has {len(transient_indices)} transient terms; pass "
                f"term_index to select which one to profile"
            )
        term_index = transient_indices[0]
    if not 0 <= term_index < len(model.terms):
        raise ValueError(
            f"term_index {term_index} out of range for {len(model.terms)} terms"
        )
    term = model.terms[term_index]
    if not isinstance(term, _HOUSE_TRANSIENTS):
        raise ValueError(
            f"term_index {term_index} is a {term.kind!r} term; profiling needs "
            f"a house transient (log_transient / exp_transient) with a tau field"
        )
    lo, hi = float(tau_bounds[0]), float(tau_bounds[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError(
            f"tau_bounds must be finite with 0 < lo < hi, got {tau_bounds}"
        )
    t_ref = float(np.mean(tt))

    def with_tau(tau: float) -> TrajectoryModel:
        terms = list(model.terms)
        terms[term_index] = dataclasses.replace(term, tau=tau)
        return TrajectoryModel(terms)

    def design(theta: float) -> FloatArray:
        return with_tau(math.exp(theta)).design(tt, t_ref=t_ref)

    fit = estimate_varpro(
        design,
        yy,
        theta_bounds=(math.log(lo), math.log(hi)),
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        n_grid=n_grid,
    )
    tau_hat = math.exp(fit.theta)
    n_params = model.n_params
    m = model.uncentering(t_ref)
    m_ext = np.eye(n_params + 1, dtype=np.float64)
    m_ext[:n_params, :n_params] = m
    params_abs = np.asarray(m @ fit.params, dtype=np.float64)
    cov_abs = np.asarray(m_ext @ fit.covariance @ m_ext.T, dtype=np.float64)

    model_hat = with_tau(tau_hat)
    result = TransientTauFit(
        tau=tau_hat,
        tau_sigma=tau_hat * fit.theta_sigma,
        tau_interval=(math.exp(fit.theta_interval[0]), math.exp(fit.theta_interval[1])),
        interval_open_lower=fit.interval_open_lower,
        interval_open_upper=fit.interval_open_upper,
        model=model_hat,
        term_index=term_index,
        params=params_abs,
        covariance=cov_abs,
        chisq=fit.chisq,
        scale_sq=fit.scale_sq,
        n_obs=fit.n_obs,
        varpro=fit,
    )
    j = _amp_index(model_hat, model_hat.terms[term_index])
    t_post = float(np.max(tt)) - float(term.epoch)
    amp = float(params_abs[j])
    amp_sigma = float(np.sqrt(cov_abs[j, j]))
    snr = abs(amp) / amp_sigma if amp_sigma > 0.0 else float("inf")
    if t_post < 5.0 * tau_hat or snr < 5.0:
        warnings.warn(
            f"profiled tau is marginally identified (T_post = {t_post:.2f} yr "
            f"vs 5*tau = {5.0 * tau_hat:.2f} yr, amplitude SNR = {snr:.1f} vs "
            f"5) - publish a tau bound, not an estimate, and carry the caveat "
            f"into the amplitude",
            UserWarning,
            stacklevel=2,
        )
    return result
