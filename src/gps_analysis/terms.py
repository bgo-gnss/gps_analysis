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
  ``Step(epoch)`` carries its epoch as data. ``select_terms`` removes a term
  by ZEROING its coefficients, so a shape parameter sitting in the vector
  would be zeroed too — silently changing the basis instead of removing the
  term. Keeping shape parameters off the vector makes that unreachable
  rather than guarded.

Everything is pure, float64, unit-agnostic, inputs never mutated (R2/R6).
"""

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import _LINEAR_DESIGN_ATTR, ModelFunc, _LinearDesign
from .models import FloatArray

__all__ = [
    "GROUP_ORDER",
    "Polynomial",
    "Seasonal",
    "Step",
    "Term",
    "TrajectoryModel",
    "term_from_spec",
]

GROUP_ORDER: tuple[str, ...] = ("secular", "periodic", "step", "transient")
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
        Bevis & Brown 2014, *J. Geodesy* 88 — the jump term; identical to
        ``models.heaviside_steps`` / ``fitting.with_steps`` (exact ``t >= t_k``
        comparison, no tolerance).

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


_TERM_KINDS: dict[str, Any] = {
    "polynomial": Polynomial,
    "seasonal": Seasonal,
    "step": Step,
}
"""Deserialization registry, ``kind`` → class.

Transient terms (exponential / logarithmic postseismic decay) are
deliberately absent until Bevis & Brown 2014 can be cited by equation number
— MATH_STANDARDS §2.4 requires it and the paper is not yet in ``reference/``.
"""


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
        """Positional parameter names, with step amplitudes renumbered.

        Numbering runs across ALL step terms in canonical order, which is
        what makes ``lineperiodic + K steps`` reproduce the names
        ``fitting.with_steps`` synthesizes byte-for-byte — the stored
        records and ``trajectory_from_record``'s cross-check depend on it.
        """
        out: list[str] = []
        n_step = 0
        for term in self.terms:
            for name in term.names:
                if name == "step_amp":
                    n_step += 1
                    out.append(f"step_amp_{n_step}")
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
        return evaluate

    def to_spec(self) -> list[dict[str, Any]]:
        """JSON-ready term list for a stored record."""
        return [term.to_spec() for term in self.terms]

    @classmethod
    def from_spec(cls, spec: Sequence[Mapping[str, Any]]) -> "TrajectoryModel":
        return cls([term_from_spec(s) for s in spec])
