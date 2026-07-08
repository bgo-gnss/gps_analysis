"""Transient detection — GBIS4TS-style velocity break-point analysis.

Planned surface (plan §10.7, Phase 2):

- Two-stage detection: cheap daily triage (moving-window / CUSUM over all
  173×3 series) → Bayesian confirm on flagged candidates only.
- Python port of GBIS4TS (2023GL103432, Yang / Sigmundsson / Geirsson,
  MATLAB/BSD-2) with the per-sample O(N³) inverse replaced by a
  Cholesky/Toeplitz solve. Ported in collaboration with the local authors —
  not a solo port.

Nothing lands here in Phase 1; the module exists so the package layout is
stable from day one.
"""

__all__: list[str] = []
