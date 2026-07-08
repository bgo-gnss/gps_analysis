"""Deformation-source forward models and inversion.

Planned surface (plan §10.2, sequence locked Mogi → Okada → joint):

- ``mogi_forward`` / ``mogi_invert`` — point pressure source, scipy-based,
  GPS-only (Phase 1). **Weakest-grounded module** (see
  ``docs/CONSOLIDATION_MAP.md``): the workspace only *consumes* Vincent's
  operational inversion output (``inv_volume_mogi.dat``, rsynced from
  insar.vedur.is in ``volume_curve.py``) — the inversion itself is new
  work here and must be reconciled against Vincent's result before it
  ships (plan §11).
- Volume-rate helpers — ports of ``svartsengi_model.fitting.rate_to_m3s``/
  ``rate_from_m3s``/``time_for_rate`` (Phase 1–2).
- Physical parameter estimation — ``svartsengi_model.physics`` (Segall
  conduit-recharge + viscoelastic relaxation, Monte-Carlo importance
  sampling) is pure numpy and leaf-compatible **as-is**; it lands here or
  as a sibling ``physics`` module when the Svartsengi workflow moves over
  (Phase 2).
- ``okada_*`` — rectangular dislocation (Phase 2; the acknowledged theory
  gap — see vault note ``exponential-deformation-physics``).
- Joint GPS+InSAR inversion (Phase 2+, after the InSAR lane exists).

Geometry helpers may come from ``geofunc`` later; if so it joins the
dependency list explicitly — nothing else from the ecosystem does.
"""

__all__: list[str] = []
