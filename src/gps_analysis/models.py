"""Trajectory models for GNSS coordinate time series.

Planned surface (plan §10.2), grounded in real code from
``~/work/projects/gps_data_analyses`` (see ``docs/CONSOLIDATION_MAP.md``):

- ``linear``, ``periodic``, ``lineperiodic`` — secular rate plus
  annual (2π) / semiannual (4π) cosine+sine terms. Port of the
  ``line``/``periodic``/``lineperiodic`` trio from ``detrend_rnes.py``,
  copy-pasted verbatim across 7+ files of the ``detrend-*`` family —
  consolidating them here is the single highest-value target (Phase 1).
- ``exp_linear`` — postseismic / transient exponential relaxation. Port of
  ``svartsengi_model.fitting.expf_long`` / ``expf_short`` (+ derivatives
  ``dexpf``/``dexpf_short``) (Phase 2).
- ``poly2`` — quadratic transient, port of ``polynomial_transient`` (+
  ``polynomial_peak_time``/``polynomial_peak_value``/``halflife_days``)
  (Phase 2).
- ``TrajectoryParams`` — typed parameter container shared by
  :mod:`gps_analysis.fitting` and :mod:`gps_analysis.velocity`; descendant
  of the ``FitParameters``/``LongTermParams``/``TransientParams``
  dataclasses in ``svartsengi_model.model``.

All functions are pure and unit-agnostic: they evaluate model values on
numeric time arrays (fractional years by convention, ``yearf``) in whatever
unit the caller supplies, and never do I/O. The legacy ``expf``/
``secondorder`` compatibility aliases stay in the old code — new names only
here.
"""

__all__: list[str] = []
