"""Model fitting and detrending for GNSS coordinate time series.

Planned surface (plan §10.2), grounded in ``gps_data_analyses`` (see
``docs/CONSOLIDATION_MAP.md``):

- ``fit_components`` — per-component least-squares fit of a trajectory
  model (Phase 1). Clean rewrite of the legacy ``fittimes`` loop
  (``detrend_rnes.py``, duplicated across the ``detrend-*`` family) whose
  half-refactored descendants are ``svartsengi_model.fitting.fit_curve`` /
  ``fit_three_components``.
- ``detrend_fit`` / ``remove_trend`` — fit-then-subtract convenience layer
  (Phase 1); replaces the legacy ``detrend(...)`` (the one with the empty
  docstring and the function-body imports).
- Robust fitting — ``scipy.optimize.least_squares`` with ``soft_l1``/
  ``huber`` loss plus ``reject_outliers`` iterative rejection (Phase 2);
  the legacy running-median filter (``RunningMedian``) may survive as a
  pre-filter helper.
- Step handling — equipment steps (TOS session boundaries) and earthquake
  steps (skjálftalísa catalog) enter as fixed epochs in the design matrix
  (Phase 2); the step catalog itself is produced *outside* this package.

API convention (decision recorded in the consolidation map): the core is
**array-first** — inputs are numeric arrays (``(3, N)`` N/E/U or per-
component 1-D) plus model callables from :mod:`gps_analysis.models`. The
pandas ``DataFrame`` convenience layer (``fit_dataframe``,
``VolcanicDeformationModel``) stays *outside* the leaf. Units and
thresholds are the caller's business; no I/O, no logging config, no
in-place mutation.
"""

__all__: list[str] = []
