"""Velocity estimation for GNSS coordinate time series.

Planned surface (plan §10.2):

- ``estimate_velocity`` — fixed-window weighted least squares with formal
  uncertainties (Phase 1). The secular rate is the ``p1`` term of
  ``models.lineperiodic`` (the estimator every ``detrend-*`` script
  re-implements today); the same entry point is upgraded in place to
  colored-noise MLE (white + flicker/random-walk, Hector/CATS-style) in
  Phase 2 so callers never switch functions.
- ``sliding_velocity`` — sliding-window dynamic velocity series (Phase 2).
- Detectability floor — per-station velocity-change alarm threshold derived
  from the GBIS4TS noise model (Phase 2, plan §10.7).

Colored-noise MLE and the detectability floor have **no ancestor** in
``gps_data_analyses`` — net-new work, not extraction (consolidation map §6).
Honest uncertainties are the point: WLS formal sigmas are placeholders until
the MLE noise model lands; both are always reported with their method tag.
"""

__all__: list[str] = []
