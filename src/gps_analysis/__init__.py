"""gps_analysis — general GNSS time-series analysis for the IMO network.

Leaf math package (Tier 1) of the gpslibrary ecosystem: trajectory models,
robust detrending, velocity estimation (WLS → colored-noise MLE), baseline
utilities, and deformation-source inversion (Mogi → Okada → joint GPS+InSAR).

Consolidates the proven analysis code from ``~/work/projects/gps_data_analyses``
(``svartsengi-model``, the ``detrend-*`` family) into small, pure, unit-agnostic,
tested functions. Module plan: PLAN-postprocessing-revamp.md §10.2 in the
gpslibrary_new collection.

Dependency policy (hard rule, plan risk R6): numpy / scipy / gtimes only —
this package must never import geo_dataread, gps_parser, tostools, receivers,
gps_plot or gps_api.
"""

__version__ = "0.1.0"
