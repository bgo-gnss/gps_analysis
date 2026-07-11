# Consolidation map — `gps_data_analyses` → `gps_analysis`

> Source→target mapping from a full survey of `~/work/projects/gps_data_analyses`
> (2026-07-08, Phase 0). This is the harvest plan for Phases 1–2; the plan doc
> (`gpslibrary_new/PLAN-postprocessing-revamp.md` §10.2) stays authoritative for scope.

## Where the good code lives

- **`svartsengi-model/`** — the only packaged, typed, partly-tested project
  (git submodule → `github.com/bennigo/svartsengi-model`). Its `src/` layout,
  Google-style docstrings, and ruff/mypy config are the **quality target**.
  Caveat: only `events.py` + `fitting.py` are tested (26 tests);
  `model.py` / `physics.py` / `cli.py` have **zero** coverage — do not treat
  their behavior as golden without characterization tests first.
- **`detrend-reykjanes/detrend_rnes.py`** — the most-developed legacy script;
  its `volcanic_model/` subpackage is the direct ancestor of svartsengi-model
  (byte-identical `model.py`/`events.py`).
- **The `detrend-*` family** (`-bb`, `-katla`, `-OLAC`, `-oraef`, `-tjorn`,
  bare `detrend/`, plus `pygmt_test/reykjanes.py`) — the same core functions
  copy-pasted verbatim into **7+ files**, differing only in hardcoded
  `stalist` and date windows. Highest-value dedup target.

## Source → target

| gps_analysis target | Source (actual names) | Notes |
|---|---|---|
| `models.linear/periodic/lineperiodic` | `line`/`periodic`/`lineperiodic` (`detrend_rnes.py`, ×7 copies) | secular + annual(2π) + semiannual(4π); `p1` = rate |
| `models.exp_linear` | `svartsengi_model.fitting.expf_long`/`expf_short` + `dexpf*` | keep derivatives (velocity of transient) |
| `models.poly2` | `polynomial_transient` + `dpolynomial_transient`, `polynomial_peak_time/value` | ValueError on `p2≈0` behavior worth keeping |
| `models.TrajectoryParams` | `FitParameters`/`LongTermParams`/`TransientParams` dataclasses (`svartsengi_model.model`) | merge into one typed container |
| `fitting.fit_components` | `fittimes` (legacy ×7) → `fit_curve`/`fit_three_components` (`svartsengi_model.fitting`) | note: `fit_three_components` missing from `__all__` upstream — fix at port |
| `fitting.detrend_fit/remove_trend` | legacy `detrend(...)` | drop its function-body imports + empty docstring habits |
| `fitting.reject_outliers` (Ph2) | `RunningMedian` (median pre-filter) + new robust-loss work | `least_squares(loss=…)` is net-new |
| `baseline.estimate_offset/remove_offset` | `find_exp_offsets`/`find_line_offsets` (`detrend_rnes.py`) | fit adjacent periods, difference at midpoint |
| `baseline.slice_window` | `dPeriod`/`vshift` usage patterns (gtimes) | pure-array replacement |
| `preprocess.screen_uncertainty`, `prep_plot_series`/`prep_neu_series` | `iprep`/`vshift` (`geo_dataread.gps_read`, refactor-B slice 2) | formal-σ screen + reference-shift; **two explicit consumer profiles kept per D1** (plot/getData vs .NEU/gamittoNEU) — bit-parity on live paths, pinned by geo_dataread goldens; mm conversion + refdate→yearf stay in geo_dataread shims |
| `deformation.rate_to_m3s/rate_from_m3s/time_for_rate`, `halflife_days` | `svartsengi_model.fitting` | trivial ports |
| `deformation.mogi_forward/invert` | **nothing** — workspace only rsyncs Vincent's `inv_volume_mogi.dat` output (`volume_curve.py`) | net-new; reconcile against Vincent (plan §11) |
| `deformation`/`physics` (Ph2) | `svartsengi_model.physics` (Segall + viscoelastic MC importance sampling) | pure numpy, **leaf-compatible as-is**; best-documented module in the workspace |
| `velocity.estimate_velocity` MLE, `sliding_velocity`, detectability floor | **nothing** | net-new (Hector/CATS-style, GBIS4TS) |
| `transient.*` (GBIS4TS) | **nothing** | net-new, port with the NVC authors |
| event timeline / quiet periods | `svartsengi_model.events` (`VolcanicEvent`, `SUNDHNUKUR_EVENTS`, `get_quiet_periods`) | data+logic; likely feeds the `analysis.yaml`/steps config, not this leaf |

Not harvested into this package: `tplt`/`gps_plot.timesmatplt` plotting calls
(→ gps_plot Phase 3), `getData`/`read_gps_data` I/O (→ geo_dataread),
`llh`/`xyzDict` pyproj transforms (→ geofunc territory), `getDetrFit`'s
`itrf08det` reader (**file I/O — belongs in geo_dataread**, but the fixed-width
format + its role seeding `lineperiodic` `p0` is worth preserving), RTK
`.pos`/baseline readers (`pygmt/functions.py`), Chow-test (`offset_test/`,
third-party educational copy), qc-xml (`flask/`, abandoned 2017 generateDS
experiment).

## Decisions recorded

1. **Array-first leaf (pandas stays out).** The most usable refactored API
   (`fit_dataframe`, `VolcanicDeformationModel`) is DataFrame-based, but this
   package's locked dependency policy is numpy/scipy/gtimes. Core functions
   take arrays (`(3, N)` N/E/U or per-component 1-D + `yearf`); a
   DataFrame-aware convenience layer, if wanted, lives in the precompute job
   or a thin adapter module *outside* the leaf. (pandas is transitively
   present via gtimes, but the API contract is arrays.) — *Revisit only if
   Phase 1 shows the adapters dominate the code.*
2. **"baseline" naming.** Three meanings exist in the workspace (reference
   offset/window; RTK rover–base vectors; inter-station vectors). This
   package's `baseline.py` = reference-offset/window only.
3. **New names only.** The `expf`/`secondorder` compatibility aliases (kept
   upstream to match `geo_dataread.gps_read` naming) do not migrate; shims
   live in `geo_dataread` during its refactor (plan §10.1), not here.
4. **Style to port**: svartsengi-model `src/` conventions (typed, Google
   docstrings, physical references in module docs). **Style to drop**: empty
   `""" """` docstrings, function-body imports, commented-out code blocks,
   hardcoded absolute paths, config-by-comment-toggling `main()`s.

---
*Created 2026-07-08 from the Phase 0 workspace survey. Update when functions actually move.*
