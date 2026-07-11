# CLAUDE.md — gps_analysis

Tier-1 **leaf math package** (plan §10.2): trajectory models, robust
detrending, velocity estimation (WLS → colored-noise MLE), deformation-source
inversion (Mogi → Okada → joint GPS+InSAR), GBIS4TS transient detection.
Consolidates `~/work/projects/gps_data_analyses` (`svartsengi-model`,
`detrend-*` family) into small, pure, unit-agnostic, tested functions.

> **Read first:** `../PLAN-postprocessing-revamp.md` — §10.2 (module plan),
> §6 (Phase 1 slice + the explicit NOT-in-Phase-1 list), §13 (risks R2/R6) —
> and `docs/CONSOLIDATION_MAP.md` (source→target harvest map from the
> gps_data_analyses survey + the recorded array-first/naming decisions).

## Hard rules

- **Leaf (R6):** deps are numpy/scipy/gtimes only (geofunc may join, explicitly).
  Never import geo_dataread/gps_parser/tostools/receivers/gps_plot/gps_api.
  Enforced by `tests/test_leaf_guard.py` in CI — keep that test alive.
- **Pure + unit-agnostic (R2):** functions take numeric arrays and callables;
  units, thresholds and I/O are the caller's business. No file reads, no
  logging config, no in-place mutation of inputs.
- **Consolidate, don't reinvent:** harvest from `gps_data_analyses`; golden
  behaviour is defined by the existing scripts until tests say otherwise.

## Module map

`models` · `fitting` · `velocity` · `baseline` · `deformation` · `transient`
— fill surfaces in place; don't rename modules without updating plan §10.2.
All math is atomic + referenced per [`docs/MATH_STANDARDS.md`](docs/MATH_STANDARDS.md) (binding).

| Module | Status | Contents |
|---|---|---|
| `models` | ✅ implemented | `linear`/`periodic`/`lineperiodic`, `exp_linear`(+rate), `poly2`(+rate/peak), `TrajectoryParams` |
| `fitting` | ✅ implemented | `fit_components`, `detrend_fit`, `remove_trend`, `reject_outliers` (robust) |
| `baseline` | ✅ implemented | `slice_window`, `estimate_offset`/`remove_offset`, `estimate_step_offset` |
| `velocity` | ✅ implemented (WLS) | `estimate_velocity` (WLS + formal σ, `method="wls"`), `sliding_velocity`, magnitude/azimuth (+σ); `detectability_floor` stub |
| `transient` | ✅ ported (GBIS4TS), CI-parity green | `bpd1/bpd2_forward`, `noise_covariance` (Williams 2003), `log_likelihood`, `run_inversion` (MCMC+annealing), `detect_breakpoints`. BPD1+BPD2 recover; **TS14 windowed parity matches MATLAB** (dv/tb; window zero-referenced — the upstream ±5 mm intercept prior demands inputs referenced near zero). Full-fidelity `test_ts14_full_reference` (~2.5 h) still **opt-in/unrun**. Vendored MATLAB + map: `reference/gbis4ts/` |
| `deformation` | ⏳ backburnered scaffold | Mogi→Okada→joint — parked (plan §9b); don't fill until revived |

> **Analysis-lane re-scope (2026-07-10, [`../PLAN-analysis-lane.md`](../PLAN-analysis-lane.md)):**
> `transient` (GBIS4TS) is **un-backburnered and ported** (H1). `velocity` ships **WLS**;
> the colored-noise/honest-σ upgrade is GBIS itself (`method="gbis"`, later slice).
> Base functions H4 + velocity H5 landed 2026-07-10. **Only `deformation` stays parked.**
> H1's full-fidelity `Verification/TS14` numerical parity gate is opt-in
> (`GPS_ANALYSIS_RUN_VERIFICATION=1`, ~2 h) — confirm before relying on `transient`.

## Commands

```bash
uv sync --all-groups
uv run ruff check src tests && uv run black --check src tests
uv run mypy src tests && uv run pytest
```

- Python ≥3.13, hatchling, uv; ruff+black+mypy(strict) zero warnings.
- Home: **GitHub** (libs); CI: `.github/workflows/ci.yml`.

---
*Last reviewed: 2026-07-11 (analysis lane: models/fitting/baseline/velocity(WLS)/transient(GBIS4TS) implemented; 132 fast tests green; transient test audit — no port bug, v assertions calibrated to exact GLS/profile-ML).*
