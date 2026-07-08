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

## Module map (stable since Phase 0)

`models` · `fitting` · `velocity` · `baseline` · `deformation` · `transient`
— planned surfaces are in each module's docstring; fill them in place, don't
rename modules without updating plan §10.2.

## Commands

```bash
uv sync --all-groups
uv run ruff check src tests && uv run black --check src tests
uv run mypy src tests && uv run pytest
```

- Python ≥3.13, hatchling, uv; ruff+black+mypy(strict) zero warnings.
- Home: **GitHub** (libs); CI: `.github/workflows/ci.yml`.

---
*Last reviewed: 2026-07-08*
