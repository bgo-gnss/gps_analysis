# gps_analysis

General GNSS time-series analysis for the IMO 173-station network:
trajectory models, robust detrending, velocity estimation (WLS →
colored-noise MLE), and deformation-source inversion (Mogi → Okada → joint
GPS+InSAR).

Tier-1 **leaf math package** of the gpslibrary ecosystem — depends on
numpy/scipy/gtimes only, never on the I/O or operational tiers. It
consolidates the proven analysis code from `~/work/projects/gps_data_analyses`
(`svartsengi-model`, the `detrend-*` family) into small, pure, unit-agnostic,
tested functions.

**Status: Phase 0 scaffold.** The module map is in place; Phase 1 fills
`models` / `fitting` / `velocity` / `baseline` and the Mogi part of
`deformation` for the Reykjanes/Svartsengi vertical slice. Roadmap:
`gpslibrary_new/PLAN-postprocessing-revamp.md` (§10.2).

## Modules

| Module | Contents (planned) | Phase |
|---|---|---|
| `models` | `linear`, `periodic`, `lineperiodic`, `exp_linear`, `poly2`, `TrajectoryParams` | 1–2 |
| `fitting` | `fit_components`, `detrend_fit`, `remove_trend`, robust loss, `reject_outliers` | 1–2 |
| `velocity` | `estimate_velocity` (WLS → MLE), `sliding_velocity` | 1–2 |
| `baseline` | `estimate_offset`, `remove_offset`, `slice_window` | 1 |
| `deformation` | `mogi_forward`/`mogi_invert` → `okada_*` → joint | 1–2 |
| `transient` | GBIS4TS two-stage triage → Bayesian confirm | 2 |

## Development

```bash
uv sync --all-groups

uv run ruff check src tests
uv run black --check src tests
uv run mypy src tests
uv run pytest            # includes the leaf-dependency guard (CI-enforced)
```

## Hard rule

`gps_analysis` never imports `geo_dataread`, `gps_parser`, `tostools`,
`receivers`, `gps_plot` or `gps_api` — that would put a cycle in the tier
graph. `tests/test_leaf_guard.py` fails CI on any violation.
