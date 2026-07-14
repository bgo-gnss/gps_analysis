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

`models` · `fitting` · `velocity` · `baseline` · `preprocess` · `outliers` · `detrend` · `deformation` · `joint` · `transient`
— fill surfaces in place; don't rename modules without updating plan §10.2
(`preprocess` added by refactor-B slice 2, `joint` by the GPS+InSAR slice,
`outliers` by the outlier-detection slice, `detrend` by the detrend-leaf
slice — flag all four to plan §10.2 at next plan edit).
All math is atomic + referenced per [`docs/MATH_STANDARDS.md`](docs/MATH_STANDARDS.md) (binding).
Shared internal: `_mcmc` — the ONE GBIS Metropolis/annealing/adaptive-step core
(Bagnardi & Hooper 2018 §3; `T_SCHEDULE`, `sensitivity_schedule`, `metropolis`,
`PriorBounds`/`InversionConfig` homes) — `transient` (via fidelity hooks,
**bit-parity preserved**, pinned by `tests/test_mcmc.py`) and `deformation`
(`mogi_invert_bayes`, hook-free) both delegate to it; never re-duplicate the loop.

| Module | Status | Contents |
|---|---|---|
| `models` | ✅ implemented | `linear`/`periodic`/`lineperiodic`, `exp_linear`(+rate), `poly2`(+rate/peak), `heaviside_steps` (known-step term, H(0)=1), `TrajectoryParams` (+ `to_record`/`from_record` JSON-ready per-component serialization — full vector incl. intercept, upper-triangle covariance; the CSV 5-of-6 defect is dead) |
| `detrend` | ✅ implemented (2026-07-14, `docs/DESIGN_live_detrending.md` §0/§2/§5, detrend-leaf slice) | **Stored-parameter detrending — estimate once, apply anywhere**: `estimate_detrend` (window + §2.2 validity gates → `with_steps` step augmentation → `detect_outliers` BEFORE the fit (BGÓ hard rule) → final clean WLS; `detrend_method` tag `"step_augmented_robust"`/`"plain_wls"`, opaque `frame` passthrough, loud abort degrade), `DetrendEstimate.to_record` (self-contained station record: model code, step epochs, `fitted_at`/`frame`/`record_version`/`borrowed`/`refs` provenance), `trajectory_from_record`/`evaluate_record`/`apply_detrend` (pure apply at ANY epoch incl. future/borrowed-station; frame-mismatch guard; exactly invertible, raw never mutated), `select_terms` (secular/periodic views by coefficient+covariance zeroing; steps stay secular). Estimation caller (precompute/config/steps.csv/staleness) + `geo_dataread` delivery = later slices |
| `fitting` | ✅ implemented | `fit_components`, `detrend_fit`, `remove_trend`, `reject_outliers` (light exploratory clip — production path is `outliers.detect_outliers`), `with_steps` (step-augmented model factory; linear models keep the closed-form WLS path) |
| `baseline` | ✅ implemented | `slice_window`, `estimate_offset`/`remove_offset`, `estimate_step_offset` |
| `outliers` | ✅ implemented (2026-07-13, `docs/DESIGN_outlier_detection.md` §3–§4) | **Model-aware detection + signal protection**: atomics `mad_scale`/`qn_scale` (Qn = non-default O(N²) reference, Phase 1.5)/`whiten`/`standardize_robust`/`rolling_median`/`rolling_mad`/`hampel_mask`/`candidate_clusters`/`step_evidence`; `detect_outliers` orchestration — Huber step-augmented fit → global (k_g=5) + windowed-Hampel (k_w=4, time windows) identifiers → §3.4 protection (`PROTECT_FLOOR/RUN/STEP/WINDOW` + elevated-background arm) → conservative iteration with the >f_max **candidate**-fraction abort (all-False flags, loud). Always returns mask + `REASON_*`/`PROTECT_*` bitmasks + `SuspectedEvent` hints + fits; never filters, never mutates. `epoch_policy` per_component (default)/union. §8.3 release gate test-pinned (`tests/test_outliers.py`) |
| `preprocess` | ✅ implemented (refactor-B slice 2) | `screen_uncertainty` (formal-σ epoch screen), `prep_plot_series`/`prep_neu_series` — the two explicit legacy profiles (D1: plot/getData `iprep` vs .NEU/gamittoNEU `vshift`), bit-parity on live paths, geo_dataread goldens pin them end-to-end |
| `velocity` | ✅ implemented (WLS) | `estimate_velocity` (WLS + formal σ, `method="wls"`), `sliding_velocity`, magnitude/azimuth (+σ); `detectability_floor` stub |
| `transient` | ✅ ported (GBIS4TS), CI-parity green, **H3-optimized** | `bpd1/bpd2_forward`, `noise_covariance` (Williams 2003), `log_likelihood`, `run_inversion` (MCMC+annealing), `detect_breakpoints`. BPD1+BPD2 recover; **TS14 windowed parity matches MATLAB** (dv/tb; raw window — the leaf auto-conditions the ±5 mm-prior zero-reference contract, `y_ref` provenance). **H3 (2026-07-11):** hot loop uses exact O(N²) generalized-Schur likelihood (`_schur_logdet_quad`; C is NOT Toeplitz — Hosking 1981); 27.9× @ N=1825 (248.8→8.9 ms/sample; 1e6-run chain 2.9 d→2.5 h); parity ≤4e-12 in lnP; C/Rust settled: stay NumPy (PLAN-analysis-lane §3). **Full-fidelity `test_ts14_full_reference` PASSED 2026-07-11** (125 s on the Schur path; posterior optimum + 95% intervals inside SI Table S4 for v/dv/tb/κ/amp). Vendored MATLAB + map: `reference/gbis4ts/` |
| `deformation` | ✅ implemented (GPS-only) | Mogi/McTigue/Okada forwards + NLLS inversions (`mogi_invert`, `okada_invert`), Bayesian `mogi_invert_bayes` (shared `_mcmc` sampler), **distributed slip** (`discretize_fault` → `okada_greens` → `okada_invert_slip`: Laplacian-regularized ± non-negative linear inversion, `slip_lcurve`/`lcurve_corner` λ selection — Okada 1985; Harris & Segall 1987; Jónsson et al. 2002; Aster et al. 2018 ch. 4), ΔV↔ΔP/rate products. Joint GPS+InSAR → `joint` |
| `joint` | ✅ implemented (2026-07-12) | **Joint GPS+InSAR Mogi inversion** — `los_unit_vector`/`los_project` (ground→satellite ENU unit vector; **positive LOS = motion toward satellite/range decrease** — Hanssen 2001 ch. 2, Fialko et al. 2001 eq. 1), `InsarLos` leaf contract (downsampled points + σ or dense cov; quadtree/variogram = reader's job, Lohman & Simons 2005), per-track offset/linear-ramp nuisance (`ramp_design`, Bagnardi & Hooper 2018), `mogi_invert_joint` (whitened NLLS, analytic LOS-projected Jacobian) with **Helmert VCE** dataset weighting (`variance_components` — Sudhaus & Jónsson 2009; Koch 1999 ch. 3). **Depth–ΔV trade-off break test-pinned** (`test_joint.py` exit gate): 5-station synthetic — empirical σ_depth 986→252 m (×0.26) single descending track; formal σ_depth 523→215 m, cov(d,ΔV) ×0.14, ρ 0.972→0.917; asc+desc (Wright et al. 2004) tightens further (σ_depth 171 m). Bayesian joint variant on `_mcmc` = next slice; real InSAR via KITE reader (geo_dataread) = later |

> **Analysis-lane re-scope (2026-07-10, [`../PLAN-analysis-lane.md`](../PLAN-analysis-lane.md)):**
> `transient` (GBIS4TS) is **un-backburnered and ported** (H1). `velocity` ships **WLS**;
> the colored-noise/honest-σ upgrade is GBIS itself (`method="gbis"`, later slice).
> Base functions H4 + velocity H5 landed 2026-07-10. **Only `deformation` stays parked.**
> H1's full-fidelity `Verification/TS14` numerical parity gate **PASSED 2026-07-11**
> (`GPS_ANALYSIS_RUN_VERIFICATION=1`, now 125 s via H3's Schur path) — port confirmed
> vs SI Table S4. **Zero-reference input contract (Option B, 2026-07-11):** the leaf
> auto-conditions — `run_inversion`/`detect_breakpoints` fit `y − median(y[:30])`
> (`baseline_epochs`, 0 disables) against the fixed ±5 mm intercept prior and report
> intercepts back in the input frame (`InversionResult.y_ref`, the `velocity.t_ref`
> precedent); a saturation guard raises `ValueError` instead of returning a fit whose
> offset leaked into the trend. Shift-invariance + guard are test-pinned.
> **Seasonal-aware variant (#2, 2026-07-11):** new model codes `BPD1S`/`BPD2S`
> (`detect_breakpoints(..., seasonal=True)`, `run_inversion(model="BPD1S")`,
> `bpd1_seasonal_forward`) add annual+semiannual cos/sin terms (Blewitt & Lavallée 2002,
> mirrors `models.periodic`) between the trajectory and noise blocks and **co-estimate**
> them with break/rate/noise (joint, not pre-removal). Seasonal-blind BPD1/BPD2 stay
> byte-identical (TS14 optimum unchanged); H3 Schur likelihood + #1 conditioning inherited
> (seasonal intercept prior widened ±5→±(5+2√2·A_max)≈47 mm, sized so the guard
> never false-fires on legitimate high-amplitude vertical seasonal, since the median baseline
> carries the seasonal window-mean). Debiasing test-pinned: on a break+seasonal+colored-noise synthetic
> the blind rate-change error is ~57× the seasonal-aware one and the break is misplaced ~0.4 yr.

## Commands

```bash
uv sync --all-groups
uv run ruff check src tests && uv run black --check src tests
uv run mypy src tests && uv run pytest
```

- Python ≥3.13, hatchling, uv; ruff+black+mypy(strict) zero warnings.
- Home: **GitHub** (libs); CI: `.github/workflows/ci.yml`.

---
*Last reviewed: 2026-07-14 (NEW `detrend` module — stored-parameter detrending leaf per
`docs/DESIGN_live_detrending.md` §0 locked decisions: `estimate_detrend` (gates → steps →
`detect_outliers` → clean WLS, method tag + frame passthrough), record serialization
(`TrajectoryParams.to_record`/`from_record` + self-contained station records with borrowed
provenance), pure apply-by-record at any epoch, `select_terms`. Previous 2026-07-13: `outliers` module — model-aware outlier detection with signal protection per `docs/DESIGN_outlier_detection.md`: Hampel + global identifiers on studentized residuals of a Huber step-augmented trajectory fit, PROTECT_* bitmask protection stage, candidate-fraction abort, SuspectedEvent hints; `models.heaviside_steps` + `fitting.with_steps` added, `reject_outliers` demoted to exploratory. Previous 2026-07-12: `joint` module — joint GPS+InSAR Mogi inversion, Helmert-VCE weighting, depth–ΔV trade-off break test-pinned; shared `_mcmc` core; Okada distributed slip.)*
