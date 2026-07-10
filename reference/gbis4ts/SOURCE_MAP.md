# GBIS4TS MATLAB → `gps_analysis.transient` source map

> Vendored 2026-07-10 from [osf.io/q35tw](https://osf.io/q35tw/overview) (Yang,
> Sigmundsson, Geirsson 2023 — *2023GL103432*). License **BSD-2** (see
> `GBIS4TS_copyright.txt`, `GBIS_copyright.txt` — keep them). GBIS4TS is a
> time-series fork of GBIS (Bagnardi & Hooper 2018). This maps each `.m` to the
> Python port target in `src/gps_analysis/transient.py` (task **H1/H2/H3**).

## The algorithm in one screen

Per station, per component (N/E/U), fit a **piecewise-linear trajectory with 1–2
velocity break points** under a **power-law + white colored-noise** model, by
**Metropolis-Hastings MCMC + simulated annealing**, returning posterior samples
(hence honest uncertainties on the rate change and the break epoch).

- **Model params** — `BPD1`: `[intercept, trend1, trend_change, breakpoint, kappa, amp]`;
  `BPD2` inserts a 2nd break: `[intercept, trend1, trend_change1, breakpoint1,
  trend_change2, breakpoint2, kappa, amp]`. A **hyperparameter** is appended
  (`prepareModel_ts` lines 120–124) but is effectively fixed to 1 in the current code.
- `kappa` = power-law spectral index; `amp` = power-law amplitude; `wn_amp` = fixed
  white-noise amplitude (read from `WNlist`, per station, NOT inverted for).

## File → port target

| MATLAB file | Role | Python target |
|---|---|---|
| `GBISrun_ts.m` | driver: read lists, per-station loop, save `.mat` | the precompute caller (NOT the leaf) — `transient.detect_breakpoints` orchestrates one station |
| `PrepareData/ts_rd.m` | read `time N sigma` (skip 1 header line) | `geo_dataread` I/O — NOT the leaf; tests read `Verification/TS14.txt` directly |
| `PrepareData/prepareModel_ts.m` | build start `m`, `step`, `lower`, `upper`, param names; **search ranges = Table S2** | `PriorBounds` + `InversionConfig` construction in `transient.py` |
| `DeformationSources/ts/BPD1.m` | forward model, 1 break | `bpd1_forward(params, t)` |
| `DeformationSources/ts/BPD2.m` | forward model, 2 breaks | `bpd2_forward(params, t)` |
| `DeformationSources/ts/UniVarMatrix.m` | Williams-2003 var-cov: `wn²·I + amp²·(T·Tᵀ)`, `T` lower-tri **Toeplitz** | `noise_covariance(n, wn_amp, kappa, pln_amp)` — **H3 Toeplitz/Cholesky target** |
| `logdet.m` | `2·Σ log diag(chol(A))` | `numpy`/`scipy.linalg.cho_factor` — the per-sample O(N³) hotspot |
| `PrepareData/runInversion_ts.m` | **MCMC core**: annealing schedule, likelihood, accept/reject, adaptive step (sensitivity test → 77% rejection), bounds reflection, BPD2 break-ordering | `run_inversion(t, y, wn_amp, config, model)` |
| `Variogram/*` (`variogram.m`, `fitVariogram.m`, `variogramfit.m`, `FMINSEARCHBND/`) | empirical estimation of `wn_amp`, `kappa`, `amp` from residuals (pre-step, feeds `WNlist`/start params) | later slice — `noise.estimate_variogram(...)`; NOT required for the first draft (Verification provides `wnList`/`startPara`) |
| `PrepareData/quadtree*.m` | spatial (InSAR) downsampling — **not used** in the TS path | skip |

## Exact equations to preserve (fidelity gates)

- **BPD1 forward** (`BPD1.m` l.15, continuity-preserving ramp):
  `y = m1 − m2·t₀ + m2·t + m3·H(t−m4)·t − m3·H(t−m4)·t*` where `t* = first t ≥ m4`,
  `H` is Heaviside with **`H(0)=1`** (`sympref('HeavisideAtOrigin',1)`). BPD2 adds the
  analogous 2nd-break terms with `m5,m6`.
- **Covariance** (`UniVarMatrix.m`): recursion `b₁=1`, `bᵢ = ((i−2−kk)/(i−1))·bᵢ₋₁`,
  `kk = kappa/2`; `T` is lower-triangular with column-shifted `b` (**Toeplitz**);
  `T1 = (1/365)^(−kappa/4)·T`; `Cov = wn_amp²·I + amp²·(T1·T1ᵀ)`. *(The commented CATS /
  Williams-2008 angle form is disabled upstream — do not port it; note it exists.)*
- **Log-likelihood** (`runInversion_ts.m` l.132):
  `P = −resExp/2 − logdet(Cov)/2 − nObs·log(2π)/2`, `resExp = rᵀ·Cov⁻¹·r`, `r = y − model`.
- **Accept ratio** (l.138): `PRatio = (hyperPrev/hyperParam)^(nObs/2)·exp((P−PPrev)/T)`;
  accept if `PRatio ≥ rand()`. `T` from annealing schedule `10.^(3:-0.2:0)`, `TRuns=1000`.
- **Adaptive step** (l.145–163): sensitivity test targets **77% rejection**; breakpoint
  step floored at `0.0027` yr (~1 day). Bounds handled by **reflection** (l.249–253).

## ⚠️ Fidelity-vs-bug flags (raise with the NVC authors, preserve first)

- `runInversion_ts.m` l.258/264: the BPD2 break-ordering guard compares
  `trial(3)` vs `trial(5)` — those indices are **TrendChange1/TrendChange2**, not the
  breakpoints (`trial(4)/trial(6)`). Reproduce **as-is** for numerical parity against
  `Verification/`, but flag it — likely an index slip.
- `invCov = Cov^(-1)` (l.113) is an explicit inverse each sample → replace with a
  Cholesky solve (`cho_factor`/`cho_solve`) that also yields `logdet` for free (H3).

## Acceptance harness (`Verification/`)
`tsList` → `TS14.txt` (one station, `time N sigma`), `wnList` → white-noise amp,
`startPara` → start params. **H1 exit gate:** a Python `BPD1` run reproduces the MATLAB
posterior (optimal params + marginal spreads) on `TS14` within MCMC tolerance.

---
*Read 2026-07-10 from the vendored tree. Big PDFs (`SupportingInformation.pdf`,
paper PDF) hold Table S2 priors + methods — `.gitignore`d to keep the repo light.*
