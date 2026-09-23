# gps_analysis — package-wide code review, 2026-09-13

Six independent reviewers, one per coupled module cluster, over 17,256 LOC / 16 modules.
Branch `groups-block-donor-drift`. Baseline at review time: `704 passed, 1 skipped`
(the skip is the intentional hours-long `test_ts14_full_reference` H1 gate).
Read-only review; ~42 findings, almost all reproduced with numbers rather than asserted.

> **Status 2026-09-13:** the four branch blockers (§P2) are **implemented** —
> `staged.py` +295, `tests/test_staged.py` +357, 11 new tests of which 9 fail against the
> pre-fix source. `gps_analysis` 715 pass, `geo_dataread` 439, `gps_api` 144. Committed
> 2026-09-23 (`809dd25`).
>
> **Status 2026-09-23:** finding **#1 (datum leak) fixed** on both routes — gps_api
> `_borrowed_record` now calls the new leaf `reanchor_record` (datum re-anchored on the
> borrower's own series, donor steps dropped), and `estimate_staged` refuses a
> datum-carrying `HeldExplicit` unless `local_datum=True` (fail-closed). Tests assert
> `mean(y − model) ≈ 0` on a recipient 30 mm from the donor and a true 5 mm step recovered
> as 5 mm. No deployed record was affected (72 records, none borrowed; `use_sta: {}`).
> Everything else here is outstanding.

---

## The meta-finding: seven tests assert the report, not the behaviour

This is the most important result, because it explains how a green 704-test suite coexists
with six-figure numerical errors. In seven places a test observes the value the code
*reports* rather than the value it *used*:

| Test | What it asserts | What it therefore cannot see |
|---|---|---|
| `TestFreeze` (outliers) | the reported `scale_global` scalar | that the global identifier used a *different*, unfrozen scale |
| `test_fitting.py:558` | `isfinite(rates[isfinite(rates)])` — a tautology | that 2 of 4 sliding windows are NaN |
| `TestApplyOnlyStage._held()` | holds `TRUTH[SECULAR]`, the truth that generated `y` | donor datum ≡ recipient datum *by construction* |
| 570d1ba parametrization | that `poly_3`/`poly_4` *classify* | that they cannot be *fitted* |
| `detrend.py:429` writer + `:1151` reader | that writer and reader agree | that both compute the same wrong `param_names` |
| `noise.py:252-254` docstring | claims parity "test-pinned at rtol 1e-11" | no test references `_schur_whiten` at all |
| `test_chain_shapes_and_bounds` | the BPD1 box | 13,906/20,000 BPD2 samples outside their declared box |

**Recommendation:** for every diagnostic a record or result object exposes, add one test that
*reconstructs* it from raw inputs rather than reading it back. `det.z` must equal
`(w − center)/det.scale_global`; a borrowed record must satisfy `mean(y − model) ≈ 0` on the
*recipient's* data; `param_names` must come from the augmented callable's signature.

---

## Triage: wrong *now* vs. wrong *if called*

**Reaching production today**
- **#1** datum leak — `gps_api/precompute/detrend.py:161` confirmed open
- **#3** BPD2 freeze — `precompute/breaks.py:181-195` turns a frozen chain into z ≈ 6.8e15
- **#8** `freeze_scale` — *every* `detect_outliers` call runs with an unfrozen global identifier
- **#9** median baseline — refuses |v| ≳ 113 mm/yr, and Svartsengi stations are that fast

**Latent — real defects, not currently hit**
- **#2** shielded by `geo_dataread` calling `resolve_stage_plan` per component
- **#5** `job.py:364` passes string model names, so no stepped-model caller today
- **#6** only if someone fits degree ≥ 3 — which 570d1ba just made *advertised*
- **#4** only via `okada_invert` with free dip (but TRF drives straight into the bad region)

Spot-checked directly against source and confirmed: the `_COS_DIP_EPS` gate and its four
branches, the 1695-before-1712 ordering in `outliers.py`, and the zero call sites for
`trajectory_from_record` in `geo_dataread/src/`.

---

## P0 — wrong numbers on reachable production paths

### 1. Borrowed-secular datum leak, now with a leaf route that stores it
`staged.py` apply path; root cause `staged.py:127-129` + `detrend.py:181`

Holds are term-granular by contract, so holding `"secular"` necessarily transfers `offset` —
the intercept at t=0, ~10³ years from the data. The leaf does no re-anchoring.

Reproduced (and **re-verified after the P2 fixes landed** — untouched by them):
recipient datum 30 mm from the donor → `mean(y − model)` = **30.00 mm**; with `free=("step",)`
a **true 5.0 mm step is estimated as −25.00 mm** (5 − 30). The datum lands entirely in the
free term.

- The apply-only path is the first leaf route that *stores* an all-donor record, guardless.
- The 148 branch test lines cannot see it (see meta-finding).
- `geo_dataread` correctly refuses a cross-station hold without `anchor_t/anchor_y`.
  `gps_api/precompute/detrend.py:161 _borrowed_record` still deep-copies the donor verbatim
  with `terms: "all"` — **the open production leak.**

**Fix:** either give the leaf a datum-free hold (a rate-only secular vector with `offset`
explicitly freed), or make the apply path refuse a `secular` hold whose source is not marked
anchored. Add a test with donor datum ≠ recipient datum asserting `mean(y − model) ≈ 0`.

### 2. `HeldExplicit` broadcasts one donor vector across N, E and U
`staged.py` — `values[gm] = np.asarray(src.values)` inside `for c in range(n_components)`

`HeldExplicit.values` has no component axis. A (C, N) call applies the same coefficients to
every component: north's seasonal `[2.0, 1.5, 0.4, -0.3]` lands in the east fit (truth
`[-1.0, 0.5, 0.1, 0.2]`) and in up (truth `[6.0, -3.0, 1.0, 0.5]`). No error, no warning.
A (C, k) array raises a numpy `TypeError`, not a leaf `ValueError`. Survives only because
`geo_dataread` calls `resolve_stage_plan` per component and every existing test uses 1-D `y`.

### 3. BPD2 chain freezes on inflate→deflate, reported as infinitely significant
`transient.py:1294-1301` (guard), `:1435-1437` (seed relabel), `_mcmc.py:332`

The guard forces every *proposal* to satisfy `g1 <= g2`. `_preliminary_start` relabels breaks
ascending **in time**, so any accelerate-then-decelerate series seeds `g1 > g2`. Every
proposal is swapped into a catastrophically wrong model and rejected; the chain never leaves
the start.

- truth g1=+12@2019.3, g2=−8@2020.8 → **1 unique kept column out of 6000**, post-anneal std 1.8e-15
- `precompute/breaks.py:181-195` computes `|mean|/std` → **z ≈ 6.8e15**
- breaks 0.35 yr apart → 1/3600 unique post-anneal, z ≈ 5.6e15

Exactly the Svartsengi/Askja signal class. MATLAB behaves the same given such a start, but its
seeds were hand-supplied; the port generates the trapping seed itself.

**Parity-safe mitigation:** reject a BPD2 start with `g1 > g2`; raise when post-anneal
acceptance is 0. The real fix (order by `tb`, not `g`) must be opt-in — it changes chains.

### 4. Okada strike-slip/tensile kernels destroyed near vertical dip
`deformation.py:119` — `_COS_DIP_EPS = float(np.finfo(np.float64).eps)`, gating `:433/456/470/498`

The switch to Okada eq. (29) vertical forms fires only when cosδ < 2.2e-16, i.e. for
`dip == 90.0` exactly. The general-dip I-terms carry `1/cosδ`, amplifying cancellation error
without bound:

| dip | strike-slip rel. err | tensile rel. err |
|---|---|---|
| 90−1e-5° | 7.8e-2 | 3.1e-2 |
| 90−1e-6° | **5.6** | **3.1** |
| 90−1e-9° | **3.3e6** | **1.9e6** |

A vertical dike at `dip=89.9999999` returns `u = [3.26, 21.08, 0.030]`; `dip=90.0` gives
`[0.0169, 0.0060, 0.030]`. Finite, no warning, inside the documented domain `0 < dip ≤ 90`.

**Exposure:** `okada_invert` with free dip defaults `bounds["dip"] = (1e-3, 90.0)`; scipy TRF's
`make_strictly_feasible` keeps iterates strictly interior (rstep 1e-10 → dip = 90−9e-9°,
cosδ = 1.6e-10). Three test inversions recovered correctly — luck of the step sequence, not a
guard. Dip-slip is immune (kernels multiply by sinδ·cosδ).

**Fix:** one constant — threshold at cosδ ≲ 1e-6. `okada85.m` shares the weakness; parity with
the consulted reference is not a defence.

### 5. `estimate_velocity` zeroes step columns via time re-referencing
`velocity.py:838-855` — `t_local = t_win − t_ref` then `fit_components(model_func, t_local, …)`

`with_steps` builds indicators as `t ≥ t_k` on whatever `t` it is handed. With
`t_local ∈ [−3, 3]` and `t_k ≈ 2021` the column is identically zero, the design is
rank-deficient, and the min-norm solution is the un-stepped fit.

```
estimate_velocity(t, y, model=with_steps(lineperiodic, [2021.3]))
  -> rates [21.908]  sigmas [inf]  step_amp 0.0   + OptimizeWarning
fit_components(same model, absolute t)
  -> rate 11.937  sigma 0.049  step_amp 40.23
```

Rate wrong by **+10 mm/yr**. `estimate_velocity_mle` and `sliding_velocity` build on absolute
`tt` and are unaffected.

### 6. `Polynomial(degree ≥ 3)` fits are silently wrong
`terms.py:661-665` + `fitting.py:582`

`as_modelfunc` declares only `trend_column`/`intercept_column`, so only the rate column is
centered. `Polynomial.uncentering` exists — its docstring says "this term OWNS the centering"
— but is never used on the fit path.

| degree | cond | curve error | rms(fit−clean) vs reference |
|---|---|---|---|
| 2 | 9.9e11 | 2.5e-07 mm | 0.092 vs 0.092 (survives on SVD rcond) |
| 3 | 2.1e18 | **1.7 mm** | **0.63 vs 0.07** |
| 4 | 3.4e24 | **4.2 mm** | 1.15 vs 0.04 |

Semi-loud (`OptimizeWarning`, `cov_upper: None`) but `estimate_detrend(detect=False)` **still
writes the record**. 570d1ba deliberately made `poly_3`/`poly_4` classifiable — the algebra
now advertises a model the estimator cannot fit. Separate storage ceiling: absolute-t
coefficients lose 0.08 mm at degree 5 regardless.

---

## P1 — systematically wrong uncertainties and unreachable safeguards

### 7. VARPRO amplitude σ understated by up to 53 %
`varpro.py:486-497`, consumed at `terms.py:1175`, gating at `terms.py:1196-1198`

`H = [Φ_w, J_varpro]` borders with the derivative of the *concentrated* residual. At the
optimum `Φ_wᵀJ = −Dᵀr_w ≈ 0`, so H is block-orthogonal and the amplitude block collapses to
`ŝ²(Φ_wᵀΦ_w)⁻¹` — the covariance *conditional on* θ = θ̂. The full-problem Gauss–Newton column
is `D_w·ĉ`, and it is the `Φ_wᵀ(D_w ĉ)` coupling that carries the amplitude–τ trade-off.

400 realizations, N=1461 daily, τ=0.4 yr, σ=1.5:

| | c₀ | c₁ (rate) | c₂ (transient amp) | θ |
|---|---|---|---|---|
| Monte-Carlo empirical | 0.107 | 0.088 | 0.255 | 0.0333 |
| bordered (shipped) | 0.095 | **0.058** | **0.189** | 0.0331 |
| full GN `[Φ_w, D_w ĉ]` | 0.109 | 0.090 | 0.255 | 0.0331 |
| `curve_fit` full model | 0.110 | 0.089 | 0.252 | 0.0334 |

σ_θ is fine; the Jacobian is correct (FD 1e-8). The bug is in *what gets bordered*.
`TransientTauFit.covariance` inherits it and the `snr < 5` gate is judged on the understated σ.
Existing tests cannot see it — with a single θ-dependent column `Dᵀr_w = 0` exactly, so B ≡ 0.

### 8. `freeze_scale` does not freeze the global identifier
`outliers.py:1688-1695` vs `:1712-1715`

```python
z, center, s_global = standardize_robust(w, ...)   # 1688 -- FRESH scale
global_mask = np.abs(z) > params.global_n_sigma    # 1695 -- uses fresh z
...
if frozen is not None:                             # 1712 -- too late
    s_global, s_local_frozen = frozen
```

Only the local Hampel scale, thin-window fallback and D/B normaliser are frozen. The
`|ẑ| > k_g` identifier re-estimates every sweep — precisely the feedback the knob removes.
With `enable_window=False, enable_protection=False`, `freeze_scale=True` and `False` give
**byte-identical flags** on every series tried; `det.z ≠ (w − med w)/det.scale_global` by up
to 0.43 %. One-line fix: compute `z` after the `frozen` block.

### 9. `median(y[:30])` baseline biases the intercept; guard then refuses fast stations
`transient.py:1084-1123`, `:1241`, `:1328-1342`

`_start_baseline` is the median of 30 *raw* samples, so `r − y(t₀) ≈ v·14.5 d`. With the fixed
±5 mm prior and 0.5 mm margin, the guard fires for |v| ≳ 113 mm/yr on clean daily data.

- v=150 mm/yr → `ValueError: intercept optimum -4.991 mm saturates its prior`.
  Same series with `baseline_epochs=1` → v=150.02, dv=−60.01, tb=2020.498 (exact recovery).
- v=40 mm/yr with a 120-day gap after sample 5 → `median − y[0] = 15.7 mm` → `ValueError`.
- v=60 mm/yr passes but the intercept sits halfway to the bound; the truncated prior leaks in.

Docstring 1115-1116 ("~2 mm at 50 mm/yr") holds only for slow, gap-free starts.
**Fix:** `r = median(y[:k] − v̂·(t[:k] − t[0]))`, shift-invariant in v̂ so no circularity.
Changes `y_ref` for default runs → re-pin two tests; MATLAB parity (`baseline_epochs=0`) untouched.

### 10. Helmert VCE collapses on a low-redundancy track
`joint.py:495` (guards only `red <= 0`), `:822-828`

A 4-point track with a linear ramp has partial redundancy 0.988 — a single-dof χ² draw fed
back into the weights every iteration.

| n_k | VCE factors (GPS, track) | iterations |
|---|---|---|
| 4 | **[1.24, 0.049]** | **20 (= max, not converged)** |
| 5 | [1.23, 4.75] | 3 |
| 30 | [1.33, 3.98] | 3 |

The 4-point track ends with σ shrunk to 0.44 mm — up-weighted 20× against 4 mm data — while
`JointFit` reports `chi2_reduced=1.000`. The only signal is `vce_iterations == vce_max_iter`,
which the docstring never tells the caller to check.

**Related:** at `vce_max_iter` the reported `variance_components` and the reported covariance
describe *different* weightings (`s = s·factors` is computed, then the loop exits, so `res`
belongs to the pre-update `s`). Reported `s=[1.248, 102.7]` with `chi2_reduced=92.9`, vs
`[1.192, 101.9]` / 0.99 when allowed to converge.

**Fix:** add `vce_converged: bool` and per-group `red_k` to `JointFit`; skip or raise below a
documented minimum redundancy (Koch/Förstner ~3–5).

### 11. `sliding_velocity` NaNs every window that doesn't straddle a step
`velocity.py:1759-1788`, masked by `tests/test_fitting.py:538-560`

In a step-free window the step column is all-0 (rank-deficient) or all-1 (collinear with the
intercept); `_wls_solve` warns, `simplefilter("error")` demotes the window to NaN.

```
stepped model:  2019 NaN    2020 NaN    2021 12.26±0.25   2022 11.70±0.17
plain model:    2019 12.11  2020 11.88  2021 39.43        2022 24.07
```

No model gives a usable series. **Fix for 5+11 together:** drop step/indicator columns that
are constant over the window, keeping the rate slot by name.

---

## P2 — branch blockers ✅ IMPLEMENTED 2026-09-13

All four were in the uncommitted diff on `groups-block-donor-drift`. All four are now fixed in
`staged.py`, each pinned by a test that fails against the pre-fix source.

### 12. Apply-path regression: freeing a group the model lacks silently succeeded
`group_parameter_mask` returns all-False for a group with no term — documented as "the caller
should treat as an error". On main, `Stage("fit", ("transient",), held={secular, periodic})`
on `lineperiodic` reached `fit_held_partition` and raised "holds every column". With the
apply-only path, `free_mask` is empty → apply path → `held_covariance="applied"`, no error:
a plan asking to *fit* a transient produced a record indistinguishable from a deliberate
ELDC-style borrow.

**Fixed:** raises when any freed group resolves to zero columns. Checked **per group**, not on
the union, so a partly resolvable `free=("secular", "transient")` cannot quietly fit half of
what was asked. Pinned by `TestFreeingAnAbsentGroupIsRefused` (2 tests).

### 13. `evaluate_group_values` raised on `curvature` — *found independently by two lanes*
`probe = Polynomial(degree=max(max_degree, 1))` has names `("offset", "rate")`, so `curvature`
fell into the `# pragma: no cover` "unrecognised" branch — which was the live path. Also
order-dependent: `['poly_3','curvature']` worked, `['curvature','poly_3']` raised.

**Real consumer:** `geo_dataread/stage_plan.py:831-846` filters donor `param_names` through
`_staged_group_of`, so a degree-2 donor yields exactly `("offset","rate","curvature")` — the
borrow re-anchoring this function was written for raised on its most ordinary input.

**Fixed:** the probe is a fixed `Polynomial(degree=2)` — the widest *named* tier; every higher
monomial is `poly_<m>` and parsed separately, so the probe is complete and order-independent.
Pinned by `TestEvaluateGroupValuesCoversCurvature` (3 tests, incl. order-invariance).

### 14. `record_group_mask` ignored `record_version` and never validated `param_names`
`trajectory_from_record` enforces both; `record_group_mask` did not — and it is the reader the
borrow path actually uses (`geo_dataread` reaches a donor via `record_group_mask` →
`donor_group_values`, and has **zero** `trajectory_from_record` call sites). A record with
`record_version: 0` and `param_names` permuted to `["rate","offset",…]` returned a happy
`[T,T,F,…]` mask, and the positional `params[mask]` then borrowed `(12.5, 3.0)` as
`(offset, rate)`.

**Fixed, partially — read this before assuming #14 is closed:**
- `record_version` is enforced **when present**.
- The `param_names` head is checked against the model's positional names **when
  `len(param_names) >= len(model params)`** (a shorter list is a different malformation that
  `secular_from_record` diagnoses better as "no background to save").
- **Remaining gap:** a record carrying NO `record_version` key still passes. Closing it means
  updating 19 hand-written fixtures in `geo_dataread`, a cross-package change deliberately not
  made here. Production records always carry a version (`to_record` writes one).
- The gap is pinned as a *decision* by `test_a_record_without_a_version_is_still_tolerated`,
  so it cannot drift silently.

Pinned by `TestRecordGroupMaskEnforcesTheRecordContract` (4 tests).

### 15. `HeldFromStage` from a stage that never estimated the group composed zeros
A stage's vector is zero outside what it freed or held, and its covariance block for such a
group is zero rather than `None` — so `all(b is not None)` passed and the record labelled an
invented zero `held_covariance="propagated"`. The docstring had always promised such a plan is
"rejected up front rather than silently emitting a zero"; the ownership check could not see
it, because it counts a group as owned as soon as *any* stage frees it, without asking whether
the stage being borrowed *from* is that stage.

**Fixed:** validated before any fitting — a `HeldFromStage` must name a stage that actually
frees or holds that group. Pinned by `TestHeldFromStageMustNameAnEstimator` (1 negative + 1
positive control asserting a legitimately relayed coefficient is still a real estimate).

---

## P3 — correctness debt, worth a backlog entry each

**Detrend / records**
- Step admissibility admits **one-epoch flanks** (`detrend.py:927`, `:936-948`): two declared
  steps one day apart bracketing a single 200 mm blunder fit it exactly; `detect_outliers` sees
  a zero residual and does not flag it; amplitudes **+199.69 / −199.60** stored with finite
  covariance and no warning. The guard runs on `t_win` (pre-outlier) while the fit runs on
  inliers. *The `segments=` hull/gate interaction itself is sound* — a step in an excised gap
  recovers 7.01 ± 0.18 for a true 7.0.
- Duplicate `step_amp_1` in record `param_names` when a `Step` term sits in the spec alongside
  `step_epochs` (`detrend.py:429` / `:1151`) — `dict(zip(param_names, params))` silently drops
  a 10 mm step.
- Non-finite `HeldExplicit` values pass straight through → composed params full of NaN,
  detonating later at `from_record`.
- Two conventions for "no estimate" coexist: `cov_upper: None → +inf` vs apply-only `→ 0`.
  Zero propagates as *certainty* into any downstream σ.
- Provenance is spelled two ways in one record (`groups[g].provenance` bare vs
  `stage_plan[i].held[g]` prefixed `"explicit:"`); `geo_dataread` stamps `borrowed` only when
  *every* stage is apply-only, so a partial borrow serves `borrowed: None` while
  `groups.secular.provenance = "donor:SKSH@…"` — **the served provenance says "not borrowed".**
  Nothing flags a rate borrow across an incompatible pair (SENG/SKSH, 5.7 mm/yr E).

**Outliers** — *reconciled against `REVIEW_outlier_backlog.md`; several SHARPEN an existing entry*
- Aborted components lose their `SuspectedEvent`s (`outliers.py:2104`, `:2108-2109`) — an
  undeclared 80 mm step on N aborts N; `suspected_events` returns entries for E and U only.
  Zero hints on the component where "which step did I miss?" is the question.
- The §3.4.2 run rule is **verdict-inert at defaults**: over 175 mixed cases,
  `max_run_days=1e6, run_sign_fraction=1.0` changes 0/175 flag sets while setting 2,381
  `PROTECT_RUN` bits. And `max_run_days=2.0` sits on a floating-point boundary on a daily grid
  — `t[i+2]−t[i] > 2/365.25` is True at 2,156 positions and False at 1,842: a stored bitmask
  bit that depends on fp rounding. Give it 2.5 (`cluster_gap_days=1.5` has headroom).
  *Sharpens backlog #10*, which calls `run_sign_fraction` merely "unidentifiable" — it is
  logically dead at defaults.
- Leaf accepts N ≤ P: `lineperiodic` (P=6) at N=3…6 fits an underdetermined design and yields
  ŝ = 2e-15…7e-6 with |ẑ| up to 260; `linear` at N=3 aborts 20/20 with ŝ ≈ 1e-12 — pure
  `least_squares` tolerance reported as `excess_flag_abort` ("unmodeled signal").
  *Adjacent to backlog #4* (the 90-day identifiability case); this is the N ≲ 2P regime.
- `protect_windows` with NaN bounds are silently inert — `nan < nan` is False, so the bad
  window passes the ordering check and protects nothing. `step_epochs` gets "must be finite".
- Union breaks `flags ⊆ candidates` — *extends logged 11.1*: flagged epochs with `reasons == 0`;
  a `REASON_GROSS` despike on E propagates to N/U as flags with no reason. (Partial-abort union
  itself works as §12 specifies.)
- Documented `ValueError` on σ ≤ 0 is actually a `RuntimeWarning` then `LinAlgError`.
- *Not a defect, but noted:* the elevated-background arm is the last rule no threshold can
  release. A 100σ spike (ẑ = 98.4) on a 90-day +4σ plateau is `PROTECT_RUN`'d with D = 0.13;
  only `step_evidence_sigma ≥ 4.5` releases it, which also loosens the step rule. Test-pinned
  as intended, but §3.4.2b gave the *step* arm a magnitude-relative release and this arm has
  none while sharing its threshold. Own bit (`PROTECT_BACKGROUND`, backlog #7) + own threshold.

**Estimation core**
- Fixed-window lane returns a min-norm rate with σ = inf where the sliding lane demotes to NaN
  — two lanes, two degeneracy policies. `rates=[21.9]` with `sigmas=[inf]` reaches
  `VelocityEstimate` unchallenged.
- `fit_components` closed-form path doesn't validate `sigma` despite a comment saying it
  mirrors `check_finite`: `sigma[10]=0` → params NaN with only a RuntimeWarning;
  `sigma[10]=nan` → bare `LinAlgError`.
- `_rate_param_count` rejects `f(t, *p)` callables despite the documented contract.
- FD derivative evaluated outside `theta_bounds` (`varpro.py:371-374`), unlike `concentrated`.

**Deformation / joint**
- `lcurve_corner` has no "no corner" outcome — on a corner-less curve it returns an
  over-smoothed λ at ρ = 0.80 of ‖d_w‖ (77 % of signal unexplained). Robust with realistic
  noise; the docstring's "all collinear → index 1" claim is false in float.
- Laplacian edge condition is global (all four edges), so a surface-rupturing fault cannot have
  the top edge free — Jónsson et al. 2002, the cited source, does exactly that. Bottom-row slip
  0.498–0.798 (`edge="zero"`) vs 0.704–0.932 (`edge="free"`) against truth 1.0.
- McTigue guard admits a/d → 1⁻ where the O(ε⁶) expansion is meaningless: at a/d = 0.967 the
  "correction" is **+93 %**. (The docstring's 13 % at a/d = 0.5 is correct — measured 0.1291.)
- Non-negativity applies to *all* slip components; the docstring's "flip the sign at the call
  site" is impossible through the API for multi-component inversions.

**Transient / MCMC / noise**
- No finiteness / monotonic-`t` validation: one NaN in `y` → no exception, 1 unique chain
  column, `optimal == start`. Reversed `t` → garbage optimum (v=−2.99 for truth 5).
  `noise.estimate_noise_mle` *does* validate; `run_inversion` doesn't. Production input is
  `series.t[keep]` after an inlier mask with no explicit NaN drop.
- Index-grid covariance on gapped series (documented flag, unvalidated, exercised in
  production): 3-yr flicker with a 180-day outage → true-grid σ_v 0.433, index-grid 0.402
  (−7 %), `powerlaw_rate_sigma(n_obs)` 0.520 (+20 %) — opposite directions. Across the gap the
  index grid assigns cov 1.96 where truth is 0.63.
- BPD2 kept samples leave the declared prior box: reflection happens per-coordinate *before*
  the swap, so **13,906/20,000** samples have g2 outside its box (max 39.99 for [−3,6]).
  `prepare_bounds`'s SI-Table-S3 ranges are not the sampled prior for BPD2/BPD2S.
- `t_runs` colliding with the sensitivity schedule (`t_runs | 100k+2`: 3, 6, 7, 11, 13, 17, 34,
  51, 101…) re-fires the anneal branch on sweep passes, corrupting `prob_sens` and the step
  retune. Safe for 1000 and every value the tests use. Guarding with `sensitivity_test == 0` is
  bit-identical for all non-colliding `t_runs`.
- Step adaptation continues through the T=1 sampling phase (~98 retunes in a 1e6 chain) —
  non-diminishing adaptation, so the tail is not strictly ergodic. Cannot change without
  breaking parity; the module docstring should say "approximately".
- `_schur_logdet_quad` accepts κ=+0.5 and κ=−4 without complaint (MATH_STANDARDS §3 asks for guards).
- `detect_breakpoints` has no 0-vs-1-vs-2 criterion and defines an asymmetric, data-derived
  prior `(−ĝ, 2ĝ)` from an unpenalized min-RSS seed — double use of the data biases z upward
  for null stations.

---

## Verified clean (checked hard; worth not re-checking)

- **LOS sign convention** — S1 descending θ=39°, α=192°: `u = (+0.6156, −0.1308, +0.7771)`;
  uplift 10 mm → +7.771, east 10 mm → +6.156. Matches Fialko 2001 eq. (1) computed
  independently. Positive = toward satellite = range decrease, end to end. **The "unverified
  LOS sign" flag in the project notes is a reader-side question (which convention KITE/ISCE
  ships), not a leaf defect.**
- **Analytic LOS-projected Jacobian** — FD agreement 2.9e-10…8e-12 at the optimum across all 8
  columns; 1.6e-9…1e-11 at points perturbed 10–200 % off it.
- **Okada frame, strike/dip/rake signs, singular-point handling** — centroid→bottom-edge
  transform re-derived; eqs. (25)–(27) and I₁–I₅ match the paper line by line; Table 2 cases
  2–4 test-pinned.
- **McTigue coefficients** — a/d = 0.2 gives McTigue/Mogi − 1 = 8.3e-3 ≈ ε³. (Primary PDF
  blocked by Wiley after the full retrieval ladder + firecrawl stealth, so "eqs. 52–53" is
  verified against vmod, not the paper.)
- **χ²/dof scaling** — bit-for-bit consistent across `estimate_velocity`, `sliding_velocity`
  and `fit_components` on the same window (σ_v 0.1051961782706 in all three). VARPRO uses N−P−1.
- **Rate-slot-by-name (825be4d)** — every `_RATE_INDEX` read sits downstream of the by-name
  check. No unguarded positional read remains.
- **Classifier totality (570d1ba)** — enumerated `Polynomial(0..7)`, `Seasonal(1..5)`, `Step`,
  `LogTransient`, `ExpTransient` plus `with_steps` names: none unclassified.
- **Leaf guard** — no `importlib`/`__import__`/`TYPE_CHECKING`/`exec` anywhere in `src/`; the
  AST walk covers nested nodes; the `__init__` diff does not weaken it.
- **Serialization round-trip** — spec → JSON → `from_spec` bit-identical; `to_record`/
  `from_record` params and covariance bit-identical and exactly symmetric.
- **Column ordering** — deterministic stable sort; no dict/set iteration affects it.
- **`preprocess.py`** — parity traced line-by-line against the legacy `vshift` copy. Clean.
- **Step-flank fallback discriminator** — no off-by-one; a fast transient cannot be misread as
  a gap. False `PROTECT_STEP` on an isolated 30σ blunder: 0.5 % at 20-d cadence, 0 % daily.
- **MAD-implosion guards** — no inf/NaN ẑ reachable; exact-zero MAD returns zeros.
- **MCMC accept ratio / reflection / tempering** — symmetric kernel, correct `exp(Δ/T)`,
  tempering applied consistently to acceptance and the sensitivity ratio; sweep index
  bookkeeping matches `runInversion_ts.m` l.145-163 line for line.
- **Schur likelihood in-domain** — |Δ ln P| ≤ 1.1e-11 at N=1825, ≤ 5.7e-10 at N=7300. The
  "≤4e-12" claim is N=1825-specific but growth is benign.
- **`segments=` hull/gate interaction** — sound; no silent inadmissible-step admission.

---

## Cross-package items (outside the leaf, found while tracing)

- `gps_api/precompute/detrend.py:161 _borrowed_record` — deep-copies the donor verbatim with
  `terms: "all"`. The open production datum leak.
- `gps_api/precompute/breaks.py:181-195` — `|mean|/std` maps a point-mass chain to `inf`
  (P0 #3), and discards `min(16·t_runs, n_kept//2)`; the `n_kept//2` clamp lets tempered
  samples into short screens.
- `geo_dataread` imports three private names as de-facto cross-package API:
  `staged._staged_group_of` (2 files), `detrend._resolve_model`, `fitting._resolve_linear_design`.
- `staged.group_parameter_mask` / `record_group_mask` are in `staged.__all__`, documented as
  public, imported by `geo_dataread` **and** `gps_plot` — yet not re-exported at package level,
  while their sibling `evaluate_group_values` is.
- `tests/test_package.py:7 EXPECTED_MODULES` omits `staged`, `terms`, `varpro`, `noise`.
- 19 hand-written record fixtures in `geo_dataread/tests/` carry no `record_version` — the
  reason #14's fix is partial.

---

## Suggested order of remaining work

1. ~~Branch blockers #12–#15~~ — **done 2026-09-13.**
2. **Datum leak end to end:** #1 in the leaf + `gps_api/precompute/detrend.py:161`, with the
   donor≠recipient test that would have caught it.
3. **One-constant / one-line fixes with large blast radius:** #4 (`_COS_DIP_EPS` → 1e-6),
   #8 (move the `z` computation after the frozen block).
4. **Component-axis contract:** #2 — `HeldExplicit` takes a component axis or hard-refuses
   `n_components > 1`.
5. **Uncertainty correctness:** #7 (VARPRO bordering), #10 (VCE convergence flag + `red_k`).
6. **Time-frame contract in `velocity`:** #5 + #11 together.
7. **Transient robustness:** #3 (start-ordering rejection + zero-acceptance detector, both
   parity-safe), #9 (detrended baseline, re-pin two tests).
8. **`Polynomial` centering:** #6 — give `_LinearDesign` a full uncentering hook, or refuse
   degree ≥ 3 until it has one.
9. **Package-wide degenerate-input policy** — five different behaviours for bad input were
   found (raise / warn+NaN / silent NaN / abort-misdiagnosis / min-norm with σ=inf). Pick one.

## Tests worth pinning regardless

- Near-vertical Okada: `dip ∈ {89.9, 89.999, 89.99999}` vs `dip=90` at 1e-4 relative — fails today.
- FD-vs-analytic Jacobian for `_mogi_jacobian` and the joint closure (cheap, currently absent).
- One *fitting* test per polynomial degree the classifier accepts.
- `np.all(np.isfinite(rates))` replacing the tautology at `test_fitting.py:558`.
- A borrowed-record test with donor datum ≠ recipient datum asserting `mean(y − model) ≈ 0`.
- `_schur_whiten` vs `_schur_logdet_quad` (m=1) — the parity the `noise.py` docstring claims.
- A BPD2 box test asserting the *actual* invariant (sorted image of box₁×box₂).
- `classify_cluster(span, sign_fraction, D, B, params) -> (bits, kind)` extracted as a pure
  function so the protection truth table can be table-tested; 13-arg `_protect_component` cannot.
