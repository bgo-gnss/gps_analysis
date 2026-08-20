# REVIEW — Outlier detection backlog (unactioned)

> **Status:** UNACTIONED BACKLOG REVIEW, dated 2026-07-28. Critical technical
> review of `DESIGN_outlier_detection.md` (incl. §10/§11 addenda) and
> `src/gps_analysis/outliers.py` (2 045 LOC). Findings are ranked by severity
> and are deliberately **not** softened for implementability — they feed a
> backlog, not a sprint. All "measured" numbers were produced 2026-07-28 on
> live data via `geo_dataread.gps_read.getData(sta, ref="plate")`,
> `lineperiodic`, default `OutlierParams()` unless stated. Experiment scripts:
> session scratchpad (`exp1_rhof.py`, `exp2_more.py`).

---

## 1. CRITICAL — Whitening by formal σ suppresses exactly the epochs the detector exists to catch

**The central design error candidate.** §3.1 asserts formal σ contributes the
"epoch-to-epoch quality *ratio*". The data say it does not — and worse, that
the σ inflation at bad epochs *cancels* the residual inflation, because both
are outputs of the same daily estimation. The ratio r/σ is approximately
invariant to how bad the solution was. Whitening is close to circular for the
gross-blunder class.

**Measured (RHOF, uncert=100, N=4738):**

- The known 2023-02-28 excursion, t=2023.1603: **U r=+83.1 mm**, σ=13.2 mm
  (×4.1 median) → whitened z=+3.32, **not even a candidate**; unwhitened
  z=+12.9 → flagged. Same epoch **E r=+29.8 mm** → z=2.21 whitened vs 13.0
  unwhitened.
- Fleet-direction: `sigma=None` raises flags **+28 %** (uncert=10: 99→127) to
  **+55 %** (uncert=100: 139→216) on RHOF.
- σ is nearly uninformative about epoch quality: Spearman(σᵢ, |rᵢ|) =
  **−0.06 / +0.24 / +0.12** (N/E/U, RHOF full span).
- Whitening fails its own stated purpose (homogenising quality across
  seasons): CV of monthly MAD, raw → whitened: N 0.101→**0.134 (worse)**,
  E 0.165→0.106, U 0.085→0.079 (marginal).
- Tail suppression: among the 30 largest-|r| epochs on U, **5** exceed
  |z|>5 unwhitened, **0** whitened.

**Interaction defect:** the upstream `screen_uncertainty` (uncert=10) would
have removed the σ=13.2 mm epoch; with any looser screen the epoch survives
*and* whitening then shields it. The two stages assume each other's behavior
without a contract: whitening is only defensible for epochs the σ-screen
already vouched for, and the screen threshold is set independently.

**Alternatives (concrete):**
1. **Clipped whitening**: `w_i = r_i / min(σ_i, c·med(σ))` with c ≈ 1.5–2 —
   keeps the seasonal quality ratio, removes the blunder-epoch self-pardon.
2. **Dual statistic**: candidate iff `max(|z_whitened|, |z_raw|) > k` — one
   extra MAD, catches both failure directions.
3. Drop whitening from the *identifiers* entirely (keep σ in the fit
   weights), and let `screen_uncertainty` own the σ dimension — arguably the
   cleanest separation of concerns, and the +28–55 % measured recall gain.
Any of these needs the §8.2 `test_sigma_weighting` scenario re-examined: as
written that test enshrines the failure mode (large σ + large r ⇒ not
flagged) as the desired behavior.

---

## 2. HIGH — The abort rule is non-monotone, has component-level blast radius, and is "loud" only in principle

**Measured non-monotonicity (RHOF, k_w = `window_n_sigma` sweep):**
5.0→62 flags, 4.0→99, 3.5→156, 3.0→252, 2.5→414, **2.0→0** (candidate
fraction 0.065 > 0.05 ⇒ abort). Loosening a threshold discontinuously drops
the output to zero, and **0 flags is indistinguishable from "clean series"**
in every consumer that only looks at the mask.

**Measured blast radius (SAUD, defaults):** per-component candidate fractions
**[0.100, 0.009, 0.006]** — only N is pathological, but the abort zeroes
**all three** components (flags=[0,0,0]). E and U lose perfectly reasonable
cleaning because N has an unmodeled-signal problem. The spec's own §3.5
wording ("per-component candidate fraction") reads as if the guard were
per-component; the action is global. GFUM aborts the same way
([0.077, 0.045, 0.092]).

**Loudness in practice:** `detrend.estimate_detrend` (the production caller)
converts abort into a `RuntimeWarning` + plain-WLS fallback
(`detrend.py:694`) — invisible in a rendered figure or a stored record unless
the consumer explicitly checks. "Do nothing loudly" is the right principle;
the current implementation is *do nothing, whisper*.

**Is 5 % defensible?** As a tripwire for "wrong model", directionally yes
(SAUD-N at 10 % genuinely has unmodeled signal). But it is a cliff on a
continuous quantity, uncalibrated against any measured fleet distribution,
and the candidate fraction is itself a function of every other threshold —
so f_max=0.05 means a different thing at k_w=4 than at k_w=3. The measured
RHOF sweep crosses it purely by turning one knob.

**Alternatives:** (a) per-component abort — zero only the offending
component; (b) monotone degrade instead of cliff: on excess candidates fall
back to the *global identifier only* at k_g (gross blunders still caught,
Hampel disabled), stamped `degraded=True`; (c) make abort a first-class,
unmissable output: plot annotation, API provenance field, non-zero exit
class in precompute — not a warning; (d) hysteresis or a top-N cap
(flag only the f_max·N largest |z|) if any masking is better than none.

---

## 3. HIGH — The spec-pinned idempotence invariant does not hold on real data

Spec §3.5: "running detection on a series with its flags applied must
reproduce the same mask (test-pinned, §8)".

**Measured:** RHOF full span converges (iter=3, converged=True, 99 flags);
re-running detection on its inlier subset produces **16 new flags** (expected
0). Mechanism: removing 99 epochs shrinks the MAD and shifts the fit → new
exceedances. The §8.4 `test_idempotent` evidently holds on synthetics only.

**Consequence:** the requirement-5 iterative workflow (clean → refit →
re-detect) is *divergent* on real series — a slow-motion masking cascade
across pipeline runs, exactly what the abort rule was designed to prevent
within one call. Combined with finding 4 (window dependence) the flag set is
not a stable property of the data; it is a property of (data × invocation
history × window).

**Also measured:** HOFN and AKUR return `converged=False` at the iteration
cap and their non-fixed-point flags are served with no downstream
distinction whatsoever. Nothing in the store schema or provenance marks a
non-converged mask.

**Alternatives:** freeze ŝ (per component) on the first sweep and reuse it on
re-entry (scale pinning removes the shrinkage feedback); pin idempotence
tests on real-archive fixtures, not synthetics; propagate
`converged`/`n_iterations` into stored provenance and treat
`converged=False` as a reviewable state; or honestly demote the spec claim
to "approximately idempotent" with a measured bound.

---

## 4. HIGH — Verdicts are a function of the analysis window, and the pipeline bakes that in

**Measured (RHOF U):** flags from 1-yr sliding windows vs the full span
agree at **Jaccard 0.50** (9 both / 5 full-only / 4 window-only) — half the
verdicts change with the analysis window at identical parameters.

**Mechanism, measured:** `scale_global` = **1.91** at full span vs
**0.88–1.00** at 90 d/1 yr/3 yr around the same epoch — the full-span MAD of
whitened residuals is inflated ≈2× by 25 years of σ-calibration drift
(GAMIT eras, antenna changes), so the global identifier is effectively 2×
looser at full span than windowed. Additionally, `lineperiodic` on a 90-d
window is unidentifiable (annual + semiannual terms fitted to a quarter
period) and the code fits it without complaint — the z at the probe epoch
swung 1.66↔4.93 across windows.

**Pipeline consequence:** `detrend.estimate_detrend` runs detection on the
*requested window* (detrend.py:678, after windowing), so the stored
`DetrendEstimate` for the same station differs by window choice, and the
§11 caveat about mixed-vintage records applies to every window change too.

**Alternatives:** detect **once per station on the full archive** in
precompute, persist the mask, and make every windowed consumer slice the
persisted flags (the store schema §5.2 already supports exactly this);
if windowed detection must exist, (a) estimate ŝ on a fixed-length trailing
context rather than the window, (b) refuse (or degrade the model for)
windows shorter than the model's identifiability span, (c) handle σ-era
heteroscedasticity explicitly (per-era scale, or robust scale on
era-detrended w).

---

## 5. MEDIUM-HIGH — No physical magnitude trigger: `min_outlier` is one-sided by design

`min_outlier` only ever *protects* (`|r| < a_min` unflags,
`_protect_component` line 1636). Every route to a flag passes through the
z-statistics — there is no rule "a residual this large is always at least a
candidate". This is why finding 1 is unrecoverable in-band: an 83 mm
excursion with a co-inflated σ has no non-z path into the candidate set.

**Alternative:** a physical magnitude *ceiling* `a_max` (config, e.g.
50 mm horizontal / 100 mm vertical) that forces **candidacy** — not flagging
— above it, regardless of z. Because it only creates candidates, the full
protection stage still applies (a real 200 mm coseismic step is still
step-protected), so it cannot clip signal that the current rules would keep.
It is the exact dual of the floor, closes the whitening hole as
defense-in-depth, and costs one comparison. The absence of the dual while
the floor exists is an asymmetry the spec never argues for.

---

## 6. MEDIUM — `f_scale=1.0` is not scale-adaptive: the "Huber" fit is effectively L1, and the σ=None path measurably breaks unit-agnosticism

**Measured:** MAD of whitened residuals (RHOF) = 1.33 / 2.51 / 1.93 (N/E/U),
so the Huber transition at 1.0 sits at **0.40–0.75× the actual noise scale**
— well inside the core, i.e. the fit is essentially L1 (robust, but
inefficient, and not the documented 95 %-efficiency Huber regime; the
canonical tuning is 1.345·σ).

**Measured contract violation:** with `sigma=None`, scaling y ×1000
(mm→µm) changes the mask — **127 vs 124 flags, 7 differing epochs** (RHOF).
The module header claims "pure and unit-agnostic"; §8.4 `test_invariances`
only covers the *joint* (y, σ, floors) scaling where w is invariant by
construction. The unwhitened path — which finding 1 recommends using
*more* — is not unit-agnostic because `f_scale=1.0` is an absolute constant
in whatever unit w happens to carry.

**Alternative:** `f_scale = 1.345 · mad_scale(seed residuals)` computed per
fit (the seed WLS residuals are already available in
`_component_candidates`). One line, restores both the Huber regime and unit
invariance.

---

## 7. MEDIUM — The protection stage is an entangled rule bundle: not separable, not attributable, and diverging from the spec

Six interacting arms resolve inside one ~90-line loop
(`_protect_component`): floor, operator window, run rule, step rule
(determinate + indeterminate policy), elevated-background arm,
conclusive-blunder release, magnitude-ratio release. Specific defects:

- **Bit conflation:** the elevated-background arm sets `PROTECT_RUN` — the
  same bit as the span+sign run rule (line 1719-1720). Two rules with
  different rationales are indistinguishable in diagnostics, in stored
  provenance, and in the `--outlier-param` isolation lane that §11 says
  motivated `protect_on_indeterminate`. The background arm also exists only
  in code+docstring; spec §3.4 derives it as part of the blunder-release
  precedence, not as a standalone protection that fires with `run_rule`
  False (e.g. a *single* candidate with elevated flanks gets PROTECT_RUN —
  span 0 < L_max, no run in any meaningful sense).
- **Spec/code divergence on D's input:** spec §3.4.2/§4.1 define the flank
  medians on **r** (raw residuals); the orchestration computes them on **w**
  (whitened) via an *inlined duplicate* of the flank logic
  (lines 1651–1667) instead of calling the public `step_evidence(t, r, …)`
  primitive — which takes r. The duplication already realized the
  divergence it risks. Under seasonal σ, a constant-amplitude step has
  season-dependent D in w — the verdict depends on when the step happened.
- **Noisy verdict statistic:** with the §3.4.2a fallback, D can rest on two
  3-sample medians (std ≈ 0.9·ŝ under white noise) yet gates three distinct
  decisions (protect, run-release, background). The spec admits this
  ("materially noisier") but still defaults the fallback ON
  (`step_flank_max_reach_days=60`) while the §11 measured impact (+10.8 %
  flags, 14 stations up / 1 down) was observed on 20 stations without a
  ground-truth comparison — the +10.8 % is asserted as improvement, measured
  only as *change*.
- **Testability:** the arms are only testable through `detect_outliers`
  end-to-end; there is no pure function
  `classify_cluster(cluster, flanks, params) → (protect_bits, kind)` that
  the six rules could be table-tested against.

**Alternatives:** give the background arm its own bit (`PROTECT_BACKGROUND`);
extract cluster classification into a pure, exhaustively table-testable
function; make the orchestration call `step_evidence` (deleting the inline
copy) and decide explicitly — with a stated reason — whether D lives in r or
w; record per-cluster D, B, span, sign-fraction in `SuspectedEvent` (only D
is kept today) so operators can audit which arm fired and why.

---

## 8. MEDIUM — Scale-floor and thin-window fallback inconsistencies

- **`scale_floor` defaults to 0.0** while spec §3.3 states "`s_floor > 0`
  guards the well-known Hampel degeneracy (≥ 50 % of a window identical ⇒
  MAD = 0 flags everything)". The shipped default disables the guard the
  spec calls essential. NEU products are quantized (0.1 mm); a quiet winter
  window on a very stable station can tie ≥ 50 % of values. The guard should
  either default > 0 (in whitened units, e.g. 0.1) or the spec should stop
  claiming it.
- **Thin-window fallback is *stricter* than the global identifier:** when a
  window is thin, the Hampel test becomes `|w − med(w)| > k_w · ŝ_global`
  with k_w = 4 — a global test at threshold 4, while the actual global
  identifier uses k_g = 5. Epochs in sparse regions (gappy eras, series
  edges) are systematically *easier* to flag than dense-region epochs, the
  opposite of the documented "conservative degradation". Fallback should
  use k_g, or not flag at all (NaN ⇒ False, as `hampel_mask` already does
  when the caller does not substitute).

---

## 9. MEDIUM — Pipeline robustness: NaN input crashes per-station (REYK), and the leaf/caller contract has no owner

**Measured:** `getData("REYK", ref="plate", uncert=10)` returns NaNs in y;
`detect_outliers` raises `ValueError: y must be finite`. The leaf contract
(validate + raise) is correct in isolation, but no caller in the chain owns
NaN removal — a fleet precompute run dies on the first such station. Either
`detrend`/precompute documents and implements the NaN pre-drop, or the leaf
accepts a `valid` mask. Silent NaN-dropping inside the leaf would be wrong
(index bookkeeping); an explicit contract is required either way.

---

## 10. LOW-MEDIUM — Parameter surface: 26 knobs, 3 hard mirrors, no sensitivity map

- **Mirroring (verified):** the field list is hard-copied in
  `gps_parser/src/gps_parser/outlier_catalogs.py`,
  `gps_api/src/gps_api/precompute/config.py`, and
  `config-templates/analysis-lane/analysis.yaml`;
  `gps_plot/plot_gps_timeseries.py` introspects
  `dataclasses.fields(OutlierParams)` dynamically (good — the model for the
  others). A new `OutlierParams` field is silently absent from three config
  surfaces with no parity test. **Alternative:** generate the YAML schema
  and the config dataclass from `OutlierParams` (single source), plus one
  cross-repo test asserting field-name parity.
- **Identifiability:** several parameters are practically unidentifiable
  from data and interact non-linearly (`run_sign_fraction`,
  `despike_return_sigma`, `step_flank_max_reach_days`,
  `window_robust_iterations`, `cluster_gap_days` vs `max_run_days`). The
  defaults are justified by Gaussian tail arithmetic (§6) that the spec's
  own colored-noise discussion (§3.3) invalidates; the single most
  output-sensitive knob is `window_n_sigma` (measured: 62→414 flags over
  5.0→2.5) and there is no guidance for setting it beyond the default.
  No parameter sensitivity study exists. **Alternative:** a fleet-scale
  sweep harness (flags vs each knob, per station class) before per-station
  overrides are handed to operators; freeze the unidentifiable knobs out of
  the operator-facing config into an "expert" tier.

---

## 11. LOW — Assorted invariant and consistency defects

1. **§8.4 `flags ∧ (protected ≠ 0) = ∅` is violated under
   `epoch_policy="union"`** by construction (union overrides other
   components' protection — documented in the code docstring, claimed
   unconditionally in the spec). The spec should scope the invariant to
   `per_component`, or union should respect protection (union of
   *unprotected* candidates).
2. **Abort zeroes Stage-0 despikes** although they are "decided blunders"
   explicitly excluded from the abort numerator for that very reason —
   the rationale for exclusion contradicts the rationale for zeroing.
3. **Final-sweep diagnostics don't correspond to final flags** when the loop
   exits at the iteration cap: `z`, `candidates`, `reasons`, `protected`
   come from the fit *before* the last flag update, while `fits` are refit
   on the final flags. On non-converged series (HOFN, AKUR) the reported z
   is one fit stale.
4. **`converged=False` has no consumer**: nothing downstream distinguishes a
   fixed-point mask from an iteration-cap mask (see finding 3).
5. **`qn_scale` is O(N²) memory** (≈213 MB at N=7300, documented) — fine
   while non-default, but §9-Q6 answered "yes" to shipping Qn in 1.5; the
   O(N log N) implementation is a prerequisite for defaulting it, not an
   optimization.

---

## Summary table

| # | Severity | Finding | Key measurement |
|---|----------|---------|-----------------|
| 1 | CRITICAL | Whitening self-pardons co-inflated bad epochs | RHOF 83 mm → z=3.3 (unwhitened 12.9); +28–55 % flags without σ; ρ(σ,|r|)≈0 |
| 2 | HIGH | Abort: non-monotone cliff, all-component blast radius, quiet in practice | k_w 2.5→414 flags, 2.0→0; SAUD [0.100, 0.009, 0.006] → all zeroed |
| 3 | HIGH | Idempotence claim false on real data; non-convergence served silently | RHOF inlier re-run → 16 new flags; HOFN/AKUR converged=False |
| 4 | HIGH | Verdicts window-dependent; pipeline detects on the analysis window | Jaccard 0.50 (1-yr vs full); ŝ_global 1.91 vs 0.9 |
| 5 | MED-HIGH | No physical magnitude trigger (floor has no dual ceiling) | 83 mm epoch has no non-z path to candidacy |
| 6 | MEDIUM | f_scale=1.0 → effectively L1 fit; σ=None path not unit-agnostic | transition at 0.4–0.75×MAD(w); y×1000 changes mask (7 epochs) |
| 7 | MEDIUM | Protection rules entangled; D on w vs spec's r; PROTECT_RUN bit conflated | code inspection; inline duplicate of step_evidence |
| 8 | MEDIUM | scale_floor=0 default contradicts spec; thin-window fallback stricter than global | code inspection |
| 9 | MEDIUM | NaN input crashes per-station; no owner for the pre-drop | REYK ValueError |
| 10 | LOW-MED | 3 hard config mirrors; 26 knobs, no sensitivity map | grep-verified |
| 11 | LOW | Union breaks §8.4 invariant; stale final-sweep diagnostics; abort zeroes despikes | code inspection |

---

*Review performed 2026-07-28 against `outliers.py` @ working tree and
`DESIGN_outlier_detection.md` (2026-07-27 §11 state). Experiments run in the
`gpslibrary` mamba env on live archive data (RHOF, VMEY, HOFN, AKUR, REYK,
SAUD, GFUM). Nothing here has been actioned.*
