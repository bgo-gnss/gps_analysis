# PLAN — Outlier detection: correctness, then consistency

> **Status:** agreed implementation plan, 2026-07-28. Not started.
> Alignment artifact: `gps_plot/.interrogate-canonical-outlier-detection.md`
> (gitignored — local). Backlog of deliberately-unactioned findings:
> [`REVIEW_outlier_backlog.md`](REVIEW_outlier_backlog.md).
> Spec being amended: [`DESIGN_outlier_detection.md`](DESIGN_outlier_detection.md).

Ordering decision (BGÓ, against evidence, not a default): **correctness
first**. Canonicalising a detector whose verdicts are known-wrong would only
make the misses consistent. The evidence that flipped the original
consistency-first ordering is backlog finding 1 — Spearman(σ, |r|) =
−0.06 / +0.24 / +0.12, i.e. the formal σ we whiten by carries almost no
epoch-quality information.

Repos and branches (each package is its own git repo; all editable in the
`gpslibrary` mamba env, so a leaf change is live everywhere immediately):

| repo | branch | state |
|---|---|---|
| `gps_analysis` | `outlier-step-protection-flanks` | pushed |
| `geo_dataread` | `provisional-outlier-state` | pushed |
| `gps_plot` | `plot-hide-outlier-overlay` | pushed |
| `gps_api` | **on `main`** | needs `outlier-abort-granularity`, `outlier-whitening-clip` |

## Pin table (corrected — the first version was wrong in three places)

| pin | breaks? | note |
|---|---|---|
| `gps_analysis/tests/test_outliers_stage0.py:506` golden order-0 | **No** | Calls `detect_outliers(lineperiodic, t, y)` with **no σ**, so `whiten(r, None)` returns `r` — every whitening mechanism is a no-op on it. 3 flags of 900 ⇒ the abort branch is never entered either. The real defect: this pin is **blind to the whole σ path**. |
| `gps_analysis/tests/test_outliers.py:579 test_sigma_weighting` | **Yes** | The pin that actually breaks, and it was unnamed. Asserts `not flags[i_noisy]` for large-σ + large-residual — exactly what DoD 2 inverts. **Split, do not delete.** |
| `gps_api/src/gps_api/precompute/outliers.py:211` params hash | **Yes** | Hashes `dataclasses.asdict(params)` — any new field re-hashes every station. |
| `geo_dataread/src/geo_dataread/gps_write.py:80 outlier_params_hash` | **Yes** | Fourth mirror, previously unlisted. |
| `gps_api/tests/test_outlier_config_parity.py:13` | **Yes** | Breaks *silently* on a new leaf field. |

Rule for all of them: **re-establish, never loosen.** A pin that gets weakened
to pass is a pin deleted with extra steps.

---

## T0 — Fleet measurement harness (enabler)

`geo_dataread/tests/characterize_fleet_outliers.py`, non-`test_` prefix so
pytest does not collect it; modelled on
`gps_analysis/tests/verification_outlier_ab.py`. Lives in `geo_dataread`
because the measurement path is `gps_read.getData(...)` and the leaf must not
import a downstream package.

Per station × window × component: N, n_candidates, candidate fraction,
n_flagged, abort, converged, `scale_global`, n_provisional, wall-clock.
Windows `90d | 1yr | full`. Accepts `--outlier-param NAME=VALUE` (same
coercion as `plot_gps_timeseries._build_outlier_params`) and `--baseline FILE`.

**REYK comes free.** `detect_view_outliers` already computes
`finite = isfinite(t) & all(isfinite(y)) & (sigma > 0)` (`gps_views.py:919`)
and excludes those epochs, so the harness never hits the leaf's NaN crash. The
leaf-level `ValueError` is still real for the `estimate_detrend`/precompute
lane — that stays backlog #9. **Do not "fix" it here.**

**Acceptance** — DONE 2026-07-28, `geo_dataread/tests/characterize_fleet_outliers.py`

```bash
python geo_dataread/tests/characterize_fleet_outliers.py \
  --stations RHOF VMEY HOFN SAUD GFUM AKUR REYK THEY \
  --windows 90d,1yr,full --ref plate --uncert 10 \
  --save tests/data/fleet_baseline.tsv
```

Reproduced exactly: SAUD full-span per-component candidate fractions
`[0.1005, 0.0085, 0.0060]`; GFUM `[0.0774, 0.0451, 0.0925]`; GFUM 90 d = 4
candidates of 63; **REYK included** (the finite mask is mirrored from
`detect_view_outliers`, so the leaf NaN crash never fires — that leaf
fragility stays backlog #9, unworked-around).

⚠ **The plan's HOFN criterion was itself defective and is replaced.** It said
"HOFN 90 d → 7 flagged, 1 yr → 6", measured against `currDatetime(-1)`. That
anchor MOVES, and on a detector we already know is window-dependent the counts
move with it: measured 8 / 6 / 5 flagged at 90 d for three end dates days
apart. The full-span checks reproduced precisely because they are
anchor-insensitive; only the windowed ones drifted. **A baseline tied to
"yesterday" is not a baseline** — this is the T3 defect surfacing inside T0's
own acceptance test. All windowed acceptance numbers must pin `--end`
explicitly (the harness takes it as a fractional year); windows otherwise
anchor on each station's own last epoch, which keeps a station that stopped
years ago measurable.

**Config-file lever (BGÓ's stated end goal: per-station control from a config
file rather than flags).** `--outlier-overrides PATH` resolves through the
production chain — `station_step_epochs` + `resolve_protect_windows` +
`resolve_outlier_detection` — so a candidate `outlier_overrides.csv` is
evaluated exactly as deployment would apply it. Demonstrated: a CSV with rows
for RHOF and VMEY only moved those two stations (+21/+22/+8 and +13/+9/+3
flags) and left THEY untouched.

**Coverage gap found while wiring it:** the per-station catalog
(`gps_parser.outlier_catalogs.OUTLIER_OVERRIDE_COLUMNS` — note: *not*
`OUTLIER_OVERRIDE_KEYS`, as an earlier revision of this file said) carries only
**5 of 26** `OutlierParams` fields: `despike`, `despike_n_sigma`,
`window_order`, `window_robust_iterations`, `epoch_policy`, plus the
per-component `min_outlier_{n,e,u}` floor. **`despike_gap_days` is NOT among
them** — the S0 coverage knob has no per-station route today, nor do
`global_n_sigma`, `window_n_sigma` or any `step_*`. Extending that column set
is a prerequisite for tuning S0 from config, and it touches all four
`OutlierParams` mirrors.

**Pins:** none. **Detrend records:** not moved.

**Also do here:** commit `gps_analysis/tests/characterize_survivors.py` and
`characterize_stepedge_union.py` (currently untracked). They are the
provenance for §11's +10.8 % and for backlog findings 2/4 — without them the
numbers every later ticket cites have no reproducer in the repo.

---

## T1 — Per-component abort + small-N floor + abort visible in the figure

Satisfies **DoD 1**. Depends on T0.
Order: `gps_analysis` → `gps_api` → `geo_dataread` → `gps_plot`.

**Leaf (`outliers.py`)**

- `OutlierDetection` gains `component_abort: NDArray[np.bool_]`, shape `(C,)`.
  **`excess_flag_abort` keeps its meaning as `component_abort.any()`** — this
  is the compatibility hinge and the reason T1 moves no detrend records.
- Sweep loop (1913–1988): scalar `aborted` → `(C,)`; zero only the aborting
  components and let healthy ones continue to their own fixed point instead of
  `break`ing the loop. Per-component inlier sets are **already independent**
  (line 1925), so this is loop control, not new math.
- `converged` becomes "every non-aborted component reached a fixed point" —
  a semantic change to an existing field; document it.
- **`epoch_policy="union"` under a partial abort is undefined today.** Decide:
  union over non-aborted components only. The alternative lets an aborted
  component's all-False row poison the union, reintroducing the exact blast
  radius this ticket removes.
- `OutlierParams.min_abort_candidates: int = 0` — abort iff
  `n/N > f_max` **and** `n >= min_abort_candidates`. Default 0 = bit-identical
  to today. Production value 10 (GFUM 90 d: 4 of 63 = 6.3 %).
- **The leaf must not gain a `warnings.warn` on abort.** It has none today.
  `gps_analysis` and `gps_api` run `filterwarnings=["error"]`; keeping
  visibility a consumer responsibility is what stops this failing two suites.

**`geo_dataread` (`gps_views.detect_view_outliers`)**

- Replace the whole-station early return (939–945). Read the new field with the
  codebase's own degrade idiom — `getattr(detection, "component_abort", None)`,
  as `_provisional_mask` already does — so an older leaf still works and the
  slice stays independently mergeable.
- Healthy components get flags *and* `_provisional_mask`; aborted get all-False.
- Provenance gains `component_abort` + `n_aborted_components`; `outlier_abort`
  keeps `any()` so `gps_write.py:396` is untouched.

**`gps_plot` (`timesmatplt.py`)**

- `_mask_outliers` discards `prov` except `provisional` (line 339); it must
  return abort status too. Private, single caller — contained.
- Badge on **the aborted component's axis only**, not a figure-level banner,
  or the granularity this ticket creates is invisible in the artifact meant to
  show it.
- **The badge must survive `hide_outliers=True`** — an aborted component has no
  overlay to hide, and suppressing its badge is precisely the failure this
  ticket exists to fix. Same argument the code already makes for gold markers.
- CLI needs nothing: `--outlier-param min_abort_candidates=10` works via
  `dataclasses.fields` introspection.

**`gps_api`** — `StationOutliers` gains `component_aborted`; delete the
whole-station `flags[:] = False` (~line 427); no new store column in T1 (the
flags already encode it). `OutlierConfig` + `analysis.yaml` +
`gps_parser/outlier_catalogs.py` gain `min_abort_candidates`.

**Acceptance**

```bash
# granularity witness — only SAUD-N is pathological
python geo_dataread/tests/characterize_fleet_outliers.py --stations SAUD --windows full \
  --baseline geo_dataread/tests/data/fleet_baseline.txt
#   expect abort [True, False, False]; E and U n_flagged > 0 (baseline 0,0,0)

# small-N floor — a DIFFERENT mechanism (GFUM full-span fractions all exceed
# f_max, so granularity alone does not save it)
python geo_dataread/tests/characterize_fleet_outliers.py --stations GFUM --windows 90d \
  --outlier-param min_abort_candidates=10     # expect no abort, 4 flagged
python geo_dataread/tests/characterize_fleet_outliers.py --stations GFUM --windows 90d
#   expect still aborts (default 0 preserves baseline)

# visibility
plot-gps-timeseries SAUD --ref plate --view cleaned --special full --save png --name saud_abort
plot-gps-timeseries SAUD --ref plate --view cleaned --special full --hide-outliers --save png --name saud_abort_hidden
#   North axis badged in both; East/Up carry grey overlays in the first
```

Plus all four suites, and a `gps_plot` test asserting the badge artist exists
on exactly the aborted axis.

**Pins:** golden order-0 must pass **untouched** — if it moves, the change is
not default-preserving and *that* is the bug. Params hash (both mirrors) and
config parity break; re-capture, and **add `test_outlier_field_parity`**
asserting field-name parity between `dataclasses.fields(OutlierParams)` and
`OUTLIER_OVERRIDE_KEYS` / `OutlierConfig` / the yaml template. That converts a
pin that broke into a pin that catches the next one — which is what pays for
itself in T2.

**Detrend records: NOT moved.** Verify rather than assert: re-run
`gps-estimate-detrend` over the working set and diff `detrend_params.json` —
expect byte-identical except `fitted_at`.

**Spec:** amend §3.5; add §8.3 rows `test_abort_is_per_component` /
`test_small_n_floor`; add a **§12 addendum** with the SAUD/GFUM measurements.

---

## T2a — Whitening statistic, param-gated, default bit-identical

First half of **DoD 2**. Depends on T0; sequence after T1.

**Mechanism: clipped whitening.** New atomic `clip_sigma(sigma, c)` →
`σ'_i = min(σ_i, c · med(σ))`, plus `OutlierParams.whiten_sigma_clip: float =
0.0` (0 = off = today).

Why not the alternatives: **drop-σ** is the biggest measured gain (+28–55 % on
RHOF) but lands the detector on the σ=None path, which backlog #6 measured as
**not unit-agnostic** (`f_scale=1.0`; y×1000 moved 7 epochs) — that couples T2
to an out-of-scope fix. **Dual statistic** `max(|z_w|, |z_raw|)` needs a second
scale and its raw arm carries the same unit problem; keep it as the named
fallback.

Why a new primitive rather than a `whiten(..., clip=)` kwarg: MATH_STANDARDS §1
has the orchestrator compose atomics; `whiten`'s published equation stays
`w = r/σ`.

**Call site:** `_component_candidates:1543` →
`w = whiten(r, clip_sigma(sigma_c, params.whiten_sigma_clip))`.
**The fit is untouched** (1529–1541 keep raw σ) — only the *identifier* path
changes, so §3.1's WLS weighting survives and backlog #6 stays out of scope.

**⚠ Step 1 is a falsifiable gate, before any other work.** Compute the RHOF
2023-02-28 U statistic at c ∈ {1.5, 2.0}. σ = 13.2 mm ≈ 4.1× median, so σ'
shrinks by 4.1/c while ŝ(w) barely moves (MAD ignores the clipped tail);
expected z ≈ 9.1 at c=1.5, ≈ 6.8 at c=2.0 against k_g = 5.0. **If z does not
clear k_g with margin at either c, stop and switch to the dual statistic** —
and note that doing so drags backlog #6 into scope.

**Honesty note:** clipping is monotone in `w` per epoch but **not** in `ẑ` (the
global median/MAD and the local Hampel centre/scale all move). Flags will move
in **both** directions — which is why T2b's network measurement is mandatory
rather than confirmatory.

**Acceptance**

```bash
python geo_dataread/tests/characterize_fleet_outliers.py --stations RHOF --windows full \
  --outlier-param whiten_sigma_clip=1.5 --probe-epoch 2023.1603
#   expect U z >= 5.0 (baseline 3.32) and the epoch in the flagged set

plot-gps-timeseries RHOF --ref plate -s 2023-01-01 -e 2023-04-01 --view cleaned \
  --outlier-param whiten_sigma_clip=1.5 --save png --name rhof_clip
#   2023-02-28 renders grey; without the flag it renders red

cd gps_analysis && python -m pytest tests/ -q     # green, zero expected-value edits
```

**Pins:** golden order-0 — **extend, don't update**: keep the σ-free fixture
byte-identical and add `test_golden_whitened_fixture`, a seeded σ-bearing
fixture with a co-inflated epoch (large r, σ ≈ 4× median), pinned at
clip-off. It becomes the pin T2b legitimately breaks, and it closes the hole
that today's golden fixture cannot see the σ path at all.
`test_sigma_weighting` — **split now, in T2a**: `test_sigma_weighting_default`
keeps current assertions with `whiten_sigma_clip=0.0` *explicit*, and
`test_sigma_clip_releases_coinflated_epoch` runs the same fixture at 1.5
asserting `flags[i_noisy]` is **True**.

**Detrend records: NOT moved** (default off).

**Spec:** §3.1 must be **rewritten, not amended** — "σ contributes the
epoch-to-epoch quality *ratio*" is falsified. Add a **§13 addendum**.

---

## T2b — Flip the whitening default + quantify network-wide

Second half of **DoD 2**, and all of **DoD 3**. Depends on T2a, T0.

Scope: `whiten_sigma_clip` `0.0 → 1.5` (or 2.0, decided by T2a's gate), same
default in `OutlierConfig` and `analysis.yaml`. No code beyond defaults and
pins — but this is where the detector's real behaviour changes for every
station and every cleaned figure.

**Acceptance**

```bash
python geo_dataread/tests/characterize_fleet_outliers.py \
  --stations RHOF VMEY HOFN SAUD GFUM AKUR REYK --windows full \
  --baseline geo_dataread/tests/data/fleet_baseline.txt --list-moved
#   committed as gps_analysis/docs/MEASURE_whitening_flip.md: per station
#   Δn_flagged, Δcandidate fraction, Δabort, and WHICH epochs moved each way

cd gps_analysis && python -m pytest tests/ -q -k coinflated
#   test_coinflated_blunder_is_flagged: synthetic reconstruction of the RHOF
#   case (r = 83, σ = 4.1× median) flagged at defaults. The leaf suite must
#   not read the archive; the real-epoch assertion lives in the harness output.
```

**Pins:** `test_golden_whitened_fixture` re-captured at the new default, old
values kept in the comment per the fixture's house style. `test_sigma_weighting`
split completes. Params hash + config parity re-captured — write **all three
generations** (pre-T1, post-T1, post-T2b) into §13, so a stored product's
vintage is identifiable from its hash. That is the cheap substitute for the
`record_version` bump the doc puts out of scope.

**⚠ Stored detrend records MOVE.** `estimate_detrend` detects before it fits
(`detrend.py:668–703`). Options per §11 precedent: (a) re-estimate a whole
`detrend_params.json` in one run — **recommended**; (b) accept mixed vintage
distinguished only by `fitted_at`. The movement must be measured and attached,
not suppressed: diff before/after over the working set.

**Also flag in the addendum, don't fix:** T2b raises flag counts network-wide,
which per backlog #3 makes the non-idempotence cascade *larger* (RHOF inlier
re-run already yields 16 new flags). Out of scope by decision — but T2b is the
ticket that makes it worse, and that belongs in writing now.

---

## T3 — Canonicalization, fork A (detect on full series, slice to window)

**DoD 4.** Depends on T1 and T2b. Fork A settled by the destination doc.
**No leaf change** — the defect lives in the two consumer layers.

**`geo_dataread`** — keep `detect_view_outliers` pure (array-in/array-out; it
is testable precisely because it is). Add the canonical entry point one level
up: `detect_station_outliers(sta, ...)` reads the **full span** itself and
returns full-span flags + epoch index; `read_gps_view` and the plot path
consume and slice. The structural point: **the caller must stop windowing
before detection.** Cost 0.2–0.5 s/station at `window_order=0`; add an
**in-process LRU** on `(sta, ref, tType, uncert, params_hash)`. A *disk* cache
is fork B in disguise — do not. Provenance gains `detection_span` and
`detection_window_mode`, written into the cleaned-NEU sidecar.

**`gps_plot`** — `plotTime:472` windows at `getData(..., fstart, fend)` before
`_mask_outliers:480`: the identical defect already falsified in
`read_gps_view`. For `view="cleaned"`, read full span, detect, slice series
*and* flags for display.
**Trap:** `highlight_last` / `firstpoint, lastpoint` (492–499) must keep using
the **displayed** window's last epoch. The code comment there already insists
the green-title and green-marker decisions cannot disagree; a full-span read is
exactly what would break that.
New `--detect-span {full,view}`, default `full`, `view` retained as escape
hatch **and** as the A/B lever in the acceptance check.

**Acceptance**

```bash
python geo_dataread/tests/characterize_fleet_outliers.py --stations HOFN GFUM RHOF \
  --windows 90d,1yr,full --detect-span full
#   expect flag sets identical across windows after intersecting (Jaccard 1.00);
#   baseline HOFN 7 vs 6, RHOF U Jaccard 0.50
python geo_dataread/tests/characterize_fleet_outliers.py --stations HOFN GFUM RHOF \
  --windows 90d,1yr,full --detect-span view
#   expect the T0 baseline exactly — the defect, on demand

plot-gps-timeseries HOFN --ref plate --view cleaned --special 90d  --save png --name hofn_90d
plot-gps-timeseries HOFN --ref plate --view cleaned --special year --save png --name hofn_1yr
#   epochs flagged inside the overlapping 90 d are the same set in both
```

Plus a cost gate (< 1 s/station at `window_order=0`), and GFUM's abort becomes
a property of the station rather than of the plotted window.

**Pins disturbed: none** — no new `OutlierParams` field, no leaf math change.
A direct dividend of the correctness-first ordering: canonicalization last is
the cheapest of the three in pin terms.

**Detrend records: not moved.** But state the consequence: after T3 the plot
and the stored detrend record can disagree about the same epoch and nothing
surfaces it. `detection_window_mode` in the sidecar is the minimum legibility.

**Tail policy (DoD 4) — resolved for fork A.** There is no stored cutoff:
detection runs on the full archive at read time, so epochs newer than any
previous run are detected, never reverted to live windowed detection — the
failure DoD 4 names is structurally impossible under A. The residual
epistemics (the trailing ~14 d where verdicts are genuinely unstable) are
already carried by the provisional/gold mask (`PROVISIONAL_DAYS`,
`gps_views.py:789`). Write this into the spec as a **§14 addendum**, together
with the statement that a tail policy **is** required before fork B and is not
yet chosen.

---

## T-S — Stage isolation (`--stages`) + the S0 calibration question

Added 2026-07-28 from BGÓ's annotated RHOF 2013–2016 figure. Slots **before
T2a**: being able to run S0 alone is how you measure whether the whitening fix
is even needed for a given class of outlier.

### Naming

**Stages are S0–S5, tickets are T0–T3.** BGÓ's original phrasing used T0/T1 for
stages, which collides with the ticket names. Stages, per the spec:

| stage | what | spec |
|---|---|---|
| **S0** | gross-blunder despike — robust first differences of RAW data | §3.0 |
| **S1** | step-augmented robust `lineperiodic` fit → residuals | §3.1 |
| **S2** | whitening by formal σ | §3.1 |
| **S3** | global identifier, \|ẑ\| > k_g | §3.3a |
| **S4** | windowed Hampel identifier | §3.3b |
| **S5** | signal protection (floor / run / step / window) | §3.4 |

### The finding that motivates it

**S0 is σ-free.** It works on robust first differences of the raw series — no
model fit, no whitening — so it is structurally immune to the σ-blindness that
backlog #1 calls the central design error. That makes it a cheap *partial*
answer to "obvious outliers survive", independent of T2.

**S0 currently fires on almost nothing.** Default `despike_n_sigma` (k_d) = 10,
chosen for "gross blunders only" (A ≳ 14σ). Measured on BGÓ's labelled figure
(RHOF, `ref=plate`, `uncert=10`, 2013-01-01 → 2016-01-01), the S0 statistic
`min(|δ⁻|,|δ⁺|)/ŝ_Δ` at each annotated point — **all six pass the sign-flip and
return-to-baseline conditions**, so magnitude alone separates them:

| point | date | statistic | BGÓ label |
|---|---|---|---|
| E +8.7 | 2013-10-02 | **8.4** | correctly detected (by S3/S4) |
| N −7.3 | 2013-10-02 | **5.0** | MISSED — should be caught |
| U +12.5 | 2013-08-23 | **4.2** | MISSED |
| U −31 | 2013-10-02 | **4.1** | MISSED |
| U +13 | 2015-10-03 | **3.9** | MISSED (15 d gap after ⇒ also needs `despike_gap_days`) |
| U −24 | 2015-12-31 | **3.2** | provisional/gold — correctly undecided |

`k_d = 3.5` + `despike_gap_days = 20` reproduces the labelling exactly: all
four misses flagged, the already-caught epoch still flagged, the gold epoch
still undecided. Network cost of S0 alone at k_d = 3.5: **0.21–0.74 %** of
component-epochs (RHOF 0.53, AKUR 0.74, THEY 0.21, HOFN 0.32, REYK 0.38,
VMEY 0.32) — far under `max_flag_fraction`.

### Recommendation on the default: **do not lower it yet**

The separation is clean but rests on **one station-window and six points**.
Flipping the default would move stored detrend records (S0 flags feed the fit
exclusion), re-hash provenance in both mirrors, and require rewriting §3.0's
justification for k_d = 10 — all from a sample of six.

Instead: **label 5–10 more station-windows first**, then set the default from
the pooled boundary in a measured ticket shaped like T2b.

### `despike_gap_days` is a DIFFERENT question — and it is a defect, not tuning

Measured 2026-07-28 (correcting an earlier claim in this file that the gap was
"the larger semantic change of the two" — it is the opposite).

The two S0 knobs are not equivalent. **`despike_gap_days` controls COVERAGE**
(what fraction of epochs S0 can judge at all — outside it δ is NaN and no
despike is possible however extreme the point); **`despike_n_sigma` controls
SENSITIVITY** within that coverage.

S0 coverage — % of epochs with BOTH neighbours inside the gap:

| sta | gap=1.5 (default) | gap=5 | gap=20 | gap=60 |
|---|---|---|---|---|
| RHOF | 71.4 % | 94.4 % | 99.0 % | 99.5 % |
| HOFN | **58.5 %** | 93.8 % | 99.4 % | 99.7 % |
| REYK | **57.0 %** | 94.1 % | 99.4 % | 99.8 % |
| AKUR | 63.8 % | 92.4 % | 98.9 % | 99.4 % |

**At the default, S0 is structurally blind to 29–43 % of epochs.** Cost of
lifting it, at k_d = 3.5: RHOF 0.53 % → 0.61 % despiked (1.5 → 20 d), and
20 → 60 adds nothing (0.61 → 0.61) — it saturates by ~20 d because only 0.51 %
of epochs have a predecessor further back than that.

Structurally this is the same shape as §11's unreachable NaN branch and
`step_flank_max_reach_days`: a fixed window measuring elapsed time rather than
evidence.

### …but the coverage fix is INERT on its own — measured, correcting the above

An earlier revision of this file recommended treating 1.5 → 20 as a defect fix
"on its own evidence, separately from the k_d question". **That recommendation
was wrong** and is withdrawn. Measured over the working set, full span,
`ref=plate`, `uncert=10`:

| config | total flagged (6 stations) |
|---|---|
| baseline, S0 off | 688 |
| S0 on, gap 1.5 (default) | 703 (+15) |
| **S0 on, GAP 20 only** | **701 (+13)** |
| S0 on, gap 20 + k_d 3.5 | 952 (+264) |

Raising coverage from ~65 % to ~99 % moved the total by **−2**, and contributed
exactly **zero** on AKUR. Per labelled point, no gap value recovers anything at
k_d = 10:

| labelled point | gap1.5/k10 | gap20/k10 | gap60/k10 | gap60/k6 | gap60/k3.5 |
|---|---|---|---|---|---|
| RED N −7.3 (5.0) | – | – | – | – | **FLAG** |
| RED U +12.5 (4.2) | – | – | – | – | **FLAG** |
| RED U −31 (4.1) | – | – | – | – | **FLAG** |
| RED U +13 (3.9, gap 15 d) | – | – | – | – | **FLAG** |
| RED N −11.8 (6.4, gap 36 d) | – | – | – | **FLAG** | **FLAG** |
| GRN E +8.7 (8.4) | FLAG | FLAG | FLAG | FLAG | FLAG |
| GOLD U −24 (3.2) | – | – | – | – | – |

**Conclusion: the two knobs are coupled, not separable.** k_d is the binding
constraint everywhere; the gap is a necessary enabler for one subclass only
(isolated points after long gaps — `RED N −11.8` needs gap ≥ 40 **and**
k_d < 6.4, neither alone). The coverage statistic is true but operationally
inert at k_d = 10, because the blind region contains almost nothing that clears
that threshold.

Method note, since this is the third instance in this work: the error was
reasoning from a structural fact (29–43 % blind) to an operational
recommendation without measuring the consequence — the same pattern as the
falsified `window_order=1` invariance claim and the `steps.csv` premise.
**Measure the consequence, not the mechanism.**

The separation boundary has now held across **7 labelled points in 2 windows**:
reds at 3.9–6.4 all flagged at k_d = 3.5, gold at 3.2 still correctly
undecided. Cost of the combined change: 688 → 952 (+38 %), 0.46–2.20 % per
station. Still short of the 5–10 windows this file requires before a default
moves.

Fixture #2 (BGÓ, RHOF 2013–2020): **2016-04-02 north −11.8 mm**, gap_before
35.9 d, gap_after 4.0 d. At gap ≥ 60 its statistic is **6.39** with a near
perfect return (0.01) — the strongest S0 signature in either fixture, invisible
solely because its predecessor is 36 days away.

Caveats to carry into implementation: no case for > 20 d as a default (the
2016-04-02 point at 36 d is the tail, not the norm); and a 36-day-old neighbour
is a weaker baseline reference than a 1-day one (seasonal motion over that span
is ~1 mm against ŝ_Δ ≈ 1.5 mm, so ~1σ of bias in the return test). **Nothing
currently records the reach used** — as with the flank fallback, the reach
should be reported so a far-reaching verdict is legible as less reliable.

### No new option is needed to do that

Already works, via `dataclasses.fields` introspection in
`plot_gps_timeseries._build_outlier_params` — verified 2026-07-28:

```bash
figview.sh RHOF --ref plate --view cleaned --uncert 10 -s 20130101 -e 20160101 \
  --outlier-param despike=true \
  --outlier-param despike_n_sigma=3.5 \
  --outlier-param despike_gap_days=20
```

### Scope of T-S

1. **`--stages S0[,S1,…]` / `--stages all`** as a first-class selector,
   replacing the sentinel-threshold hack (`global_n_sigma=1e9`). Sentinels do
   not express intent and cannot be asserted on. Needs explicit enable flags on
   `OutlierParams` (or an equivalent), since S5's indeterminate arm is the one
   branch already shown to be unreachable by thresholds (§11 §3.4.2a-i).
2. **A labelled-fixture harness**: record `(station, window, component, epoch,
   label)` from human annotation, and report per-stage attribution — which
   stage caught each labelled epoch, and which labelled epochs nothing caught.
   BGÓ's figure above is fixture #1.
3. Do **not** change any default in T-S. Defaults move in a later measured
   ticket, once the fixture set is large enough to justify them.

**Pins:** adding enable flags to `OutlierParams` disturbs both params-hash
mirrors and config parity, exactly as T1 does — same re-establishment recipe,
and T1's `test_outlier_field_parity` (if landed first) catches it.
**Detrend records:** not moved, provided every stage defaults to enabled.

---

## S-N — Network-coherent epoch rejection (DEFERRED — separate project)

> **Status 2026-07-28: OUT OF SCOPE for this plan.** BGÓ's decision —
> network/regional-wide effects (atmospheric loading, reference-frame and
> orbit artifacts, common-mode) are their own project for a later time, not a
> branch of the per-station outlier work. Recorded here because the evidence
> was gathered and should not be lost, **not** as a live lead.
>
> Consequently the k_d default question below is **deferred pending more
> labelled windows** (as originally stated), NOT suspended pending an S-N
> feasibility check. Ignore any wording to the contrary.

Found 2026-07-28 while labelling VMEY. This is the design doc's own **Q7**
("signal coherence lives across stations, which is the caller's network-level
domain") — anticipated in §3.4.4 and never built.

### The finding

BGÓ labelled exactly one obvious outlier on VMEY 2013–2020: **north
2016-04-02**. That is the *same date* as RHOF's isolated −11.8 mm north point.
Checking the network:

| 2016-04-02, north residual vs local 15-d median | |
|---|---|
| RHOF | −11.8 mm |
| AKUR | −7.0 mm (−13.1σ) |
| HOFN | −5.8 mm (−37.1σ) |
| VMEY | −4.3 mm (−6.1σ) |
| REYK | −3.1 mm (−7.1σ) |
| Up component | unaffected everywhere (0.0, −0.5, +0.7, +0.9σ) |

Every station, north only, same sign, from Vestmannaeyjar to Akureyri to Höfn.
That is a **solution-day artifact**, not station physics.

**Two bad days account for 4 of the 7 labelled points.** 2013-10-02 shows in
all three components at RHOF; 2016-04-02 in north at RHOF and VMEY.

| labelled epoch | n_sta | median z | frac \|z\|>3 | verdict | BGÓ |
|---|---|---|---|---|---|
| 2016-04-02 N | 4 | **−10.09** | **100 %** | network-coherent | OBVIOUS |
| 2013-10-02 E | 7 | **+10.55** | 86 % | network-coherent | already caught |
| 2013-10-02 U | 7 | **−4.69** | 86 % | network-coherent | missed |
| 2013-10-02 N | 7 | **−3.23** | 57 % | network-coherent | missed |
| 2013-08-23 U | 6 | +0.56 | 17 % | station-local | missed |
| 2015-10-03 U | 12 | −1.27 | 8 % | station-local | missed |
| 2015-01-07 N (VMEY) | 9 | −1.90 | 22 % | station-local | **not** circled |
| 2017-09-18 U (VMEY) | 8 | −0.15 | 12 % | station-local | **not** circled |
| 2019-11-11 U (VMEY) | 11 | +0.44 | 18 % | station-local | **not** circled |

### Why this beats tuning k_d

On VMEY the S0 statistic cannot separate the labels at all — BGÓ's obvious
point scores **4.06**, the three he did *not* circle score **3.74 / 3.78 /
3.75**. There is no defensible threshold in that gap, and the clean RHOF
boundary (3.9 vs gold 3.2) **does not transfer**. Network coherence separates
the same set by an order of magnitude (|med z| 3.2–10.6 at 57–100 % vs 0.15–1.9
at 8–22 %).

Structural reason: on any single station the artifact is modest (VMEY −4.3 mm,
REYK −3.1 mm). It is unambiguous only in aggregate, so **no per-station
threshold can recover it without also flagging genuine noise.**

### Consequences for the rest of this plan

- **2 of 7 labelled points are genuinely station-local** (RHOF U 2013-08-23,
  2015-10-03) — only S0 will ever catch those, whatever happens network-side.
- The remaining 4 belong to the deferred class. **They should not be used as
  evidence for or against a k_d default**, since the lever that fits them is
  out of scope. The usable labelled set for the k_d question is therefore
  much smaller than 7 — another reason not to move that default yet.

### Sweep result (archive-wide, for the future project)

Run 2026-07-28 over 18 stations, full archive. Residual vs a centred 25-day
rolling median, robust-scaled; a day is a candidate when ≥ 5 stations have data,
|median z| > 2 and > 50 % of stations exceed 3σ.

**138 distinct candidate days** (165 component-days); **20 hit more than one
component** — whole-solution failures, the least ambiguous class:

```
2021-12-03  north(−11.0, 100%)  east(+66.6, 100%)  up(+10.7, 100%)
2022-09-08  north(−29.7, 100%)  east(+26.9, 100%)  up( +6.3,  88%)
2022-06-30  north( −9.7, 100%)  east(+42.2, 100%)  up( +6.4, 100%)
2013-10-02  north( −4.4, 100%)  east( +8.5, 100%)  up( −5.0, 100%)
```

Component split: **east 70, north 70, up 25** — i.e. NOT north-specific. The
earlier speculation that north was systematically hit came from reading two
hand-found samples as a pattern; it is withdrawn.

**Three caveats that must travel with these numbers** — they are not clean:

1. **The sweep misses the strongest known positive.** `2016-04-02` (median z
   −10.09, 100 % of stations) was NOT recovered, because only 4 stations had
   data and the gate required ≥ 5. 138 is an undercount and the threshold is
   doing unexamined work.
2. **The year distribution is uninterpretable as a trend** (2001–2013: 5 days;
   2022–2026: 120). Few of the 18 stations existed before ~2015, so the n ≥ 5
   gate can rarely fire early. Almost certainly a method artifact.
3. **The 2022–2026 concentration overlaps the Reykjanes/Grindavík unrest**, and
   a dike intrusion moves many stations coherently — the exact false-positive
   mode. That 100 % of stations exceed 3σ argues artifact (real deformation
   decays with distance, and RHOF/HOFN should not respond identically to a
   Reykjanes source) — but it argues, it does not demonstrate.

**The single check that would separate the two populations** and was not run:
whether z varies with geography on the strongest days. Uniform across the
country ⇒ processing artifact; decaying from a source ⇒ real deformation. That
turns the geographic-dispersion discriminator from an assumption into a
measurement, and is the natural first step whenever this project is picked up.

### Open questions for that future project

1. **Network-coherent is not automatically a blunder.** A large earthquake
   moves many stations at once. What makes these diagnostic is
   single-component, same-sign, across stations hundreds of km apart in
   different tectonic settings. That discriminator must be *encoded*, not
   assumed — geographic dispersion and component-selectivity are the
   candidate features.
2. Evidence so far is **2 dates, 4–12 stations**. Needs a sweep over the full
   archive: how many such days exist, and do they correlate with known
   processing events (orbit products, frame changes, GAMIT/GLOBK reruns)?
3. Architecturally this cannot live in `gps_analysis.outliers` — the leaf is
   per-station by contract. It belongs in the caller
   (`gps_api.precompute`/`geo_dataread`), consistent with §3.4.4's Q7 note.
4. Interaction with detection: is a network-flagged epoch removed before
   per-station detection runs (changing every fit), or masked after? The
   former moves stored detrend records.

---

## Dependencies and merge order

```
T0 ──► T1 ──► T-S ──► T2a ──► T2b ──► T3
        │
        └─ T1b (optional store column for component_abort) — independent after T1
```

T-S sits after T1 (it wants T1's field-parity test) and before T2a (running S0
alone is how you measure whether the whitening fix is needed for a given
outlier class).

- **T0** first: every later acceptance check is a T0 command.
- **T1** before **T2a**: T1's field-parity test is what stops T2a's new field
  going missing in the mirrors, and T1 establishes hash-regeneration discipline
  on a ticket that moves no detrend records.
- **T2a** before **T2b** by construction (param before default).
- **T3** last, per the ordering decision.

**Within a ticket:** `gps_analysis` (leaf, default-preserving) → run all four
suites → `gps_api` + `geo_dataread` → `gps_plot`. Because all nine packages are
editable, a leaf merge is live everywhere immediately — so the leaf change must
be default-preserving *at merge time*, and consumers must use the existing
degrade idiom (`getattr(detection, ...)`, `prov.get(...)`). That idiom is not a
nicety; it is the mechanism that makes "independently mergeable" true rather
than aspirational, and it is already why the provisional slice merged cleanly.

## Risks / open decisions

1. **Fork A vs B** — settled A-now/B-later. Unsettled and not blocking T3:
   whether B's provisional state is a third store column or provenance-only,
   and whether B's reader lives in `geo_dataread` or a `gps_api` client.
2. **Tail policy** — resolved by construction for A, **genuinely open for B**:
   (i) unflagged + `beyond_run` marker; (ii) live detection on the tail,
   stitched — which manufactures a seam whose statistics differ from the stored
   body, i.e. the window-dependence defect in miniature; (iii) refuse to serve
   the tail cleaned. **Decide before B, not during.**
3. **Whitening mechanism and c** — clipped at c = 1.5 recommended, but T2a
   step 1 is a real gate. If it fails, the fallback pulls backlog #6 into
   scope. The one branch where scope could legitimately have to reopen.
4. **`epoch_policy="union"` under partial abort** — undefined. Recommendation
   (union over non-aborted components) is a decision, not a derivation.
5. **Small-N floor value and shape** — 10 extrapolated from a single GFUM
   datapoint. Absolute count (recommended, simple) vs a one-sided binomial tail
   on `f_max` (principled, backlog territory). Do one or the other, not half.
6. **T1 does not fix the abort's non-monotonicity.** Backlog #2's cliff (RHOF
   k_w 2.5 → 414 flags, 2.0 → **0**) survives granularity and the small-N floor
   untouched. If "abort granularity" was expected to remove it, that is a
   misread — cheaper surfaced now than after T1 merges.
7. **`converged=False` remains invisible** (HOFN, AKUR). T1 already touches
   convergence semantics so it is the cheapest moment to propagate
   `converged`/`n_iterations` into provenance — but that is creep on a ticket
   already four repos wide. An offer, not folded in.
8. **`gps_api` is on `main`** and needs two branches. Its
   `filterwarnings=["error"]` means any *new* warning on a degrade path fails
   the suite; T1 is designed to add none.
