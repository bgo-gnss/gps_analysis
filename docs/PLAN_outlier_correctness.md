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

**Acceptance**

```bash
python geo_dataread/tests/characterize_fleet_outliers.py \
  --stations RHOF VMEY HOFN SAUD GFUM AKUR REYK --windows 90d,1yr,full --ref plate --uncert 10
```

Exits 0 with REYK included and reproduces: SAUD full-span per-component
fractions ≈ `[0.100, 0.009, 0.006]`; GFUM ≈ `[0.077, 0.045, 0.092]`; GFUM 90 d
= 4 candidates of 63; HOFN 90 d → 7 flagged, 1 yr → 6. Table committed as
`geo_dataread/tests/data/fleet_baseline.txt`.

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

## Dependencies and merge order

```
T0 ──► T1 ──► T2a ──► T2b ──► T3
        │
        └─ T1b (optional store column for component_abort) — independent after T1
```

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
