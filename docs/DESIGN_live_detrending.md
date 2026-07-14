# DESIGN — Live-data detrending: estimate, store, apply

## 0. Decisions locked (BGÓ, 2026-07-14) — READ FIRST, overrides conflicting text below

**Data-flow architecture (binding).** Two layers, do NOT conflate:
- **Internal = direct.** The analysis code AND BGÓ's primary dev workflow
  (nvim + terminal + PDF/EPS in zathura) read time series **directly through
  `geo_dataread`** (flat files today → a database later). No API in the loop.
  This is the PRIORITY path for detrend + the raw↔cleaned toggle.
- **Outward = API, added last.** `gps_api` is a read-only public projection over
  the store/DB, inserted toward the end for external consumers + the eventual
  public aflogun portal. Internal systems talk to files/DB directly, never
  through the API. The A8 outlier `clean=` param (slice 2) stays as the OUTWARD
  surface; it is NOT how the internal workflow toggles cleaning.

**Locked decisions:**
1. **Detrend lands in `geo_dataread` (+ the leaf), not gated behind `gps_api`.**
   The estimation caller may still run inside the precompute job for the store,
   but the apply-on-read + raw/detrended delivery + toggle are `geo_dataread`
   first-class, feeding `gps_plot`.
2. **Method tag (was open #2):** every detrended series carries a
   `detrend_method` provenance tag — `"step_augmented_robust"` (outlier-robust
   step-augmented fit, the new default when the outlier stage is on) vs
   `"plain_wls"` (legacy) — plus frame + params-version. Resolves the floating
   detrended-column semantics from the outlier wiring slice.
3. **Raw↔cleaned toggle is first-class in `geo_dataread` + `gps_plot`** (turn
   cleaned series on/off in the nvim+PDF workflow), independent of the API.
   Raw is ALWAYS retrievable. Per-component clean-serving stays deferred.
4. **Graceful degrade (was open #4):** on a detrend failure OR an outlier abort,
   return a WARNING + the **undetrended / raw** series — never hard-fail, never
   silently clip. Alerting policy revisited from ops experience.
5. **Plate-first:** detrend only AFTER plate removal; ITRF frame is integrity-
   check only, no tectonic relevance (§ answers Q4). Params stored in the
   plate-removed processing frame.
6. **`UseSTA` borrowing is first-class:** reuse a nearby pre-activity station's
   params for a station in an active area (the CSV + `geo_dataread` precedent) —
   the store must record borrowed provenance.
7. **Flexible per-station pinning:** operator can FIX/pin params and mark a
   station "do not auto-refit"; fresh fits run only every few months and must be
   individually overridable. Fit window as long as possible (min 1–2 yr).
8. **Config (was open #1):** add the `outliers:` + `detrend:` blocks to the
   in-project template `config-templates/analysis-lane/analysis.yaml` (in-project
   edit, safe); the out-of-project DEPLOY to `gps-config-data`/`~/.config` still
   waits for BGÓ.

---

**Status:** DRAFT for review — spec only, no implementation.
**Scope:** the detrend-parameter lifecycle across `gps_analysis` (leaf math),
`gps_api.precompute` (estimation caller + store), `geo_dataread` (apply-on-read),
and the gpsconfig deployment (`gps-config-data` → `~/.config/gpsconfig/`).
**Conforms to:** `docs/MATH_STANDARDS.md` (binding — §2 documentation contract,
§6 derived-product provenance), plan §10.4 (no-hardcoding), leaf rules R2/R6.
**Supersedes (on adoption):** the first-draft mechanism
`geo_dataread.gps_read.getDetrFit`/`convconst`/`save_detrend_const` +
`detrend_itrf2008.csv`.

---

## 0. Problem statement

Detrending a GNSS displacement series means subtracting a fitted trajectory
model — secular rate plus annual/semiannual seasonal terms (Bevis & Brown 2014,
J. Geodesy 88, eq. 1; Blewitt & Lavallée 2002, JGR 107(B7)) — so that transient
deformation (unrest, intrusions) stands out. Today every consumer that wants a
detrended view either re-fits the whole series on every read
(`read_gps_data`, the precompute job) or seeds a re-fit from a flat CSV of
half-specified coefficients. For **live data** this is wrong twice over:

1. Re-fitting on every read means the "background" definition _changes with
   every new epoch_ — during unrest the transient itself contaminates the fit,
   and yesterday's plot and today's plot disagree about what "detrended" means.
2. The fit is O(N) work repeated everywhere, with no provenance: nobody can
   say _when_ the parameters were estimated, over _which window_, in _which
   frame_, or with which software.

The correct model is the one already adopted for outliers: **the raw series is
the durable record; detrending is a _view_ computed from separately stored,
versioned parameters.** Estimation is a deliberate, occasional act; application
is a cheap pure evaluation `y − f(t; p̂)` valid for any epoch, including epochs
that arrive after the fit.

### 0.1 The first draft and its shortcomings

`getDetrFit()` (`geo_dataread/src/geo_dataread/gps_read.py:239`) reads
`~/.config/gpsconfig/detrend_itrf2008.csv` (source of truth:
`~/git/gps-config-data/detrend_itrf2008.csv`). Format: one row per station,
columns

```
STA, Nrate,Nacos,Nasin,Nscos,Nssin, Erate,…, Urate,…, Sitename, Starttime, Endtime, UseSTA, Fit
```

i.e. 15 numbers — rate [mm/yr] + annual cos/sin + semiannual cos/sin
[mm] per component — plus a fit window in fractional years and a borrowing
mechanism (`UseSTA`/`Fit`: copy another station's periodic or full
coefficients). Audit of the draft:

| #   | Shortcoming                                                                                                                                                                                                                                        | Consequence                                                                                                                                         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **No intercept.** The `lineperiodic` model has 6 parameters per component; the CSV stores 5. `convconst` prepends `0`.                                                                                                                             | Stored params cannot reconstruct the model value; the apply path papers over it with a `vshift` re-referencing pass. Application ≠ pure evaluation. |
| 2   | **Params are seeds, not results.** `read_gps_data` uses them as `p0` for a fresh `fittimes` fit on every call.                                                                                                                                     | Live view is unstable; transients contaminate the "background".                                                                                     |
| 3   | **No uncertainties, no covariance.**                                                                                                                                                                                                               | Cannot propagate errors into detrended products; violates MATH_STANDARDS §6.                                                                        |
| 4   | **No provenance.** No `fitted_at`, no software version, no estimator tag, no outlier policy; frame lives only in the _filename_; `Fit` column is semi-dead (the `useFIT="periodic"` branch crashed for years and is now an explicit `ValueError`). | Nobody can audit or reproduce an entry.                                                                                                             |
| 5   | **Stale and inconsistent data.** Reykjanes rows are all-zero (placeholder = "no detrend"), several `Endtime`s stop in 2021, borrowed rows carry copied numbers with drifted precision.                                                             | All-zero and NaN and "borrowed" are three different semantics squeezed into one flat table.                                                         |
| 6   | **Mutable in place, no atomicity.** `save_detrend_const` rewrites the CSV where it stands.                                                                                                                                                         | No history, no review, torn writes possible; the deployed file and the git source diverge silently.                                                 |
| 7   | **Frame baked into the filename** (`itrf2008`), no path for ITRF2014/2020 coexistence or plate-fixed variants.                                                                                                                                     | Frame migration requires a parallel file and code changes.                                                                                          |
| 8   | **Column-name case chaos** (`STA` vs `Sitename` vs `useSTA`/`UseSTA`) already caused the dead-branch `KeyError('Fit')`.                                                                                                                            | Fragile parsing.                                                                                                                                    |

Everything below is designed so each row of this table has an explicit fix.

### 0.2 What already exists and is reused (not reinvented)

- **Estimation math** — `gps_analysis.fitting`: `reject_outliers` (robust
  M-estimate + MAD rejection + final clean WLS, Huber 1964; Rousseeuw & Croux 1993) and `fit_components` (closed-form SVD WLS for the linear-in-parameters
  house models, exact at absolute `yearf` via internal re-centering). The
  estimator below is a thin orchestrator over these two, exactly like
  `velocity.estimate_velocity` is.
- **Application math** — `gps_analysis.fitting.remove_trend(model, t, y, fits)`
  already evaluates the model at **arbitrary epochs** and subtracts. Applying
  stored parameters to live data requires _no new math_ — only
  reconstruction of `TrajectoryParams` from the store and one term-selection
  convenience (§5.3).
- **Windowing** — `gps_analysis.baseline.slice_window`.
- **Model registry** — `models.linear` / `models.periodic` /
  `models.lineperiodic` with the fixed parameter order
  `[offset, rate, cos_annual, sin_annual, cos_semiannual, sin_semiannual]`
  and the **absolute-`yearf` phase convention** (trig arguments `2πt`, `4πt`
  on absolute fractional year — the same convention the legacy CSV
  coefficients use, which is what makes migration possible at all).
- **Provenance precedent** — `VelocityEstimate.t_ref`/`span`/`method` and the
  store's `Provenance(method, frame, fitted_at, source, extra)`.
- **Steps sidecar** — `config-templates/analysis-lane/steps.csv` (known
  equipment/earthquake offsets; manual-first).

---

## 1. Architecture in one paragraph

The **precompute job** (`gps_api.precompute`, the only cross-package caller)
runs an explicit **estimation** command that fits the configured trajectory
model per station over a configured, step-free fit window using the leaf's
robust fit, and writes a **candidate parameter file** with full provenance.
BGÓ reviews the diff and commits the file into **`gps-config-data`**, from
where it deploys to `~/.config/gpsconfig/detrend_params.json` like every other
config artifact. **Application** is pure evaluation: any consumer
(`geo_dataread` readers, the precompute job's detrended-series product) loads
the deployed parameters via `gps_parser` and calls the leaf's
`remove_trend`/`apply_detrend` on whatever epochs it has — including epochs
newer than the fit. The **raw series is never modified**: detrended values are
always additional columns/arrays alongside the raw ones, and the operation is
exactly invertible (`raw = detrended + f(t; p̂)`). A **staleness policy**
(age, extrapolation span, residual drift, new steps, frame change) is
evaluated on every precompute run and reported — refresh is always a
deliberate, reviewed act, never an automatic overwrite, because silently
changing detrend parameters silently changes every consumer's view.

```
                    ESTIMATE (occasional, reviewed)
  .NEU raw series ──► gps_api.precompute detrend-estimate ──► candidate detrend_params.json
                          │  (leaf: reject_outliers → fit_components)      │ human review
                          ▼                                                ▼
                    staleness report                    ~/git/gps-config-data (git) ──deploy──►
                                                        ~/.config/gpsconfig/detrend_params.json
                    APPLY (every read / every run, pure)                   │
  .NEU raw series ──► geo_dataread / precompute ── gps_parser reads ◄──────┘
                          │ leaf: remove_trend(model, t, y, fits)
                          ▼
                    raw + detrended view (raw preserved, view invertible)
```

---

## 2. (a) Estimation procedure

### 2.1 Inputs

Per station, from the caller (precompute job):

- `t` — epochs, fractional years (`yearf`, gtimes convention), the raw series
  as read from `.NEU` (or the API store), **outlier-unscreened** (the
  estimator does its own rejection) but uncertainty-screened per
  `preprocess.screen_uncertainty` if the reading profile already does so.
- `y` — (3, N) N/E/U displacements [mm], **in the declared reference frame**
  (§2.5).
- `sigma` — (3, N) formal 1-σ uncertainties [mm] (from the `.NEU` columns);
  used as WLS weights.
- `model` — from `analysis.yaml` `detrend.default_model` /
  `detrend.overrides.<STA>.model` (today: `lineperiodic`; `linear` allowed).
- `fit_window` — §2.2.
- `steps` — the station's rows from `steps.csv` (§2.3).

### 2.2 Fit-window selection

The window defines what "background" means; choosing it is a _policy_
decision, so it lives in config, with a mechanical validity check in the leaf:

1. **Explicit per-station window** (preferred where it matters):
   `detrend.overrides.<STA>.fit_window: [start_yearf, end_yearf]` in
   `analysis.yaml`. This is the migration target for the CSV's
   `Starttime`/`Endtime` columns and the operator's tool for "fit the
   pre-unrest interval".
2. **Default policy** (no override): the trailing
   `detrend.default_fit_window_years` (proposed default: **4.0 yr**) ending at
   the last epoch, _clipped back_ so the window contains no step from
   `steps.csv` (§2.3) — i.e. the window becomes the longest step-free suffix
   of the default window.
3. **Validity gates** (leaf-enforced, hard errors — a silently bad background
   is worse than no background):
   - span ≥ `min_span_years` (default **2.5 yr** — shorter windows alias the
     annual signal into the rate; Blewitt & Lavallée 2002 show rate bias is
     unbounded below ~2.5 yr of data),
   - `n_epochs ≥ min_epochs` (default **365**),
   - largest gap ≤ `max_gap_years` (default **0.5 yr**) so the seasonal terms
     are constrained around the year.

Stations that fail the gates get **no parameters** (absent from the store),
and consumers fall back to the raw view — the explicit successor of the
all-zero placeholder rows (§0.1 #5).

### 2.3 Steps (offsets) inside the window

v1 keeps the leaf model step-free and demands a step-free window (rule 2
above): equipment/coseismic offsets are _excluded by window choice_, sourced
from `steps.csv`. This is honest and simple, and matches the current
`steps.csv` "manual-first" state.

v2 (explicit non-goal now, but the schema reserves room): extend the design
matrix with Heaviside columns at the known step epochs
(Bevis & Brown 2014, eq. 1 full form) so long windows spanning steps become
usable; the stored record then carries the estimated step amplitudes. The
schema's per-station `steps_in_window` list (§3) is written even in v1
(empty ⇒ window verified step-free) so v2 is additive.

### 2.4 Estimator

Per station, per component (N, E, U), reusing the leaf verbatim:

1. **Window**: `mask = slice_window(t, start, end)`.
2. **Robust fit + outlier rejection**: `reject_outliers(model, t[mask],
y[:, mask], sigma[:, mask], loss="soft_l1", f_scale=1.0, n_sigma=3.0,
max_iterations=5)` — M-estimate, normalized-MAD rejection, then a final
   plain WLS on inliers so the reported covariance is Gauss–Markov
   (`fitting.py` steps 1–4). These defaults are config-surfaced
   (`detrend.estimator:` block) but not per-station tunables.
3. **Result**: per component `TrajectoryParams` — `params` (P,) in the
   model's canonical **absolute-t parameterization** (the leaf already maps
   its internal centering back exactly, `_fit_linear_design`), `covariance`
   (P, P) with the reduced-χ² rescaling (`absolute_sigma=False`, the legacy
   convention).
4. **Diagnostics** recorded for provenance and staleness baselining:
   `n_epochs`, `n_rejected`, inlier residual RMS per component [mm], window
   actually used `(t_first, t_last)`.

**Which parameters:** all P model parameters _including the intercept_ —
fixing draft-shortcoming #1. The intercept is meaningless across reference
re-zeroing but costs one number and makes the stored record a complete,
directly evaluable model; consumers that want a different zero apply their
own re-referencing (`baseline.remove_offset` / `vshift`) _after_ detrending,
as an explicitly separate view step.

**Uncertainties:** the WLS covariance is stored (upper triangle). Formal
white-noise σ is optimistic for GNSS (Williams 2003, J. Geodesy 76) — the
record carries `sigma_kind: "wls_formal"` so a later colored-noise MLE
estimate (`estimate_noise_mle`, `powerlaw_rate_sigma`) can coexist as
`sigma_kind: "mle"` without a schema change.

### 2.5 Reference-frame handling

Decision: **estimate and store in the raw processing frame** (the frame of
the `.NEU` products, currently ITRF2008 for the legacy chain and ITRF2014 for
the analysis lane; per-region `default_reference_frame` in `analysis.yaml`).

Rationale: the secular rate of the trajectory model _already contains_ plate
motion — detrending in the raw frame removes plate motion and local secular
motion and seasonal terms in one subtraction. Plate-motion removal
(`geofunc.plateVelo`, MORVEL/ITRF Euler poles) remains a **separate,
model-based view** for users who want "motion relative to the plate" _with_
local secular signal retained. The two views must not be conflated:

- `ref="itrf20xx"` (raw) — no subtraction.
- `ref="plate"` — subtract Euler-pole prediction (existing
  `_remove_plate_velocity`; unchanged).
- `ref="detrend"` — subtract stored trajectory model (this design; replaces
  the dead `ref="detrend"` branch removed in refactor-B).

Consequences:

- The parameter file declares its frame **once, at document level** (`frame:
"ITRF2014"`), fixing draft-shortcoming #7. One file per frame; coexisting
  frames = coexisting files (`detrend_params.json` naming stays frame-free;
  the frame is _inside_ the document, and the deploy chooses which document
  ships). A frame change (ITRF2020 adoption, reprocessing campaign) is a
  full re-estimation campaign by definition (§6, trigger T5).
- Applying parameters to a series in a different frame is an error the
  apply-side reader must detect (compare series frame tag vs document
  `frame`) and refuse, not fudge.

### 2.6 Borrowing (the `UseSTA` mechanism, kept but explicit)

New/short-history stations near an established one may borrow coefficients —
the legitimate need behind `UseSTA`/`Fit`. Redesign:

- Config-side _intent_: `detrend.overrides.<STA>.borrow_from: DYNG` +
  `borrow_terms: periodic | all` in `analysis.yaml`.
- The **estimator resolves borrowing at estimation time** and writes a
  _complete, self-contained record_ for the borrower (donor's coefficients
  copied for the borrowed terms; own WLS fit for the rest — e.g.
  `borrow_terms: periodic` fits own offset+rate with donor seasonal terms
  held fixed, via the linear model's design-matrix partition: subtract the
  fixed seasonal prediction, fit `linear` to the remainder).
- Provenance records `borrowed: {from: DYNG, terms: periodic, donor_fitted_at: …}`.
- The apply path never chases donor references — no cross-row lookups at read
  time (kills the `KeyError('Fit')` class of bugs, #8).

---

## 3. (b) Stored-parameter schema, location, provenance

### 3.1 Format decision: structured JSON document, not a wider CSV

The record per station is nested (per-component parameter vectors +
covariances + provenance + borrowing). Flattening into CSV needs ≥ 60
columns and reintroduces the case/semantics chaos of #5/#8. Decision:
**one JSON document per frame**, `detrend_params.json`, schema-versioned,
diff-able and reviewable in git, atomically replaced on deploy (write temp +
rename). YAML was considered (house style for `analysis.yaml`) and rejected
for the _data_ file: float round-tripping and programmatic writing are
first-class in JSON, and this file is machine-written/human-reviewed, not
human-authored. (`analysis.yaml` stays YAML — it is policy, hand-edited.)

### 3.2 Schema (v1)

```jsonc
{
  "schema_version": 1,
  "frame": "ITRF2014",                    // document-level; apply must match series frame
  "units": { "displacement": "mm", "rate": "mm/yr", "time": "yearf" },
  "phase_convention": "absolute_yearf",   // trig args 2πt, 4πt on absolute fractional year
                                          // == gps_analysis.models.periodic convention
  "generated_at": "2026-07-13T12:00:00Z", // document write time
  "generator": "gps_api.precompute detrend-estimate",
  "software": { "gps_analysis": "0.1.0", "gps_api": "0.3.0" },

  "stations": {
    "SENG": {
      "model": "lineperiodic",            // gps_analysis.models registry name
      "param_names": ["offset", "rate", "cos_annual", "sin_annual",
                       "cos_semiannual", "sin_semiannual"],  // == model positional order
      "fit_window": [2015.0000, 2019.9000],   // requested window [yearf]
      "span_used": [2015.0863, 2019.8964],    // first/last inlier epoch actually used
      "fitted_at": "2026-07-13T11:58:41Z",
      "estimator": {
        "name": "reject_outliers+wls",
        "loss": "soft_l1", "f_scale": 1.0, "n_sigma": 3.0,
        "absolute_sigma": false,
        "sigma_kind": "wls_formal"        // "mle" reserved (Williams 2003 honesty upgrade)
      },
      "n_epochs": 1734,
      "n_rejected": [23, 19, 41],          // per component N,E,U
      "rms_mm": [1.21, 1.05, 4.4],         // inlier residual RMS per component
      "steps_in_window": [],               // v1: must be empty (window verified step-free);
                                           // v2 reserves step records here
      "borrowed": null,                    // or {"from": "DYNG", "terms": "periodic",
                                           //     "donor_fitted_at": "..."}
      "components": {
        "north": {
          "params": [1.234567890123, 15.395505050635, 1.061041857709,
                     -0.384356564020, 0.193717165083, 0.011476751780],
          "cov_upper": [ /* P(P+1)/2 = 21 numbers, row-major upper triangle */ ]
        },
        "east":  { "params": [...], "cov_upper": [...] },
        "up":    { "params": [...], "cov_upper": [...] }
      },
      "source": "cdn .NEU rapid",          // which series product was fitted
      "notes": null                        // free text (migration flags live here too)
    }
  }
}
```

Rules:

- `params` order **is** the model callable's positional order — the same
  contract as `TrajectoryParams`; `param_names` is redundant-by-design
  (validation + human readability), and a reader must verify it against the
  model registry.
- Full covariance (upper triangle), not just diagonals: 21 numbers per
  component buys exact linear error propagation into any derived view
  (GUM/JCGM 100:2008 §5.1.2) and costs nothing. `sigma` vectors are derived,
  not stored (no redundancy to drift).
- Floats serialized with `repr`-full precision (17 significant digits) so
  store→load→apply is bit-identical to fit→apply.
- A station **absent** from `stations` = "no background model" (consumers
  serve raw only) — the explicit replacement for all-zero rows.
- Unknown `schema_version`, `frame` mismatch, `param_names` mismatch, or
  non-finite `params` ⇒ reader raises; no silent NaN-tolerant paths
  (draft-shortcoming #3/#5).

### 3.3 Location and ownership — and the config-deploy overlap

Two candidate homes were weighed:

| Home                                                       | Pro                                                                                                                                                                                                       | Con                                                                                                                     |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **gpsconfig** (`gps-config-data` → `~/.config/gpsconfig/`) | Git provenance + human review; deployed everywhere `geo_dataread` runs (incl. hosts without the API store); same channel as `stations.cfg`/`analysis.yaml`/`steps.csv`; matches how the CSV already flows | Machine-written data in a "config" repo (mitigated: candidate-file + review workflow, §6.2)                             |
| Precompute store (`$GPS_API_STORE`)                        | Naturally machine-written                                                                                                                                                                                 | Not deployed to legacy hosts; cache-semantics (deletable); no review gate; `geo_dataread` would need a store dependency |

**Decision: gpsconfig, via `gps-config-data`** — the parameters are
slow-moving, review-worthy _reference data_ (like `steps.csv`), not run
artifacts. The precompute store additionally stamps a **copy of each used
station record** into its product provenance (`meta/run.json` +
per-series Parquet metadata) so API consumers can audit the view without
reaching into gpsconfig.

**Relation to `analysis.yaml` (the in-flight config deploy — the
recommendation BGÓ asked for):**

- **Policy vs data split.** `analysis.yaml` keeps _policy only_: model
  choice (`detrend.default_model`, `overrides.<STA>.model` — already
  deployed keys, unchanged), plus **new policy keys** `default_fit_window_years`,
  per-station `fit_window` / `borrow_from` / `borrow_terms`, the
  `estimator:` sub-block, and the staleness thresholds (§6). The **fitted
  numbers never go into `analysis.yaml`** — they live in
  `detrend_params.json`. This keeps the hand-edited file hand-editable and
  the machine-written file atomic.
- **Wiring key:** `analysis.yaml` gains `detrend.params_file:
detrend_params.json` (bare filename, resolved against the gpsconfig dir —
  same pattern as `store.path`/`data.neu_dir`), mirrored by an additive
  `postprocess.cfg` key `[FILES] detrend_params = detrend_params.json` for
  the legacy `gps_parser` reader surface.
- **Legacy key:** `postprocess.cfg`'s existing `detrendFile =
detrend_itrf2008.csv` **stays untouched** until the `geo_dataread`
  migration lands, then is removed in a follow-up deploy. Old and new files
  coexist during transition; nothing breaks on either side of the deploy.
- **Concrete recommendation for the config deploy:** ship it as **one
  additive bundle** — `analysis.yaml` (with the new `detrend:` keys),
  `steps.csv`, and the **migrated** `detrend_params.json` (§3.4) — into
  `~/git/gps-config-data` templates and deploy together. The deploy is safe
  to do _before_ any code lands: every key is additive, and no current code
  reads the new file. Do **not** fold the CSV's numbers into `analysis.yaml`,
  and do not wait for the estimator to exist — the migrated file gives the
  apply-side work a real target immediately.

**The deployed `detrend:` section, concretely** (the exact shape the deploy
should ship; existing keys unchanged, new keys additive — the precompute
`config.py` reader ignores unknown keys today, so nothing breaks before the
reader slice lands):

```yaml
detrend:
  default_model: lineperiodic # existing key — unchanged
  params_file:
    detrend_params.json # NEW: fitted-numbers document, bare
    #   filename resolved against gpsconfig
  default_fit_window_years: 4.0 # NEW: trailing window when no override (§2.2)
  estimator: # NEW: leaf reject_outliers settings (§2.4)
    loss: soft_l1
    f_scale: 1.0
    n_sigma: 3.0
    min_span_years: 2.5 # validity gates (§2.2 rule 3)
    min_epochs: 365
    max_gap_years: 0.5
  staleness: # NEW: refresh-trigger thresholds (§6.1)
    max_age_years: 2.0 # T1
    max_extrapolation_years: 3.0 # T2
    drift_window_days: 90 # T3
    drift_sigma: 3.0 # T3
  overrides: # existing key — value shape extended
    SENG:
      model: exp_linear # existing sub-key
      fit_window: [2015.0, 2019.9] # NEW: explicit per-station window (§2.2 rule 1)
    OLAC:
      borrow_from: DYNG # NEW: borrowing intent (§2.6)
      borrow_terms: periodic #      "periodic" | "all"
```

No fitted coefficient ever appears in this block — it is policy top to
bottom; the numbers live exclusively in `detrend_params.json` (§3.2).

### 3.4 Migration from `detrend_itrf2008.csv`

One-shot converter (lives in `gps_api.precompute`, run once, output reviewed):

- `frame: "ITRF2008"` (from the filename — now explicit).
- Per row: `model: "lineperiodic"`; `params = [0.0, rate, acos, asin, scos,
ssin]` per component (**offset explicitly 0.0** — preserving the legacy
  `convconst` semantics); `cov_upper: null` (unknown); `fit_window` from
  `Starttime`/`Endtime` (null-tolerant → `notes`); `fitted_at: null`;
  `estimator: {name: "legacy_csv"}`; `notes: "migrated from
detrend_itrf2008.csv"`.
- All-zero rows (the Reykjanes placeholders) → **omitted** (absent =
  no model), with the omission list in the converter report.
- `UseSTA`/`Fit` rows → coefficients are already materialized in the CSV
  row, so they migrate as own records with `borrowed: {from: <UseSTA>,
terms: <fit>, donor_fitted_at: null}`.
- `cov_upper: null` and `fitted_at: null` are the only permitted nulls, and
  only via the migration path; they mark records for early refresh (§6,
  trigger T1 fires immediately for null `fitted_at`).

Because the legacy stored offset is 0 and the legacy apply then re-zeroes
with `vshift`, applying a migrated record + `vshift` reproduces the legacy
detrended view exactly — this is the migration parity test (§7).

---

## 4. (c) Apply-to-live-data path — raw preserved

### 4.1 Semantics

Application is stateless pure evaluation: given stored `(model, params)` and
_any_ epochs `t` (historical, today's, tomorrow's),

```
detrended(t) = y(t) − f(t; p̂)          — remove_trend, already exists
raw(t)       = detrended(t) + f(t; p̂)  — exactly invertible
```

There is no incremental state, no fitting, no window — new epochs are simply
new arguments. Extrapolation beyond the fit window is _by design_ (that is
what live detrending is) but is bounded by the staleness policy (§6): the
apply path always works; the _freshness_ of the parameters is a reported,
monitored property, never a silent one.

**Raw preservation (the outlier precedent, made a rule):**

1. `.NEU` files and the Parquet raw columns are never rewritten by any
   detrend operation.
2. Every product that carries detrended values carries the raw values beside
   them — the Parquet series already does this (`north` /
   `north_detrended`); `geo_dataread`'s DataFrame gains the same convention
   (§4.2) instead of the legacy in-place overwrite of `data`.
3. Every detrended view carries provenance sufficient to invert it (model
   name + params reference), so "detrended" is always a derived, reversible
   annotation of the raw record — exactly like the outlier mask never
   deleting samples.

### 4.2 Consumer paths

**`geo_dataread` (interactive/live reads, gpsplot, scripts):**

- New reader `read_detrend_params(path | None) -> DetrendParamsDoc` (path
  default resolved via `gps_parser` `[FILES] detrend_params`). Replaces
  `getDetrFit`/`convconst`; a thin compat shim may keep `getDetrFit` alive
  reading the _old CSV only_ until its callers are migrated, then both die
  with `save_detrend_const`.
- `read_gps_data(..., ref="detrend")` is **revived** (the branch is currently
  an explicit `ValueError`): loads the station's stored record, checks frame
  compatibility, evaluates and subtracts via the leaf, and returns the
  DataFrame with **both** raw (`north`, …) and detrended (`north_detrended`,
  …) columns plus the record's provenance attached (`DataFrame.attrs`
  or the returned `const` slot). Missing station record ⇒ warn + raw-only
  columns (no crash, no zero-fit fallback — the legacy "estimate from the
  whole dataset and detrend" fallback is **dropped**: silent refits are the
  disease this design cures; callers who want a fresh fit call the leaf
  explicitly).
- Partial views (the old `detrend_periodic` / `detrend_line` switches) map to
  `terms="periodic"` / `terms="secular"` of §5.3 — same stored params, terms
  selected at apply time, no separate parameter sets.

**`gps_api.precompute` (batch, feeds the API/aflogun):**

- The per-station stage switches its detrended-series product from
  fit-per-run to **stored-params apply** when a record exists
  (`detrend_source: "stored"` + record provenance stamped into the product);
  stations without a record keep the current fresh fit
  (`detrend_source: "fitted_this_run"`) so the fleet keeps working during
  rollout. This makes the detrended view _stable across daily runs_ and
  _identical_ to what `geo_dataread` serves — one background definition
  everywhere.
- The run _also_ performs a fresh fit regardless (it already does, cheaply)
  — not to serve, but to power the residual-drift staleness check (§6, T3)
  by comparing fresh vs stored predictions over the trailing window.

**`aflogun` / `gps_plot`:** consume raw + detrended columns from the API
store / Parquet; never fit, never read the params file directly.

---

## 5. (e) Atomic leaf functions and caller responsibilities

Leaf rules R2/R6 hold: everything below is pure, array-first, unit-agnostic,
numpy/scipy/gtimes only, no file I/O, no config reads. All new functions get
MATH_STANDARDS §2-compliant docstrings.

### 5.1 `gps_analysis.detrend` (new thin module) — estimation orchestrator

```python
@dataclasses.dataclass(frozen=True)
class DetrendEstimate:
    """Stored-detrend estimation result for one station (all components)."""
    fits: tuple[TrajectoryParams, ...]      # per component, absolute-t parameterization
    inliers: NDArray[np.bool_]              # (C, N_window) mask (raw-preserving, outlier precedent)
    span_used: tuple[float, float]          # first/last inlier epoch [yearf]
    n_epochs: int                           # window epochs offered to the fit
    n_rejected: tuple[int, ...]             # per component
    rms: tuple[float, ...]                  # inlier residual RMS per component [L]
    window: tuple[float, float]             # requested window [yearf]

def estimate_detrend(
    model: ModelFunc,
    t: ArrayLike,                 # epochs, yearf [yr]
    y: ArrayLike,                 # (C, N) or (N,) observations [L]
    sigma: ArrayLike | None = None,
    *,
    window: tuple[float | None, float | None] = (None, None),
    min_span_years: float = 2.5,  # Blewitt & Lavallée 2002 rate-bias floor
    min_epochs: int = 365,
    max_gap_years: float = 0.5,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    n_sigma: float = 3.0,
    max_iterations: int = 5,
    absolute_sigma: bool = False,
    names: Sequence[str] | None = ("north", "east", "up"),
) -> DetrendEstimate
```

Composition (no new math): `slice_window` → validity gates (raise
`ValueError` with the failed gate named) → `reject_outliers` → package.
Mirrors the `estimate_velocity` precedent.

### 5.2 Serialization: `TrajectoryParams.to_record()` / `from_record()`

```python
class TrajectoryParams:
    def to_record(self) -> dict[str, Any]:
        """{'params': [...], 'cov_upper': [...] | None, 'component': ...} — JSON-ready,
        full-precision floats; pure (no I/O — file writing is the caller's business)."""
    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrajectoryParams":
        """Inverse of to_record; validates shapes; cov_upper=None -> covariance
        filled with +inf (the 'could not be estimated' convention)."""
```

Round-trip is bit-exact (test-pinned, §7). The _document_ schema (§3.2) is
owned by the caller (`gps_api.precompute`); the leaf only owns the
per-component record shape.

### 5.3 Term-selected application: `select_terms` + existing `remove_trend`

```python
def select_terms(
    model: ModelFunc,
    fits: TrajectoryParams | Sequence[TrajectoryParams],
    terms: str,                    # "all" | "secular" | "periodic"
) -> list[TrajectoryParams]:
    """Zero out the non-selected coefficients of a house-model fit so that
    remove_trend(model, t, y, select_terms(model, fits, "periodic"))
    subtracts only the seasonal part. Defined for the registered
    linear-in-parameters models (parameter-index masks per model:
    lineperiodic — secular={offset,rate}, periodic={4 trig coeffs});
    raises for unregistered callables. Covariance rows/cols of zeroed
    parameters are zeroed too (the subtracted quantity's covariance)."""
```

Because the house models are linear in parameters, zeroing coefficients _is_
term removal — no new evaluator needed; `remove_trend` stays the single
application primitive. (`"secular"` keeps the offset with the rate so the
line-only view matches the legacy `detrend_line` behavior of subtracting
`p[0:2]`.)

### 5.4 Caller responsibilities (who does what)

| Concern                                                                                                        | Owner                                                                                                         |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Fit-window policy, model choice, estimator settings, staleness thresholds                                      | `analysis.yaml` (policy), read by `gps_api.precompute.config`                                                 |
| Running estimation, steps-aware window clipping, candidate-file writing, staleness report, migration converter | `gps_api.precompute` (new `detrend.py` stage + `gps-api-detrend-estimate` CLI)                                |
| `detrend_params.json` schema validation, document read/write, atomic write                                     | `gps_api.precompute` (write); `geo_dataread` (read); shared _shape_ defined by leaf `to_record`/`from_record` |
| Params file deployment + review                                                                                | `~/git/gps-config-data` (git) → `~/.config/gpsconfig/`                                                        |
| Config resolution (`params_file` key)                                                                          | `gps_parser` (`getAnalysisConfig()` / `[FILES]` key — part of the in-flight config task)                      |
| Apply-on-read, raw+detrended DataFrame, frame check                                                            | `geo_dataread`                                                                                                |
| Apply-in-batch, Parquet raw+detrended, provenance stamping, drift monitoring                                   | `gps_api.precompute`                                                                                          |
| All math (fit, reject, evaluate, term selection, serialization shape)                                          | `gps_analysis` (leaf) — **never** reads files or config                                                       |

---

## 6. (d) Staleness / refresh policy

### 6.1 Triggers (evaluated by every precompute run; reported, never auto-acted)

| #   | Trigger                                                                           | Default threshold (policy keys under `detrend.staleness:`) |
| --- | --------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| T1  | **Age**: `now − fitted_at > max_age_years` (null `fitted_at` ⇒ immediately stale) | 2.0 yr                                                     |
| T2  | **Extrapolation**: `last_data_epoch − span_used[1] > max_extrapolation_years`     | 3.0 yr                                                     |
| T3  | **Residual drift**: over the trailing `drift_window_days`, `| median(y − f(t; p̂_stored))                                 | > drift*sigma × rms_mm` per component (median, not mean — robust to remaining outliers). Detects both genuine transients and background change; a \_flag*, not an oracle — the human decides which it is (during unrest T3 firing is _expected_ and is exactly when you must NOT refit). | 90 d, 3.0 |
| T4  | **New step**: `steps.csv` gains a step inside `[span_used[0], last_data_epoch]`   | — (any)                                                    |
| T5  | **Frame change**: series frame tag ≠ document `frame`                             | — (hard error on apply, campaign-level refresh)            |

Flags land in `meta/run.json` per station
(`detrend_stale: ["T1", "T3:north"]`) and in the API provenance so aflogun
can badge the view.

### 6.2 Refresh procedure (deliberate, reviewed — the write path)

1. Operator runs `gps-api-detrend-estimate [--stations …|--region …|--stale-only]`.
2. The tool writes a **candidate** `detrend_params.candidate.json` (never the
   live file) + a human-readable diff report: per station, old vs new rate
   (with σ), seasonal amplitude change, window change, and the trigger that
   motivated the refresh.
3. BGÓ reviews, copies into `~/git/gps-config-data`, commits (git history =
   parameter history — fixes #6), deploys to `~/.config/gpsconfig/`.
4. Consumers pick the new file up on next read/run; the store's provenance
   `fitted_at`/`generated_at` change makes the view change auditable.

No automatic overwrite path exists **by design**: a detrend-parameter change
moves every plot and every API series at once; that is an operator decision.
(If a scheduled refresh is ever wanted, it schedules _step 1–2 + a
notification_, never step 3.)

---

## 7. (f) Test plan

### 7.1 Leaf (`gps_analysis`, `tests/test_detrend.py`)

1. **Recovery** — synthetic `lineperiodic` truth + white noise + injected
   outliers (5 %, 10× σ): `estimate_detrend` recovers each parameter within
   3σ of its reported uncertainty; outlier mask catches ≥ 95 % of injected
   points, ≤ 1 % false positives (tolerances stated per MATH_STANDARDS §4).
2. **Validity gates** — window shorter than `min_span_years`, too few
   epochs, an over-long gap: each raises `ValueError` naming the gate.
3. **Invertibility (raw preserved)** — `remove_trend(model, t, y, fits) +
model(t, *p̂) == y` to float64 eps; inputs never mutated (id + content
   checks — regression against the legacy in-place behavior).
4. **Live-apply property** — fit on `t ≤ T`, apply on `t > T`:
   `detrended(t_new) == y(t_new) − f(t_new; p̂)` exactly; and _restriction
   consistency_: applying to a sub-window equals slicing the full
   application (no window dependence in apply).
5. **Term selection** — `select_terms(..., "periodic")` +
   `select_terms(..., "secular")` predictions sum to the `"all"` prediction
   (linearity); `"secular"` on `lineperiodic` matches a manual `p[0:2]`
   line subtraction (legacy `detrend_line` parity); unregistered model raises.
6. **Serialization round-trip** — `from_record(to_record(x))` bit-identical
   (`params`, `covariance`, incl. the ±inf covariance case and
   `cov_upper: null`); JSON dump→load preserves float64 exactly
   (17-significant-digit repr).
7. **Absolute-t / phase convention** — a fit made on epochs shifted by an
   integer year changes only offset-related parameters as predicted;
   evaluating stored params at the original epochs reproduces the original
   prediction (guards the `absolute_yearf` phase contract the CSV migration
   depends on).

### 7.2 Migration (converter in `gps_api`, `tests/test_detrend_migration.py`)

8. **Converter golden** — run on a vendored copy of the real
   `detrend_itrf2008.csv`: row counts match (minus all-zero omissions,
   which are listed); DYNG/OLAC/TANC spot-check values bit-equal; borrow
   rows carry `borrowed` provenance.
9. **Legacy parity** — for a fixture station (e.g. DYNG with its archived
   `.NEU`): migrated-params apply + `vshift` reproduces the legacy
   `read_gps_data`-detrended output within float64 round-off (pins the
   offset-0 + re-zero semantics before the legacy path is deleted).

### 7.3 Reader/apply (`geo_dataread`)

10. **Stored apply** — `read_gps_data(..., ref="detrend")` returns raw _and_
    `_detrended` columns; raw columns byte-equal to `ref="itrf..."` read of
    the same window.
11. **Fallbacks** — station absent from the params file: warn + raw-only,
    no zero-fit, no crash; frame mismatch: hard error naming both frames.
12. **Schema rejection** — unknown `schema_version`, `param_names` mismatch,
    NaN params each raise.

### 7.4 Precompute (`gps_api`, extend `test_precompute.py`)

13. **Stored-vs-fitted switch** — with a params record present the Parquet
    `*_detrended` columns come from stored params (provenance
    `detrend_source: "stored"`, record copied into run meta); without one,
    fresh-fit path with `"fitted_this_run"` — both asserted.
14. **Cross-consumer consistency** — the API-served detrended series and a
    `geo_dataread` stored-apply of the same raw window are identical (one
    background definition everywhere — the core promise of the design).
15. **Staleness triggers** — synthetic cases fire exactly T1 (old
    `fitted_at`), T2 (long extrapolation), T3 (injected post-window ramp;
    median-based flag fires on the ramped component only), T4 (step added
    to steps fixture); flags appear in `meta/run.json`.
16. **Estimation CLI end-to-end** — `gps-api-detrend-estimate` on the
    synthetic fleet writes a schema-valid candidate document + diff report;
    the live file is untouched.

### 7.5 Quality gates

Everything under the house rules: ruff + black + mypy(strict) zero warnings;
new leaf math docstrings pass the MATH_STANDARDS §2 checklist (equation,
symbols→args with units, reference, numerical notes); tolerances stated and
justified in each test.

---

## 8. Rollout order (small, independently landable slices)

1. **Leaf slice** — `estimate_detrend`, `select_terms`,
   `to_record`/`from_record` + tests 1–7. No consumers change.
2. **Migration + config deploy** — converter, generate
   `detrend_params.json` from the CSV, ship the config bundle
   (`analysis.yaml` new keys + `steps.csv` + params file) to
   `gps-config-data` (⚠️ needs user — out-of-project boundary), tests 8–9.
3. **Apply-side** — `geo_dataread` reader + `ref="detrend"` revival
   (tests 10–12); precompute stored-apply + provenance (tests 13–14).
4. **Estimation + staleness** — `gps-api-detrend-estimate` CLI + staleness
   reporting (tests 15–16). First real refresh: re-estimate the Reykjanes
   stations that today carry all-zero placeholder rows, in ITRF2014.
5. **Cleanup** — delete `getDetrFit`/`convconst`/`save_detrend_const`,
   remove the legacy `detrendFile` key from `postprocess.cfg` (second
   config deploy), retire `detrend_itrf2008.csv`.

---

## 9. Open questions (for review)

1. `max_age_years` / `max_extrapolation_years` defaults (2.0 / 3.0 yr) —
   BGÓ's call; secular Iceland background is stable enough that longer may
   be fine outside unrest zones.
   - bgo reply: as many years as possible for the estimate minimum 1-2 years
   - we need to be able to estimate periodic signal even during active periods if the wavelengh of the activity signal difference. usually manual (geo_dataread has some draft of this)
   - harder and only rarely possible estimate background signal during unrest periods, but we should try to do it if possible.
   - however we also have the option of reusing parameter between stations and you have example of this in the \*csv file and you can find in geo_dataread how I implement this
2. Should the precompute _fresh fit_ be dropped entirely once stored params
   cover the fleet (saving the per-run fit cost), keeping only the trailing
   drift check? (Design keeps it because it powers T3 cheaply.)
   a. yes we shold run a fresh fit only every few months to improve paramaeters but it has to be manuverable as well as some need updating other can be hard to estimate and we need to be able to fix the parameters so it has to flexible.
3. v2 step columns in the design matrix (§2.3): pull forward if a long
   Reykjanes background window spanning the 2023–24 dikes is wanted, or keep
   windows step-free and short?
   a. not sure I understand the question estimating bacrkround during intence activity is almost impossible but if there are long time series predating the activity in the visinity we sometimes can apply those parameters to the station in the active area and get a good estimate of the background signal.
4. One params document per frame (§2.5) — is a plate-fixed params document
   ever wanted, or is `ref="plate"` + `ref="detrend"` composition always
   sufficient? (Design assumes the latter; the rate difference is exactly
   the Euler-pole prediction, so a plate-fixed document is derivable.)
   a. makes no since to detrend untill after plate removal. (ITRF reference frame reprecentation is hardly ever used except to check integrity of the time series it has no relevance for tectonic movements)

---

_References: Bevis & Brown 2014 (J. Geodesy 88, eq. 1 — trajectory model);
Blewitt & Lavallée 2002 (JGR 107(B7) — seasonal/rate aliasing, window floor);
Williams 2003 (J. Geodesy 76 — colored-noise honesty of formal σ);
Huber 1964; Rousseeuw & Croux 1993 (robust fit/rejection, as implemented in
`gps_analysis.fitting`); JCGM 100:2008 (GUM §5.1.2 — covariance propagation)._

_Written 2026-07-13 (spec-first design; no implementation)._
