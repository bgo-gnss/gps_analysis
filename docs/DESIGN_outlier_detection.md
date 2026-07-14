# DESIGN — Outlier detection & removal for GNSS displacement time series

> **Status:** SPEC — no implementation yet. Target module: `gps_analysis.outliers`
> (leaf, Tier 1) + consumer wiring in `gps_api.precompute` and the series endpoint.
> Binding standards: [`MATH_STANDARDS.md`](MATH_STANDARDS.md) (atomicity, docstring
> contract, references, numerical notes). Author: analysis lane, 2026-07-13.

## 1. Requirements (BGÓ, hard)

1. **Proper statistics.** Robust, defensible estimators with literature references —
   no ad-hoc thresholds baked into code. Every threshold is a passed-in parameter
   with a documented default and a justification.
2. **Signal protection.** A coseismic offset, a post-seismic/inflation transient, or
   a slow-slip event must NOT be clipped as an "outlier". The design treats
   _protection of real signal_ as a first-class stage, not a side effect of tuning.
3. **Non-destructive, always.** Outlier handling FLAGS epochs; it never overwrites,
   drops, interpolates or replaces the original observations. The raw series is
   preserved end-to-end (leaf → store → API), and every consumer can request it.
4. **BGO additon**. we need to have the option of ither removing/marking outlierss when plotting and also the opiton of creating clean time series files with the outliers removed
5. outlier detection needs to have the possibility of being iterative in a since that say rough large outliers removal before detrending is estimated, the outlier detection again after detrending, adding the backround signal back in and re detrend. this sort of workflow that can converge on a solution

## 2. Prior art and why it is insufficient

| Source                                                         | Approach                                                               | Defect                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gps_data_analyses/svartsengi-model/examples/multi_station.py` | a-priori station exclusion set + MAD sigma-clip on fitted _parameters_ | hand-listed exclusions carry no provenance; clip is on cross-station parameter scatter, not per-epoch data; `sigma_clip=2.0` is aggressive                                                                                                                                              |
| `gps_data_analyses/rtk-postproc/gps/ParaRTK/plot/*.py`         | `df.X.rolling(k).median()` — rolling-median **replacement**            | destroys data (violates req. 3); biases every epoch, not just bad ones; smears real steps by ±k/2 samples                                                                                                                                                                               |
| `gps_analysis.fitting.reject_outliers` (current)               | robust M-fit + global MAD clip, iterate                                | model-aware ✓, but: **global** scale only (misses local outliers under colored-noise wander, over-flags during unmodeled transients); no signal-protection stage; no known-step terms; returns an inlier mask but no diagnostics/reasons; no stopping safeguard beyond mask fixed point |
| `gps_analysis.preprocess.screen_uncertainty`                   | formal-σ threshold screen                                              | quality screen, not outlier detection — kept as a _separate_ upstream stage with its own reason code                                                                                                                                                                                    |

The new design keeps what works (residual-based detection after a robust trajectory
fit; normalized MAD) and adds: known-step model terms, windowed (Hampel) detection,
formal-σ studentization, an explicit signal-protection stage, a conservative
iteration/stopping rule, reason-coded diagnostics, and non-destructive plumbing.

## 3. Methodology

### 3.1 Principle: model-aware detection on studentized residuals

Outliers are identified **against a robustly fitted trajectory model**, never
against the raw series. The model absorbs everything we _know_ is real —
secular rate, annual + semiannual seasonal terms, and **known steps** (equipment
changes from TOS, coseismic offsets from skjálftalísa, via the deployed
`steps.csv`) — so that real signal captured by the model can never be flagged.

Per component c ∈ {N, E, U} of one station:

```
f(t; p, a) = f_traj(t; p) + Σ_k a_k · H(t − t_k)        (step-augmented trajectory)
r_i        = y_i − f(t_i; p̂, â)                          (residuals)
w_i        = r_i / σ_i           (σ_i formal 1-σ; σ_i ≡ 1 when absent)
ẑ_i        = (w_i − med(w)) / ŝ                          (robust standardization)
```

- `f_traj` is the configured trajectory model (`lineperiodic` default; overrides per
  station as today) — Bevis & Brown 2014, J. Geodesy 88, eq. (1).
- `H` is the Heaviside step; step epochs `t_k` come from the caller (leaf never
  reads config). Step amplitudes `a_k` are _estimated_, epochs are fixed.
- The fit is an M-estimator (Huber loss, `scipy.optimize.least_squares`, Huber 1964)
  so gross outliers cannot drag the trajectory; final reported fits are plain WLS
  on inliers (Gauss–Markov covariance), exactly the existing
  `reject_outliers` convention.
- **Formal σ is used for relative weighting only.** GAMIT/GipsyX formal errors are
  optimistic by a variable factor, so the absolute detection scale ŝ is always
  estimated robustly from the whitened residuals themselves; σ_i contributes the
  epoch-to-epoch quality \_ratio_ (a noisy winter epoch needs a larger raw residual
  to be flagged than a quiet summer epoch). When σ is absent the same machinery
  runs unwhitened.

### 3.2 Robust scale

Two estimators, selected by config:

- **Normalized MAD** (default): `ŝ = 1.4826 · med_i |x_i − med(x)|` —
  Rousseeuw & Croux 1993, JASA 88. Breakdown 50 %, Gaussian efficiency 37 %.
- **Qn** (optional): `Qn = 2.2191 · c_N · {|x_i − x_j| : i < j}_(k)`,
  `k = C(h,2)`, `h = ⌊N/2⌋ + 1`, with the finite-sample factors `c_N` —
  Rousseeuw & Croux 1993; O(N log N) algorithm Croux & Rousseeuw 1992
  (_Computational Statistics_ 1). Breakdown 50 %, efficiency 82 %, and — unlike
  MAD — no implicit symmetry assumption, which matters for residuals during
  one-sided unmodeled signal.

### 3.3 Two identifiers: global (epoch-wise) AND windowed (Hampel)

GNSS daily residuals are **not white**: flicker/power-law noise (Williams 2003)
produces multi-day correlated wander. A single global test either misses local
outliers riding on the wander or over-flags the wander itself. Therefore:

**(a) Global identifier** — for gross blunders:

```
G_i : |ẑ_i| > k_g            (default k_g = 5.0)
```

Under Gaussian noise the expected number of false exceedances is
`N · 2Φ(−k_g)`: for N = 7300 (20 yr daily), k = 3 flags ≈ 20 good points,
k = 4 ≈ 0.5, k = 5 ≈ 0.004. Colored noise inflates the tails further, so the
_global_ threshold is deliberately high; the identifier-consistency framing is
Davies & Gather 1993, JASA 88 ("The identification of multiple outliers").

**(b) Windowed Hampel identifier** — for local outliers under wander and
nonstationary scale (winter snow noise, unrest periods):

```
m_i = med { w_j : |t_j − t_i| ≤ h }                     (local median, h = half-window)
s_i = 1.4826 · med { |w_j − m_i| : |t_j − t_i| ≤ h }    (local MAD about m_i)
L_i : |w_i − m_i| > k_w · max(s_i, s_floor)             (default k_w = 4.0, h = 15.5 d)
```

Hampel 1974 identifier; generalized-Hampel-filter framing: Pearson, Neuvo,
Astola & Gabbouj 2016, EURASIP J. Adv. Signal Process. 2016:87. The local median
tracks the low-frequency noise wander, so k_w can sit below k_g without
over-flagging. Windows are defined **in time, not samples** (daily series have
gaps): the window at t_i is `{j : |t_j − t_i| ≤ h}`. If the window holds fewer
than `n_min` samples (default 11), s_i falls back to the global ŝ (documented
degradation, never a silent zero). `s_floor > 0` guards the well-known Hampel
degeneracy where ≥ 50 % of a window is identical (MAD = 0 flags everything).

**Candidates** = `G ∪ L`, with a per-epoch uint8 reason bitmask
(`REASON_GLOBAL = 1`, `REASON_LOCAL = 2`).

Leverage note: externally studentized residuals with the hat-matrix correction
`(1 − h_ii)^{1/2}` (Belsley, Kuh & Welsch 1980, _Regression Diagnostics_, ch. 2)
are supported via an optional `leverage` argument; for N ≫ P daily series
h_ii ≈ P/N is negligible and the default omits it — the docstring must state this.

### 3.4 Signal protection (the decision stage)

Candidates are **not** flags. Every candidate must survive four protection rules
before it may be flagged. Each protection is recorded in a `protected` bitmask
(`PROTECT_FLOOR = 1`, `PROTECT_RUN = 2`, `PROTECT_STEP = 4`, `PROTECT_WINDOW = 8`).

1. **Physical magnitude floor.** A candidate with `|r_i| < a_min` is unflagged
    (defaults: 5 mm horizontal, 10 mm vertical — open question Q4). Rationale: on a
    very quiet station the robust scale collapses and a 3 mm wiggle can exceed
    5·ŝ; a residual that small is _never_ worth masking, and masking it biases
    noise-model estimation (κ, amplitudes) downstream.

2. **Run-length / same-sign guard (transient & step protection).** Candidates are
    grouped into clusters: consecutive candidate epochs with gaps ≤ `g` days
    (default 1.5). A cluster is **protected as suspected signal** — every member
    unflagged and reported in `suspected_events` — when _either_: - its **time span exceeds `L_max` days** (default 2.0) **and** ≥ `q` (default
    80 %) of its members share the residual sign — a genuine blunder is isolated
    or a very short burst; a step/transient/SSE produces a _sustained one-sided_
    residual run (this is precisely why offset detection is hard — Gazeaux
    et al. 2013, JGR 118, the DOGEx experiment); _or_ - the **step-evidence statistic** fires: with `r̄_pre = med{r_j : t_j ∈
[t_start − W_pre, t_start)}` and `r̄_post = med{r_j : t_j ∈ (t_end,
t_end + W_post]}`,

                   ```
                   D = |r̄_post − r̄_pre| / ŝ > k_step        (default k_step = 3.0, W = 10 d each)
                   ```

                   i.e. the series _stays offset_ after the cluster — a step, not a blunder.
                   A blunder cluster has `D ≈ 0` (series returns to the model).

    **Run-rule / blunder-cluster precedence (2026-07-14).** The span+sign run
    rule is _released_ when the step-evidence conclusively marks the cluster a
    blunder: both flanks present (`D` determinate), `D ≤ k_step`, **and** the
    background `B = max(|r̄_pre|, |r̄_post|)/ŝ ≤ k_step` (flanks at the model).
    Such a cluster returns to baseline with no net offset and no elevated
    background — a decided blunder — so it is flagged rather than
    `PROTECT_RUN`'d. A NaN/thin flank keeps `D` indeterminate → not conclusive →
    the run rule stands (conservative). This fixed a real miss: the SENG 2016
    −450 mm Up streak (4 consecutive days, returning to 0) was run-protected
    purely on span+sign. Genuine transients/SSEs have an elevated or
    indeterminate flank → never "conclusive blunder" → still protected.

3. **Protected event windows.** The caller may pass intervals
    `[t_a, t_b]` (eruption onsets, dike intrusions, configured
    `outliers.protect_windows`) which are treated as an operator-declared
    anomalous-signal interval and are excluded from **three** stages:
    (i) flagging is disabled outright inside the window; (ii) the window's
    epochs are excluded from the robust trajectory **fit** (so a meter-scale
    transient does not distort the background model, keeping quiet-region
    detection sane — the SENG/Svartsengi lesson, 2026-07-14); and (iii) the
    window's candidates are excluded from the §3.5 abort numerator. Only
    operator-declared `protect_windows` get this treatment — the auto
    protections (FLOOR/RUN/STEP) do NOT, since a series the leaf must
    auto-protect wholesale still signals a wrong model and should still abort.
4. **Cross-component policy** (`epoch_policy`):
    - `per_component` (default): each component keeps its own flags. N/E/U noise
      levels differ by ~3× and snow/multipath can hit U alone.
    - `union`: an epoch flagged in any component is flagged in all — appropriate
      when bad daily solutions are known to be epoch-wise (all-component) events.
      Cross-component **agreement is deliberately NOT used to protect signal**: a bad
      daily solution is also 3-D-coherent, so component coherence cannot separate
      signal from blunder. Signal coherence lives across _stations_, which is the
      caller's (network-level) domain — see Q7.

Protected clusters are surfaced as `SuspectedEvent` records (interval, sign,
step-evidence D, component) — operator-facing hints for `steps.csv` and the
GBIS4TS break-detection lane, never silent.

### 3.5 Iteration and stopping rule (conservative by construction)

```
sweep:  robust fit (Huber, step-augmented) on current inliers
        → residuals on ALL epochs → §3.3 candidates → §3.4 protection → new flags
until:  flags unchanged (fixed point)  OR  max_iterations (default 3)
guard:  if (candidate fraction OUTSIDE protect_windows) > f_max (default 0.05)
        at any sweep →
        ABORT MASKING: return flags = all-False, excess_flag_abort = True,
        diagnostics (candidates, reasons, suspected events) fully populated
finish: plain WLS refit on final inliers (reported fits + Gauss–Markov cov)
```

- Flagging is evaluated on **all** epochs each sweep, so a previously flagged
  point may re-enter (no ratchet, no ordering artifacts).
- The `f_max` abort is the "when in doubt, do nothing" rule: a series where > 5 %
  of epochs look like outliers almost certainly contains _unmodeled signal_
  (missed step, transient) — masking there deletes signal. The abort is loud
  (provenance-stamped, logged), never a silent cap.
- **Protect-window exemption (2026-07-14):** epochs inside an operator-declared
  `protect_windows` interval are excluded from the abort numerator (denominator
  stays N) — the operator has already declared *this is signal here*, so those
  candidates are expected and must not trip the abort. An active-unrest station
  (SENG) whose unrest interval is declared thus **cleans** (blunders outside the
  window removed, signal inside preserved) instead of degrading. The window is
  also excluded from the fit (§3.4.3), so the background model stays sane and the
  quiet region cleans correctly. Blunders *inside* the window are left uncleaned
  (flagging is off there) — the future local-identifier-inside-window refinement
  addresses that; plain suppression is the conservative first step.
- Idempotence requirement: running detection on a series with its flags applied
  must reproduce the same mask (test-pinned, §8).

### 3.6 What this stage is NOT

- Not step/break _detection_ — unknown breaks are GBIS4TS's job
  (`gps_analysis.transient`); this stage only _protects_ them and hints at them.
- Not formal-σ screening — `preprocess.screen_uncertainty` runs upstream with its
  own reason; its mask is an input, not an output, of this stage.
- Not imputation/replacement — flagged epochs are never filled.

## 4. Leaf API (`gps_analysis/outliers.py`)

All functions: pure, float64, unit-agnostic ([L] = caller's unit), inputs never
mutated, no I/O, no config reads. Every docstring follows the MATH_STANDARDS §2
contract (quantity, explicit equation, symbols→args with units, reference,
numerical notes). Signatures below are the spec; equations/references as in §3.

### 4.1 Atomic primitives

```python
def mad_scale(x: ArrayLike, *, center: float | None = None) -> float
    # ŝ = 1.4826 · med|x − med(x)|                (Rousseeuw & Croux 1993)
    # center=None → med(x); NaN-free input required; returns 0.0 on degenerate x
    # (caller guards with s_floor). N < 3 raises ValueError.

def qn_scale(x: ArrayLike) -> float
    # Qn = 2.2191 · c_N · {|x_i − x_j|}_(k)       (Rousseeuw & Croux 1993;
    # O(N log N) per Croux & Rousseeuw 1992, incl. finite-sample c_N table)

def whiten(r: ArrayLike, sigma: ArrayLike | None) -> FloatArray
    # w_i = r_i / σ_i (elementwise); sigma=None → copy of r. σ_i ≤ 0 raises.

def standardize_robust(
    x: ArrayLike, *, scale: str = "mad"
) -> tuple[FloatArray, float, float]
    # ẑ = (x − med(x)) / ŝ ; returns (ẑ, center, ŝ). scale ∈ {"mad", "qn"}.

def rolling_median(
    t: ArrayLike, x: ArrayLike, *, half_window: float, min_count: int
) -> FloatArray
    # m_i = med{x_j : |t_j − t_i| ≤ h} [same units as x]; windows in TIME (t units,
    # fractional years — pass half_window in yr, e.g. 15.5/365.25).
    # Fewer than min_count in-window samples → NaN at i (caller substitutes the
    # global fallback). Two-pointer sweep over sorted t: O(N · w̄) worst case;
    # t must be sorted ascending (raises otherwise).

def rolling_mad(
    t: ArrayLike, x: ArrayLike, center: ArrayLike, *, half_window: float,
    min_count: int
) -> FloatArray
    # s_i = 1.4826 · med{|x_j − center_i| : |t_j − t_i| ≤ h}; NaN where the
    # window is thin (same rule as rolling_median).

def hampel_mask(
    x: ArrayLike, center: ArrayLike, scale: ArrayLike, *, n_sigma: float,
    scale_floor: float = 0.0
) -> NDArray[np.bool_]
    # L_i = |x_i − center_i| > n_sigma · max(scale_i, scale_floor)
    # (Hampel 1974; Pearson et al. 2016, eq. 4). NaN scale_i (thin window) →
    # comparison against the caller-supplied fallback is the caller's business:
    # here NaN scale ⇒ not flagged (conservative).

def candidate_clusters(
    t: ArrayLike, candidates: NDArray[np.bool_], *, max_gap: float
) -> list[tuple[int, int]]
    # Group candidate epochs into [i_start, i_end] index clusters where
    # consecutive candidate epochs are ≤ max_gap apart in t. Pure indexing op.

def step_evidence(
    t: ArrayLike, r: ArrayLike, i_start: int, i_end: int, *, window: float,
    scale: float, exclude: NDArray[np.bool_] | None = None
) -> float
    # D = |med(r_post) − med(r_pre)| / ŝ over [t_start − W, t_start) and
    # (t_end, t_end + W]; `exclude` masks other candidates out of the medians.
    # Returns NaN when either side holds < 3 samples (caller treats NaN as
    # "cannot rule out a step" ⇒ protect). Gazeaux et al. 2013 motivates the
    # conservatism.
```

### 4.2 Step-augmented trajectory (in `fitting.py` / `models.py`)

```python
# models.py
def heaviside_steps(t: ArrayLike, epochs: ArrayLike, amplitudes: ArrayLike) -> FloatArray
    # Σ_k a_k · H(t − t_k), H(0) = 1 (epoch t_k belongs to the post-step side —
    # convention documented; gtimes yearf epochs). Atomic model term.

# fitting.py
def with_steps(model: ModelFunc, step_epochs: ArrayLike) -> ModelFunc
    # Factory: returns f(t, *p, *a) = model(t, *p) + heaviside_steps(t, epochs, a).
    # When `model` is registered in _LINEAR_DESIGNS the returned callable is too
    # (design gains K Heaviside columns H(t − t_k) — still linear ⇒ closed-form
    # WLS path preserved). Step epochs are FIXED data; amplitudes are parameters.
    # Numerical note: a step column with no observations on one side is
    # rank-deficient → existing _wls_solve inf-cov warning path applies.
```

### 4.3 Parameters, results, orchestration

```python
@dataclasses.dataclass(frozen=True)
class OutlierParams:
    """Detection thresholds — constructed by the CALLER (config → here)."""
    scale_estimator: str = "mad"          # "mad" | "qn"
    global_n_sigma: float = 5.0           # k_g
    window_days: float = 31.0             # full Hampel window (h = /2), days
    window_n_sigma: float = 4.0           # k_w
    window_min_count: int = 11            # n_min → global-scale fallback
    scale_floor: float = 0.0              # s_floor, whitened-residual units
    min_outlier: float = 0.0              # a_min, [L] — per-component value;
                                          # the (C,)-vector form is detect_outliers'
                                          # min_outlier argument
    max_run_days: float = 2.0             # L_max
    cluster_gap_days: float = 1.5         # g
    run_sign_fraction: float = 0.8        # q
    step_evidence_sigma: float = 3.0      # k_step
    step_window_days: float = 10.0        # W_pre = W_post
    max_flag_fraction: float = 0.05       # f_max abort
    max_iterations: int = 3
    loss: str = "huber"                   # robust-fit loss
    f_scale: float = 1.0

@dataclasses.dataclass(frozen=True)
class SuspectedEvent:
    component: int                        # row index into y
    i_start: int
    i_end: int
    t_start: float                        # [yr]
    t_end: float                          # [yr]
    sign: int                             # dominant residual sign (−1/+1)
    step_evidence: float                  # D (NaN = indeterminate)
    kind: str                             # "step" | "transient_run"

@dataclasses.dataclass(frozen=True)
class OutlierDetection:
    flags: NDArray[np.bool_]              # shape of y — True = OUTLIER (final)
    candidates: NDArray[np.bool_]         # pre-protection union G ∪ L
    reasons: NDArray[np.uint8]            # REASON_* bitmask, shape of y
    protected: NDArray[np.uint8]          # PROTECT_* bitmask, shape of y
    z: FloatArray                         # final-sweep detection statistic ẑ
    scale_global: FloatArray              # (C,) global ŝ per component
    scale_local: FloatArray               # local s_i (NaN where fallback), shape of y
    fits: list[TrajectoryParams]          # final inlier WLS fits (step-augmented)
    step_amplitudes: FloatArray | None    # (C, K) fitted â_k, input order
    suspected_events: list[SuspectedEvent]
    n_iterations: int
    converged: bool
    excess_flag_abort: bool               # True ⇒ flags is all-False by rule §3.5
    params: OutlierParams                 # echo — provenance building block

def detect_outliers(
    model: ModelFunc,
    t: ArrayLike,                         # (N,) yearf [yr], sorted ascending
    y: ArrayLike,                         # (N,) or (C, N) [L] — NEVER mutated
    sigma: ArrayLike | None = None,       # formal 1-σ, shape of y [L]
    *,
    step_epochs: ArrayLike | None = None, # known steps t_k [yr] (per-station table)
    protect_windows: Sequence[tuple[float, float]] = (),
    min_outlier: ArrayLike | None = None, # a_min per component (C,) [L]
    p0: ArrayLike | None = None,
    params: OutlierParams = OutlierParams(),
    names: Sequence[str] | None = None,
) -> OutlierDetection
    # Thin orchestration (MATH_STANDARDS §1) of §3.1→§3.5: with_steps → robust
    # fit → whiten → standardize_robust (global) + rolling_median/rolling_mad/
    # hampel_mask (local) → candidate_clusters/step_evidence protection →
    # iterate → final WLS refit. Returns MASK + diagnostics; never a filtered
    # array. Raises ValueError on shape mismatch, unsorted t, non-finite y/t.
```

**Relationship to `fitting.reject_outliers`:** kept as the lightweight
model-robust clip (used internally by exploratory code); `detect_outliers` is the
production path. `reject_outliers`'s docstring gains a pointer; deprecation is Q3.

## 5. Non-destructive data flow (precompute → store → API)

### 5.1 Precompute (`gps_api.precompute`, the caller — reads config, leaf never does)

Pipeline order per station (job.py):

```
load .NEU → screen_uncertainty (existing, formal-σ)     [epoch-quality mask]
         → detect_outliers(model, t, y, σ,
               step_epochs = steps.csv rows for marker,  [via gps_parser]
               protect_windows / params / min_outlier = analysis.yaml outliers:)
         → Parquet: RAW columns unchanged + flag columns (§5.2)
         → downstream stages consume the INLIER view:
             fit/detrend product, WLS/MLE velocity, GBIS4TS breaks, Mogi/Okada
             (mask applied at the call site; the store keeps everything)
         → detrended columns = residuals of the outlier-robust step-augmented fit,
           evaluated at ALL epochs (flagged epochs included — they carry flags)
```

An `excess_flag_abort` result is recorded per station in `meta/run.json`
(`outliers_aborted: [markers]`) and the station proceeds **unmasked** — loud,
never silent. `suspected_events` are written into the run meta as operator hints
(candidate `steps.csv` entries / GBIS4TS gating hints).

### 5.2 Store schema (`series/<MARKER>.parquet`)

New columns, additive and optional (absent ⇒ product predates this feature):

| column                                        | type  | meaning                                     |
| --------------------------------------------- | ----- | ------------------------------------------- |
| `north_outlier`, `east_outlier`, `up_outlier` | bool  | final per-component flags                   |
| `north_outlier_reason`, …                     | uint8 | `REASON_*` bitmask (0 on inliers)           |
| `outlier_epoch`                               | bool  | union over components (serving convenience) |

Raw `north/east/up`, `sigma_*` columns are byte-identical to today — **the raw
series is always in the product**. Parquet provenance metadata
(`gps_api_provenance`) gains an `outliers` object: method tag
(`"hampel-trajectory"`), the full `OutlierParams` dict, per-component
`n_flagged`/`n_candidates`/`n_protected`, `suspected_events`, `aborted`,
`gps_analysis` version (MATH_STANDARDS §6).

### 5.3 API (`GET /v1/stations/{marker}/series`) — contract amendment (A8)

- New query parameter **`clean: bool = false`**. Default serves the **raw**
  series (requirement 3: raw is the default truth; cleaning is opt-in).
  - `clean=false`: all epochs; response points carry an `outlier` boolean
    (schema addition to `SeriesResponse`, nullable for pre-A8 products).
  - `clean=true`: flagged epochs are dropped **before** LTTB downsampling
    (outlier spikes otherwise dominate LTTB's triangle selection — the exact
    plotting artifact aflogun must avoid).
  - `detrended=true` composes with either (detrended values exist at all epochs).
- Response `provenance` echoes the store's `outliers` object so any consumer can
  see what would be/was removed and with which parameters.
- `docs/API_CONTRACT.md` and `schemas.py` change together (contract rule).
- aflogun UI: cleaned view as the _display_ default is a UI decision (Q2); the
  API default stays raw.

### 5.4 Config (`analysis.yaml` — template `config-templates/analysis-lane/analysis.yaml`)

```yaml
outliers:
  enabled: true
  scale_estimator: mad # mad | qn
  global_n_sigma: 5.0
  window_days: 31
  window_n_sigma: 4.0
  window_min_count: 11
  min_outlier_mm: # physical floor a_min (protection §3.4.1)
    horizontal: 5.0
    vertical: 10.0
  max_run_days: 2.0
  cluster_gap_days: 1.5
  run_sign_fraction: 0.8
  step_evidence_sigma: 3.0
  step_window_days: 10.0
  max_flag_fraction: 0.05
  max_iterations: 3
  epoch_policy: per_component # per_component | union   (§3.4.4)
  protect_windows: [] # [{start: 2023.85, end: 2023.95, comment: "..."}]
  overrides: {} # per-station: { SENG: { window_n_sigma: 4.5 } }
```

`precompute/config.py` gains a frozen `OutlierConfig` (validated like
`BreakpointConfig`: thresholds > 0, `epoch_policy` enum, fraction in (0, 1]) that
maps 1:1 onto `OutlierParams` + the per-component floor vector. Known steps keep
riding the existing `steps.csv` (`gps_parser` reader — already planned §10.4);
component-specific rows (`N|E|U|ALL`) map to per-component step lists. CLI:
`gps-api-precompute --no-outliers` mirrors `--no-deformation`.

## 6. Defaults — justification summary

| Parameter    | Default | Why                                                                                                           |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------- |
| k_g (global) | 5.0     | N·2Φ(−5) ≈ 0.004 false flags per 20 yr; colored-noise tails make 3–4 unsafe globally                          |
| k_w (Hampel) | 4.0     | local median absorbs the wander ⇒ near-Gaussian local residuals; N·2Φ(−4) ≈ 0.5/20 yr                         |
| window       | 31 d    | ≳ 2× typical outlier-cluster span, ≪ seasonal period and typical transient τ, ≥ n_min even at 60 % data yield |
| L_max        | 2 d     | multi-day one-sided runs are the signature of signal, not blunders (Gazeaux et al. 2013)                      |
| f_max        | 5 %     | a series flagging > 5 % has an unmodeled-signal problem, not an outlier problem                               |
| a_min        | 5/10 mm | below GNSS daily repeatability nothing is worth masking; prevents scale-collapse flagging                     |
| loss         | huber   | bounded influence with exact WLS behavior in the core (Huber 1964)                                            |

All defaults are config, not code (leaf takes them via `OutlierParams`).

## 7. References (add to MATH_STANDARDS §5 on implementation)

- **Rousseeuw & Croux 1993** — _Alternatives to the median absolute deviation_,
  JASA 88(424). (MAD normalization; Qn definition, efficiency/breakdown.)
- **Croux & Rousseeuw 1992** — _Time-efficient algorithms for two highly robust
  estimators of scale_, Computational Statistics 1, 411–428. (O(N log N) Qn +
  finite-sample correction factors.)
- **Hampel 1974** — _The influence curve and its role in robust estimation_,
  JASA 69. (Hampel identifier.)
- **Pearson, Neuvo, Astola & Gabbouj 2016** — _Generalized Hampel filters_,
  EURASIP J. Adv. Signal Process. 2016:87. (windowed median/MAD decision rule,
  MAD-collapse degeneracy.)
- **Davies & Gather 1993** — _The identification of multiple outliers_, JASA
  88(423). (outlier-identifier consistency; threshold-vs-N reasoning.)
- **Huber 1964** — Ann. Math. Statist. 35. (M-estimation; already listed.)
- **Gazeaux et al. 2013** — _Detecting offsets in GPS time series: First results
  from the detection of offsets in GPS experiment (DOGEx)_, JGR 118. (steps are
  hard to separate from outliers/noise ⇒ conservative protection + known-step
  tables beat automation.)
- **Bevis & Brown 2014** — _Trajectory models and reference frames for
  crustal motion geodesy_, J. Geodesy 88. (step-augmented trajectory model.)
- **Belsley, Kuh & Welsch 1980** — _Regression Diagnostics_ (Wiley), ch. 2.
  (studentized residuals, hat-matrix leverage.)
- **Nikolaidis 2002** — _Observation of geodetic and seismic deformation with the
  GPS_, PhD thesis, UC San Diego. (canonical GPS time-series cleaning practice —
  the baseline this design improves on.)
- Williams 2003, Blewitt & Lavallée 2002 — already in the shared list.

## 8. Test plan (`tests/test_outliers.py` + precompute integration)

### 8.1 Synthetic generator (test helper, in-tests only)

`_synthetic(...)`: daily `t` (yearf) with configurable gap mask; truth
trajectory = linear + annual/semiannual (per component, U amplitude ×3);
noise = white + power-law (colored sample via Cholesky of
`transient.noise_covariance` — Williams 2003, reusing the tested leaf);
formal σ log-normal around the white amplitude. Seeded, reproducible.
Injectors (composable, each returns truth bookkeeping):

- `spikes(n, amp, sign, cluster_len)` — isolated ±spikes, 1–2-epoch clusters,
  U-only and all-component variants, amplitude sweep 3–50 mm;
- `step(t0, amp)` — coseismic/equipment offset;
- `transient(t0, tau, amp)` — one-sided exponential (post-seismic/inflation);
- `sse(t0, dur, amp)` — linear ramp over 5–20 d (slow slip);
- `rate_change(t0, dv)`.

### 8.2 Detection quality

| test                                 | assertion                                                                                                                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_spike_recall`                  | ≥ 90 % of injected spikes with amp ≥ 5·ŝ flagged (each seed); 100 % at ≥ 8·ŝ                                                                                                  |
| `test_clean_series_false_positives`  | white noise, no injections: E[#flags] ≤ 2× the Gaussian expectation N·2Φ(−k_w) over 50 seeds; **zero** flags above the magnitude floor on a 2 mm-noise series with floor 5 mm |
| `test_colored_noise_false_positives` | flicker κ = −1: #flags ≤ small documented bound (records the measured rate — the honest cost of colored noise)                                                                |
| `test_sigma_weighting`               | epoch with large σ_i and large residual NOT flagged; same residual with small σ_i flagged (studentization works)                                                              |
| `test_qn_matches_mad_gaussian`       | Qn and MAD scales within 10 % on clean Gaussian; identical flag sets for well-separated spikes                                                                                |

### 8.3 Signal protection — the release gate

| test                             | scenario                                       | assertion                                                                                                   |
| -------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `test_known_step_survives`       | 40 mm step, epoch in `step_epochs`             | **zero** flags in [t0 − 30 d, t0 + 30 d]; fitted â within 3σ of truth                                       |
| `test_unknown_step_protected`    | same step, NOT declared                        | zero flags in the interval; `suspected_events` contains a `kind="step"` event covering t0 with D > k_step   |
| `test_transient_survives`        | exp transient τ ∈ {10, 30, 60} d, amp 30 mm    | zero flags in [t0, t0 + 3τ]                                                                                 |
| `test_sse_survives`              | 15 mm ramp over 10 d                           | zero flags in the ramp ± 10 d; suspected event reported                                                     |
| `test_spike_on_transient_caught` | transient + 25 mm single-epoch spike inside it | the spike IS flagged; no other epoch of the transient is (the compound case that kills global-only schemes) |
| `test_protect_window`            | spike inside a configured protect window       | not flagged; `PROTECT_WINDOW` bit set                                                                       |
| `test_excess_abort`              | undeclared 100 mm step → > f_max candidates    | `excess_flag_abort=True`, `flags` all-False, candidates populated                                           |
| `test_protect_window_clears_abort` | same step, unrest declared `protect_windows` | `excess_flag_abort=False`, converged, nothing flagged inside, `PROTECT_WINDOW` set                          |
| `test_protect_window_excluded_from_fit` | 500 mm ramp in a protect window + quiet blunder | no abort; quiet blunder flagged; quiet region otherwise clean (fit not distorted)                    |
| `test_floor_protection`          | quiet series (ŝ → 1 mm), 3 mm wiggles          | zero flags (`PROTECT_FLOOR` recorded on candidates)                                                         |
| `test_returning_blunder_cluster_not_run_protected` | 4-day same-sign cluster that returns to baseline | flagged, NOT `PROTECT_RUN` (run-rule released; transients still protected)                        |

### 8.4 Contract & property tests

- `test_mask_never_filters` — input `y`, `t`, `sigma` byte-identical after the
  call; every returned array has the input shape; `flags ⊆ candidates`;
  `flags ∧ (protected ≠ 0) = ∅`.
- `test_idempotent` — re-run with previous flags' inliers ⇒ identical mask.
- `test_invariances` — mask invariant under: adding a constant / linear trend
  (model absorbs); joint scaling of (y, σ, floors) by 1000 (m ↔ mm,
  unit-agnostic); time origin shift.
- `test_determinism` — same inputs ⇒ bit-identical `OutlierDetection`.
- `test_reject_outliers_superset_sanity` — on a spike-only white-noise series the
  new detector flags at least the (protection-surviving) flags of the legacy
  `reject_outliers` with matching thresholds.
- Analytic checks per MATH_STANDARDS §4: `mad_scale`/`qn_scale` against known
  constants on N(0,1) samples and small closed-form vectors; `rolling_median` vs
  brute-force windowing; `step_evidence` exact on a noiseless step.

### 8.5 Integration (gps_api side, `tests/test_precompute.py` extension)

- Store round-trip: flag columns present + typed; raw columns byte-identical to a
  no-outlier run; provenance `outliers` object complete.
- Endpoint: `clean=false` (default) returns all epochs + `outlier` field;
  `clean=true` returns none of the flagged epochs; LTTB applied after cleaning;
  both validate against the amended `SeriesResponse`.
- Abort path: aborted station appears in `meta/run.json.outliers_aborted` and its
  Parquet has all-False flags.
- Downstream: velocity/breaks stages on a spiked synthetic move toward truth when
  flags are applied (velocity bias with spikes ≫ without).

## 9. Open questions for BGÓ

1. **`epoch_policy` default** — `per_component` (spec'd) or `union`? Do we know
   whether the GAMIT daily bad-solutions dominate over per-component (snow-on-
   antenna U) failures in the 173-station fleet?
   - depends however there are varialbles in GAMIT that can be used to idendify bad solutions, suspected icinging, snow and such, check vault note "..check-station-for-suspect-icing-signal.md it is refering to okada:/D/WEEKS/WWWW/../\
2. **aflogun display default** — API default stays `clean=false`; should the
   portal default its toggle to cleaned with a visible "N points hidden" badge?
   - good question can we address it later ?
3. **`fitting.reject_outliers`** — keep as the light exploratory variant, or
   deprecate once `detect_outliers` lands?
   - somtimes we remove it somtimes we might just want to mark it and it in the plot. but we always remove outliers before estimating a parameter
4. **Magnitude floors** — 5 mm horizontal / 10 mm vertical acceptable fleet-wide,
   or per-station-class (e.g. higher for high-latitude/snow sites)? They are
   config either way.
   - start with this. we will need to ajust as we test and apply this to time series. so keep parameters like this in config it might have to be tailored to some stations.
5. **`suspected_events` → `steps.csv`** — should the precompute write a candidate
   file (`meta/suspected_steps.csv`) for operator review, and/or gate GBIS4TS
   triage on stations with suspected events?
   - yea this is sort of what I was thinking with this 4. requrement. I want to be able to asses visually susbpected outliers and suspeced icing and so on
6. **Qn in Phase 1?** — MAD-only ships first (zero new numerics risk); Qn needs a
   careful O(N log N) implementation + finite-sample factors. Phase 1.5?
   - yes
7. **Network-level common-mode check** (later slice): a day flagged at many
   stations ⇒ orbit/frame problem (flag with a distinct reason); a multi-day
   run at one station corroborated by neighbors ⇒ signal (strengthen protection).
   Worth a design addendum once per-station lands?
   - definatly systematic changes accross region or country wide are common and quite important to idendify in order not to wrongly flag a change in velosity as geophysical signal. so yes we should add this in the future. I do this manually now by looking at the time series of all stations in a region and comparing them to each other. so yes we should add this in the future.
8. **GBIS4TS input** — run break detection on the cleaned (inlier) series
   (recommended: its likelihood assumes no blunders) — confirm, and whether the
   breaks provenance should record the outlier params hash it consumed.
   - yes this is correct. we should also record the outlier params hash in the break detection output.
9. **`max_flag_fraction`** — 5 % abort threshold OK, or per-station override
   needed for known-pathological stations?
   - possibly we can have a per station override for this. but I think 5% is a good starting point. we can adjust it later if needed.

## 10. Addendum — two upgrades (branch `outlier-prefilter-localfit`, 2026-07-14)

Two **param-gated** additions to `detect_outliers`; the `OutlierParams()`
defaults reproduce the §3 behavior bit-identically (proven: the 423 §8 tests
pass unchanged, and a golden order-0 fixture is pinned in
`tests/test_outliers_stage0.py::test_golden_order0_fixture`).

### §3.0 — Stage-0 gross-blunder despike (pre-fit; `despike`, default off)

A cheap surgical pre-filter run ONCE on the raw per-component observations
**before** the model fit, so obvious extreme single-epoch blunders (the FAGD
East ~150 mm case) cannot corrupt the robust fit or the MAD scale. It is NOT
the main detector — only "obvious" isolated extremes.

Spike-vs-step rule (`neighbor_differences` → `spike_mask`). With gap-aware
first differences `δ⁻_i = y_i − y_{i−1}`, `δ⁺_i = y_{i+1} − y_i` (NaN across a
gap wider than `despike_gap_days`, so adjacency is temporal, not index-based)
and the robust difference scale `ŝ_Δ = mad_scale(δ⁻)`, epoch `i` is a gross
blunder iff

```
min(|δ⁻_i|, |δ⁺_i|) > k_d·ŝ_Δ   ∧   δ⁻_i·δ⁺_i < 0   ∧   |δ⁻_i + δ⁺_i| ≤ c_r·ŝ_Δ
```

— strong deviation from BOTH neighbors, opposite directions (up-then-down),
and the neighbors agree (`δ⁻ + δ⁺ = y_{i+1} − y_{i−1}`, the series RETURNS to
baseline). A persistent offset (real step) has `δ⁺ ≈ 0` at the jump and fails
the first two conditions; a linear/curved trend gives same-sign differences and
fails the second — steps and transients are never despiked. Refs: Goring &
Nikora 2002 (difference despiking); Pearson et al. 2016 §1 (spike returns,
level shift persists); Gazeaux et al. 2013 (why steps must not be auto-clipped).

Defaults: `k_d = despike_n_sigma = 10` (a spike of amplitude A scores ≈ A/(√2σ)
under white noise σ, so k_d = 10 ⇒ A ≳ 14σ — deliberately conservative);
`c_r = despike_return_sigma = 4`; `despike_gap_days = 1.5`. Despiked epochs are
masked (never deleted), tagged `REASON_GROSS = 4`, excluded from the fit and the
identifiers, force-flagged, counted in `OutlierDetection.n_despiked`, and
respect `protect_windows`. They do **not** count toward the `f_max` abort
fraction (decided blunders, not "epochs that look like outliers").

### §3.3b-poly — windowed identifier: local constant → robust local polynomial

The windowed (Hampel) identifier generalizes from an order-0 local constant
(`rolling_median`/`rolling_mad`) to a **robust local polynomial** f(x) of order
`window_order ∈ {0, 1, 2}` (`rolling_polyfit`; LOWESS — Cleveland 1979, JASA
74:829: tricube distance weights + `window_robust_iterations` bisquare
robustness passes). Decision unchanged: `|w_i − m_i| > k_w·max(s_i, s_floor)`
with `m_i = f̂_i(t_i)` and `s_i` the MAD of the local fit residuals.
`window_order = 0` (default) takes the exact `rolling_median` path.

Rationale and honest scope (A/B evidence,
`tests/verification_outlier_ab.py`): during a steep transient the order-0 local
MAD is **inflated by the in-window signal slope**, raising the local threshold
so genuine outliers riding the ramp are masked (0/3 spikes caught, local scale
14.3 mm on a 3 mm/d ramp); a robust local line/parabola restores an honest local
scale (order-2: 1.86 mm, 3/3 caught). This is the real, demonstrated benefit —
**recall on transients**, not abort avoidance: on a strongly unmodeled
meter-scale ramp the candidate population is ~97 % *global*-identifier, which no
window order can change, so order ≥ 1 does NOT by itself prevent the §3.5 abort
(that needs the transient declared as a step / transient term, or a per-station
`max_flag_fraction`). **Caveat (test-pinned):** at an *undeclared* slope
discontinuity the local line cannot straddle the kink → the onset epochs become
candidates; the existing PROTECT_STEP guard still runs, and in production the
onset is a declared step or a `protect_window`. **Decision for BGÓ:** order ≥ 1
is safe to enable only where the transient/step is declared or protected.

### §8.4 amendment — candidates includes gross

To preserve the §8.4 property invariants with despike on (`flags ⊆ candidates`;
`reasons == 0` exactly off candidates), the returned `candidates` mask includes
the Stage-0 despiked epochs (distinguished by `REASON_GROSS`), even though they
are excluded from clustering, protection and the abort fraction.

---

_Spec created 2026-07-13 (analysis lane). §10 addendum implemented 2026-07-14
(branch `outlier-prefilter-localfit`): Stage-0 despike + local-polynomial
windowed identifier, param-gated, defaults bit-identical. Module lands under the
usual gates: MATH_STANDARDS-conformant docstrings, ruff/black/mypy(strict) zero
warnings, the §8 test plan green (423 unchanged + new), contract A8 amendment in
`gps_api/docs/API_CONTRACT.md` + `schemas.py` together (still pending — API not
in this change)._
