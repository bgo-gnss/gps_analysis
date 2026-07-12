# Mathematical documentation & rigor standard — `gps_analysis`

> Binding for **all** math functions, methods and derived products in this
> package (and the analysis lane generally). Audience: a geophysicist (BSc
> physics / MSc geophysics). **Do not dumb the math down** — make it precise,
> atomic, and followable both logically and mathematically. Every derived
> quantity must be traceable to an equation and a reference.

## 1. Atomicity
- **One mathematical operation per function.** A function computes *one* well-defined
  quantity (a model value, a design matrix, a covariance, a log-likelihood, a rate).
  If a docstring needs "and" to describe the math, split it.
- Composition happens in thin orchestration functions that call atomic pieces; the
  math primitives stay pure, side-effect-free, and independently testable.
- No hidden unit conversions or thresholds inside a math primitive — pass them in.

## 2. Documentation contract (every math function)
Each math function's docstring MUST contain, in this order:
1. **What it computes** — one precise sentence naming the quantity and its symbol.
2. **The equation** — written explicitly (Unicode math or clear ASCII), e.g.
   `C = σ_w²·I + σ_p²·(T₁T₁ᵀ)`, not "the covariance". Show the form actually
   implemented, including any convention choices (e.g. `H(0)=1`).
3. **Symbols → args** — every symbol mapped to a parameter, **with units**
   (mm, mm/yr, fractional year, dimensionless spectral index, …).
4. **Reference** — author, year, and the specific equation/section number
   (e.g. "Williams 2003, J. Geod., eq. 4"; "Bagnardi & Hooper 2018, G³, §2.2";
   "Yang et al. 2023, 2023GL103432, Table S2"). Use the shared list in §5.
5. **Numerical notes** — stability/conditioning choices and why (e.g. "Cholesky
   solve, not explicit inverse: O(N³)→ stable, and gives logdet for free";
   domain caveats: "requires `t` sorted ascending"; documented tolerances).

Module docstrings lay out the **derivation chain** — how the atomic pieces compose
into the estimator — so the reader can follow the flow top-down before reading code.

## 3. Numerical precision
- Prefer numerically stable formulations: Cholesky/`cho_solve` over `inv`; `logdet`
  via `2·Σ log diag(L)`; log-sum-exp where probabilities underflow; `math.fsum`/
  pairwise summation for long reductions where it matters.
- State the working dtype (float64) and any tolerance/`rtol` used in tests.
- Guard and document degeneracies (rank-deficient design matrix, `t*` at the break,
  near-singular covariance for extreme `kappa`).

## 4. Testing the math (not just "it runs")
- **Analytic checks**: closed-form cases (e.g. zero rate-change ⇒ pure linear;
  white-noise limit `amp→0` ⇒ `C = σ_w²I`; known integral/derivative identities).
- **Reference parity**: reproduce the source implementation's output where one exists
  (GBIS4TS `Verification/`; legacy `detrend-*` rates as golden values).
- **Property tests**: symmetry/positive-definiteness of covariances, continuity of
  the break model at the breakpoint, invariance/units.
- Document the tolerance and *why* it is acceptable (MCMC sampling error, float eps).

## 5. Reference list (cite by these keys; add as needed)
- **Williams 2003** — S.D.P. Williams, *The effect of coloured noise on the
  uncertainties of rates estimated from geodetic time series*, J. Geodesy 76 (2003).
  (power-law covariance construction — `UniVarMatrix`.)
- **Williams 2008 / CATS** — S.D.P. Williams, *CATS: GPS coordinate time series
  analysis software*, GPS Solutions 12 (2008). (white+coloured MLE; the disabled
  angle form in `UniVarMatrix`.)
- **Bagnardi & Hooper 2018** — *Inversion of surface deformation data …: A Bayesian
  approach*, G³ 19, doi:10.1029/2018GC007585. (GBIS MCMC engine.)
- **Yang, Sigmundsson, Geirsson 2023** — *Joint Bayesian Modeling of Velocity Break
  Points, Noise Characteristics, and Their Uncertainties in GNSS Time Series*,
  GRL, 2023GL103432 (+ Supporting Information, Table S2 priors). (GBIS4TS.)
- **Blewitt & Lavallée 2002** — *Effect of annual signals on geodetic velocity*,
  JGR 107(B7). (annual/semiannual terms in `lineperiodic`; why window length matters.)
- **Segall 2010** — *Earthquake and Volcano Deformation* (Princeton). (Mogi/Okada,
  trajectory models — backburnered lane, cite when it revives.)
- **Hosking 1981** — J.R.M. Hosking, *Fractional differencing*, Biometrika 68(1),
  eq. 2.4. (ψ coefficients of `(1−L)^(κ/2)`; nonstationarity for κ ≤ −1 — why the
  power-law covariance is NOT Toeplitz.)
- **Kailath & Sayed 1995** — *Displacement structure: theory and applications*,
  SIAM Review 37(3), §1–4. (generalized Schur algorithm — the exact O(N²)
  factorization behind `transient._schur_logdet_quad`, task H3; displacement rank:
  Kailath, Kung & Morf 1979, J. Math. Anal. Appl. 68.)
- **Chandrasekaran & Sayed 1996** — *Stabilizing the generalized Schur algorithm*,
  SIAM J. Matrix Anal. Appl. 17(4). (backward stability of the positive-definite
  displacement case — only orthogonal rotations.)
- **Bos et al. 2013** — *Fast error analysis of continuous GNSS observations with
  missing data*, J. Geodesy 87. (Hector's O(N²)-class fast MLE — the comparison
  point for the H3 fast-likelihood path.)
- **Okada 1985** — *Surface deformation due to shear and tensile faults in a
  half-space*, BSSA 75(4), 1135–1154, eqs. (24)–(30). (rectangular dislocation —
  `okada_forward` kernels, patch Green's functions.)
- **Harris & Segall 1987** — *Detection of a locked zone at depth on the Parkfield,
  California, segment of the San Andreas fault*, JGR 92(B8), 7945–7962. (regularized
  distributed-slip inversion from geodesy — smoothing + positivity; `okada_invert_slip`.)
- **Jónsson et al. 2002** — *Fault slip distribution of the 1999 Mw 7.1 Hector Mine,
  California, earthquake, estimated from satellite radar and GPS measurements*,
  BSSA 92(4), 1377–1389. (Laplacian smoothing with zero-slip edges + NNLS —
  `patch_laplacian`, the non-negative option.)
- **Aster, Borchers & Thurber 2018** — *Parameter Estimation and Inverse Problems*
  (3rd ed., Elsevier), ch. 4 (Tikhonov regularization, L-curve), §9.2 (nonlinear-LSQ
  covariance). (`_solve_regularized`, `_lsq_covariance`.)
- **Hansen 1992** — *Analysis of discrete ill-posed problems by means of the L-curve*,
  SIAM Review 34(4), 561–580. (L-curve corner criterion — `lcurve_corner`.)
- **Hanssen 2001** — *Radar Interferometry: Data Interpretation and Error Analysis*
  (Kluwer), ch. 2. (SAR imaging geometry — `los_unit_vector`.)
- **Fialko, Simons & Agnew 2001** — *The complete (3-D) surface displacement field in
  the epicentral area of the 1999 Mw 7.1 Hector Mine earthquake …*, GRL 28(16),
  3063–3066, eq. (1). (LOS decomposition of an ENU displacement — `los_project`.)
- **Wright, Parsons & Lu 2004** — *Toward mapping surface deformation in three
  dimensions using InSAR*, GRL 31, L01607. (multi-viewing-geometry 3-D retrieval —
  why a second track approaches full 3-D in `mogi_invert_joint`.)
- **Lohman & Simons 2005** — *Some thoughts on the use of InSAR data to constrain
  models of surface deformation: Noise structure and data downsampling*, G³ 6,
  Q01007. (quadtree downsampling + data covariance — the `InsarLos` input contract;
  downsampling itself is the reader's job, NOT the leaf's.)
- **Sudhaus & Jónsson 2009** — *Improved source modelling through combined use of
  InSAR and GPS under consideration of correlated data errors: application to the
  June 2000 Kleifarvatn earthquake*, GJI 176(2), 389–404. (joint InSAR+GPS NLLS with
  variance-component-estimated dataset weights — `mogi_invert_joint`.)
- **Koch 1999** — *Parameter Estimation and Hypothesis Testing in Linear Models*
  (2nd ed., Springer), ch. 3. (Helmert variance component estimation, partial
  redundancies — `variance_components`.)
- Add domain refs (Hackl et al., Bevis & Brown trajectory model, Hector) as used —
  always with the specific equation/section.

## 6. Derived products
Any product written to the store (velocities, break epochs, detrended series, model
params) carries **provenance**: method tag (`wls`/`gbis`), reference frame, fit type,
software version, `fitted_at`, and the reference for the estimator that produced it.
The API `method` field and the precompute metadata are where this surfaces.

---
*Created 2026-07-10 (analysis lane, per BGÓ's rigor requirement). Binding on H1/H2/H3/H4/H5.*
