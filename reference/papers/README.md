# Cited papers

PDFs here are **gitignored** — same policy as `reference/gbis4ts/*.pdf` ("keep
code, drop heavy PDFs"), and most publisher PDFs may not be redistributed even
when they are readable from this network.

Fetch any of them in one command:

```bash
python3 ~/.claude/skills/search-scholar/scripts/paper-get.py <doi> --out reference/papers
```

`MATH_STANDARDS.md` §2.4 requires citing a **specific equation or section
number**. Those can only be verified against the primary PDF — citing papers
restate a model's prose but almost never preserve its original numbering. This
directory exists so that verification is a one-command operation rather than a
research project.

## Papers whose equation numbers this package cites

| DOI | Paper | Access from this host | Cited for |
|---|---|---|---|
| `10.1007/s00190-013-0685-5` | Bevis & Brown 2014, *Trajectory models and reference frames for crustal motion geodesy*, J. Geodesy 88(3) 283–311 | **open access, CC BY 2.0** | eq. (8) SLTM; eq. (9) `A·log(1+Δt/T)`; eq. (10) ETM; eq. (3) Heaviside `H(0)=1/2` |
| `10.1007/s10589-012-9492-9` | O'Leary & Rust 2013, *Variable projection for nonlinear least squares problems*, Comput. Optim. Appl. 54(3) 579–593 | closed, but downloads via the institutional path | eq. (8) VARPRO Jacobian, p. 585; §2.5 bordered covariance, p. 587 |
| `10.1002/2015JB012552` | Blewitt et al. 2016, *MIDAS robust trend estimator…*, JGR Solid Earth 121 | open access via PMC5024356 | eqs. (1)–(8), the MIDAS estimator |
| `10.1007/s00190-002-0283-4` | Williams 2003, *The effect of coloured noise…*, J. Geodesy 76 | closed, institutional path | power-law covariance |
| `10.1002/2013JB010569` | Reverso et al. 2014, *A two-magma chamber model … Grímsvötn Volcano, Iceland*, JGR Solid Earth 119, 4666–4683 | OA; Wiley 403s to curl — use firecrawl stealth | **eq. (20)** the exp+linear inflation fit (`models.py::exp_linear`); eqs. (11)–(12) the physics; eq. (17) displacement ∝ overpressure |

### Verified conventions worth not re-deriving

**Bevis & Brown 2014** — its seasonal terms use **absolute** `sin(ω_k t)` while
the polynomial uses `(t − t_R)`, *in the same equation* (eqs. 5 and 10). That is
the citation for this package's "Seasonal ignores `t_ref`" convention. It also
fixes τ as station metadata (default `T = 1 yr`, the linear variant called the
**ELTM**) and refines it by a 1-D grid search — and its Appendix 1 shows the fit
is strikingly insensitive to `T`, because the SLTM coefficients absorb the error.

⚠ The paper contains **no exponential transient** — the word does not occur in
its 23 pages. Do not cite it for one. `models.py::exp_linear` carried exactly
that false attribution until 2026-08-02.

### `exp_linear` — a general ODE solution, NOT a geophysical result

⚠ **Corrected 2026-08-02 after BGÓ's review.** An earlier version of this
section, and of the `models.py` docstring, presented `exp_linear` as
"Reverso et al. 2014 eq. (20)". That over-attributes it.

`x₀ + v·t + A·e^(−k·t)` is the general solution of a **first-order linear ODE
with a linearly drifting equilibrium**. With `u = x − (x₀ + v·t)`:

```
du/dt = −k·u   ⟹   u = A·e^(−k·t)      equivalently   dx/dt = v − k·(x − x₀ − v·t)
```

That is two lines of textbook ODE theory, and it is the honest provenance.
Relaxation toward a moving equilibrium is among the most common behaviours in
nature — Newton's cooling, RC charging under a ramp, decay with resupply,
compartmental and biological kinetics, and any number of unrelated geophysical
systems yield this identical curve.

**Therefore nothing about a process may be inferred from the fit.** A good fit
says only that *something* relaxes at rate `k` toward a trend `v`. Reverso's own
§5 makes the point from the other side: a hydraulically-fed reservoir and one in
a viscous shell fit equally well (Dzurisin et al. 2009), as does
thermo-poro-elastic injection (Fournier & Chardot 2012). Identifying the
mechanism requires independent knowledge of the system, never the curve shape.

Reverso et al. 2014 eq. (20) remains worth citing as **one worked instance**:
they fit posteruptive GNSS at Grímsvötn with `Φ(1 − e^(−t/τ)) + U̇_∞·t + C`
(this function under `x₀ = C + Φ`, `v = U̇_∞`, `A = −Φ`, `k = 1/τ` — which is
where the `A < 0` sign convention comes from) and measure τ = 0.33 ± 0.08 and
0.13 ± 0.04 yr. The same shape appears at Axial Seamount (Nooner & Chadwick
2009), Westdahl and Okmok (Lu et al. 2003, 2010). Examples of use, not
derivations, and not evidence about mechanism.

Retrieval note: Wiley 403s a plain request from here, but
`firecrawl_scrape` with `parsers:["pdf"]` + `proxy:"stealth"` returns the
18-page PDF. The HTML full text renders formulas as GIFs — read the PDF if you
need the symbolic content, not the rendered article.

**Svartsengi specifically**: Parks et al. 2025, *2023–2024 inflation-deflation
cycles at Svartsengi …*, EPSL 658, 119324, doi:10.1016/j.epsl.2025.119324
(hybrid OA) documents the observed behaviour — "during each inflation period,
there is tendency for the inflow rate to decrease as the pressure builds within
the magma domain", volume-change rates falling ~7–9 → ~2.4–4 m³/s — but
publishes volumes and forecasting, **not** an exp+linear fitting equation.
Fitting this form to the Svartsengi GNSS series is ours; the form itself is
nobody's.
