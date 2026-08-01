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
