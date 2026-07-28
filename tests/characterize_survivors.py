"""Characterization harness — WHERE do outlier survivors live?

NOT collected by pytest (no ``test_`` prefix) and NOT part of the shipped
library: reads real ``.NEU`` files (read-only) and prints a distribution
table. Run manually from the gps_analysis package root:

    uv run python tests/characterize_survivors.py

Goal (the go/no-go gate before building any edge/gap remedy): the three
proposed follow-on #1 remedies — (a) declared-step segment-local fallback,
(b) edge/gap-aware windowing, (c) union epochs — fix *different* failure
modes. We do not yet know which dominates. This harness measures it.

Method (non-circular): run the SAME production code path twice per station,
differing only in the windowing knobs —

  - ``prod``    — production-faithful: ``window_order=1`` robust local line,
                  per-component floors ``min_outlier=[5, 5, 10]`` mm.
  - ``aggr``    — identical, but ``window_min_count`` dropped to ``order+2``
                  and the window widened: this is exactly "production WITH
                  edge/gap-aware windowing" — near a boundary or across a
                  gap it fits a one-sided local line where ``prod`` fell back
                  to the (meaningless-on-active-stations) global scale.

  SURVIVORS = ``aggr.flags & ~prod.flags`` — the epochs a windowing remedy
  would recover. Each survivor is then split two ways:

  1. ``prod.scale_local`` NaN (the thin-window global fallback fired here)
     → the #1/#2 windowing target.  finite → a DENSE window that ``aggr``
     only caught by widening/lowering threshold → closer to a floor/threshold
     issue that windowing does NOT fix.
  2. local context — series-edge / gap-edge / sparse / step-edge / dense —
     from the epoch spacing around it (shared across components).

No declared steps / protect_windows are supplied (station-specific); step
context is inferred from a robust local median jump. That undercounts
step-edge survivors on stations with real declared steps — noted in output.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Reuse the read-only .NEU reader from the A/B harness (same tests dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verification_outlier_ab import read_neu  # noqa: E402

from gps_analysis.models import lineperiodic  # noqa: E402
from gps_analysis.outliers import (  # noqa: E402
    OutlierParams,
    detect_outliers,
)

FloatArr = NDArray[np.float64]
DAYS_PER_YEAR = 365.25
COMP = ("N", "E", "U")
MIN_OUTLIER = np.array([5.0, 5.0, 10.0])  # per-component floors [mm]

# Candidate stations: active/gappy series across the reference blob.
BLOB = Path("/home/bgo/work/projects/gpslibrary/gps_data_analyses")
STATIONS: dict[str, Path] = {
    "SENG": Path("/home/bgo/Downloads/SENG-plate.NEU"),
    "HOFN": BLOB / "detrend-oraef/HOFN-plate.NEU",
    "KVSK": BLOB / "detrend-oraef/KVSK-plate.NEU",
    "SKFC": BLOB / "detrend-oraef/SKFC-plate.NEU",
    "KALF": BLOB / "detrend-oraef/KALF-plate.NEU",
    "HAUC": BLOB / "detrend-bb/HAUC-plate.NEU",
    "KIDC": BLOB / "detrend-bb/KIDC-plate.NEU",
    "URHC": BLOB / "detrend-bb/URHC-plate.NEU",
}

PROD = OutlierParams(window_order=1, window_robust_iterations=2, despike=True)
# Windowing-ONLY variant: one-sided-capable windows, SAME threshold as prod.
# Extra flags here = epochs recovered purely by edge/gap-aware windowing
# (remedy #2). This is the clean test of the follow-on #1 hypothesis.
WIDE = OutlierParams(
    window_order=1,
    window_robust_iterations=2,
    despike=True,
    window_min_count=3,  # order + 2 — fit wherever >=3 neighbors exist
    window_days=61.0,  # wider reach across gaps / into boundaries
    window_n_sigma=PROD.window_n_sigma,  # threshold held fixed (no confound)
)
# Threshold-ONLY variant: prod windows, looser Hampel k_w. Extra flags here =
# epochs recoverable by threshold/floor tuning, NOT windowing.
LOOSE = OutlierParams(
    window_order=1,
    window_robust_iterations=2,
    despike=True,
    window_n_sigma=3.0,
)

# Context-classification thresholds (on the yearf axis).
H_PROD = PROD.window_days / 2.0 / DAYS_PER_YEAR  # prod half-window [yr]
GAP_DAYS = 20.0  # a one-side spacing wider than this => "gap", not "boundary"
STEP_JUMP_MM = np.array([8.0, 8.0, 15.0])  # local-median jump => step-edge


def _context(t: FloatArr, i: int) -> str:
    """Classify the local sampling context around epoch ``i`` (shared N/E/U).

    Order of precedence: series-edge (a true series end) → gap-edge (a wide
    one-side spacing) → sparse (both sides present but the prod window is
    thin) → dense (a full two-sided prod window). ``step-edge`` is layered on
    top by the caller from the residual jump, since it can co-occur.
    """
    n = t.size
    left_gap = np.inf if i == 0 else t[i] - t[i - 1]
    right_gap = np.inf if i == n - 1 else t[i + 1] - t[i]
    n_left = int(np.count_nonzero((t < t[i]) & (t >= t[i] - H_PROD)))
    n_right = int(np.count_nonzero((t > t[i]) & (t <= t[i] + H_PROD)))
    gap_yr = GAP_DAYS / DAYS_PER_YEAR
    if i == 0 or i == n - 1:
        return "series-edge"
    if left_gap > gap_yr or right_gap > gap_yr:
        return "gap-edge"
    if n_left + n_right + 1 < PROD.window_min_count:
        return "sparse"
    return "dense"


def _is_step_edge(t: FloatArr, y_c: FloatArr, i: int, jump_mm: float) -> bool:
    """Robust local-median jump across epoch ``i`` exceeds ``jump_mm``.

    Proxy for an (undeclared) step/offset near the survivor, using the same
    time-windowed rolling median the detector uses. Compares the median of
    the ``[i-W, i)`` flank to ``(i, i+W]``; ``W`` = the prod half-window.
    """
    lo = t[i] - H_PROD
    hi = t[i] + H_PROD
    pre = y_c[(t >= lo) & (t < t[i])]
    post = y_c[(t > t[i]) & (t <= hi)]
    if pre.size < 3 or post.size < 3:
        return False
    return bool(abs(float(np.median(post)) - float(np.median(pre))) > jump_mm)


def characterize(label: str, path: Path) -> None:
    if not path.exists():
        print(f"\n== {label}: {path} not found — skipped ==")
        return
    t, y, sigma = read_neu(path)
    prod = detect_outliers(
        lineperiodic, t, y, sigma, min_outlier=MIN_OUTLIER, params=PROD
    )
    wide = detect_outliers(
        lineperiodic, t, y, sigma, min_outlier=MIN_OUTLIER, params=WIDE
    )
    loose = detect_outliers(
        lineperiodic, t, y, sigma, min_outlier=MIN_OUTLIER, params=LOOSE
    )
    n = t.size
    span = t[-1] - t[0]
    thin_frac = float(np.mean(np.isnan(prod.scale_local)))
    print(
        f"\n== {label} ({path.name}): N={n}, {t[0]:.2f}-{t[-1]:.2f} "
        f"({span:.1f} yr), abort(prod/wide/loose)="
        f"{int(prod.excess_flag_abort)}/{int(wide.excess_flag_abort)}/"
        f"{int(loose.excess_flag_abort)} | thin-window frac={thin_frac:.1%} =="
    )
    print(
        f"  prod flags: N={int(prod.flags[0].sum())} E={int(prod.flags[1].sum())} "
        f"U={int(prod.flags[2].sum())}"
    )
    # Windowing-only survivors (the follow-on #1 target) with context split.
    ctx_all: Counter[str] = Counter()
    thin_split: Counter[str] = Counter()
    win_surv = 0
    examples: list[str] = []
    for c in range(3):
        idx = np.flatnonzero(wide.flags[c] & ~prod.flags[c])
        win_surv += idx.size
        for i in idx:
            ctx = _context(t, int(i))
            if _is_step_edge(t, y[c], int(i), STEP_JUMP_MM[c]):
                ctx = "step-edge"
            ctx_all[ctx] += 1
            thin_split["thin" if np.isnan(prod.scale_local[c, i]) else "dense"] += 1
            if len(examples) < 6:
                examples.append(
                    f"      {COMP[c]} @ {t[i]:.3f}  y={y[c, i]:+8.1f}mm  "
                    f"ctx={ctx:<11} loc="
                    f"{'thin' if np.isnan(prod.scale_local[c, i]) else 'dense'}"
                )
    thr_surv = sum(
        int((loose.flags[c] & ~prod.flags[c]).sum()) for c in range(3)
    )
    print(f"  WINDOWING-recoverable survivors (wide & ~prod): {win_surv}")
    if win_surv:
        print(
            "    by context: "
            + "  ".join(f"{k}={v}" for k, v in ctx_all.most_common())
        )
        print(
            "    by location: "
            + "  ".join(f"{k}={v}" for k, v in thin_split.most_common())
        )
        print("\n".join(examples))
    print(f"  THRESHOLD-recoverable survivors (loose k_w=3.0 & ~prod): {thr_surv}")


def main() -> None:
    print(
        "Survivor characterization — prod (order-1, floors 5/5/10) vs "
        "aggr (min_count=3, window=61d).\n"
        "SURVIVOR = flagged by aggr, missed by prod. loc=thin => prod fell "
        "back to global scale here (the windowing target);\n"
        "loc=dense => prod had a full local window yet missed => "
        "threshold/floor, NOT a windowing fix."
    )
    agg_ctx: Counter[str] = Counter()
    for label, path in STATIONS.items():
        characterize(label, path)
    print(
        "\n[note] no declared steps/protect_windows supplied — step-edge "
        "survivors on stations with real offsets are undercounted."
    )
    _ = agg_ctx


if __name__ == "__main__":
    main()
