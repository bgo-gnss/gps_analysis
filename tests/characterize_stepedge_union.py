"""Close the two holes the survivor sweep left open (step-edge + union).

NOT collected by pytest; read-only. Run from the gps_analysis package root:

    uv run python tests/characterize_stepedge_union.py

The continuous-station survivor sweep (``characterize_survivors.py``) showed
thin-window/edge-gap survivors are ~non-existent (thin fraction 0.1-2%). But
it ran every station STEPLESS and SENG aborted to 0/0/0, so two of follow-on
#1's three sub-parts had no data:

  1. STEP-EDGE — synthetic hard-step truth here: spikes sitting right at a
     40 mm offset, detected with the step UNDECLARED vs DECLARED (the existing
     ``step_epochs`` lever). Question: do step edges spawn survivors that a
     windowing remedy (not the existing step lever) would need to fix?

  2. UNION — flip ``epoch_policy`` on the real stations. Question: does the
     already-shipped ``union`` policy deliver the "coherent all-component
     epochs" sub-part #3 wanted, with no new code?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verification_outlier_ab import read_neu  # noqa: E402

from gps_analysis.models import lineperiodic  # noqa: E402
from gps_analysis.outliers import OutlierParams, detect_outliers  # noqa: E402

FloatArr = NDArray[np.float64]
DAY = 1.0 / 365.25
TRUE_LP = (12.0, -3.5, 4.0, -2.0, 1.0, 0.5)

PROD = OutlierParams(window_order=1, window_robust_iterations=2, despike=True)
WIDE = OutlierParams(
    window_order=1,
    window_robust_iterations=2,
    despike=True,
    window_min_count=3,
    window_days=61.0,
)

BLOB = Path("/home/bgo/work/projects/gpslibrary/gps_data_analyses")
STATIONS: dict[str, Path] = {
    "HOFN": BLOB / "detrend-oraef/HOFN-plate.NEU",
    "KVSK": BLOB / "detrend-oraef/KVSK-plate.NEU",
    "SKFC": BLOB / "detrend-oraef/SKFC-plate.NEU",
    "KALF": BLOB / "detrend-oraef/KALF-plate.NEU",
    "HAUC": BLOB / "detrend-bb/HAUC-plate.NEU",
    "KIDC": BLOB / "detrend-bb/KIDC-plate.NEU",
}


def step_edge_truth() -> None:
    """Hard 40 mm step; spikes at the step edge, step undeclared vs declared."""
    print("=" * 70)
    print("STEP-EDGE (synthetic truth): 40 mm offset at t0, 30 mm spikes on it")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n = 1500
    t = 2021.0 + np.arange(n, dtype=np.float64) * DAY
    y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, 3.0, n)
    i0 = 750
    y[i0:] += 40.0  # hard step
    # spikes: just-before / on / just-after the step edge, + one far control
    spikes = np.array([i0 - 1, i0, i0 + 1, n // 4], dtype=np.intp)
    y[spikes] += np.array([30.0, -30.0, 30.0, 30.0])
    t0 = t[i0]

    def run(title: str, params: OutlierParams, steps: FloatArr | None) -> None:
        res = detect_outliers(
            lineperiodic, t, y, min_outlier=5.0, params=params, step_epochs=steps
        )
        caught = int(res.flags[spikes].sum())
        near = (t >= t0 - 15 * DAY) & (t <= t0 + 15 * DAY)
        near[spikes] = False
        false_near = int(res.flags[near].sum())
        edge = "  ".join(
            f"{lbl}={'Y' if res.flags[j] else '.'}"
            for lbl, j in zip(("pre", "on", "post", "ctrl"), spikes, strict=True)
        )
        print(
            f"  {title:<28} caught={caught}/4  [{edge}]  "
            f"false@step±15d={false_near}  abort={int(res.excess_flag_abort)}"
        )

    print("\n-- step UNDECLARED (fit sees the offset as signal) --")
    run("prod (win 31d)", PROD, None)
    run("wide (win 61d, min_count 3)", WIDE, None)
    print("\n-- step DECLARED via step_epochs=[t0] (the existing lever) --")
    run("prod + declared step", PROD, np.array([t0]))
    run("wide + declared step", WIDE, np.array([t0]))


def union_check() -> None:
    """Per-component vs union epoch policy on the real stations."""
    print("\n" + "=" * 70)
    print("UNION (sub-part #3): epoch_policy per_component vs union, real data")
    print("=" * 70)
    per = PROD
    uni = OutlierParams(
        window_order=1, window_robust_iterations=2, despike=True, epoch_policy="union"
    )
    mo = np.array([5.0, 5.0, 10.0])
    print(f"  {'station':<8}{'per_comp NEU':>16}{'union NEU':>16}{'added by union':>16}")
    print("  " + "-" * 54)
    for label, path in STATIONS.items():
        if not path.exists():
            continue
        t, y, sigma = read_neu(path)
        rp = detect_outliers(lineperiodic, t, y, sigma, min_outlier=mo, params=per)
        ru = detect_outliers(lineperiodic, t, y, sigma, min_outlier=mo, params=uni)
        pc = [int(rp.flags[c].sum()) for c in range(3)]
        uc = [int(ru.flags[c].sum()) for c in range(3)]
        # epochs newly covered in some component by union but not per_component
        added = int((ru.flags & ~rp.flags).sum())
        print(
            f"  {label:<8}{str(pc):>16}{str(uc):>16}{added:>16}"
        )
    print(
        "\n  union := an epoch flagged in ANY component is flagged in ALL "
        "(coherent all-component epochs). It only PROMOTES existing\n"
        "  per-component flags across components; it never finds a NEW epoch "
        "no component flagged. So it answers sub-part #3's\n"
        "  'coherent epochs' goal but does NOT recover survivors."
    )


def main() -> None:
    step_edge_truth()
    union_check()


if __name__ == "__main__":
    main()
