"""A/B verification harness — Stage-0 despike + local-polynomial Hampel.

NOT collected by pytest (no ``test_`` prefix) and NOT part of the shipped
library: this module may read real ``.NEU`` files (read-only) and prints
comparison tables. Run manually:

    uv run python tests/verification_outlier_ab.py [--real]

Compares three detector configurations on synthetic truth cases (and,
with ``--real``, on HOFN/SENG .NEU series):

- ``current``   — ``OutlierParams()`` defaults (order-0 window, no despike)
- ``order1``    — ``window_order=1`` (robust local line, Cleveland 1979)
- ``o1+despike``— ``window_order=1, despike=True`` (Stage-0 pre-filter)

Cases:
- A. clean lineperiodic series + one 150 mm single-epoch spike (the FAGD
  East blunder), both mid-series and near the series end (the live case:
  a fresh blunder whose post-flank is too thin for step evidence);
- B. fast-ramp (SENG-like unrest onset) + undeclared step + one genuine
  25 mm spike riding the ramp.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gps_analysis.models import lineperiodic
from gps_analysis.outliers import OutlierDetection, OutlierParams, detect_outliers

FloatArr = NDArray[np.float64]

DAY = 1.0 / 365.25
TRUE_LP = (12.0, -3.5, 4.0, -2.0, 1.0, 0.5)
WN = 3.0

HOFN_NEU = Path(
    "/home/bgo/work/projects/gpslibrary/gps_data_analyses/detrend-oraef/HOFN-plate.NEU"
)
SENG_NEU = Path("/home/bgo/Downloads/SENG-plate.NEU")

VARIANTS: dict[str, OutlierParams] = {
    "current": OutlierParams(),
    "order1": OutlierParams(window_order=1),
    "order2": OutlierParams(window_order=2),
    "o1+despike": OutlierParams(window_order=1, despike=True),
}


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """Metrics of one detector variant on one synthetic truth case."""

    variant: str
    n: int
    n_candidates: int
    n_flags: int
    n_despiked: int
    true_positive: int
    false_positive: int
    false_negative: int
    aborted: bool
    rate_error: float  # |fitted rate - true rate| [mm/yr]
    scale_global: float


def _metrics(
    name: str,
    res: OutlierDetection,
    truth_idx: NDArray[np.intp],
    true_rate: float,
) -> CaseResult:
    flags = np.asarray(res.flags, dtype=np.bool_)
    truth = np.zeros(flags.size, dtype=np.bool_)
    truth[truth_idx] = True
    return CaseResult(
        variant=name,
        n=int(flags.size),
        n_candidates=int(res.candidates.sum()),
        n_flags=int(flags.sum()),
        n_despiked=int(res.n_despiked.sum()),
        true_positive=int(np.count_nonzero(flags & truth)),
        false_positive=int(np.count_nonzero(flags & ~truth)),
        false_negative=int(np.count_nonzero(~flags & truth)),
        aborted=bool(res.excess_flag_abort),
        rate_error=abs(float(res.fits[0].params[1]) - true_rate),
        scale_global=float(res.scale_global[0]),
    )


def _print_table(title: str, rows: list[CaseResult]) -> None:
    print(f"\n== {title} ==")
    hdr = (
        f"{'variant':<12}{'N':>6}{'cand':>6}{'flags':>6}{'desp':>6}"
        f"{'TP':>4}{'FP':>4}{'FN':>4}{'abort':>7}{'rate_err':>10}{'s_glob':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.variant:<12}{r.n:>6}{r.n_candidates:>6}{r.n_flags:>6}"
            f"{r.n_despiked:>6}{r.true_positive:>4}{r.false_positive:>4}"
            f"{r.false_negative:>4}{str(r.aborted):>7}"
            f"{r.rate_error:>10.3f}{r.scale_global:>8.3f}"
        )


# ---------------------------------------------------------------------------
# Case A — FAGD-like 150 mm single-epoch blunder
# ---------------------------------------------------------------------------


def case_a_spike(spike_at_end: bool) -> list[CaseResult]:
    """Clean series + one 150 mm spike (mid-series or 2 d before the end)."""
    rng = np.random.default_rng(101)
    n = 1500
    t = 2021.0 + np.arange(n, dtype=np.float64) * DAY
    y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, WN, n)
    i_spike = n - 2 if spike_at_end else n // 2
    y[i_spike] += 150.0
    truth = np.array([i_spike], dtype=np.intp)
    rows = []
    for name, params in VARIANTS.items():
        res = detect_outliers(lineperiodic, t, y, params=params, min_outlier=5.0)
        rows.append(_metrics(name, res, truth, TRUE_LP[1]))
    return rows


# ---------------------------------------------------------------------------
# Case B — SENG-like steep unrest ramp with modest spikes riding it
# ---------------------------------------------------------------------------


def _metrics_recall(
    name: str,
    res: OutlierDetection,
    spikes: NDArray[np.intp],
    seg: slice,
) -> tuple[str, int, int, int, float, bool]:
    """Recall-focused metrics for the ramp case: (variant, caught, total_flags,
    other_flags, median local scale on the ramp segment, abort)."""
    flags = np.asarray(res.flags, dtype=np.bool_)
    caught = int(flags[spikes].sum())
    with np.errstate(invalid="ignore"):
        s_local = float(np.nanmedian(res.scale_local[seg]))
    return (
        name,
        caught,
        int(flags.sum()),
        int(flags.sum()) - caught,
        s_local,
        bool(res.excess_flag_abort),
    )


def case_b_ramp() -> None:
    """Steep 3 mm/d ongoing ramp; 3 modest spikes ride the steep part.

    order-0's local MAD is inflated by the in-window ramp spread, raising
    the Hampel threshold so the spikes are masked (0/3); a robust local
    line/parabola (order 1/2) restores an honest local scale and recovers
    recall. Reported with the local identifier isolated (``k_g`` raised)
    so the mechanism is not masked by the global identifier, and again
    under realistic defaults.
    """
    rng = np.random.default_rng(5)
    n = 2000
    t = 2018.0 + np.arange(n, dtype=np.float64) * DAY
    y = lineperiodic(t, *TRUE_LP) + rng.normal(0.0, 2.0, n)
    i0 = 900
    y[i0:] += (t[i0:] - t[i0]) / DAY * 3.0  # 3 mm/d ongoing unrest ramp
    spikes = np.array([950, 1000, 1050], dtype=np.intp)
    y[spikes] += np.array([12.0, -13.0, 12.0])
    seg = slice(i0, 1100)

    def _run(title: str, kg: float) -> None:
        print(f"\n== Case B — {title} ==")
        hdr = (
            f"{'variant':<12}{'spikes':>8}{'total_fl':>9}{'other_fl':>9}"
            f"{'s_local':>9}{'abort':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for name, base in VARIANTS.items():
            params = dataclasses.replace(base, global_n_sigma=kg)
            res = detect_outliers(lineperiodic, t, y, params=params, min_outlier=5.0)
            _, caught, tot, other, sl, ab = _metrics_recall(name, res, spikes, seg)
            print(
                f"{name:<12}{f'{caught}/3':>8}{tot:>9}{other:>9}{sl:>9.3f}"
                f"{str(ab):>7}"
            )

    _run("local identifier isolated (k_g = 50)", 50.0)
    _run("realistic defaults (k_g = 5)", 5.0)


# ---------------------------------------------------------------------------
# Real .NEU series (read-only; harness-local reader, never in src/)
# ---------------------------------------------------------------------------


def _yearf(date: datetime.datetime) -> float:
    """Fractional year of a datetime (leap-aware; harness precision)."""
    y0 = datetime.datetime(date.year, 1, 1)
    y1 = datetime.datetime(date.year + 1, 1, 1)
    return date.year + (date - y0) / (y1 - y0)


def read_neu(path: Path) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Read a .NEU file → (t [yearf], y (3, N) [mm], sigma (3, N) [mm]).

    Handles both first-column formats: ``yyyy/mm/dd HH:MM:SS.SSS`` and
    decimal year ``yyyy.dddd``. Read-only; duplicate/unsorted epochs are
    sorted and deduplicated (first kept).
    """
    times: list[float] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "*", '"')):
            continue
        parts = line.split()
        if "/" in parts[0]:
            stamp_str = f"{parts[0]} {parts[1]}"
            fmt = "%Y/%m/%d %H:%M:%S.%f" if "." in parts[1] else "%Y/%m/%d %H:%M:%S"
            stamp = datetime.datetime.strptime(stamp_str, fmt)
            t_val = _yearf(stamp)
            vals = [float(v) for v in parts[2:8]]
        else:
            t_val = float(parts[0])
            vals = [float(v) for v in parts[1:7]]
        times.append(t_val)
        rows.append(vals)
    t = np.asarray(times, dtype=np.float64)
    data = np.asarray(rows, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, data = t[order], data[order]
    keep = np.concatenate(([True], np.diff(t) > 0.0))
    t, data = t[keep], data[keep]
    y = data[:, [0, 2, 4]].T.copy()  # dN, dE, dU
    sigma = data[:, [1, 3, 5]].T.copy()  # DN, DE, DU
    return t, y, sigma


def real_series(path: Path, label: str) -> None:
    """Run the three variants on a real .NEU series and print the table."""
    if not path.exists():
        print(f"\n== {label}: {path} not found — skipped ==")
        return
    t, y, sigma = read_neu(path)
    print(f"\n== {label} ({path.name}): N = {t.size}, {t[0]:.2f}–{t[-1]:.2f} ==")
    hdr = (
        f"{'variant':<12}{'comp':>5}{'cand':>7}{'flags':>7}{'desp':>6}"
        f"{'abort':>7}{'s_glob':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, params in VARIANTS.items():
        res = detect_outliers(
            lineperiodic, t, y, sigma, params=params, min_outlier=[5.0, 5.0, 10.0]
        )
        for c, comp in enumerate(("N", "E", "U")):
            print(
                f"{name:<12}{comp:>5}{int(res.candidates[c].sum()):>7}"
                f"{int(res.flags[c].sum()):>7}{int(res.n_despiked[c]):>6}"
                f"{str(bool(res.excess_flag_abort)):>7}"
                f"{float(res.scale_global[c]):>8.3f}"
            )


def main() -> None:
    _print_table("Case A1 — 150 mm spike MID-series", case_a_spike(False))
    _print_table("Case A2 — 150 mm spike 2 d before series END", case_a_spike(True))
    case_b_ramp()
    if "--real" in sys.argv:
        real_series(HOFN_NEU, "HOFN")
        real_series(SENG_NEU, "SENG")


if __name__ == "__main__":
    main()
