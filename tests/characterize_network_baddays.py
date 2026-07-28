"""Sweep the archive for network-wide bad solution days (S-N feasibility).

Per station/component: residual against a centred 25-day rolling median,
scaled by the robust MAD of those residuals -> z. Then for each calendar day,
aggregate z across stations. A day is a candidate when the network moves
COHERENTLY: |median z| large AND most stations exceed 3 sigma.

Reports: candidate count, distribution over years, which component is hit,
and whether the affected component varies (a systematic component points at a
specific processing cause; a random one does not).
"""

import warnings

import geo_dataread.gps_read as gpsr
import numpy as np
import pandas as pd
from gtimes.timefunc import TimefromYearf

warnings.filterwarnings("ignore")

STATIONS = [
    "RHOF", "VMEY", "HOFN", "AKUR", "REYK", "THEY", "STOR", "KIDJ", "ARHO",
    "MYVA", "GRAN", "OLKE", "HVER", "SKHA", "BLON", "HRIC", "HAMR", "SAUD",
]
COMPONENTS = ("north", "east", "up")
MIN_STATIONS = 5
WINDOW_DAYS = 25


def station_z(sta):
    """Robust z of each epoch against a local rolling median, per component."""
    yearf, data, _dd, _ = gpsr.getData(sta, ref="plate", uncert=10)
    fin = np.isfinite(yearf) & np.all(np.isfinite(data), axis=0)
    if int(fin.sum()) < 400:
        return None
    dates = pd.to_datetime([str(TimefromYearf(v))[:10] for v in yearf[fin]])
    out = {}
    for c in range(3):
        s = pd.Series(data[c, fin], index=dates).groupby(level=0).mean()
        s = s.asfreq("D") if s.index.freq else s.resample("D").mean()
        med = s.rolling(WINDOW_DAYS, center=True, min_periods=8).median()
        r = s - med
        scale = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        if not np.isfinite(scale) or scale <= 0:
            continue
        out[c] = (r / scale).dropna()
    return out


print(f"loading {len(STATIONS)} stations …")
Z = {}
for sta in STATIONS:
    try:
        z = station_z(sta)
    except Exception as exc:
        print(f"  {sta}: skip ({type(exc).__name__})")
        continue
    if z:
        Z[sta] = z
print(f"loaded {len(Z)}\n")

rows = []
for c, cname in enumerate(COMPONENTS):
    frame = pd.DataFrame({sta: z[c] for sta, z in Z.items() if c in z})
    n = frame.notna().sum(axis=1)
    med = frame.median(axis=1)
    frac = (frame.abs() > 3).sum(axis=1) / n.replace(0, np.nan)
    ok = (n >= MIN_STATIONS) & (med.abs() > 2.0) & (frac > 0.5)
    for day in frame.index[ok.fillna(False)]:
        rows.append(
            dict(day=day, comp=cname, n=int(n[day]),
                 med_z=float(med[day]), frac=float(frac[day]))
        )

df = pd.DataFrame(rows)
if df.empty:
    print("no candidate days found")
    raise SystemExit

total_days = len(set().union(*[set(z[0].index) for z in Z.values() if 0 in z]))
print(f"CANDIDATE NETWORK-WIDE BAD DAYS: {len(df)} component-days "
      f"across {df.day.nunique()} distinct days")
print(f"(archive spans ~{total_days} station-days on the north component)\n")

print("by component:")
print(df.groupby("comp").agg(days=("day", "nunique"),
                             med_abs_z=("med_z", lambda v: np.median(np.abs(v)))))

print("\nby year:")
print(df.assign(year=df.day.dt.year).groupby("year").day.nunique().to_string())

print("\ndays hit in MORE THAN ONE component (a whole-solution failure):")
multi = df.groupby("day").comp.nunique()
multi = multi[multi > 1].sort_values(ascending=False)
print(f"  {len(multi)} of {df.day.nunique()} days" if len(multi) else "  none")
for day, k in multi.head(12).items():
    sub = df[df.day == day]
    print(f"    {day.date()}  {k} components: "
          + ", ".join(f"{r.comp}({r.med_z:+.1f}, {r.frac:.0%})"
                      for r in sub.itertuples()))

print("\nstrongest 20 component-days:")
for r in df.reindex(df.med_z.abs().sort_values(ascending=False).index).head(20).itertuples():
    print(f"    {r.day.date()}  {r.comp:5s}  n={r.n:2d}  med_z={r.med_z:+7.2f}  "
          f"|z|>3 in {r.frac:.0%}")

print("\nthe two days found by hand:")
for d in ("2013-10-02", "2016-04-02"):
    sub = df[df.day == pd.Timestamp(d)]
    print(f"    {d}: " + (", ".join(f"{r.comp}({r.med_z:+.1f})" for r in sub.itertuples())
                          if len(sub) else "NOT recovered by the sweep"))
