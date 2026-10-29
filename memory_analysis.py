"""Descriptive analysis: hydrographs, map, aquifer memory and GRACE agreement.

1. figures/hydrographs.png: all study wells, excluded and flagged ones marked
2. figures/study_area.png: well locations
3. Aquifer memory: correlation between each well's deseasonalised level
   anomaly and the deseasonalised P - E sum over the previous w months,
   w = 1..240. The best w is the time scale over which the aquifer
   integrates the water balance. results/memory_windows.csv,
   figures/memory.png
4. GRACE: regional well index (mean of standardised deseasonalised
   anomalies) vs deseasonalised GRACE storage, 2002-2024, with lags.
   results/grace_agreement.csv, figures/grace_vs_wells.png

Correlations of autocorrelated series are reported without p-values, since
the effective sample size is much smaller than the number of months.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXCLUDED, FLAGGED, SHORT, WELLS, deseasonalised_anomaly, merged

RES = Path("results")
FIG = Path("figures")
WINDOWS = range(1, 241)
MEMORY_START = "1980-01"
MIN_WELLS_INDEX = 8


def deseason(s, dates):
    s = pd.Series(np.asarray(s, dtype=float))
    month = pd.Series(dates).str[5:7].astype(int).values
    return (s - s.groupby(month).transform("mean")).values


def hydrograph_figure(df):
    names = list(WELLS)
    fig, axes = plt.subplots(7, 2, figsize=(10, 12), sharex=True)
    for ax, name in zip(axes.flat, names):
        d = df[df.name == name]
        t = pd.to_datetime(d.date)
        colour = "#b0b0b0" if name in EXCLUDED else "#c0504d" if name in FLAGGED else "#2f5d8a"
        ax.plot(t, d.level_m, lw=0.7, color=colour)
        tag = " (excluded)" if name in EXCLUDED else " (flagged)" if name in FLAGGED else ""
        ax.set_title(SHORT[name] + tag, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supylabel("Water level (m above sea level)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "hydrographs.png", dpi=200)


LABEL_OFFSETS = {  # (dx, dy, ha) in points, to keep close labels apart
    "WSA Agrium 43": (-4, 4, "right"),
    "WSA Vanscoy": (4, -9, "left"),
    "WSA Conquest No.500": (4, -9, "left"),
    "WSA Conquest No.501": (-4, 4, "right"),
}


def map_figure():
    w = pd.read_csv("data/gee/wells_gee.csv").set_index("name")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, drv in WELLS.items():
        lat, lon = w.loc[drv, "lat"], w.loc[drv, "lon"]
        colour = "#b0b0b0" if name in EXCLUDED else "#c0504d" if name in FLAGGED else "#2f5d8a"
        ax.scatter(lon, lat, s=30, color=colour, zorder=3, edgecolor="white", lw=0.5)
        if name != "WSA Duck Lake No. 2":
            label = "Duck Lake 1, 2" if name == "WSA Duck Lake No. 1" else SHORT[name]
            dx, dy, ha = LABEL_OFFSETS.get(name, (4, 3, "left"))
            ax.annotate(label, (lon, lat), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, ha=ha)
    ax.scatter(-106.67, 52.13, marker="*", s=120, color="black", zorder=4)
    ax.annotate("Saskatoon (city)", (-106.67, 52.13), xytext=(6, -4), textcoords="offset points",
                fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(-107.5, -105.4, 0.5))
    ax.set_xlim(-107.8, -105.35)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(52.2)))
    ax.grid(color="0.9", lw=0.5)
    ax.set_title("WSA observation wells used (blue: core, red: flagged, grey: excluded)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "study_area.png", dpi=200)


def memory(df):
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    drivers = pd.read_csv("data/drivers_monthly.csv")
    rows = []
    for name, g in df.groupby("name"):
        # Same months for every window: the drivers start in 1960, so from
        # 1980 on even the 240-month window is available
        g = g[g.date >= MEMORY_START].sort_values("date")
        # P - E for this location over the whole driver record, so windows
        # can reach back before the first level observation
        drv = drivers[drivers.name == WELLS[name]].sort_values("date")
        pme = drv.set_index("date").p_minus_e_mm
        for w in WINDOWS:
            s = pme.rolling(w).sum()
            s = pd.Series(deseason(s.values, s.index), index=s.index)
            x = s.reindex(g.date).values
            ok = ~np.isnan(x) & g.anom.notna().values
            rows.append(dict(name=name, window=w, r=np.corrcoef(x[ok], g.anom.values[ok])[0, 1],
                             n=int(ok.sum())))
    r = pd.DataFrame(rows)
    best = r.loc[r.groupby("name").r.idxmax()].sort_values("window")
    best["flag"] = best.name.map(FLAGGED).fillna("")
    best.to_csv(RES / "memory_windows.csv", index=False, float_format="%.3f")

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = plt.get_cmap("tab20").colors
    for k, (name, g) in enumerate(r.groupby("name")):
        ls = "--" if name in FLAGGED else "-"
        ax.plot(g.window, g.r, ls, lw=1, color=colours[k % 20], label=SHORT[name])
    ax.scatter(best.window, best.r, s=12, color="black", zorder=3)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Window w (months) of the preceding P − E sum")
    ax.set_ylabel("Correlation with level anomaly")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "memory.png", dpi=200)
    return best


def grace(df):
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    df["z"] = df.groupby("name").anom.transform(lambda s: (s - s.mean()) / s.std())
    idx = df.groupby("date").agg(z=("z", "mean"), n=("z", "count"))
    idx = idx[idx.n >= MIN_WELLS_INDEX]

    gr = (df.dropna(subset=["grace_lwe_cm"]).drop_duplicates(["driver_name", "date"])
            .groupby("date").grace_lwe_cm.mean())
    gr_ds = pd.Series(deseason(gr.values, gr.index), index=gr.index)
    j = pd.concat([idx.z, gr, gr_ds.rename("grace_ds")], axis=1).dropna()

    rows = []
    full = pd.period_range(j.index.min(), j.index.max(), freq="M").astype(str)
    z = idx.z.reindex(full)
    g = gr_ds.reindex(full)
    for lag in range(0, 37):  # wells lag GRACE by `lag` months
        a, b = z.shift(-lag), g
        ok = a.notna() & b.notna()
        rows.append(dict(lag_months=lag, r=np.corrcoef(a[ok], b[ok])[0, 1], n=int(ok.sum())))
    lags = pd.DataFrame(rows)
    lags.to_csv(RES / "grace_agreement.csv", index=False, float_format="%.3f")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = pd.to_datetime(j.index)
    ax.plot(t, j.z, color="#2f5d8a", lw=1.2, label=f"Well index (mean of {MIN_WELLS_INDEX}+ wells, z-score)")
    ax.set_ylabel("Well index (z)")
    ax2 = ax.twinx()
    ax2.plot(t, j.grace_ds, "o-", ms=2, lw=0.7, color="#c0504d", label="GRACE/GRACE-FO (deseasonalised)")
    ax2.set_ylabel("GRACE storage anomaly (cm)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"Regional well index vs GRACE, r = {lags.r.iloc[0]:.2f} (lag 0)", fontsize=9)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "grace_vs_wells.png", dpi=200)
    return lags, j


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = merged(include_excluded=True)
    hydrograph_figure(df)
    map_figure()

    summary = (df.groupby("name").agg(first=("date", lambda s: s[df.loc[s.index, "level_m"].notna()].min()),
                                       last=("date", lambda s: s[df.loc[s.index, "level_m"].notna()].max()),
                                       months=("level_m", "count"),
                                       range_m=("level_m", lambda s: s.max() - s.min()))
                 .reset_index())
    summary["status"] = summary.name.map(lambda n: "excluded" if n in EXCLUDED else
                                         "flagged" if n in FLAGGED else "core")
    summary.to_csv(RES / "well_summary.csv", index=False, float_format="%.2f")
    print(summary.to_string(index=False))

    best = memory(df)
    print("\nBest P - E window per well:")
    print(best[["name", "window", "r", "n"]].to_string(index=False))

    lags, j = grace(df)
    print("\nGRACE vs well index (wells lag GRACE by lag months):")
    print(lags.to_string(index=False))
    print(f"r with raw (not deseasonalised) GRACE: {np.corrcoef(j.z, j.grace_lwe_cm)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
