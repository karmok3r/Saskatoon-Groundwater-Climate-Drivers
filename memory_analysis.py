"""Descriptive analysis: hydrographs, map, aquifer memory and GRACE agreement.

1. hydrographs.png: all study wells, excluded and flagged ones marked
2. study_area.png: well locations (province scope: coloured by best window)
3. Aquifer memory: correlation between each well's deseasonalised level
   anomaly and the deseasonalised P - E sum over the previous w months,
   w = 1..240, on months from 1980 on (so every window is available). The
   best w is the time scale over which the aquifer integrates the water
   balance. memory_windows.csv, memory.png
4. GRACE: per-well correlation between the level anomaly and deseasonalised
   GRACE at the well's location, and a regional well index (mean of
   standardised anomalies) vs mean GRACE, with lags. grace_agreement.csv,
   grace_per_well.csv, grace_vs_wells.png

Significance tests that allow for autocorrelation are in robustness.py.
Run with GW_SCOPE=province for the province-wide analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DRIVERS, EXCLUDED, FIG, FLAGGED, LOCATIONS, RES, SCOPE, SHORT, WELLS,
                    deseasonalised_anomaly, merged, window_correlations)

WINDOWS = range(1, 241)
MEMORY_START = "1980-01"
MIN_WELLS_INDEX = 8 if SCOPE == "saskatoon" else 25


def status(name):
    return "excluded" if name in EXCLUDED else "flagged" if name in FLAGGED else "core"


COLOURS = {"core": "#2f5d8a", "flagged": "#c0504d", "excluded": "#b0b0b0"}


def deseason(s, dates):
    s = pd.Series(np.asarray(s, dtype=float))
    month = pd.Series(dates).str[5:7].astype(int).values
    return (s - s.groupby(month).transform("mean")).values


def hydrograph_figure(df):
    names = sorted(WELLS, key=lambda n: (status(n) != "core", n))
    ncol = 2 if len(names) <= 14 else 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10 if ncol == 2 else 16, 1.7 * nrow), sharex=True)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        d = df[df.name == name]
        ax.plot(pd.to_datetime(d.date), d.level_m, lw=0.6, color=COLOURS[status(name)])
        tag = "" if status(name) == "core" else f" ({status(name)})"
        ax.set_title(SHORT[name] + tag, fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supylabel("Water level (m above sea level)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "hydrographs.png", dpi=200)
    plt.close(fig)


LABEL_OFFSETS = {  # (dx, dy, ha) in points, to keep close labels apart (Saskatoon map)
    "WSA Agrium 43": (-4, 4, "right"),
    "WSA Vanscoy": (4, -9, "left"),
    "WSA Conquest No.500": (4, -9, "left"),
    "WSA Conquest No.501": (-4, 4, "right"),
}


def map_figure(best=None):
    loc = pd.read_csv(LOCATIONS).set_index("name")
    fig, ax = plt.subplots(figsize=(5.5, 5.5) if SCOPE == "saskatoon" else (6, 7.5))
    window = best.set_index("name").window if best is not None else None
    sc = None
    for name, site in WELLS.items():
        lat, lon = loc.loc[site, "lat"], loc.loc[site, "lon"]
        if window is not None and name in window.index and status(name) == "core":
            sc = ax.scatter(lon, lat, s=40, c=[window[name] / 12], cmap="viridis", vmin=0, vmax=20,
                            zorder=3, edgecolor="black", lw=0.4)
        else:
            ax.scatter(lon, lat, s=30, color=COLOURS[status(name)], zorder=3, edgecolor="white", lw=0.5)
        if SCOPE == "saskatoon" and name != "WSA Duck Lake No. 2":
            label = "Duck Lake 1, 2" if name == "WSA Duck Lake No. 1" else SHORT[name]
            dx, dy, ha = LABEL_OFFSETS.get(name, (4, 3, "left"))
            ax.annotate(label, (lon, lat), xytext=(dx, dy), textcoords="offset points", fontsize=7, ha=ha)
    for city, (lat, lon) in {"Saskatoon": (52.13, -106.67), "Regina": (50.45, -104.61),
                             "Prince Albert": (53.20, -105.75)}.items():
        if SCOPE == "saskatoon" and city != "Saskatoon":
            continue
        ax.scatter(lon, lat, marker="*", s=110, color="black", zorder=4)
        ax.annotate(city, (lon, lat), xytext=(6, -4), textcoords="offset points", fontsize=8, fontweight="bold")
    if SCOPE == "saskatoon":
        ax.set_xticks(np.arange(-107.5, -105.4, 0.5))
        ax.set_xlim(-107.8, -105.35)
    else:
        ax.set_xlim(-110.2, -101.3)
        ax.set_ylim(48.9, 55.5)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Best P − E window (years), core wells")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(52.2)))
    ax.grid(color="0.9", lw=0.5)
    ax.set_title("WSA observation wells (red: flagged, grey: excluded)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "study_area.png", dpi=200)
    plt.close(fig)


def memory(df, drivers):
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    rows = []
    for name, g in df.groupby("name"):
        g = g[(g.date >= MEMORY_START) & g.anom.notna()].sort_values("date")
        drv = drivers[drivers.name == WELLS[name]].sort_values("date")
        r, n = window_correlations(drv.p_minus_e_mm.values, drv.date.tolist(),
                                   g.anom.values, g.date.tolist(), WINDOWS)
        rows += [dict(name=name, window=w, r=rr, n=int(nn)) for w, rr, nn in zip(WINDOWS, r, n)]
    r = pd.DataFrame(rows)
    best = r.loc[r.groupby("name").r.idxmax()].sort_values("window")
    best["status"] = best.name.map(status)
    best["flag"] = best.name.map(FLAGGED).fillna("")
    best.to_csv(RES / "memory_windows.csv", index=False, float_format="%.3f")

    fig, ax = plt.subplots(figsize=(7, 4))
    if SCOPE == "saskatoon":
        colours = plt.get_cmap("tab20").colors
        for k, (name, g) in enumerate(r.groupby("name")):
            ax.plot(g.window, g.r, "--" if name in FLAGGED else "-", lw=1, color=colours[k % 20], label=SHORT[name])
        ax.legend(fontsize=6, ncol=2, frameon=False)
    else:
        core = r[r.name.map(status) == "core"]
        for _, g in core.groupby("name"):
            ax.plot(g.window, g.r, lw=0.5, color="#9bbbd8")
        med = core.groupby("window").r.median()
        ax.plot(med.index, med.values, lw=2, color="#2f5d8a", label="Median of core wells")
        ax.legend(fontsize=7, frameon=False)
    bc = best[best.status == "core"] if SCOPE != "saskatoon" else best
    ax.scatter(bc.window, bc.r, s=12, color="black", zorder=3)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Window w (months) of the preceding P − E sum")
    ax.set_ylabel("Correlation with level anomaly")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "memory.png", dpi=200)
    plt.close(fig)
    return best


def lag_correlations(z, g):
    rows = []
    for lag in range(0, 37):  # wells lag GRACE by `lag` months
        a = z.shift(-lag)
        ok = a.notna() & g.notna()
        rows.append(dict(lag_months=lag, r=np.corrcoef(a[ok], g[ok])[0, 1], n=int(ok.sum())))
    return pd.DataFrame(rows)


def grace(df):
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    df["z"] = df.groupby("name").anom.transform(lambda s: (s - s.mean()) / s.std())
    gd = df.dropna(subset=["grace_lwe_cm"]).copy()
    gd["grace_ds"] = gd.groupby("name", group_keys=False).apply(
        lambda g: pd.Series(deseason(g.grace_lwe_cm.values, g.date.values), index=g.index))

    # per well, at its own location
    per = []
    for name, g in gd.groupby("name"):
        full = pd.period_range("2002-04", "2024-09", freq="M").astype(str)
        z = df[df.name == name].set_index("date").z.reindex(full)
        gg = g.set_index("date").grace_ds.reindex(full)
        lags = lag_correlations(z, gg)
        per.append(dict(name=name, status=status(name), r_lag0=lags.r.iloc[0],
                        best_lag=int(lags.r.idxmax()), r_best=lags.r.max(), n=int(lags.n.iloc[0])))
    per = pd.DataFrame(per)
    per.to_csv(RES / "grace_per_well.csv", index=False, float_format="%.3f")

    # regional index
    idx = df.groupby("date").agg(z=("z", "mean"), n=("z", "count"))
    idx = idx[idx.n >= MIN_WELLS_INDEX]
    gr = gd.drop_duplicates(["driver_name", "date"]).groupby("date").grace_lwe_cm.mean()
    gr_ds = pd.Series(deseason(gr.values, gr.index), index=gr.index)
    j = pd.concat([idx.z, gr, gr_ds.rename("grace_ds")], axis=1).dropna()
    full = pd.period_range(j.index.min(), j.index.max(), freq="M").astype(str)
    lags = lag_correlations(idx.z.reindex(full), gr_ds.reindex(full))
    lags.to_csv(RES / "grace_agreement.csv", index=False, float_format="%.3f")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = pd.to_datetime(j.index)
    ax.plot(t, j.z, color="#2f5d8a", lw=1.2, label=f"Well index (mean of ≥{MIN_WELLS_INDEX} wells, z-score)")
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
    plt.close(fig)
    return lags, per, j


def main():
    df = merged(include_excluded=True)
    drivers = pd.read_csv(DRIVERS)
    hydrograph_figure(df)

    summary = (df.dropna(subset=["level_m"]).groupby("name")
                 .agg(first=("date", "min"), last=("date", "max"), months=("level_m", "count"),
                      range_m=("level_m", lambda s: s.max() - s.min()))
                 .reset_index())
    summary["status"] = summary.name.map(status)
    summary.to_csv(RES / "well_summary.csv", index=False, float_format="%.2f")
    print(summary.status.value_counts().to_dict())

    best = memory(df, drivers)
    map_figure(best)
    core = best[best.status == "core"]
    print(f"\nBest P - E window, core wells (n={len(core)}): median {core.window.median():.0f} months, "
          f"IQR {core.window.quantile(.25):.0f}-{core.window.quantile(.75):.0f}; "
          f"median r {core.r.median():.2f}; at the 240-month limit: {(core.window >= 228).sum()}")
    print(best[["name", "window", "r", "status"]].to_string(index=False))

    lags, per, j = grace(df)
    pc = per[per.status == "core"]
    print(f"\nGRACE per core well: median r(lag 0) {pc.r_lag0.median():.2f}, "
          f"median best lag {pc.best_lag.median():.0f} months, median r(best) {pc.r_best.median():.2f}")
    print("Regional index vs GRACE:", lags.iloc[[0, 6, 12, 18, 24, 36]].to_string(index=False))
    print(f"best lag {int(lags.r.idxmax())}, r {lags.r.max():.2f}; "
          f"r with raw GRACE {np.corrcoef(j.z, j.grace_lwe_cm)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
