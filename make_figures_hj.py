"""Figs. 2-7 for the Hydrogeology Journal manuscript (Arial, journal widths).

Fig. 2  example hydrographs with the best-window P - E sum (both standardised)
Fig. 3  memory: correlation vs window, best windows (ERA5-Land, TerraClimate),
        Pastas response time vs memory window
Fig. 4  regional well index vs GRACE, lag correlation with a surrogate band
Fig. 5  forecast skill vs horizon (forecast and oracle settings)
Fig. 6  per-well NSE of forecast levels by horizon
Fig. 7  SHAP feature contributions (XGBoost) at 3 and 12 months

Reads results/province (run memory_analysis.py, forecast_models.py,
robustness.py and pastas_models.py first). Writes figures/hj/Fig0n.png
(600 dpi) and .pdf.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("GW_SCOPE", "province")
from common import (EXCLUDED, FLAGGED, RES, deseasonalised_anomaly, load_drivers,  # noqa: E402
                    merged, nse, window_correlations, WELLS)
from robustness import deseason, phase_surrogate  # noqa: E402

OUT = Path("figures/hj")
FULL, HALF = 17.6 / 2.54, 8.6 / 2.54
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# fixed categorical order (reference palette slots 1-5); markers as secondary encoding
MODELS = {"Ridge": ("#2a78d6", "o"), "Random Forest": ("#eb6834", "s"), "XGBoost": ("#1baf7a", "^"),
          "LSTM": ("#eda100", "D"), "Pastas TFN": ("#e87ba4", "v")}
SP = ("Seasonal persistence", "#8c8c8c")
EXAMPLES = ["WSA Smoky Burn A and B", "WSA Duck Lake No. 2", "WSA Melfort",
            "WSA Dalmeny", "WSA Shaunavon", "WSA Conquest No.502"]


def save(fig, n):
    fig.savefig(OUT / f"Fig0{n}.png", dpi=600)
    fig.savefig(OUT / f"Fig0{n}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"Fig0{n}.png")


def short(name):
    return name.replace("WSA ", "").replace(" A and B", "").replace("No.", "No. ").replace("  ", " ")


def pme_window_anomaly(dr, w):
    """Deseasonalised w-month rolling sum of P - E, indexed by 'YYYY-MM'."""
    s = dr.set_index("date").p_minus_e_mm.rolling(w).sum()
    m = s.index.str[5:7]
    return s - s.groupby(m).transform("mean")


def z(s):
    return (s - s.mean()) / s.std()


def fig2(df, drivers, mem):
    fig, axes = plt.subplots(3, 2, figsize=(FULL, 15 / 2.54), sharex=True)
    for ax, name in zip(axes.flat, EXAMPLES):
        g = df[(df.name == name) & (df.date >= "1980-01")].set_index("date")
        row = mem[mem.name == name].iloc[0]
        dr = drivers[drivers.driver_name == WELLS[name]].sort_values("date")
        x = pme_window_anomaly(dr, int(row.window)).reindex(g.index)
        t = pd.to_datetime(g.index)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.plot(t, z(x), color=ORANGE, lw=1.0, label=f"P − E, preceding {int(row.window)} months")
        ax.plot(t, z(g.anom), color=BLUE, lw=1.0, label="Water level anomaly")
        ax.set_title(f"{short(name)}  (window {int(row.window)} months, r = {row.r:.2f})", fontsize=8, loc="left")
        ax.set_ylabel("Standardised")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, ["P − E summed over the best window (deseasonalised)", "Water level anomaly (deseasonalised)"],
               loc="upper center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, 2)


def fig3(df, drivers, mem, rob, resp):
    core = [n for n in mem[mem.status == "core"].name]
    curves = []
    for name in core:
        g = df[(df.name == name) & (df.date >= "1980-01") & df.anom.notna()].sort_values("date")
        dr = drivers[drivers.driver_name == WELLS[name]].sort_values("date")
        r, _ = window_correlations(dr.p_minus_e_mm.values, dr.date.tolist(), g.anom.values,
                                   g.date.tolist(), range(1, 241))
        curves.append(r)
    curves = np.array(curves)
    w = np.arange(1, 241) / 12

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 6.2 / 2.54))
    ax = axes[0]
    for c in curves:
        ax.plot(w, c, color="#c9c8c3", lw=0.5)
    ax.plot(w, np.median(curves, 0), color=BLUE, lw=1.6, label="Median of core wells")
    ax.set_xlabel("Window length (years)")
    ax.set_ylabel("Correlation with level anomaly")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[1]
    c = rob[rob.flag.isna()]
    bins = np.arange(0, 21, 2)
    ax.hist([c.window / 12, c.window_tc / 12], bins=bins, color=[BLUE, ORANGE], rwidth=0.85,
            label=["ERA5-Land P − E", "TerraClimate P − AET"])
    ax.set_xlabel("Best window (years)")
    ax.set_ylabel("Number of core wells")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[2]
    x = resp.merge(c, on="name")
    ax.scatter(x.window_tc / 12, x.t50_days / 365.25, s=16, color=BLUE, edgecolor="white", lw=0.5,
               label="TerraClimate window", zorder=3)
    ax.scatter(x.window / 12, x.t50_days / 365.25, s=16, marker="s", facecolor="none", edgecolor=ORANGE,
               lw=0.8, label="ERA5-Land window", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Best window (years)")
    ax.set_ylabel("Pastas t50 (years)")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.6, 0.17), frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_title("c", loc="left", fontweight="bold")
    for a in axes:
        a.grid(color=GRID, lw=0.4)
        a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 3)


def fig4(df):
    d = df[~df.name.isin(EXCLUDED) & ~df.name.isin(FLAGGED)].copy()
    d["z"] = d.groupby("name").anom.transform(z)
    idx = d.groupby("date").agg(z=("z", "mean"), n=("z", "count"))
    idx = idx[idx.n >= 25].z
    gr = (d.dropna(subset=["grace_lwe_cm"]).drop_duplicates(["driver_name", "date"])
            .groupby("date").grace_lwe_cm.mean())
    full = pd.period_range(gr.index.min(), gr.index.max(), freq="M").astype(str)
    g = pd.Series(deseason(gr.values, gr.index), index=gr.index).reindex(full)
    zi = idx.reindex(full)

    def lag_rs(gs):
        out = []
        for lag in range(37):
            a, b = zi.shift(-lag), gs.where(g.notna())
            ok = a.notna() & b.notna()
            out.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return np.array(out)

    obs = lag_rs(g)
    rng_null = np.array([lag_rs(pd.Series(phase_surrogate(g.interpolate(limit_direction="both").values),
                                          index=full)) for _ in range(500)])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 6 / 2.54), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    t = pd.to_datetime(full)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.plot(t, z(zi), color=BLUE, lw=1.1, label="Regional well index (core wells)")
    ax.plot(t, z(g), color=ORANGE, lw=1.0, label="GRACE/GRACE-FO storage anomaly")
    ax.set_ylabel("Standardised")
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.04, 1.0), ncol=2)
    ax.set_title("a", loc="left", fontweight="bold")
    ax = axes[1]
    lags = np.arange(37)
    ax.fill_between(lags, np.percentile(rng_null, 2.5, 0), np.percentile(rng_null, 97.5, 0),
                    color="#d4d3ce", label="Central 95% of surrogates")
    ax.plot(lags, obs, color=INK, lw=1.2, label="Observed")
    ax.set_xlabel("Lag of wells behind GRACE (months)")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.08, 1.0), ncol=1)
    ax.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, 4)


def fig5(skill):
    c = skill[skill.wells == "core wells"]
    hs = [1, 3, 6, 12]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 7 / 2.54), sharey=True)
    for ax, oracle, title in [(axes[0], False, "a  Forecast (information up to month t)"),
                              (axes[1], True, "b  Oracle (observed weather of the forecast window)")]:
        ax.axhline(0, color=INK, lw=0.8)
        sp = c[c.model == SP[0]].set_index("horizon").reindex(hs)
        ax.plot(range(4), sp.skill_vs_persistence, color=SP[1], ls="--", lw=1.2, marker="o", ms=4,
                label="Seasonal persistence")
        off = np.linspace(-0.16, 0.16, len(MODELS))
        for k, (m, (col, mk)) in enumerate(MODELS.items()):
            name = f"{m} (oracle)" if oracle else m
            s = c[c.model == name].set_index("horizon").reindex(hs)
            if s.skill_vs_persistence.isna().all():
                continue
            x = np.arange(4) + off[k]
            ax.errorbar(x, s.skill_vs_persistence, yerr=[s.skill_vs_persistence - s.ss_lo, s.ss_hi - s.skill_vs_persistence],
                        color=col, marker=mk, ms=4, lw=1.0, capsize=0, elinewidth=0.8, label=m)
        ax.set_xticks(range(4), [f"{h}" for h in hs])
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Skill score vs persistence")
    axes[0].legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, 5)


def fig6(preds):
    p = preds[~preds.name.isin(FLAGGED)]
    ms = ["Persistence", "Seasonal persistence", "Ridge", "Random Forest", "XGBoost", "LSTM", "Pastas TFN"]
    cols = {"Persistence": "#4d4d4d", "Seasonal persistence": SP[1], **{m: v[0] for m, v in MODELS.items()}}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 6.5 / 2.54), sharey=False)
    for ax, (h, g) in zip(axes, p.groupby("horizon")):
        vals = []
        for m in ms:
            vals.append([nse(d.level_m + d.dy, d.level_m + d[m]) for _, d in g.groupby("name")])
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for patch, m in zip(bp["boxes"], ms):
            patch.set(facecolor=cols[m], edgecolor=INK, lw=0.5, alpha=0.85)
        ax.set_xticks(range(1, len(ms) + 1), ["P", "SP", "Ridge", "RF", "XGB", "LSTM", "TFN"], rotation=90)
        ax.set_title(f"{h} month{'s' if h > 1 else ''}", fontsize=8)
        ax.axhline(np.median(vals[0]), color="#4d4d4d", lw=0.6, ls=":")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        lo = np.percentile(np.concatenate(vals), 5)
        ax.set_ylim(max(lo - 0.05, -0.5), 1.0)
    axes[0].set_ylabel("NSE of forecast levels (per well)")
    fig.tight_layout()
    save(fig, 6)


def fig7(shap):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 7 / 2.54))
    for ax, h in zip(axes, [3, 12]):
        s = shap[shap.horizon == h].nlargest(10, "mean_abs_shap_m").iloc[::-1]
        ax.barh(s.label, s.mean_abs_shap_m * 100, color=BLUE, height=0.65)
        ax.set_xlabel("Mean |SHAP| (cm)")
        ax.set_title(f"{'a' if h == 3 else 'b'}  {h}-month horizon", loc="left", fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = merged(include_excluded=True)
    df["anom"] = deseasonalised_anomaly(df)
    drivers = load_drivers()
    mem = pd.read_csv(RES / "memory_windows.csv")
    rob = pd.read_csv(RES / "robust_memory.csv")
    resp = pd.read_csv(RES / "pastas_response.csv")
    skill = pd.read_csv(RES / "pastas_skill.csv")
    preds = (pd.read_csv(RES / "forecast_predictions.csv")
             .merge(pd.read_csv(RES / "pastas_predictions.csv"), on=["horizon", "name", "date"])
             .dropna(subset=["Pastas TFN"]))
    fig2(df, drivers, mem)
    fig3(df, drivers, mem, rob, resp)
    fig4(df)
    fig5(skill)
    fig6(preds)
    fig7(pd.read_csv(RES / "shap_importance.csv"))


if __name__ == "__main__":
    main()
