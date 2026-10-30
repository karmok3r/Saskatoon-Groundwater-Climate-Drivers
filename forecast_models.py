"""Forecast experiment: baselines first, then Ridge, Random Forest, XGBoost, LSTM.

Target: the change in water level over the next h months, dy = y(t+h) - y(t),
for h = 1, 3, 6, 12. The forecast level is y(t) + predicted dy, so every
model is judged on what it adds to the current level.

Two information settings:
- forecast: only information available at month t (past levels and past or
  current climate)
- oracle: also the observed precipitation, snowmelt and P - E between t and
  t+h. This is an upper bound on what a perfect weather forecast would add.

Validation is rolling-origin: for each 5-year test block (1990-94 ... 2020-26)
models are trained only on samples whose target month is before the block.
Every model is scored on the same samples. Skill is reported against
persistence (SS = 1 - MSE_model / MSE_persistence), with a 95% interval from
resampling calendar years (1000 bootstrap draws).

Writes results/<scope>/forecast_*.csv and figures/<scope>/skill_vs_horizon.png,
skill_by_well.png and shap.png. Run with GW_SCOPE=province for all wells.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FIG, FLAGGED, RES, SHORT, merged, nse

warnings.filterwarnings("ignore")
HORIZONS = [1, 3, 6, 12]
BLOCKS = [(1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009),
          (2010, 2014), (2015, 2019), (2020, 2026)]
SEQ_LEN = 24
LSTM_SEEDS = [0, 1, 2]
N_BOOT = 1000
RNG = np.random.default_rng(42)

TAB_FEATURES = [
    "d1", "d3", "d6", "d12", "dev120", "month_sin", "month_cos",
    "precip_3", "precip_6", "precip_12", "snowmelt_3", "swe_mm",
    "pme_12", "pme_24", "pme_36", "pme_60", "pme_120",
    "soil_28_100cm", "soil_100_289cm", "temp_3",
]
ORACLE_FEATURES = ["precip_fut", "snowmelt_fut", "pme_fut"]
SEQ_FEATURES = ["d1", "dev120", "precip_mm", "snowmelt_mm", "swe_mm", "evap_mm",
                "soil_100_289cm", "temp_c", "month_sin", "month_cos"]
LABELS = {
    "d1": "Level change, last month", "d3": "Level change, last 3 months",
    "d6": "Level change, last 6 months", "d12": "Level change, last 12 months",
    "dev120": "Level vs its 10-year mean", "month_sin": "Month (sin)", "month_cos": "Month (cos)",
    "precip_3": "Precipitation, last 3 months", "precip_6": "Precipitation, last 6 months",
    "precip_12": "Precipitation, last 12 months", "snowmelt_3": "Snowmelt, last 3 months",
    "swe_mm": "Snowpack (SWE) now", "pme_12": "P − E, last 12 months",
    "pme_24": "P − E, last 24 months", "pme_36": "P − E, last 36 months",
    "pme_60": "P − E, last 60 months", "pme_120": "P − E, last 120 months",
    "soil_28_100cm": "Soil moisture 28–100 cm", "soil_100_289cm": "Soil moisture 100–289 cm",
    "temp_3": "Temperature, last 3 months", "precip_fut": "Precipitation, forecast window",
    "snowmelt_fut": "Snowmelt, forecast window", "pme_fut": "P − E, forecast window",
}


# ============================ FEATURES ======================================

def well_features(g, h):
    g = g.sort_values("date").reset_index(drop=True).copy()
    y = g.level_m
    for k in (1, 3, 6, 12):
        g[f"d{k}"] = y - y.shift(k)
    g["dev120"] = y - y.rolling(120, min_periods=60).mean()
    m = g.date.str[5:7].astype(int)
    g["month"] = m
    g["month_sin"], g["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    for k in (3, 6, 12):
        g[f"precip_{k}"] = g.precip_mm.rolling(k).sum()
    g["snowmelt_3"] = g.snowmelt_mm.rolling(3).sum()
    for k in (12, 24, 36, 60, 120):
        g[f"pme_{k}"] = g.p_minus_e_mm.rolling(k).sum()
    g["temp_3"] = g.temp_c.rolling(3).mean()
    fut = lambda s: s[::-1].rolling(h).sum()[::-1].shift(-1)  # sum over t+1..t+h
    g["precip_fut"] = fut(g.precip_mm)
    g["snowmelt_fut"] = fut(g.snowmelt_mm)
    g["pme_fut"] = fut(g.p_minus_e_mm)
    g["dy"] = y.shift(-h) - y
    g["target_month"] = (m - 1 + h) % 12 + 1
    g["t"] = pd.PeriodIndex(g.date, freq="M")
    g["seq_ok"] = y.notna().rolling(SEQ_LEN + 1).sum() == SEQ_LEN + 1
    return g


def build(h):
    df = merged()
    parts = [well_features(g, h) for _, g in df.groupby("name")]
    df = pd.concat(parts, ignore_index=True)
    ok = df[TAB_FEATURES + ORACLE_FEATURES + ["dy"]].notna().all(axis=1) & df.seq_ok
    return df, ok


def sequences(df, rows):
    """(n, SEQ_LEN, n_features) arrays of the months leading up to each sample."""
    by_well = {n: g.reset_index() for n, g in df.groupby("name")}
    pos = {n: dict(zip(g["index"], range(len(g)))) for n, g in by_well.items()}
    arrs = {n: g[SEQ_FEATURES].to_numpy(float) for n, g in by_well.items()}
    out = np.empty((len(rows), SEQ_LEN, len(SEQ_FEATURES)))
    for i, (idx, name) in enumerate(zip(rows.index, rows.name)):
        p = pos[name][idx]
        out[i] = arrs[name][p - SEQ_LEN + 1:p + 1]
    return out


# ============================ MODELS ========================================

class LSTMNet(torch.nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit_predict_lstm(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-9
    ys = ytr.std() + 1e-9
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr / ys, dtype=torch.float32)
    net = LSTMNet(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(xt)
    for _ in range(40):
        net.train()
        perm = torch.randperm(n)
        for b in range(0, n, 128):
            i = perm[b:b + 128]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xt[i]), yt[i])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy() * ys


def tab_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                               max_features=0.5, n_jobs=-1, random_state=0),
        "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, random_state=0),
    }


# ============================ EXPERIMENT ====================================

def run_horizon(h):
    df, ok = build(h)
    data = df[ok]
    preds = []
    for start, end in BLOCKS:
        test = data[(data.t.dt.year >= start) & (data.t.dt.year <= end)]
        train = data[(data.t + h).dt.year < start]
        if test.empty:
            continue
        p = test[["name", "date", "level_m", "dy"]].copy()
        p["block"] = f"{start}-{end}"
        p["Persistence"] = 0.0
        # seasonal persistence: training-period monthly climatology of each well
        clim = (df[df.t.dt.year < start].groupby(["name", "month"]).level_m.mean())
        a = clim.reindex(list(zip(test.name, test.target_month))).values
        b = clim.reindex(list(zip(test.name, test.month))).values
        p["Seasonal persistence"] = np.nan_to_num(a - b)
        for setting, feats in [("forecast", TAB_FEATURES), ("oracle", TAB_FEATURES + ORACLE_FEATURES)]:
            for name, model in tab_models().items():
                model.fit(train[feats], train.dy)
                key = name if setting == "forecast" else f"{name} (oracle)"
                p[key] = model.predict(test[feats])
        Xtr, Xte = sequences(df, train), sequences(df, test)
        p["LSTM"] = np.mean([fit_predict_lstm(Xtr, train.dy.values, Xte, s) for s in LSTM_SEEDS], axis=0)
        preds.append(p)
        print(f"  h={h} {start}-{end}: train {len(train)}, test {len(test)}")
    out = pd.concat(preds)
    out.insert(0, "horizon", h)
    return out


def skill_table(preds, models):
    rows = []
    for (h, subset), g in [((h, "all wells"), g) for h, g in preds.groupby("horizon")] + \
                          [((h, "core wells"), g[~g.name.isin(FLAGGED)]) for h, g in preds.groupby("horizon")]:
        obs_level = g.level_m + g.dy
        years = g.date.str[:4].values
        uniq = np.unique(years)
        groups = {y: np.where(years == y)[0] for y in uniq}
        draws = [np.concatenate([groups[y] for y in RNG.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]
        e_p = (g.dy - g.Persistence).values ** 2
        for m in models:
            e = (g.dy - g[m]).values ** 2
            ss = 1 - e.mean() / e_p.mean()
            boot = [1 - e[i].mean() / e_p[i].mean() for i in draws]
            rows.append(dict(horizon=h, wells=subset, model=m,
                             rmse_m=np.sqrt(e.mean()), mae_m=np.abs(g.dy - g[m]).mean(),
                             nse_level=nse(obs_level, g.level_m + g[m]),
                             skill_vs_persistence=ss,
                             ss_lo=np.percentile(boot, 2.5), ss_hi=np.percentile(boot, 97.5),
                             n=len(g)))
    return pd.DataFrame(rows)


def per_well_skill(preds, models):
    rows = []
    for (h, name), g in preds.groupby(["horizon", "name"]):
        e_p = ((g.dy - g.Persistence) ** 2).mean()
        for m in models:
            rows.append(dict(horizon=h, name=name, model=m,
                             skill_vs_persistence=1 - ((g.dy - g[m]) ** 2).mean() / e_p,
                             rmse_m=np.sqrt(((g.dy - g[m]) ** 2).mean()), n=len(g)))
    return pd.DataFrame(rows)


# ============================ FIGURES =======================================

def skill_figure(tab):
    t = tab[tab.wells == "core wells"]
    styles = {"Seasonal persistence": ("#999999", "--"), "Ridge": ("#8064a2", "-"),
              "Random Forest": ("#4f9d69", "-"), "XGBoost": ("#2f5d8a", "-"),
              "LSTM": ("#e8a33d", "-"), "XGBoost (oracle)": ("#2f5d8a", ":")}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, (c, ls) in styles.items():
        d = t[t.model == m].sort_values("horizon")
        ax.errorbar(d.horizon, d.skill_vs_persistence,
                    yerr=[d.skill_vs_persistence - d.ss_lo, d.ss_hi - d.skill_vs_persistence],
                    color=c, ls=ls, marker="o", ms=4, capsize=2, lw=1.3, label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Skill vs persistence (1 − MSE/MSE$_{pers}$)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "skill_vs_horizon.png", dpi=200)


def well_figure(pw, model):
    d = pw[pw.model == model].pivot(index="name", columns="horizon", values="skill_vs_persistence")
    d = d.loc[d[12].sort_values().index]
    fig, ax = plt.subplots(figsize=(5, max(5, 0.22 * len(d) + 1)))
    im = ax.imshow(d.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(d.columns)), [f"{h} mo" for h in d.columns])
    ax.set_yticks(range(len(d)), [SHORT[n] + (" *" if n in FLAGGED else "") for n in d.index])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Skill vs persistence")
    ax.set_title(f"{model} skill by well (* flagged)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "skill_by_well.png", dpi=200)


def shap_analysis():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, h in zip(axes, [3, 12]):
        df, ok = build(h)
        data = df[ok]
        model = tab_models()["XGBoost"].fit(data[TAB_FEATURES], data.dy)
        X = data[TAB_FEATURES].sample(min(3000, len(data)), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=TAB_FEATURES).sort_values(ascending=False)
        for f, v in imp.items():
            rows.append(dict(horizon=h, feature=f, label=LABELS[f], mean_abs_shap_m=v))
        plt.sca(ax)
        shap.summary_plot(sv, X.rename(columns=LABELS), max_display=10, show=False,
                          plot_size=None, color_bar=(h == 12))
        ax.set_title(f"XGBoost, {h}-month horizon", fontsize=10)
        ax.set_xlabel("SHAP value (m change in level)")
    fig.tight_layout()
    fig.savefig(FIG / "shap.png", dpi=200)
    pd.DataFrame(rows).to_csv(RES / "shap_importance.csv", index=False, float_format="%.4f")


def main():
    preds = pd.concat([run_horizon(h) for h in HORIZONS])
    preds.to_csv(RES / "forecast_predictions.csv", index=False, float_format="%.4f")
    models = ["Persistence", "Seasonal persistence", "Ridge", "Random Forest", "XGBoost", "LSTM",
              "Ridge (oracle)", "Random Forest (oracle)", "XGBoost (oracle)"]
    tab = skill_table(preds, models)
    tab.to_csv(RES / "forecast_skill.csv", index=False, float_format="%.4f")
    pw = per_well_skill(preds, models)
    pw.to_csv(RES / "forecast_skill_by_well.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(tab[tab.wells == "core wells"].round(3).to_string(index=False))
    print(tab[(tab.wells == "all wells") & tab.model.isin(["Persistence", "XGBoost", "LSTM"])].round(3).to_string(index=False))
    skill_figure(tab)
    best = (tab[(tab.wells == "core wells") & ~tab.model.str.contains("oracle|ersistence")]
            .groupby("model").skill_vs_persistence.mean().idxmax())
    well_figure(pw, best)
    print("Per-well figure model:", best)
    shap_analysis()


if __name__ == "__main__":
    main()
