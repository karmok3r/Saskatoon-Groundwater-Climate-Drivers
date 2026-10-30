"""Robustness checks requested in review.

1. Memory significance: each well's best-window correlation is compared with
   the best-window correlations from 500 phase-randomised surrogates of the
   monthly P - E series (Ebisuzaki 1997). Taking the maximum over all
   windows for each surrogate allows for autocorrelation and for the
   selection of the best window.
2. Memory sensitivity: (a) TerraClimate P - AET instead of ERA5-Land;
   (b) windows up to 360 months on months from 1990 on; (c) both series
   linearly detrended, to check that long windows do not just match trends.
3. GRACE significance: lag-0 and best-lag (0-36 months) correlation of the
   regional well index with GRACE against 500 phase-randomised surrogates
   of the deseasonalised GRACE series.
4. Forecast robustness: XGBoost tuned on an inner time split (last 5 years of
   each training set), and well-specific Ridge models instead of one pooled
   model.
5. Saskatoon scope only: the recovered levels compared with statements in
   WSA's well descriptions.

Writes results/<scope>/robust_*.csv. Run with GW_SCOPE=province for all wells.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (DRIVERS, EXCLUDED, FLAGGED, RES, SCOPE, WELLS, deseasonalised_anomaly,
                    merged, window_correlations)
from forecast_models import BLOCKS, HORIZONS, TAB_FEATURES, build

N_SURR = 500
WINDOWS = np.arange(1, 241)
LONG_WINDOWS = np.arange(1, 361)
RNG = np.random.default_rng(7)


def deseason(values, dates):
    s = pd.Series(np.asarray(values, dtype=float))
    month = pd.Series(dates).str[5:7].astype(int).values
    return (s - s.groupby(month).transform("mean")).values


def phase_surrogate(x):
    """Random-phase surrogate with the same power spectrum (Ebisuzaki 1997)."""
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n) + x.mean()


def detrend(y):
    t = np.arange(len(y))
    ok = ~np.isnan(y)
    b = np.polyfit(t[ok], y[ok], 1)
    return y - np.polyval(b, t)


def memory_tests(df, drivers):
    rows = []
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    for name, g in df.groupby("name"):
        g = g[g.anom.notna()].sort_values("date")
        g80, g90 = g[g.date >= "1980-01"], g[g.date >= "1990-01"]
        drv = drivers[drivers.name == WELLS[name]].sort_values("date")
        dates = drv.date.tolist()
        pme = drv.p_minus_e_mm.to_numpy(float)

        r, _ = window_correlations(pme, dates, g80.anom.values, g80.date.tolist(), WINDOWS)
        obs, w_obs = r.max(), int(WINDOWS[r.argmax()])
        null = np.array([window_correlations(phase_surrogate(pme), dates, g80.anom.values,
                                             g80.date.tolist(), WINDOWS)[0].max() for _ in range(N_SURR)])

        r360, _ = window_correlations(pme, dates, g90.anom.values, g90.date.tolist(), LONG_WINDOWS)

        tc = drv.dropna(subset=["tc_pr", "tc_aet"])
        gt = g80[g80.date <= tc.date.max()]
        rtc, _ = window_correlations((tc.tc_pr - tc.tc_aet).to_numpy(float), tc.date.tolist(),
                                     gt.anom.values, gt.date.tolist(), WINDOWS)

        # detrended: level anomaly detrended over its record; each rolling sum
        # detrended inside window_correlations is not possible, so detrend P - E
        # cumulative departures instead by removing the linear trend of the
        # anomaly and comparing against detrended rolling sums
        ad = detrend(g80.anom.values)
        rd = []
        for w in WINDOWS:
            s = pd.Series(pme, index=dates).rolling(w).sum()
            s = pd.Series(deseason(s.values, s.index), index=s.index).reindex(g80.date).to_numpy()
            ok = ~np.isnan(s)
            sd = np.full_like(s, np.nan)
            sd[ok] = detrend(s[ok])
            rd.append(np.corrcoef(sd[ok], ad[ok])[0, 1])
        rd = np.array(rd)

        rows.append(dict(name=name, flag=FLAGGED.get(name, ""), window=w_obs, r=obs,
                         p_surrogate=(1 + np.sum(null >= obs)) / (N_SURR + 1),
                         null_95=np.percentile(null, 95),
                         window_360=int(LONG_WINDOWS[r360.argmax()]), r_360=r360.max(),
                         window_tc=int(WINDOWS[rtc.argmax()]), r_tc=rtc.max(),
                         window_detrended=int(WINDOWS[rd.argmax()]), r_detrended=rd.max()))
        x = rows[-1]
        print(f"  {name}: w={w_obs} r={obs:.2f} p={x['p_surrogate']:.3f} | 360: w={x['window_360']} "
              f"| TC: w={x['window_tc']} | detrended: w={x['window_detrended']} r={x['r_detrended']:.2f}",
              flush=True)
    return pd.DataFrame(rows)


def grace_tests(df):
    df = df[~df.name.isin(EXCLUDED)].copy()
    df["anom"] = deseasonalised_anomaly(df)
    df["z"] = df.groupby("name").anom.transform(lambda s: (s - s.mean()) / s.std())
    idx = df.groupby("date").agg(z=("z", "mean"), n=("z", "count"))
    idx = idx[idx.n >= (8 if SCOPE == "saskatoon" else 25)].z
    gr = (df.dropna(subset=["grace_lwe_cm"]).drop_duplicates(["driver_name", "date"])
            .groupby("date").grace_lwe_cm.mean())
    full = pd.period_range(gr.index.min(), gr.index.max(), freq="M").astype(str)
    g = pd.Series(deseason(gr.values, gr.index), index=gr.index).reindex(full)
    g_filled = g.interpolate(limit_direction="both")  # surrogates need a gap-free series
    z = idx.reindex(full)

    def lag_rs(gs):
        out = []
        for lag in range(0, 37):
            a, b = z.shift(-lag), gs.where(g.notna())
            ok = a.notna() & b.notna()
            out.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return np.array(out)

    obs = lag_rs(g)
    null = np.array([lag_rs(pd.Series(phase_surrogate(g_filled.values), index=full)) for _ in range(N_SURR)])
    return pd.DataFrame([dict(
        r_lag0=obs[0], p_lag0=(1 + np.sum(null[:, 0] >= obs[0])) / (N_SURR + 1),
        r_best=obs.max(), best_lag=int(obs.argmax()),
        p_best=(1 + np.sum(null.max(axis=1) >= obs.max())) / (N_SURR + 1))])


XGB_GRID = [dict(max_depth=d, n_estimators=n, learning_rate=lr, min_child_weight=mcw)
            for d in (2, 4, 6) for n, lr in ((300, 0.03), (300, 0.1)) for mcw in (1, 10)]


def tuned_xgb(train, feats):
    """Chooses hyperparameters on the last 5 years of the training data."""
    cut = train.t.dt.year.max() - 4
    inner_tr, inner_va = train[train.t.dt.year < cut], train[train.t.dt.year >= cut]
    best, best_mse = None, np.inf
    for p in XGB_GRID:
        m = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=0, **p)
        m.fit(inner_tr[feats], inner_tr.dy)
        mse = np.mean((inner_va.dy - m.predict(inner_va[feats])) ** 2)
        if mse < best_mse:
            best, best_mse = p, mse
    model = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=0, **best)
    return model.fit(train[feats], train.dy), best


def forecast_robustness():
    rows, chosen = [], []
    for h in HORIZONS:
        df, ok = build(h)
        data = df[ok]
        preds = []
        for start, end in BLOCKS:
            test = data[(data.t.dt.year >= start) & (data.t.dt.year <= end)]
            train = data[(data.t + h).dt.year < start]
            p = test[["name", "dy"]].copy()
            model, params = tuned_xgb(train, TAB_FEATURES)
            p["XGBoost (tuned)"] = model.predict(test[TAB_FEATURES])
            chosen.append(dict(horizon=h, block=f"{start}-{end}", **params))
            local = np.zeros(len(test))
            for name in test.name.unique():
                tr, te = train[train.name == name], (test.name == name).values
                if len(tr) < 60:  # too little history for a well-specific model: persistence
                    continue
                m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[TAB_FEATURES], tr.dy)
                local[te] = m.predict(test.loc[te, TAB_FEATURES])
            p["Ridge (per well)"] = local
            preds.append(p)
        pr = pd.concat(preds)
        for subset, d in [("core wells", pr[~pr.name.isin(FLAGGED)]), ("all wells", pr)]:
            e_p = (d.dy ** 2).mean()
            for m in ["XGBoost (tuned)", "Ridge (per well)"]:
                rows.append(dict(horizon=h, wells=subset, model=m, n=len(d),
                                 skill_vs_persistence=1 - ((d.dy - d[m]) ** 2).mean() / e_p))
        print("  forecast h=%d: " % h + ", ".join(f"{r['model']} ({r['wells']}) {r['skill_vs_persistence']:.2f}"
                                                  for r in rows[-4:]), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(chosen)


def wsa_text_checks(df):
    """Compares recovered levels with statements in WSA's well descriptions."""
    out = []
    d = df[df.name == "WSA Dalmeny"].dropna(subset=["level_m"]).sort_values("date").copy()
    # mean seasonal cycle after removing multi-year variation (13-month centred mean)
    d["resid"] = d.level_m - d.level_m.rolling(13, center=True, min_periods=13).mean()
    cycle = d.groupby(d.date.str[5:].astype(int)).resid.mean()
    out.append(dict(check="Dalmeny: seasonal minimum usually April-May; maximum August-December (WSA)",
                    result=f"mean seasonal cycle: minimum in month {cycle.idxmin()}, maximum in month "
                           f"{cycle.idxmax()} (amplitude {cycle.max() - cycle.min():.2f} m)"))
    s = df[df.name == "WSA Saskatoon"].set_index("date").level_m
    rise = s.loc["2010-01":"2010-12"].max() - s.loc["2009-07":"2010-03"].min()
    pre = s.loc["1975-01":"2004-12"]
    out.append(dict(check="Saskatoon: level rose 3.5 m in 2010 (WSA)",
                    result=f"recovered rise from the 2009-10 low to the 2010 high: {rise:.2f} m"))
    out.append(dict(check="Saskatoon: fluctuated within about 1 m for 30 years before 2005 (WSA)",
                    result=f"recovered range 1975-2004: {pre.max() - pre.min():.2f} m"))
    return pd.DataFrame(out)


def main():
    df = merged(include_excluded=True)
    drivers = pd.read_csv(DRIVERS)

    if SCOPE == "saskatoon":
        checks = wsa_text_checks(df)
        checks.to_csv(RES / "robust_wsa_text_checks.csv", index=False)
        print(checks.to_string(index=False), flush=True)

    g = grace_tests(df)
    g.to_csv(RES / "robust_grace.csv", index=False, float_format="%.4f")
    print(g.to_string(index=False), flush=True)

    m = memory_tests(df, drivers)
    m.to_csv(RES / "robust_memory.csv", index=False, float_format="%.4f")

    f, chosen = forecast_robustness()
    f.to_csv(RES / "robust_forecast.csv", index=False, float_format="%.4f")
    chosen.to_csv(RES / "robust_xgb_params.csv", index=False)
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
