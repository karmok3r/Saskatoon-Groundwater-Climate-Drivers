"""Transfer-function noise (TFN) baseline with Pastas, scored like the ML models.

For each well a Pastas model is fitted: recharge R = P_liquid - f * E (ERA5-Land
rain plus snowmelt, minus f times actual evaporation), routed to the well by a
Gamma response function. Stresses are monthly totals spread evenly over the
days of the month. Two configurations are fitted:

- joint (the forecast benchmark, specified before any test scores were seen):
  response and AR(1) noise model fitted together, the standard Pastas set-up
  for forecasting; alpha is the fitted noise decay time
- deterministic: response fitted without a noise model, then an AR(1) decay
  time taken from the lag-1 autocorrelation of the monthly calibration
  residuals. At many wells the joint fit lets the noise model absorb the slow
  signal (explained variance near zero), so the response times of the
  deterministic fit are the ones used to characterise each well; its forecasts
  are reported as a sensitivity test ("Pastas TFN det").

Forecasts use the rolling-origin blocks of forecast_models.py: for each 5-year
test block the model is calibrated only on levels before the block. For an
origin t (mid-month) and target T = t + h months, with S the simulation from
observed stresses, b the block response and r(t) = y(t) - S(t):

  forecast:  dy = S(T) - S(t) - sum_{t<d<=T} a(d) b(T-d) - r(t) (1 - exp(-(T-t)/alpha))
  oracle:    dy = S(T) - S(t)                            - r(t) (1 - exp(-(T-t)/alpha))

where a(d) = R(d) - Rclim(d) is the recharge anomaly over the forecast window
(Rclim: mean daily recharge of the calendar month in the training period), so
the forecast setting uses average weather after t and the oracle the observed
weather. The noise term is the AR(1) decay of the current model error.

Also fits each well on its whole record and reports the response times
(t50, t90 of the step response) and explained variance.

Needs Pastas, which does not yet support pandas 3; run it in the separate
environment:  GW_SCOPE=province .venv_pastas/Scripts/python pastas_models.py
Writes results/<scope>/pastas_predictions.csv, pastas_skill.csv,
pastas_response.csv.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import pastas as ps

from common import FLAGGED, RES, load_drivers, load_levels, WELLS

warnings.filterwarnings("ignore")
logging.getLogger("pastas").setLevel(logging.ERROR)
ps.set_log_level("ERROR")

HORIZONS = [1, 3, 6, 12]
BLOCKS = [(1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009),
          (2010, 2014), (2015, 2019), (2020, 2026)]
MIN_CALIB_MONTHS = 120
N_BOOT = 1000
RNG = np.random.default_rng(42)


def daily_stresses(d):
    """Daily rain + snowmelt and evaporation (mm/day) from monthly totals."""
    d = d.sort_values("date")
    idx = pd.PeriodIndex(d.date, freq="M")
    days = idx.days_in_month.values
    liquid = (d.precip_mm - d.snowfall_mm).clip(lower=0) + d.snowmelt_mm.clip(lower=0)
    monthly = pd.DataFrame({"prec": liquid.values / days, "evap": d.evap_mm.clip(lower=0).values / days},
                           index=idx.to_timestamp())
    full = pd.date_range(monthly.index[0], idx[-1].to_timestamp(how="end").normalize(), freq="D")
    return monthly.reindex(full, method="ffill")


def head_series(g):
    g = g.dropna(subset=["level_m"])
    return pd.Series(g.level_m.values, index=pd.PeriodIndex(g.date, freq="M").to_timestamp() + pd.Timedelta(days=14),
                     name="head")


def residual_alpha(ml):
    """AR(1) decay time (days) of the monthly calibration residuals."""
    r = ml.residuals()
    m = pd.Series(r.values, index=r.index.to_period("M"))
    nxt = m.reindex(m.index + 1)
    ok = nxt.notna().values
    phi = np.corrcoef(m.values[ok], nxt.values[ok])[0, 1]
    phi = min(max(phi, 1e-3), 0.999)
    return -30.44 / np.log(phi)


def fit(head, st, tmax=None, noise=True):
    ml = ps.Model(head if tmax is None else head[head.index < tmax], name="m")
    ps.RechargeModel(ml, st.prec, st.evap, rfunc=ps.Gamma(), recharge=ps.rch.Linear(), name="rch")
    if noise:
        ps.ArNoiseModel(ml)
    ml.solve(report=False)
    ml.alpha_days = ml.parameters.optimal["noise_alpha"] if noise else residual_alpha(ml)
    return ml


def predict_block(ml, head, st, rows, train_end):
    """dy forecasts (forecast and oracle settings) for the given origins."""
    p = ml.parameters.optimal
    f = p["rch_f"]
    alpha = ml.alpha_days
    sim = ml.simulate(tmin=st.index[0] + pd.Timedelta(days=3650), tmax=st.index[-1])
    b = ml.get_block_response("rch")
    R = st.prec + f * st.evap  # Pastas' Linear recharge: R = P + f E, with f < 0
    train = R[R.index < train_end]
    clim = train.groupby(train.index.month).mean()
    anom = (R - clim.reindex(R.index.month).values).values
    pos = {d: i for i, d in enumerate(st.index)}
    out = []
    for date, h in rows:
        t = pd.Period(date, "M").to_timestamp() + pd.Timedelta(days=14)
        T = (pd.Period(date, "M") + h).to_timestamp() + pd.Timedelta(days=14)
        if t not in sim.index or T not in sim.index or t not in head.index:
            out.append((np.nan, np.nan))
            continue
        it, iT = pos[t], pos[T]
        lags = iT - np.arange(it + 1, iT + 1)
        bb = np.where(lags < len(b), b.values[np.clip(lags, 0, len(b) - 1)], 0.0)
        weather = np.sum(anom[it + 1:iT + 1] * bb)
        r = head[t] - sim[t]
        decay = np.exp(-(T - t).days / alpha)
        base = sim[T] - sim[t] - r * (1 - decay)
        out.append((base - weather, base))
    return out


def response_times(ml):
    step = ml.get_step_response("rch")
    s = step.values / step.values[-1]
    t = step.index.values
    return dict(t50_days=float(t[np.searchsorted(s, 0.5)]), t90_days=float(t[np.searchsorted(s, 0.9)]),
                evp=float(ml.stats.evp()), rsq=float(ml.stats.rsq()), f=float(ml.parameters.optimal["rch_f"]),
                n=float(ml.parameters.optimal["rch_n"]), a=float(ml.parameters.optimal["rch_a"]),
                a_at_bound=bool(ml.parameters.optimal["rch_a"] >= 0.99 * ml.parameters.pmax["rch_a"]),
                alpha=float(ml.alpha_days))


def skill(preds):
    rows = []
    for h, g0 in preds.groupby("horizon"):
        for subset, g in [("all wells", g0), ("core wells", g0[~g0.name.isin(FLAGGED)])]:
            years = g.date.str[:4].values
            uniq = np.unique(years)
            groups = {y: np.where(years == y)[0] for y in uniq}
            draws = [np.concatenate([groups[y] for y in RNG.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]
            e_p = (g.dy - g.Persistence).values ** 2
            for m in [c for c in g.columns if c not in ("horizon", "name", "date", "level_m", "dy", "block", "Persistence")]:
                e = (g.dy - g[m]).values ** 2
                boot = [1 - e[i].mean() / e_p[i].mean() for i in draws]
                rows.append(dict(horizon=h, wells=subset, model=m, n=len(g),
                                 skill_vs_persistence=1 - e.mean() / e_p.mean(),
                                 ss_lo=np.percentile(boot, 2.5), ss_hi=np.percentile(boot, 97.5)))
    return pd.DataFrame(rows)


def main():
    levels = load_levels()
    drivers = load_drivers()
    ml_preds = pd.read_csv(RES / "forecast_predictions.csv")
    resp, out = [], []
    for name, g in levels.groupby("name"):
        st = daily_stresses(drivers[drivers.driver_name == WELLS[name]])
        head = head_series(g)
        try:
            resp.append(dict(name=name, **response_times(fit(head, st, noise=False))))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: full-record fit failed ({e})")
        mine = ml_preds[ml_preds.name == name]
        for start, end in BLOCKS:
            blk = mine[mine.block == f"{start}-{end}"]
            train_end = pd.Timestamp(f"{start}-01-01")
            if blk.empty or (head.index < train_end).sum() < MIN_CALIB_MONTHS:
                continue
            b = blk[["horizon", "name", "date"]].copy()
            try:
                for label, noise in [("Pastas TFN", True), ("Pastas TFN det", False)]:
                    ml = fit(head, st, tmax=train_end, noise=noise)
                    res = predict_block(ml, head, st, list(zip(blk.date, blk.horizon)), train_end)
                    b[label], b[f"{label} (oracle)"] = zip(*res)
            except Exception as e:  # noqa: BLE001
                print(f"  {name} {start}: fit failed ({e})")
                continue
            out.append(b)
        print(f"{name}: done", flush=True)
    pd.DataFrame(resp).to_csv(RES / "pastas_response.csv", index=False, float_format="%.4f")
    p = pd.concat(out)
    p.to_csv(RES / "pastas_predictions.csv", index=False, float_format="%.4f")
    # score on the samples where Pastas has a prediction, next to the ML models
    j = ml_preds.merge(p, on=["horizon", "name", "date"]).dropna(subset=["Pastas TFN"])
    s = skill(j)
    s.to_csv(RES / "pastas_skill.csv", index=False, float_format="%.4f")
    print(s[s.wells == "core wells"].to_string(index=False))


if __name__ == "__main__":
    main()
