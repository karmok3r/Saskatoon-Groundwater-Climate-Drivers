"""Shared well lists, screening and loaders for the analysis scripts.

Two scopes, chosen with the GW_SCOPE environment variable:
- "saskatoon" (default): the 14 wells within 120 km of Saskatoon
- "province": every WSA well with a recovered hydrograph of at least 20 years

Outputs go to results/<scope>/ and figures/<scope>/.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCOPE = os.environ.get("GW_SCOPE", "saskatoon")
MIN_RECORD_MONTHS = 240

if SCOPE == "saskatoon":
    # Saskatoon-region wells with an extractable hydrograph, mapped to the
    # location used for the climate drivers (Duck Lake No. 1 and No. 2 share
    # one). WSA Warman 2 and Crawford Lake have no hydrograph on wsask.ca.
    WELLS = {
        "WSA Saskatoon": "WSA Saskatoon",
        "WSA Goodale Farm 009": "WSA Goodale Farm 009",
        "WSA Dalmeny": "WSA Dalmeny",
        "WSA Vanscoy": "WSA Vanscoy",
        "WSA Agrium 43": "WSA Agrium 43",
        "WSA Blucher No. 3 and No. 4": "WSA Blucher No. 3 and No. 4",
        "WSA Hague": "WSA Hague",
        "WSA Swanson": "WSA Swanson",
        "WSA Tessier": "WSA Tessier",
        "WSA Conquest No.500": "WSA Conquest No. 500",
        "WSA Conquest No.501": "WSA Conquest No. 501",
        "WSA Bruno": "WSA Bruno",
        "WSA Duck Lake No. 1": "WSA Duck Lake",
        "WSA Duck Lake No. 2": "WSA Duck Lake",
    }
    DRIVERS = Path("data/drivers_monthly.csv")
    LOCATIONS = Path("data/gee/wells_gee.csv")
else:
    _w = pd.read_csv("wells_province.csv")
    _n = pd.read_csv("data/wsa_levels_monthly.csv").groupby("name").date.nunique()
    _w = _w[_w.name.map(_n).fillna(0) >= MIN_RECORD_MONTHS]
    WELLS = dict(zip(_w.name, _w.location))
    DRIVERS = Path("data/drivers_monthly_province.csv")
    LOCATIONS = Path("data/gee_province/wells_gee.csv")

# Wells with an abrupt drawdown of several metres and slow recovery, typical
# of pumping or dewatering: left out of model training and pooled scores.
EXCLUDED = [w for w in [
    "WSA Vanscoy",                  # ~14 m drawdown 2005-2008, recovery to 2020
    "WSA Estevan No. 1 and No. 2",  # ~25 m drawdown around 1988, slow recovery
    "WSA Outram",                   # ~35 m drawdown around 1990, slow recovery
    "WSA Instow",                   # ~8 m decline 1985-1990
] if w in WELLS]

# Behaviour that climate is unlikely to explain on its own: kept in the "all
# wells" scores but left out of the "core wells" scores.
FLAGGED = {k: v for k, v in {
    "WSA Hague": "abrupt ~1 m drop around 1991",
    "WSA Conquest No.500": "steady 3.5 m rise since 2010 not seen at other wells",
    "WSA Saskatoon": "step rise in 2010 and large month-to-month swings in 2011-2016",
    "WSA Conquest No.504": "~3 m decline in the 1980s and partial recovery",
    "WSA Coronach 1": "~3 m decline 1975-1995 and recovery",
    "WSA Riceton": "monotonic rise over the whole record",
    "WSA Verlo": "monotonic decline over the whole record",
    "WSA Regina 530": "sustained rise since 1990",
    "WSA Armley": "~2.5 m decline to 1995 and steady 3 m rise since",
}.items() if k in WELLS}

SHORT = {name: name.replace("WSA ", "").replace(" and No. 4", "/4").replace("No.", "No. ").replace("  ", " ")
         for name in WELLS}
SHORT.update({"WSA Blucher No. 3 and No. 4": "Blucher", "WSA Goodale Farm 009": "Goodale Farm",
              "WSA Duck Lake No. 1": "Duck Lake 1", "WSA Duck Lake No. 2": "Duck Lake 2",
              "WSA Conquest No.500": "Conquest 500", "WSA Conquest No.501": "Conquest 501"})

RES = Path("results") / SCOPE
FIG = Path("figures") / SCOPE
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load_levels(include_excluded=False):
    """Monthly levels for the study wells on a complete monthly index (NaN gaps)."""
    m = pd.read_csv("data/wsa_levels_monthly.csv")
    m = m[m.name.isin(WELLS)]
    if not include_excluded:
        m = m[~m.name.isin(EXCLUDED)]
    m = m.groupby(["name", "date"], as_index=False).level_m.mean()
    out = []
    for name, g in m.groupby("name"):
        idx = pd.period_range(g.date.min(), g.date.max(), freq="M").astype(str)
        g = g.set_index("date").reindex(idx).rename_axis("date").reset_index()
        g["name"] = name
        out.append(g)
    return pd.concat(out, ignore_index=True)


def load_drivers():
    d = pd.read_csv(DRIVERS)
    return d.rename(columns={"name": "driver_name"})


def merged(include_excluded=False):
    """Levels joined to the climate drivers of each well's location."""
    lv = load_levels(include_excluded)
    lv["driver_name"] = lv.name.map(WELLS)
    return lv.merge(load_drivers(), on=["driver_name", "date"], how="left")


def deseasonalised_anomaly(df, col="level_m"):
    """Level minus the well's mean for that calendar month (whole record)."""
    month = df.date.str[5:7].astype(int)
    clim = df.groupby([df.name, month])[col].transform("mean")
    return df[col] - clim


def window_correlations(series, dates, anom, anom_dates, windows):
    """Correlation of the deseasonalised w-month rolling sum of `series` with
    `anom`, for every w in `windows` (vectorised; returns an array).

    series, dates: the full monthly driver record ('YYYY-MM' strings)
    anom, anom_dates: the level anomaly and its months (subset of dates)
    """
    x = np.asarray(series, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    windows = np.asarray(list(windows))
    t = np.arange(len(x))
    sums = c[t + 1][None, :] - c[np.clip(t + 1 - windows[:, None], 0, None)]
    sums[t[None, :] + 1 < windows[:, None]] = np.nan  # window not yet full
    month = np.array([int(d[5:7]) for d in dates])
    for m in range(1, 13):
        cols = month == m
        sums[:, cols] -= np.nanmean(sums[:, cols], axis=1, keepdims=True)
    pos = pd.Series(np.arange(len(dates)), index=dates).reindex(anom_dates).to_numpy()
    s = sums[:, pos]
    a = np.broadcast_to(np.asarray(anom, dtype=float), s.shape)
    ok = ~np.isnan(s) & ~np.isnan(a)
    n = ok.sum(1)
    sm = np.where(ok, s, 0).sum(1) / n
    am = np.where(ok, a, 0).sum(1) / n
    ds, da = np.where(ok, s - sm[:, None], 0), np.where(ok, a - am[:, None], 0)
    return (ds * da).sum(1) / np.sqrt((ds ** 2).sum(1) * (da ** 2).sum(1)), n


def nse(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
