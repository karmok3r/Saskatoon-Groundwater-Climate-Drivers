"""Shared well list and loaders for the analysis scripts."""
import numpy as np
import pandas as pd

# Saskatoon-region wells with an extractable hydrograph, mapped to the
# location used for the climate drivers (Duck Lake No. 1 and No. 2 share one).
# WSA Warman 2 and Crawford Lake have no hydrograph on wsask.ca.
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

# Wells with a clear non-climatic signal (see figures/hydrographs.png):
# Vanscoy has a ~14 m drawdown and recovery in 2005-2015, so it is left out
# of model training and pooled scores. The others stay in but are flagged.
EXCLUDED = ["WSA Vanscoy"]
FLAGGED = {
    "WSA Hague": "abrupt ~1 m drop around 1991",
    "WSA Conquest No.500": "steady 3.5 m rise since 2010 not seen at other wells",
    "WSA Saskatoon": "step rise in 2010 and large month-to-month swings in 2011-2016",
}

SHORT = {
    "WSA Saskatoon": "Saskatoon", "WSA Goodale Farm 009": "Goodale Farm",
    "WSA Dalmeny": "Dalmeny", "WSA Vanscoy": "Vanscoy", "WSA Agrium 43": "Agrium 43",
    "WSA Blucher No. 3 and No. 4": "Blucher", "WSA Hague": "Hague",
    "WSA Swanson": "Swanson", "WSA Tessier": "Tessier",
    "WSA Conquest No.500": "Conquest 500", "WSA Conquest No.501": "Conquest 501",
    "WSA Bruno": "Bruno", "WSA Duck Lake No. 1": "Duck Lake 1",
    "WSA Duck Lake No. 2": "Duck Lake 2",
}


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
    d = pd.read_csv("data/drivers_monthly.csv")
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


def nse(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
