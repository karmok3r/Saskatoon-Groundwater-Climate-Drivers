"""Merges the Earth Engine exports into one monthly driver table per well.

Reads data/gee/*.csv (from groundwater_drivers.js) and writes
data/drivers_monthly.csv: one row per well and month, 1960-01 to the last
ERA5-Land month. TerraClimate columns are prefixed tc_; GRACE is NaN outside
2002-2024 and in missing months; GHSL built-up share is interpolated
linearly between the 5-yearly epochs.

Also adds cumulative water-balance anomalies (precipitation minus
evaporation, 12/24/36/60 months), the slow driver expected for prairie
aquifers, and draws figures/regional_storage.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEE = Path("data/gee")
OUT = Path("data/drivers_monthly.csv")
FIG = Path("figures/regional_storage.png")
WINDOWS = [12, 24, 36, 60]  # months
PLOT_WELL = "WSA Dalmeny"

ERA5_RENAME = {
    "total_precipitation_sum": "precip_mm",
    "snowfall_sum": "snowfall_mm",
    "snowmelt_sum": "snowmelt_mm",
    "snow_depth_water_equivalent": "swe_mm",
    "surface_runoff_sum": "runoff_surface_mm",
    "sub_surface_runoff_sum": "runoff_subsurface_mm",
    "total_evaporation_sum": "evap_mm",
    "potential_evaporation_sum": "pet_mm",
    "volumetric_soil_water_layer_1": "soil_0_7cm",
    "volumetric_soil_water_layer_2": "soil_7_28cm",
    "volumetric_soil_water_layer_3": "soil_28_100cm",
    "volumetric_soil_water_layer_4": "soil_100_289cm",
    "temperature_2m": "temp_c",
}


def load_monthly(name, rename=None, prefix=""):
    df = pd.read_csv(GEE / name).drop(columns=["year", "month"])
    df = df.rename(columns=rename or {})
    if prefix:
        df = df.rename(columns={c: prefix + c for c in df.columns if c not in ("name", "date")})
    return df


def built_monthly(dates):
    """GHSL built-up share interpolated to every month, per well."""
    b = pd.read_csv(GEE / "ghsl_builtup_wells.csv")
    x = pd.to_datetime(dates).year + (pd.to_datetime(dates).month - 0.5) / 12
    rows = []
    for name, g in b.groupby("name"):
        g = g.sort_values("epoch")
        rows.append(pd.DataFrame({"name": name, "date": dates,
                                  "built_pct": np.interp(x, g.epoch, g.built_pct)}))
    return pd.concat(rows)


def add_water_balance(df):
    """Cumulative precipitation minus evaporation, as anomalies per well."""
    df = df.sort_values(["name", "date"]).copy()
    df["p_minus_e_mm"] = df.precip_mm - df.evap_mm
    for w in WINDOWS:
        roll = df.groupby("name").p_minus_e_mm.transform(lambda s: s.rolling(w).sum())
        df[f"pme_{w}m_anom_mm"] = roll - roll.groupby(df.name).transform("mean")
    return df


def plot_storage(df):
    """Regional GRACE storage vs the 36- and 60-month water balance at one well."""
    d = df[df.name == PLOT_WELL].set_index(pd.to_datetime(df.loc[df.name == PLOT_WELL, "date"]))
    d = d.loc["1995":]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(d.index, d.pme_36m_anom_mm, color="#4a7fb0", lw=1.4, label="P − E, 36-month sum (anomaly)")
    ax.plot(d.index, d.pme_60m_anom_mm, color="#9bbbd8", lw=1.4, label="P − E, 60-month sum (anomaly)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("mm")
    ax2 = ax.twinx()
    ax2.plot(d.index, d.grace_lwe_cm, "o-", color="#c0504d", ms=2.5, lw=0.8, label="GRACE/GRACE-FO storage")
    ax2.set_ylabel("cm water equivalent")
    lines = ax.get_lines()[:2] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"Water balance (ERA5-Land) and GRACE storage at {PLOT_WELL}", fontsize=10)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=200)


def main():
    era5 = load_monthly("era5land_monthly_wells.csv", ERA5_RENAME)
    terra = load_monthly("terraclimate_monthly_wells.csv", prefix="tc_")
    grace = load_monthly("grace_monthly_wells.csv", {"lwe_thickness": "grace_lwe_cm"})
    # Some months have two GRACE solutions with the same YYYY-MM label
    grace = grace.groupby(["name", "date"], as_index=False).grace_lwe_cm.mean()

    df = (era5.merge(terra, on=["name", "date"], how="left")
              .merge(grace, on=["name", "date"], how="left")
              .merge(built_monthly(era5.date.unique()), on=["name", "date"], how="left"))
    df = add_water_balance(df)
    df.insert(2, "year", df.date.str[:4].astype(int))
    df.insert(3, "month", df.date.str[5:].astype(int))
    df.to_csv(OUT, index=False, float_format="%.4g")

    print(f"{OUT}: {len(df)} rows, {df.name.nunique()} wells, {df.date.min()} to {df.date.max()}")
    print("Non-missing months per source (per well):")
    for col in ["precip_mm", "tc_pr", "grace_lwe_cm"]:
        print(f"  {col}: {df.groupby('name')[col].count().iloc[0]}")

    d = df[(df.name == PLOT_WELL) & df.grace_lwe_cm.notna()]
    for w in WINDOWS:
        r = d.grace_lwe_cm.corr(d[f"pme_{w}m_anom_mm"])
        print(f"r(GRACE, P-E {w}-month) at {PLOT_WELL}: {r:.2f}")
    plot_storage(df)
    print(f"Figure: {FIG}")


if __name__ == "__main__":
    main()
