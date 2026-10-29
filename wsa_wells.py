"""Approximate coordinates for WSA observation wells near Saskatoon.

WSA publishes only Dominion Land Survey (DLS) locations. This converts them to
latitude/longitude with a simple grid model (township rows from 49 N, ranges
west of the meridian). Expect errors of about 1-2 km because road allowances
and correction lines are ignored. That is fine for ERA5-Land (~9 km),
TerraClimate (~4 km) and GRACE, but not for anything at field scale.

Writes wells.csv and prints the Earth Engine FeatureCollection used in
groundwater_drivers.js.
"""
import math

import pandas as pd

MERIDIANS = {"W2": 102.0, "W3": 106.0}  # degrees west
TWP_KM = 9.74  # 6 miles plus road allowances
KM_PER_DEG_LAT = 111.2
SASKATOON = (52.13, -106.67)
MAX_KM = 120

# name, quarter, LSD, section, township, range, meridian (from wsask.ca well pages)
WELLS = [
    ("WSA Saskatoon", "NW", 16, 8, 37, 4, "W3"),
    ("WSA Dalmeny", "SE", 1, 13, 38, 6, "W3"),
    ("WSA Warman 2", None, None, 9, 39, 5, "W3"),
    ("WSA Hague", "SE", 8, 6, 41, 2, "W3"),
    ("WSA Goodale Farm 009", "SW", 1, 4, 36, 4, "W3"),
    ("WSA Blucher No. 3 and No. 4", "SW", 4, 27, 35, 2, "W3"),
    ("Crawford Lake", "NW", None, 32, 35, 2, "W3"),
    ("WSA Vanscoy", "SW", 12, 14, 35, 8, "W3"),
    ("WSA Agrium 43", "NE", 16, 20, 35, 8, "W3"),
    ("WSA Swanson", "SW", 4, 15, 31, 8, "W3"),
    ("WSA Tessier", "NW", 13, 33, 33, 11, "W3"),
    ("WSA Conquest No. 500", "SW", 13, 14, 30, 9, "W3"),
    ("WSA Conquest No. 501", "NW", 4, 23, 30, 10, "W3"),
    ("WSA Duck Lake No. 1", "SE", 1, 33, 45, 2, "W3"),
    ("WSA Duck Lake No. 2", "SE", 1, 33, 45, 2, "W3"),
    ("WSA Bruno", "SE", 14, 9, 38, 25, "W2"),
]


def section_cell(sec):
    """(row from south, column from east) of a section, 0-5, in its township."""
    row, i = divmod(sec - 1, 6)
    return row, (i if row % 2 == 0 else 5 - i)


def lsd_cell(lsd):
    """(row from south, column from east) of a legal subdivision, 0-3, in its section."""
    row, i = divmod(lsd - 1, 4)
    return row, (i if row % 2 == 0 else 3 - i)


def dls_to_latlon(quarter, lsd, sec, twp, rge, mer):
    """Centre of the LSD (or quarter, or section) as (lat, lon)."""
    srow, scol = section_cell(sec)
    if lsd:
        lrow, lcol = lsd_cell(lsd)
        north, west = (lrow + 0.5) / 4, (lcol + 0.5) / 4
    elif quarter:
        north = 0.75 if quarter[0] == "N" else 0.25
        west = 0.75 if quarter[1] == "W" else 0.25
    else:
        north = west = 0.5
    sec_km = TWP_KM / 6
    lat = 49.0 + ((twp - 1) * TWP_KM + (srow + north) * sec_km) / KM_PER_DEG_LAT
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat))
    lon = MERIDIANS[mer] + ((rge - 1) * TWP_KM + (scol + west) * sec_km) / km_per_deg_lon
    return lat, -lon


def km_between(a, b):
    dlat = (a[0] - b[0]) * KM_PER_DEG_LAT
    dlon = (a[1] - b[1]) * KM_PER_DEG_LAT * math.cos(math.radians(a[0]))
    return math.hypot(dlat, dlon)


def main():
    rows = []
    for name, q, lsd, sec, twp, rge, mer in WELLS:
        lat, lon = dls_to_latlon(q, lsd, sec, twp, rge, mer)
        loc = f"{q or ''}{lsd or ''}-{sec:02d}-{twp}-{rge:02d}-{mer}".lstrip("-")
        rows.append(dict(name=name, dls=loc, lat=round(lat, 4), lon=round(lon, 4),
                         km_from_saskatoon=round(km_between((lat, lon), SASKATOON), 1)))
    df = pd.DataFrame(rows)
    df = df[df.km_from_saskatoon <= MAX_KM].sort_values("km_from_saskatoon")
    df.to_csv("wells.csv", index=False)
    print(df.to_string(index=False))

    print("\n// Paste into groundwater_drivers.js")
    print("var WELLS = ee.FeatureCollection([")
    for r in df.itertuples():
        print(f"  ee.Feature(ee.Geometry.Point([{r.lon}, {r.lat}]), {{name: '{r.name}'}}),")
    print("]);")


if __name__ == "__main__":
    main()
