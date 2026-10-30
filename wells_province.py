"""Coordinates for every WSA well with a recovered hydrograph, and the
province-wide Earth Engine script.

Reads the land location printed on each hydrograph (data/wsa_levels_meta.csv),
converts it with the same township-grid model as wsa_wells.py, and writes
wells_province.csv. It then writes groundwater_drivers_province.js: a copy of
groundwater_drivers.js with this well list and its own export folder.
Wells that share a location are sampled once (the first name is kept as the
location name; wells_province.csv maps every well to it).
"""
import re
from pathlib import Path

import pandas as pd

from wsa_wells import SASKATOON, dls_to_latlon, km_between

TEMPLATE = Path("groundwater_drivers.js")
OUT_JS = Path("groundwater_drivers_province.js")
OUT_CSV = Path("wells_province.csv")
EXPORT_FOLDER = "gee_sask_groundwater_province"


def parse_dls(text):
    """'SE01-13-38-06-W3' -> (quarter, lsd, sec, twp, rge, mer)."""
    m = re.fullmatch(r"([NS][EW]?)?-?([\d-]+)-(W\d)", text.strip())
    if not m:
        raise ValueError(f"unrecognised land location {text!r}")
    quarter, nums, mer = m.group(1), [int(n) for n in m.group(2).split("-") if n], m.group(3)
    if len(nums) == 4:
        lsd, sec, twp, rge = nums
    elif len(nums) == 3:
        lsd, (sec, twp, rge) = None, nums
    else:
        raise ValueError(f"unexpected number of fields in {text!r}")
    if not (1 <= sec <= 36 and 1 <= twp <= 126 and 1 <= rge <= 34 and (lsd is None or 1 <= lsd <= 16)):
        raise ValueError(f"out-of-range DLS values in {text!r}")
    if quarter == "S":  # south half of the section
        quarter = None if lsd else "S"
    return quarter, lsd, sec, twp, rge, mer


def latlon(quarter, lsd, sec, twp, rge, mer):
    if quarter == "S":  # half section: centre of the south half
        lat_n, lon_n = dls_to_latlon("SE", None, sec, twp, rge, mer)
        lat_w, lon_w = dls_to_latlon("SW", None, sec, twp, rge, mer)
        return (lat_n + lat_w) / 2, (lon_n + lon_w) / 2
    return dls_to_latlon(quarter, lsd, sec, twp, rge, mer)


def main():
    meta = pd.read_csv("data/wsa_levels_meta.csv")
    meta = meta[meta["first"].notna()]
    rows = []
    for r in meta.itertuples():
        lat, lon = latlon(*parse_dls(r.land_location))
        rows.append(dict(name=r.name, land_location=r.land_location, lat=round(lat, 4),
                         lon=round(lon, 4), km_from_saskatoon=round(km_between((lat, lon), SASKATOON), 1)))
    df = pd.DataFrame(rows)
    df["location"] = df.groupby(["lat", "lon"]).name.transform("first")
    df.to_csv(OUT_CSV, index=False)
    sites = df.drop_duplicates("location")
    print(f"{len(df)} wells at {len(sites)} distinct locations; "
          f"lat {df.lat.min()}-{df.lat.max()}, lon {df.lon.min()}-{df.lon.max()}")

    features = ",\n".join(f"  ee.Feature(ee.Geometry.Point([{s.lon}, {s.lat}]), {{name: '{s.location}'}})"
                          for s in sites.itertuples())
    js = TEMPLATE.read_text(encoding="utf-8")
    js = re.sub(r"var WELLS = ee\.FeatureCollection\(\[.*?\]\);",
                lambda _: "var WELLS = ee.FeatureCollection([\n" + features + "\n]);", js, flags=re.S)
    js = js.replace("// WSA observation wells within 120 km of Saskatoon (from wsa_wells.py).\n"
                    "// Duck Lake No. 1 and No. 2 share one location, so only one is listed.",
                    "// All WSA observation-well locations with a recovered hydrograph\n"
                    "// (from wells_province.py; wells sharing a location are listed once).")
    js = js.replace("var EXPORT_FOLDER = 'gee_saskatoon_groundwater';",
                    f"var EXPORT_FOLDER = '{EXPORT_FOLDER}';")
    js = js.replace("Groundwater Climate Drivers — WSA Observation Wells near Saskatoon",
                    "Groundwater Climate Drivers — WSA Observation Wells, Saskatchewan")
    js = js.replace("Map.centerObject(WELLS, 8);", "Map.centerObject(WELLS, 6);")
    OUT_JS.write_text(js, encoding="utf-8")
    print(f"wrote {OUT_JS} (export folder {EXPORT_FOLDER})")


if __name__ == "__main__":
    main()
