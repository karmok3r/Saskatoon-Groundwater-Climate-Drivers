"""Fig. 1 for Hydrogeology Journal: study area map (SAM) with a
hydrogeological location map (HLM) inset.

SAM: southern and central Saskatchewan, the WSA observation wells used
(core, flagged, excluded; sites with records shorter than 20 years shown
separately), major rivers and lakes, neighbouring provinces and states.
HLM inset: western Canada and the northern USA with Saskatchewan and the
study area outlined. Base layers: Natural Earth 1:10m (public domain), in
data/basemap/. Coordinates are geographic (latitude-longitude), as allowed
by the journal in place of a scale bar.

Writes figures/hj/Fig01.png (600 dpi) and Fig01.pdf.
"""
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

os.environ.setdefault("GW_SCOPE", "province")
from common import EXCLUDED, FLAGGED, WELLS  # noqa: E402

BASE = "data/basemap/"
OUT = Path("figures/hj")
BOX = (-110.4, 48.6, -101.0, 55.6)  # lon_min, lat_min, lon_max, lat_max
plt.rcParams.update({"font.family": "Arial", "font.size": 8})

COL = {"core": "#1f4e79", "flagged": "#c0392b", "excluded": "#7f7f7f", "short": "#ffffff"}
CITIES = {"Saskatoon": (52.13, -106.67), "Regina": (50.45, -104.61), "Prince Albert": (53.20, -105.75),
          "Moose Jaw": (50.39, -105.53), "Swift Current": (50.29, -107.80), "Yorkton": (51.21, -102.46),
          "North Battleford": (52.76, -108.29), "Estevan": (49.14, -102.99)}


def read(name, bbox=None):
    return gpd.read_file(BASE + name + ".zip", bbox=bbox)


def well_table():
    w = pd.read_csv("wells_province.csv")
    status = []
    for n in w.name:
        if n not in WELLS:
            status.append("short")
        elif n in EXCLUDED:
            status.append("excluded")
        elif n in FLAGGED:
            status.append("flagged")
        else:
            status.append("core")
    w["status"] = status
    return w.drop_duplicates(["lat", "lon", "status"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    big = (-125, 40, -88, 61)
    prov = read("ne_10m_admin_1_states_provinces", bbox=big)
    # major rivers only: the lake-centreline layer plus named rivers of the
    # North America supplement (the supplement's unnamed streams clutter the map)
    na = read("ne_10m_rivers_north_america", bbox=BOX)
    rivers = pd.concat([read("ne_10m_rivers_lake_centerlines", bbox=BOX),
                        na[na.name.isin(["Souris", "Qu'Appelle", "Assiniboine", "Battle", "Red Deer",
                                         "Frenchman", "Carrot", "Saskatchewan"])]])
    lakes = read("ne_10m_lakes", bbox=BOX)
    sask = prov[prov.name == "Saskatchewan"]

    fig = plt.figure(figsize=(17.6 / 2.54, 15.5 / 2.54))
    ax = fig.add_axes([0.07, 0.07, 0.60, 0.90])
    prov.plot(ax=ax, color="#f2f2ee", edgecolor="#9a9a9a", lw=0.6)
    sask.plot(ax=ax, color="#ffffff", edgecolor="black", lw=0.9)
    lakes.plot(ax=ax, color="#b9d7ee", edgecolor="#7fb2d9", lw=0.3)
    rivers.plot(ax=ax, color="#5b9bd5", lw=0.7)

    w = well_table()
    for status, label, marker, size in [("short", "Record < 20 yr, not analysed", "o", 16),
                                        ("excluded", "Excluded: pumping drawdown", "X", 30),
                                        ("flagged", "Flagged: non-climatic trend", "s", 20),
                                        ("core", "Core well", "o", 22)]:
        d = w[w.status == status]
        ax.scatter(d.lon, d.lat, s=size, marker=marker, color=COL[status],
                   edgecolor="#7f7f7f" if status == "short" else "black",
                   lw=0.8 if status == "short" else 0.5, zorder=4, label=f"{label} ({len(d)})")
    for city, (lat, lon) in CITIES.items():
        ax.plot(lon, lat, marker="s", ms=3, color="black", zorder=5)
        left = city in ("Moose Jaw",)
        ax.annotate(city, (lon, lat), xytext=(-3, 3) if left else (3, 3), ha="right" if left else "left",
                    textcoords="offset points", fontsize=7, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    for name, (lon, lat) in {"ALBERTA": (-109.6, 54.9), "MANITOBA": (-101.25, 51.5),
                             "SASKATCHEWAN": (-106.0, 55.25), "MONTANA (USA)": (-108.3, 48.75),
                             "NORTH DAKOTA (USA)": (-102.8, 48.75)}.items():
        ax.text(lon, lat, name, fontsize=7, ha="center", color="#555555", style="italic",
                rotation=90 if name == "MANITOBA" else 0)
    for name, (lon, lat, rot) in {"South Saskatchewan River": (-107.75, 51.05, 35),
                                  "North Saskatchewan River": (-107.5, 53.2, -20),
                                  "Lake Diefenbaker": (-106.9, 50.85, 0)}.items():
        ax.text(lon, lat, name, fontsize=6.5, color="#2e75b6", rotation=rot, ha="center")

    ax.set_xlim(BOX[0], BOX[2])
    ax.set_ylim(BOX[1], BOX[3])
    ax.set_aspect(1 / np.cos(np.radians(52)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(color="#d9d9d9", lw=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="#5b9bd5", lw=1), Line2D([], [], marker="s", ls="", color="black", ms=3)]
    labels += ["River", "City"]
    labels = [l.replace(" (", " (n = ") if l[-1] == ")" else l for l in labels]
    fig.legend(handles, labels, title="Explanation", loc="lower left", bbox_to_anchor=(0.69, 0.07), handletextpad=0.4,
               fontsize=6.5, title_fontsize=7, frameon=True, edgecolor="#bfbfbf", alignment="left")

    # HLM inset
    ins = fig.add_axes([0.71, 0.60, 0.27, 0.34])
    prov.plot(ax=ins, color="#f2f2ee", edgecolor="#b0b0b0", lw=0.3)
    countries = read("ne_10m_admin_0_countries", bbox=big)
    countries.boundary.plot(ax=ins, color="black", lw=0.6)
    sask.plot(ax=ins, color="#d9e5f2", edgecolor="black", lw=0.6)
    ins.add_patch(Rectangle((BOX[0], BOX[1]), BOX[2] - BOX[0], BOX[3] - BOX[1], fill=False, ec="#c0392b", lw=0.9))
    ins.text(-112, 58.2, "CAN", fontsize=7, fontweight="bold")
    ins.text(-112, 44.5, "USA", fontsize=7, fontweight="bold")
    ins.set_xlim(big[0], big[2])
    ins.set_ylim(big[1], big[3])
    ins.set_aspect(1 / np.cos(np.radians(50)))
    ins.tick_params(labelsize=5.5, length=2)
    ins.set_xlabel("")
    ins.set_ylabel("")
    ins.set_xticks([-120, -110, -100, -90])
    ins.set_yticks([40, 50, 60])
    ins.set_facecolor("white")
    for s in ins.spines.values():
        s.set_linewidth(0.6)

    fig.savefig(OUT / "Fig01.png", dpi=600)
    fig.savefig(OUT / "Fig01.pdf")
    print("wrote", OUT / "Fig01.png", "; sites:", w.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
