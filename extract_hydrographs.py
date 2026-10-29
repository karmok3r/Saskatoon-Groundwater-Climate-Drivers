"""Extracts monthly water levels from the WSA hydrograph PDFs.

The hydrographs are vector plots of median monthly water levels. The data
line is a set of coloured line segments, so its vertices can be read
directly and converted to (month, level) with the plot's own axes:

- x: one vertical gridline per 1 January; the year labels sit between
  gridlines and give the year of each interval
- y: least-squares fit of the axis label values to their vertical position

Each vertex is assigned to the calendar month it falls in. Writes
data/wsa_levels_monthly.csv (name, date, level_m, series) and
data/wsa_levels_meta.csv, and draws figures/extraction_check.png, which
overlays the extracted points on a rendering of the PDF for a few wells.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf

PDFS = Path("data/wsa_pdfs")
OUT_LEVELS = Path("data/wsa_levels_monthly.csv")
OUT_META = Path("data/wsa_levels_meta.csv")
FIG = Path("figures/extraction_check.png")
CHECK_WELLS = ["WSA Dalmeny", "WSA Saskatoon", "WSA Vanscoy", "WSA Duck Lake No. 1"]
NUM = re.compile(r"-?\d+(\.\d+)?")


def is_grey(colour):
    """Axes and gridlines are black or grey; the data line is coloured."""
    return colour is not None and max(colour) - min(colour) < 0.05


def spans(page):
    return [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
            for s in l["spans"] if s["text"].strip()]


def x_axis(page, sp, lines):
    """Returns f(x) -> decimal year, from the year labels and vertical gridlines."""
    labels = [s for s in sp if re.fullmatch(r"\d{2}", s["text"].strip())]
    # keep the row with the most two-digit labels (the year axis)
    rows = pd.Series([round(s["bbox"][1]) for s in labels])
    labels = [s for s, r in zip(labels, rows) if r == rows.mode()[0]]
    labels.sort(key=lambda s: s["bbox"][0])
    years, century, last = [], 1900 if int(labels[0]["text"]) >= 30 else 2000, -1
    for s in labels:
        yy = int(s["text"])
        if yy < last:
            century += 100
        last = yy
        years.append(century + yy)
    centres = np.array([(s["bbox"][0] + s["bbox"][2]) / 2 for s in labels])
    vx = np.array(sorted({round(i[1].x, 2) for i in lines
                          if abs(i[1].x - i[2].x) < 0.01 and abs(i[1].y - i[2].y) > 100}))
    per_year = np.median(np.diff(vx))
    # gridline at or just left of the first label = 1 January of that year
    left = vx[vx <= centres[0]].max()
    if np.ptp(np.diff(vx)) > 0.2 * per_year or centres[0] - left > per_year:
        raise ValueError("irregular year gridlines")
    return lambda x: years[0] + (x - left) / per_year


def y_axis(sp, x_limit):
    """Returns f(y) -> level, from the numeric labels left of the plot."""
    labels = [s for s in sp if NUM.fullmatch(s["text"].strip()) and s["bbox"][2] < x_limit]
    v = np.array([float(s["text"]) for s in labels])
    c = np.array([(s["bbox"][1] + s["bbox"][3]) / 2 for s in labels])
    slope, icpt = np.polyfit(c, v, 1)
    resid = np.abs(v - (slope * c + icpt)).max()
    if len(labels) < 3 or resid > 0.01 * np.ptp(v):
        raise ValueError(f"y labels do not fit a line (max residual {resid:.3f})")
    return lambda y: slope * y + icpt, len(labels)


def header(sp):
    text = " ".join(s["text"] for s in sp)
    def find(pat):
        m = re.search(pat, text)
        return m.group(1).strip() if m else None
    return dict(datum_m=find(r"DATUM:\s*([\d.]+)"), depth_m=find(r"DEPTH:\s*([\d.]+)"),
                land_location=find(r"LAND LOCATION:\s*(\S+)"),
                y_label=find(r"(WATER LEVEL[^\d]*?LEVEL|DEPTH TO WATER[^\d]*)"))


def extract(path, name):
    page = pymupdf.open(path)[0]
    sp = spans(page)
    draws = page.get_drawings()
    lines = [i for d in draws if is_grey(d.get("color")) for i in d["items"] if i[0] == "l"]
    to_year = x_axis(page, sp, lines)
    plot_left = min(i[1].x for i in lines if abs(i[1].x - i[2].x) < 0.01 and abs(i[1].y - i[2].y) > 100)
    to_level, n_ylabels = y_axis(sp, plot_left)

    rows = []
    colours = sorted({d["color"] for d in draws if d.get("color") and not is_grey(d["color"])})
    for k, col in enumerate(colours):
        pts = {(round(pt.x, 3), round(pt.y, 3)) for d in draws if d.get("color") == col
               for i in d["items"] for pt in i[1:] if hasattr(pt, "x")}
        for x, y in pts:
            t = to_year(x)
            rows.append(dict(name=name, year=int(t), month=int((t % 1) * 12) + 1,
                             level_m=to_level(y), series=k, x=x, y=y))
    df = pd.DataFrame(rows)
    meta = dict(name=name, file=path.name, n_series=len(colours), n_ylabels=n_ylabels, **header(sp))
    return df, meta


def check_figure(levels, index):
    fig, axes = plt.subplots(len(CHECK_WELLS), 1, figsize=(9, 2.6 * len(CHECK_WELLS)))
    for ax, name in zip(axes, CHECK_WELLS):
        f = PDFS / index.loc[index.name == name, "file"].iloc[0]
        page = pymupdf.open(f)[0]
        pix = page.get_pixmap(dpi=100)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        s = 100 / 72
        ax.imshow(img)
        d = levels[levels.name == name]
        ax.scatter(d.x * s, d.y * s, s=3, c="#e8a33d", lw=0, label="extracted months")
        ax.set_xlim(d.x.min() * s - 40, d.x.max() * s + 40)
        ax.set_ylim(d.y.max() * s + 40, d.y.min() * s - 40)
        ax.set_title(f"{name}: extracted points (orange) over the WSA PDF", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)


def main():
    index = pd.read_csv(PDFS / "index.csv")
    all_pts, metas = [], []
    for r in index.itertuples():
        try:
            pts, meta = extract(PDFS / r.file, r.name)
            all_pts.append(pts)
        except Exception as e:  # report and skip plots that do not parse
            meta = dict(name=r.name, file=r.file, error=str(e))
        metas.append(meta)
    pts = pd.concat(all_pts)
    check_figure(pts, index)

    # one value per well, series and month; a month with two vertices (rare,
    # where segments meet) gets their median
    monthly = (pts.groupby(["name", "series", "year", "month"], as_index=False)
                  .agg(level_m=("level_m", "median"), n_vertices=("level_m", "size")))
    monthly["date"] = monthly.year.astype(str) + "-" + monthly.month.astype(str).str.zfill(2)
    monthly = monthly[["name", "series", "date", "level_m", "n_vertices"]]
    monthly.to_csv(OUT_LEVELS, index=False, float_format="%.3f")

    meta = pd.DataFrame(metas)
    stats = monthly.groupby("name").agg(first=("date", "min"), last=("date", "max"),
                                        n_months=("date", "size"),
                                        months_with_2_vertices=("n_vertices", lambda s: (s > 1).sum()))
    meta = meta.merge(stats, on="name", how="left")
    meta.to_csv(OUT_META, index=False)
    print(meta[["name", "n_series", "first", "last", "n_months", "months_with_2_vertices"]
               + (["error"] if "error" in meta else [])].to_string(index=False))


if __name__ == "__main__":
    main()
