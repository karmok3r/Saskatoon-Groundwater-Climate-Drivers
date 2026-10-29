"""Downloads the hydrograph PDF of every WSA observation well.

Reads the network index page on wsask.ca, opens each well page, finds the
link to its hydrograph PDF and saves it to data/wsa_pdfs/. Writes
data/wsa_pdfs/index.csv with the well name, page URL, PDF URL and the land
location from the page.
"""
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

INDEX_URL = "https://wsask.ca/drinking-water-wastewater/groundwater/observation-well-network/"
OUT = Path("data/wsa_pdfs")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; groundwater hydrograph download)"}


def get(url):
    with urlopen(Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = get(INDEX_URL).decode("utf-8", "replace")
    pages = sorted(set(re.findall(re.escape(INDEX_URL) + r"[a-z0-9\-]+/", html)))
    rows = []
    for page in pages:
        text = get(page).decode("utf-8", "replace")
        title = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.S)
        name = re.sub(r"<[^>]+>|\s+", " ", title.group(1)).strip() if title else page
        pdfs = re.findall(r'href="([^"]+\.pdf)"', text, re.I)
        hydro = [p for p in pdfs if "hydrograph" in p.lower()]
        if not hydro:
            print(f"no hydrograph: {name}")
            continue
        url = hydro[0]
        fname = OUT / (re.sub(r"[^A-Za-z0-9]+", "_", page.rstrip("/").split("/")[-1]) + ".pdf")
        if not fname.exists():
            fname.write_bytes(get(url))
            time.sleep(0.5)
        rows.append(dict(name=name, page=page, pdf_url=url, file=fname.name))
        print(f"ok: {name}")
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    print(f"{len(rows)} hydrographs saved to {OUT}")


if __name__ == "__main__":
    main()
