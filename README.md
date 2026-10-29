# Groundwater Levels and Climate Drivers — Saskatoon Region

Code and data for studying how groundwater levels in the aquifers around Saskatoon, Saskatchewan, respond to climate, and how far ahead they can be forecast. The project recovers monthly water levels for the Water Security Agency (WSA) observation wells, pairs them with ERA5-Land, TerraClimate and GRACE/GRACE-FO data from Google Earth Engine, and compares machine-learning forecasts against persistence baselines.

## Pipeline
| Step | Script | Output |
| --- | --- | --- |
| Well locations from land-survey descriptions | `wsa_wells.py` | `wells.csv` |
| Climate, GRACE and built-up drivers (Earth Engine Code Editor) | `groundwater_drivers.js` | `data/gee/*.csv` |
| Merge drivers per well and month | `build_drivers.py` | `data/drivers_monthly.csv` |
| Download the WSA hydrograph PDFs | `fetch_hydrographs.py` | `data/wsa_pdfs/` |
| Recover monthly levels from the vector PDFs | `extract_hydrographs.py` | `data/wsa_levels_monthly.csv`, `data/wsa_levels_meta.csv` |
| Hydrographs, map, memory windows, GRACE agreement | `memory_analysis.py` | `results/`, `figures/` |
| Forecast experiment and SHAP | `forecast_models.py` | `results/`, `figures/` |
| Robustness checks | `robustness.py` | `results/`, `figures/` |

Run the scripts in this order from the repository root. `common.py` holds the well list and the wells excluded or flagged for non-climatic behaviour.

Requirements: Python with pandas, numpy, scipy, matplotlib, scikit-learn, xgboost, torch, shap and pymupdf. The Earth Engine step needs a free Earth Engine account.

## Data
- **Water levels:** recovered from the hydrograph PDFs published by the [Water Security Agency](https://wsask.ca/drinking-water-wastewater/groundwater/observation-well-network/). The plotted line is vector data, so its vertices are read directly and converted with the plot's gridlines (one per 1 January) and axis labels. The values are median monthly levels as plotted by WSA. Please credit WSA as the source of the water-level data.
- **Climate and storage:** ERA5-Land monthly aggregates, TerraClimate, JPL GRACE/GRACE-FO mascons (CRI) and GHSL built-up surface, all from the Google Earth Engine catalogue.
- Well coordinates are approximate (±1–2 km), converted from Dominion Land Survey descriptions.

## Status
Work in progress; a manuscript is in preparation.
