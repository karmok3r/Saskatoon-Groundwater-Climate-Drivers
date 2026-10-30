# Groundwater Levels and Climate Drivers — Saskatchewan

Code and data for studying how groundwater levels in Saskatchewan's prairie aquifers respond to climate, and how far ahead they can be forecast. The project recovers monthly water levels for the Water Security Agency (WSA) observation wells, pairs them with ERA5-Land, TerraClimate and GRACE/GRACE-FO data from Google Earth Engine, estimates how long the levels remember past climate, and compares machine-learning and transfer-function forecasts against persistence baselines.

The analysis runs at two scopes, chosen with the `GW_SCOPE` environment variable:
- `saskatoon` (default): the wells within 120 km of Saskatoon
- `province`: every WSA well with at least 20 years of recovered record (48 wells after screening)

Outputs go to `results/<scope>/` and `figures/<scope>/`.

## Pipeline
| Step | Script | Output |
| --- | --- | --- |
| Well locations from land-survey descriptions | `wsa_wells.py` (Saskatoon), `wells_province.py` (province) | `wells.csv`, `wells_province.csv` |
| Climate, GRACE and built-up drivers (Earth Engine Code Editor) | `groundwater_drivers.js`, `groundwater_drivers_province.js` | `data/gee/`, `data/gee_province/` |
| Merge drivers per well and month | `build_drivers.py` (`python build_drivers.py province` for the province) | `data/drivers_monthly.csv`, `data/drivers_monthly_province.csv` |
| Download the WSA hydrograph PDFs | `fetch_hydrographs.py` | `data/wsa_pdfs/` |
| Recover monthly levels from the vector PDFs | `extract_hydrographs.py` | `data/wsa_levels_monthly.csv`, `data/wsa_levels_meta.csv` |
| Hydrographs, memory windows, GRACE agreement | `memory_analysis.py` | `results/<scope>/`, `figures/<scope>/` |
| Forecast experiment (persistence, seasonal persistence, ridge, random forest, XGBoost, LSTM) and SHAP | `forecast_models.py` | `results/<scope>/` |
| Robustness: surrogate significance, TerraClimate, 360-month windows, detrending, tuned and per-well models | `robustness.py` | `results/<scope>/robust_*.csv` |
| Pastas transfer-function noise models (forecasts and response times) | `pastas_models.py` | `results/<scope>/pastas_*.csv` |
| Study area map and summary figures | `make_maps.py`, `make_figures_hj.py` | `figures/hj/` |

Run the scripts in this order from the repository root, for example `GW_SCOPE=province python memory_analysis.py`. `common.py` holds the well lists, the screening (wells excluded for pumping-type drawdowns and wells flagged for non-climatic trends or steps) and the shared loaders.

## Requirements
- Python with pandas, numpy, scipy, matplotlib, geopandas, scikit-learn, xgboost, torch, shap and pymupdf.
- `pastas_models.py` needs Pastas 2.0, which does not yet support pandas 3. Run it in a separate environment, for example:
  ```
  py -3.13 -m venv .venv_pastas
  .venv_pastas/Scripts/python -m pip install pastas "pandas<3" numba tqdm matplotlib
  GW_SCOPE=province .venv_pastas/Scripts/python pastas_models.py
  ```
- `make_maps.py` uses Natural Earth 1:10m layers (admin 0 and 1, rivers, lakes), downloaded as zip files into `data/basemap/` (not included).
- The Earth Engine step needs a free Earth Engine account.

## Data
- **Water levels:** recovered from the hydrograph PDFs published by the [Water Security Agency](https://wsask.ca/drinking-water-wastewater/groundwater/observation-well-network/). The plotted line is vector data, so its vertices are read directly and converted with the plot's gridlines (one per 1 January) and axis labels. The values are median monthly levels as plotted by WSA. Please credit WSA as the source of the water-level data.
- **Climate and storage:** ERA5-Land monthly aggregates, TerraClimate, JPL GRACE/GRACE-FO mascons (CRI) and GHSL built-up surface, all from the Google Earth Engine catalogue (`data/gee/`, `data/gee_province/`).
- Well coordinates are approximate (±1–2 km), converted from Dominion Land Survey descriptions.

## Status
Work in progress; a manuscript is in preparation.
