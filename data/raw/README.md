# Raw Datasets Inventory & Data Provenance

This directory documents and contains the raw geospatial, raster, and meteorological surface telemetry datasets used by the **Maharashtra Renewable Energy Site Finder** pipeline.

---

## 1. Primary Geospatial Rasters (World Bank / ESMAP Global Solar Atlas v2)

- **Archive**: `India_GISdata_LTAym_YearlyMonthlyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip`
- **Source**: Global Solar Atlas 2.0 (World Bank ESMAP / Solargis)
- **Coordinate Reference System (CRS)**: WGS 84 (EPSG:4326)
- **Spatial Resolution**: ~250 m ($12,800 \times 12,800$ native grid for India domain)
- **Key GeoTIFF Rasters Extracted**:
  1. `GHI.tif` — Global Horizontal Irradiation (kWh/m²/year and kWh/m²/day equivalent). Direct driver of solar PV and CSP potential.
  2. `DNI.tif` — Direct Normal Irradiation (kWh/m²/year). Critical for Concentrated Solar Power (CSP) and high-efficiency dual-axis tracking systems.
  3. `PVOUT.tif` — Specific Photovoltaic Power Output (kWh/kWp/year). Modeled system output accounting for temperature derating and spectral losses.
  4. `TEMP.tif` — Long-term Annual Air Temperature (°C at 2m above ground). Used for PV cell thermal efficiency penalties.
  5. `OPTA.tif` — Optimal tilt angle of fixed-mount PV modules (degrees from horizontal facing true south).

---

## 2. Surface Telemetry Station Records (IMD / WRIS / State Hydrology Project)

Aggregated from 608 surface telemetry and manual observatories across Maharashtra covering over 333,000 hourly timestamps:

| Filename | Variable | Measurement Unit | Records / Span | Primary Usage |
| :--- | :--- | :---: | :---: | :--- |
| `wind_speed_tel_hr_maharashtra_sw_mh_2026_2030.csv` | Wind Speed | m/s | 333,033 rows | Baseline near-surface wind speed, diurnal velocity cycles |
| `rainfall_tel_hr_maharashtra_sw_mh_2021_2025.csv` | Rainfall | mm/hr | Multi-year telemetry | Monsoon intensity, access road flood risk, seasonal soiling wash |
| `rainfall_tel_hr_maharashtra-sw_mh_1991_2020.csv` | Rainfall (Historical) | mm/hr | 30-year climatology | Long-term baseline precipitation normals |
| `humid_tele_hr_maharashtra-sw_mh_2024_2024.csv` | Relative Humidity | % | 500k+ hourly records | Atmospheric attenuation, module soiling cementation risk |
| `humid_tele_hr_maharashtra-sw_mh_2021_2021.csv` to `2026_2030.csv` | Relative Humidity | % | Multi-year telemetry | Seasonal humidity trends across Western Ghats vs Vidarbha |
| `sunshine_duration_manual_hr_maharashtra-sw_mh_1991_2020.csv` | Sunshine Duration | Minutes / hr | 300k+ records | Daily sunshine hours validation against GHI satellite rasters |
| `solar_rediation_tel_hr_maharashtra-sw_mh_1991_2020.csv` | Surface Solar Radiation | W/m² | Historical telemetry | Ground-truth calibration for GSA raster pixels |
| `temprature_tele_hr_maharashtra-sw_mh_1961_2020.csv` | Ambient Temperature | °C | 60-year climatology | Extreme heat derating factors ($P_{loss} \approx -0.4\% / ^\circ\text{C}$) |

---

## 3. Infrastructure & Terrain Calibration Data (MSETCL, NIWE & ISRO Bhuvan)

- **High-Voltage Grid Substation Proximity**: Distance in km to the nearest 400 kV / 220 kV / 132 kV MahaTransco (MSETCL) transmission substation.
- **Wasteland / Non-Agricultural Land Availability**: ISRO Bhuvan Land Use / Land Cover (LULC) wasteland percentage (barren scrub, stony terrain, unculturable land).
- **Digital Elevation Model (DEM) Slope**: NASA SRTM 30m terrain slope (degrees), where $<5^\circ$ is optimal for solar tracking and utility arrays.
- **Wind Power Density at 100m Hub-Height**: National Institute of Wind Energy (NIWE) 100m hub-height WPD (W/m²), accounting for Weibull wind shear and Rayleigh atmospheric density.
- **Operational Plant Benchmarking**: Ministry of New and Renewable Energy (MNRE) and OpenStreetMap (OSM) operational utility installations (e.g., Chalkewadi 950 MW wind, Brahmanwel 528 MW wind, Sakri 125 MW solar).

---

## 4. Pipeline Execution

The raw files are automatically ingested and preprocessed by:
```bash
python scripts/01_data_pipeline.py
```
This script handles coordinate reprojection, spatial centroid raster sampling, outlier truncation, administrative name reconciliation (e.g., Ahmadnagar $\rightarrow$ Ahilyanagar), and produces `data/processed/district_features.csv`.
