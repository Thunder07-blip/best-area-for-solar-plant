# Maharashtra Renewable Energy Site Finder (Solar & Wind)

A data-driven spatial Multi-Criteria Decision Making (MCDM) and machine learning pipeline for utility-scale solar, wind, and hybrid renewable energy site selection across 34 districts of Maharashtra, India.

---

## 1. Project Architecture

```
Raw Datasets (GeoTIFF Rasters, WRIS Telemetry CSVs, Grid Substation & Wasteland Data)
                                  │
                                  ▼
      [Step 1] Ingestion, Cleaning & Geospatial Sampling (01_data_pipeline.py)
               • High-res raster sampling (GHI, DNI, PVOUT, TEMP, OPTA from World Bank GSA)
               • Telemetry aggregation (333k+ wind observations, rainfall, humidity, sunshine)
               • District name reconciliation (e.g. Ahmadnagar ➔ Ahilyanagar)
               • Output: data/processed/district_features.csv (34 rows × 22 features)
                                  │
                                  ▼
      [Step 2] MCDM Scoring Engine with AHP (02_scoring_engine.py)
               • Saaty's Analytic Hierarchy Process (AHP) with Consistency Ratio (CR < 0.01)
               • Min-Max normalization (Benefit: GHI, WPD, Land; Cost: Grid Dist, Slope)
               • Composite 0–100 Scores for Solar, Wind, and Hybrid
               • Output: data/processed/results.csv
                                  │
                                  ▼
      [Step 3] Empirical Validation against Operational Plants (03_validation_ml.py)
               • Ground truth proxy from operational solar & wind plants in Maharashtra
               • Random Forest Classifier with Leave-One-Out Cross-Validation (LOOCV)
               • Spearman Rank-Order Correlation: Solar (ρ = 0.83), Wind (ρ = 0.76)
               • Direct AHP ROC AUC: 0.9028 (Solar), 0.9524 (Wind)
               • Output: data/processed/validation_metrics.json
                                  │
                                  ▼
      [Step 4] Robustness Assessment & Siting Tiers (04_sensitivity_analysis.py)
               • Monte Carlo sensitivity analysis (1,000 trials, ±20% weight perturbation)
               • K-Means clustering into 3 strategic investment tiers (Prime, Secondary, Constrained)
               • Output: data/processed/sensitivity_and_clusters.csv
                                  │
                                  ▼
      [Step 5] Standalone Interactive Decision Dashboard (Maharashtra Site Finder.html)
               • Zero-dependency HTML5/SVG tile layout
               • Real-time AHP weight sliders, Solar/Wind/Hybrid mode toggle
               • District detail cards with physical metrics & operational capacity
```

---

## 2. Key Findings & Ranked Siting Candidates

### Top 5 Solar Siting Districts
| Rank | District | Solar Score (/100) | GHI (kWh/m²/day) | Barren Land (%) | Substation Dist (km) | Monte Carlo Top-5 Stability |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Ahilyanagar** (Ahmednagar) | **89.3** | 5.39 | 25.0% | 10.0 km | **100.0%** |
| **2** | **Solapur** | **81.6** | 5.29 | 30.0% | 12.0 km | **100.0%** |
| **3** | **Pune** | **81.3** | 5.41 | 12.0% | 6.0 km | **100.0%** |
| **4** | **Beed** | **80.4** | 5.32 | 28.0% | 16.0 km | **100.0%** |
| **5** | **Dharashiv** (Osmanabad) | **77.4** | 5.31 | 27.0% | 18.0 km | **81.8%** |

### Top 5 Wind Siting Districts
| Rank | District | Wind Score (/100) | Wind Power Density (W/m²) | Substation Dist (km) | Existing Wind MW | Monte Carlo Top-5 Stability |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Sangli** (Jath) | **78.3** | 330 W/m² | 14.0 km | 480 MW | **100.0%** |
| **2** | **Ahilyanagar** (Supa) | **76.4** | 280 W/m² | 10.0 km | 420 MW | **100.0%** |
| **3** | **Solapur** | **75.9** | 240 W/m² | 12.0 km | 110 MW | **100.0%** |
| **4** | **Satara** (Chalkewadi) | **72.8** | 380 W/m² | 15.0 km | 950 MW | **95.7%** |
| **5** | **Dhule** (Brahmanwel) | **68.6** | 290 W/m² | 18.0 km | 528 MW | **67.6%** |

*Note: Ahilyanagar and Solapur rank in the Top 3 for both solar and wind, making them Maharashtra's foremost candidate hubs for **Hybrid Renewable Energy Parks**.*

---

## 3. Data Sources & Feature Engineering

1. **Global Solar Atlas v2 (World Bank / ESMAP)**:
   - High-resolution GIS GeoTIFF rasters (`GHI.tif`, `DNI.tif`, `PVOUT.tif`, `TEMP.tif`, `OPTA.tif`).
   - Native spatial resolution: ~250 m ($12800 \times 12800$ grid).
   - Sampled across station clusters and zonal centroid bounds for all 34 districts.
2. **Surface Telemetry Networks (WRIS / IMD / State Hydrology Project)**:
   - Over 333,000 hourly telemetry observations for wind speed, rainfall, relative humidity, and sunshine duration across 608 Maharashtra stations.
3. **National Institute of Wind Energy (NIWE) & Global Wind Atlas**:
   - Calibrated 100m hub-height Wind Power Density ($W/m^2$) and mean wind speeds ($m/s$).
4. **MahaTransco (MSETCL) High-Voltage Grid Network**:
   - Proximity to nearest 400 kV, 220 kV, or 132 kV substation ($km$).
5. **ISRO Bhuvan Wasteland Atlas of India**:
   - Non-agricultural, non-forested barren/stony wasteland percentage (%) available for large-scale installation.
6. **SRTM Digital Elevation Model (NASA)**:
   - Mean terrain slope (degrees), where flatter terrain ($\le 3-5^\circ$) reduces civil excavation costs.

---

## 4. Methodology & Mathematical Formulations

### A. Analytic Hierarchy Process (AHP)
To eliminate arbitrary weight selection, pairwise comparison matrices $A = [a_{ij}]$ were constructed based on renewable energy engineering standards. The criteria priority vector $w$ was calculated using the geometric mean approximation:

$$w_i = \frac{\left( \prod_{j=1}^n a_{ij} \right)^{1/n}}{\sum_{k=1}^n \left( \prod_{j=1}^n a_{kj} \right)^{1/n}}$$

Consistency was validated via the Consistency Index ($CI$) and Consistency Ratio ($CR$):

$$CI = \frac{\lambda_{\max} - n}{n - 1}, \quad CR = \frac{CI}{RI}$$

- **Solar Criteria Weights**: Resource ($44.8\%$), Grid Proximity ($22.4\%$), Barren Land ($18.1\%$), Terrain Slope ($14.8\%$)
  - $\lambda_{\max} = 4.0002$, $CI = 0.0001$, $CR = 0.0001 \ll 0.10$ (**Strictly Consistent**)
- **Wind Criteria Weights**: Resource ($43.6\%$), Grid Proximity ($20.2\%$), Barren Land ($17.6\%$), Terrain Slope ($18.6\%$)
  - $\lambda_{\max} = 4.0155$, $CI = 0.0052$, $CR = 0.0057 \ll 0.10$ (**Strictly Consistent**)

### B. Min-Max Normalization
- **Benefit Criteria** (Higher is better: GHI, Wind Power Density, Barren Land %):
  $$x'_{ij} = \frac{x_{ij} - x_{\min, j}}{x_{\max, j} - x_{\min, j}}$$
- **Cost Criteria** (Lower is better: Substation Distance, Terrain Slope):
  $$x'_{ij} = \frac{x_{\max, j} - x_{ij}}{x_{\max, j} - x_{\min, j}}$$

### C. Composite Siting Score
$$S_i = 100 \times \sum_{j=1}^m w_j \cdot x'_{ij}$$

---

## 5. Validation Results

Empirical validation was performed by testing whether model suitability scores accurately separate districts that have operational utility-scale plants ($\ge 50$ MW) from those that do not:

- **Solar Validation**:
  - Direct AHP Siting Score ROC AUC: **0.9028**
  - Random Forest Classifier (LOOCV) ROC AUC: **0.8889**
  - Average Precision (PR AUC): **0.9098**
  - Spearman Rank-Order Correlation: $\rho = 0.8292$ ($p = 1.37 \times 10^{-9}$)
- **Wind Validation**:
  - Direct AHP Siting Score ROC AUC: **0.9524**
  - Random Forest Classifier (LOOCV) ROC AUC: **0.9524**
  - Average Precision (PR AUC): **0.9351**
  - Spearman Rank-Order Correlation: $\rho = 0.7632$ ($p = 1.54 \times 10^{-7}$)

---

## 6. Engineering Problems Encountered & Exact Solutions

| # | Challenge Encountered | Practical Consequence | Technical Solution Implemented |
|---|---|---|---|
| 1 | **Raster Resolution Mismatch** | GHI/DNI are $12800 \times 12800$ grids, PVOUT/TEMP are $3840 \times 3840$, OPTA is $960 \times 960$. | Standardized spatial transform function mapping WGS84 $(\text{lat}, \text{lon})$ directly to pixel offsets across arbitrary grid resolutions. |
| 2 | **Administrative Name Variations** | State government CSVs used historical names (`Ahmadnagar`, `Aurangabad`, `Osmanabad`, `Buldana`, `Sindudurg`) which dropped rows during table joins. | Built a deterministic alias dictionary (`clean_district_name`) reconciling all spellings to standard names (`Ahilyanagar`, `Chh. Sambhajinagar`, `Dharashiv`). |
| 3 | **Raw Telemetry Outliers** | Ground stations had invalid readings (negative rainfall $-485$ mm, humidity $> 900\%$, solar radiation spike $> 6,500$ W/m²). | Applied domain boundary filtering ($0 \le \text{humidity} \le 100\%$, $0 \le \text{rain} \le 300$ mm/hr) and robust quantile aggregations. |
| 4 | **Arbitrary Weight Criticism** | Stakeholders and examiners challenge manually assigned factor weights ("Why 40%?"). | Formalized pairwise matrices through Saaty's AHP and verified $CR < 0.01$. Added Monte Carlo $\pm 20\%$ perturbation over 1,000 runs to prove rank stability. |
| 5 | **Class Imbalance in Ground Truth** | Only 13 of 34 districts have $\ge 50$ MW wind capacity; accuracy metric would be deceptive. | Evaluated with ROC AUC, PR AUC, and balanced Random Forest weights (`class_weight='balanced'`). |
| 6 | **Deployment / Offline Portability** | External tile servers or backends fail on isolated presentation laptops. | Designed `Maharashtra Site Finder.html` as a zero-dependency self-contained dashboard embedding real data and an SVG tile layout. |

---

## 7. Project Directory Structure

```
D:\cp-pe\
│
├── data\
│   ├── raw\                              # Archive of raw telemetry CSVs & GSA GeoTIFF archive
│   │   └── README.md                     # Raw datasets inventory, variables, and data provenance
│   └── processed\
│       ├── district_features.csv         # Clean multi-source feature table (34 districts x 22 cols)
│       ├── taluka_features.csv           # Granular sub-district feature table (150 talukas)
│       ├── results.csv                   # Siting scores, ranks, and factor breakdown
│       ├── validation_metrics.json       # ROC AUC, PR AUC, Spearman rho metrics
│       ├── sensitivity_and_clusters.csv  # Monte Carlo stability stats and K-Means tiers
│       └── sensitivity_summary.json      # Sensitivity summary metrics
│
├── scripts\
│   ├── 01_data_pipeline.py               # Ingestion, cleaning, raster sampling
│   ├── 02_scoring_engine.py              # AHP weight computation & normalization
│   ├── 03_validation_ml.py               # Empirical validation (RF, ROC AUC, Spearman)
│   ├── 04_sensitivity_analysis.py        # Monte Carlo perturbation & K-Means clustering
│   └── run_all.py                        # Master pipeline execution script
│
├── docs\
│   └── index.html                        # Comprehensive Technical Documentation & Problem Solutions
│
├── index.html                            # Primary 5-Tab Interactive Siting Studio & Decision Dashboard
├── slides.html                           # 5-Minute Presentation Slide Deck (Fullscreen, Cues & Timer)
├── presentation.html                     # Redirect / alias to slides.html
├── docs.html                             # Redirect / alias to docs/index.html
├── Maharashtra Site Finder.html          # Lightweight standalone alternative dashboard
├── presentation_guide.md                 # Slide-by-slide 5-minute presentation talk-track
├── plan.txt                              # Original project brief and MVP milestones
└── README.md                             # Comprehensive technical documentation
```

---

## 8. How to Run & Present

### Pipeline Execution
- **Prerequisites**: Python 3.10+ with `pandas`, `numpy`, `scipy`, `scikit-learn`, `pillow`
- **Execute complete pipeline**:
  ```bash
  python scripts/run_all.py
  ```

### Interactive Decision Dashboard & Siting Studio
- Simply open **`index.html`** in any web browser (Chrome, Edge, Firefox, Safari).
- Features 5 comprehensive tabs:
  1. **🗺️ Siting Studio**: Real-time AHP sliders, interactive SVG map, district/taluka detail cards.
  2. **⚙️ Engineering Solutions**: Interactive click-to-extend problem-solution cards.
  3. **📐 AHP & Math**: Saaty pairwise comparison matrices and consistency ratio formulas.
  4. **🔬 ML Validation**: Random Forest ROC AUC curves and Spearman rank correlations.
  5. **📊 Sensitivity & Tiers**: Monte Carlo simulation distribution and K-Means strategic tiers.

### Slide Deck Presentation & 5-Minute Rehearsal
- Open **`slides.html`** in any web browser.
- **Presenter Keyboard Shortcuts**:
  - `▶` / `Space` / `PageDown`: Next slide
  - `◀` / `PageUp`: Previous slide
  - `P`: Toggle speaker cue card notes
  - `F`: Toggle full-screen presentation mode
  - `D`: Jump directly to the live dashboard
  - `1` – `6`: Direct jump to any slide
  - Built-in **5:00 countdown timer** at bottom right for timed presentation rehearsal.
  - Press `Ctrl + P` to export all slides directly to a PDF slide deck.

---

## 9. Stated Limitations & Future Scope

1. **Resolution Scale**: Current analysis operates at the district level. While appropriate for macro-level state energy policy and transmission allocation, actual project development requires taluka/village-level land parcel surveys.
2. **Local Grid Transmission Capacity**: Distance to substation serves as a proximity proxy; however, actual substation spare transformer capacity (MVA headroom) must be verified with MahaTransco prior to commercial commissioning.
3. **Dynamic Seasonal Constraints**: Telemetry data captures seasonal variation, but monsoon soiling and cloud attenuation could be further refined with multi-year satellite time-series.
