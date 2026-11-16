# 5-Minute Presentation Guide & Slide Deck Outline
## Maharashtra Renewable Energy Site Finder (Solar & Wind)

This guide provides the exact numbers, slides, and talk track to rehearse and present the project in 5 minutes.

---

### Slide 1: Title & The Core Problem (0:00 - 0:45)
- **Slide Title**: Spatial Decision Support System for Utility-Scale Renewable Energy Siting in Maharashtra
- **Key Talking Points**:
  - *"Government solar and wind projects require substantial capital expenditure. Choosing the wrong district leads to transmission curtailment, land acquisition delays, or underperforming capacity factors."*
  - *"Our goal: Build an end-to-end, reproducible spatial decision pipeline and an interactive tool that identifies the most viable districts for solar, wind, and co-located hybrid renewable projects across Maharashtra."*
  - *"Key achievement: Built on real GIS rasters and meteorological telemetry, mathematically verified with AHP, validated against operational plants, with zero arbitrary weights."*

---

### Slide 2: End-to-End System Architecture (0:45 - 1:30)
- **Slide Content**: Architecture flow diagram
```
Raw GIS & Telemetry (GSA, NIWE, WRIS, ISRO)
                       │
                       ▼
         [1] Automated Data Pipeline
      (Raster sampling, outlier filtering, name reconciliation)
                       │
                       ▼
         [2] MCDM Scoring Engine
      (Saaty AHP, Consistency Ratio CR < 0.01, Min-Max Normalization)
                       │
                       ▼
         [3] Empirical & Machine Learning Validation
      (Random Forest LOOCV, ROC AUC = 0.9028 / 0.9524, Spearman rho = 0.83)
                       │
                       ▼
         [4] Sensitivity Analysis & Clustering
      (Monte Carlo 1,000 trials, K-Means Investment Tiers)
                       │
                       ▼
         [5] Standalone Interactive Dashboard
      (Zero-dependency HTML5/SVG client-side tool)
```
- **Key Talking Points**:
  - *"The architecture follows a strict separation of concerns: heavy Python data science writes clean static datasets, and a zero-dependency HTML/SVG dashboard displays the results with real-time recalculation."*
  - *"No backend failure points, no live map server dependencies—it runs offline anywhere."*

---

### Slide 3: Data Ingestion & Engineering Challenges Solved (1:30 - 2:30)
- **Data Highlights**:
  - **Solar**: World Bank Global Solar Atlas v2 high-resolution GeoTIFF rasters (GHI, DNI, PVOUT, TEMP, OPTA) at ~250m resolution.
  - **Wind**: 333,033 ground telemetry hourly wind speed observations combined with NIWE 100m hub-height Wind Power Density.
  - **Grid & Land**: MSETCL high-voltage substation network (km) and ISRO Bhuvan non-agricultural barren wasteland (%).
- **Engineering Problems Solved**:
  1. *Raster Dimension Disparity*: Handled multi-resolution grids ($12800\times12800$ down to $960\times960$) with coordinate projection sampling.
  2. *Administrative Name Mismatches*: Reconciled Maharashtra's administrative name updates (Ahmadnagar $\rightarrow$ Ahilyanagar, Aurangabad $\rightarrow$ Chh. Sambhajinagar, Osmanabad $\rightarrow$ Dharashiv) to eliminate silent merge drops.
  3. *Sensor Noise*: Filtered negative telemetry rainfall and out-of-bound humidity spikes using strict physical bounds.

---

### Slide 4: Methodology — Why These Weights? (2:30 - 3:15)
- **Slide Content**:
  - **Solar AHP Weights**: Resource ($44.8\%$), Grid Proximity ($22.4\%$), Barren Land ($18.1\%$), Terrain Flatness ($14.8\%$)
    - $\lambda_{\max} = 4.0002$, Consistency Ratio **$CR = 0.0001 \ll 0.10$**
  - **Wind AHP Weights**: Resource ($43.6\%$), Grid Proximity ($20.2\%$), Barren Land ($17.6\%$), Terrain Slope ($18.6\%$)
    - $\lambda_{\max} = 4.0155$, Consistency Ratio **$CR = 0.0057 \ll 0.10$**
- **Key Talking Points**:
  - *"If an examiner asks 'Why 45% for resource and 22% for grid?', the answer is mathematically grounded in Saaty’s Analytic Hierarchy Process with a Consistency Ratio below 0.01."*
  - *"Furthermore, in our 1,000-run Monte Carlo simulation with $\pm 20\%$ weight perturbation, our Top 4 districts remained in the Top 5 in 100% of all trials."*

---

### Slide 5: Real Siting Results & Top Candidates (3:15 - 4:00)
- **Top Candidates Table**:
  - **Top Solar**:
    1. **Ahilyanagar (Score 89.3)** — 5.39 kWh/m²/day, 25% barren land, 10 km to grid
    2. **Solapur (Score 81.6)** — 5.29 kWh/m²/day, 30% barren land, 12 km to grid
    3. **Pune (Score 81.3)** — 5.41 kWh/m²/day, 6 km to grid
    4. **Beed (Score 80.4)** — 5.32 kWh/m²/day, 28% barren land
    5. **Dharashiv (Score 77.4)** — 5.31 kWh/m²/day, 27% barren land
  - **Top Wind**:
    1. **Sangli (Score 78.3)** — 330 W/m², 14 km to grid, home to Jath wind parks
    2. **Ahilyanagar (Score 76.4)** — 280 W/m², 10 km to grid, Supa wind corridor
    3. **Solapur (Score 75.9)** — 240 W/m², 12 km to grid
    4. **Satara (Score 72.8)** — 380 W/m², Chalkewadi / Asia's largest wind park
    5. **Dhule (Score 68.6)** — 290 W/m², Brahmanwel wind hub
  - **Strategic Hybrid Hub**: Ahilyanagar and Solapur rank in the top 3 for *both* solar and wind, making them prime targets for shared grid evacuation co-located hybrid parks.

---

### Slide 6: Model Validation & Limitations (4:00 - 5:00)
- **Validation Metrics**:
  - **Solar**: Direct AHP ROC AUC = **0.9028**, Random Forest LOOCV AUC = **0.8889**, Spearman $\rho = \mathbf{0.8292}$ ($p = 1.37 \times 10^{-9}$)
  - **Wind**: Direct AHP ROC AUC = **0.9524**, Random Forest LOOCV AUC = **0.9524**, Spearman $\rho = \mathbf{0.7632}$ ($p = 1.54 \times 10^{-7}$)
- **K-Means Investment Tiers**:
  - **Tier 1 (Prime Development Zone)**: 13 districts (Western Maharashtra & Marathwada plateau)
  - **Tier 2 (Secondary / Grid Targeted)**: 16 districts (Vidarbha & Northern Maharashtra)
  - **Tier 3 (Constrained)**: 5 districts (Konkan coastal & Gadchiroli forests)
- **Honest Limitations**:
  1. *District vs Parcel Scale*: Identifies macro districts; micro-siting requires taluka/village cadastral land surveys.
  2. *Substation Headroom*: Proximity is modeled; actual spare MVA transformer capacity must be verified with MahaTransco.
  3. *Monsoon Seasonal Attenuation*: Solar generation dips during June–September due to monsoon cloud cover.

---

### Summary Card for Q&A

| Question | Short Answer |
|---|---|
| **"Where did you get the data?"** | World Bank Global Solar Atlas v2 (GeoTIFFs), NIWE wind assessments, WRIS telemetry (333k hourly records), MSETCL grid network, and ISRO Bhuvan wasteland atlas. |
| **"Why did you choose AHP?"** | Eliminates subjective weight assignment through structured pairwise comparisons, validated with Saaty's Consistency Ratio ($CR < 0.01$). |
| **"How do you know the model works?"** | Validated against operational utility-scale plants in Maharashtra; achieved ROC AUC of 0.90 for solar and 0.95 for wind, with Spearman rank correlation above 0.76. |
| **"How stable are the rankings?"** | Tested under 1,000 Monte Carlo runs with $\pm 20\%$ weight perturbation; the top 4 candidates have a 100% retention probability in the top tier. |
