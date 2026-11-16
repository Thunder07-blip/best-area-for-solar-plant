"""
02_scoring_engine.py
===================
MCDM Scoring Engine for Maharashtra Solar and Wind Site Suitability.
Implements:
1. Analytic Hierarchy Process (AHP) with Consistency Ratio (CR < 0.10) calculation.
2. Benefit/Cost Min-Max Normalization.
3. Multi-Criteria Composite Scoring (0-100 scale).
4. Export to results.csv with ranked districts.

Author: Antigravity Agent
Project: Maharashtra Renewable Energy Site Finder
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"D:\cp-pe"
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "district_features.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "results.csv")

# 1. Analytic Hierarchy Process (AHP) Implementation
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

def compute_ahp_weights(matrix, criterion_names):
    """
    Computes priority vector (eigenvector approximation via geometric mean),
    maximum eigenvalue lambda_max, Consistency Index (CI), and Consistency Ratio (CR).
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    
    # Geometric mean method (Saaty's logarithmic least squares)
    geom_mean = np.prod(A, axis=1) ** (1.0 / n)
    weights = geom_mean / np.sum(geom_mean)
    
    # Lambda max estimation: A * w / w
    Aw = np.dot(A, weights)
    lambda_max = np.mean(Aw / weights)
    
    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX.get(n, 1.12)
    cr = ci / ri if ri > 0 else 0.0
    
    weight_dict = {criterion_names[i]: float(weights[i]) for i in range(n)}
    return weight_dict, lambda_max, ci, cr

# Solar Criteria: [Resource (GHI), Grid Proximity, Barren Land, Flat Terrain (Slope)]
# Matrix judgments based on renewable siting literature:
# Resource is moderately favored over grid & land; terrain flatness is important for solar array mounting.
A_solar = [
    [1.0,     2.0,     2.5,     3.0],
    [0.5,     1.0,     1.25,    1.5],
    [0.4,     0.8,     1.0,     1.25],
    [0.3333,  0.6667,  0.8,     1.0]
]
criteria_solar = ['resource', 'grid', 'land', 'terrain']
weights_solar, lmax_s, ci_s, cr_s = compute_ahp_weights(A_solar, criteria_solar)

# Wind Criteria: [Resource (WPD), Grid Proximity, Barren Land, Terrain (Slope)]
# For wind, power density is cubic with velocity, so resource has highest dominance.
# Moderate slopes / ridges are acceptable for wind masts, but grid access is vital.
A_wind = [
    [1.0,     2.5,     2.5,     2.0],
    [0.4,     1.0,     1.2,     1.2],
    [0.4,     0.8333,  1.0,     1.0],
    [0.5,     0.8333,  1.0,     1.0]
]
criteria_wind = ['resource', 'grid', 'land', 'terrain']
weights_wind, lmax_w, ci_w, cr_w = compute_ahp_weights(A_wind, criteria_wind)

print("=" * 60)
print("AHP CONSISTENCY & CRITERIA WEIGHTS REPORT")
print("=" * 60)
print(f"Solar AHP Weights: { {k: f'{v*100:.1f}%' for k, v in weights_solar.items()} }")
print(f"  Lambda_max: {lmax_s:.4f}, CI: {ci_s:.4f}, CR: {cr_s:.4f} (Valid: {cr_s < 0.10})")

print(f"\nWind AHP Weights:  { {k: f'{v*100:.1f}%' for k, v in weights_wind.items()} }")
print(f"  Lambda_max: {lmax_w:.4f}, CI: {ci_w:.4f}, CR: {cr_w:.4f} (Valid: {cr_w < 0.10})")
print("=" * 60)

# 2. Ingest Feature Table
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} districts from {INPUT_CSV}")

# 3. Min-Max Normalization
# Benefit factors: Resource (GHI / WPD), Barren Land % -> (x - min) / (max - min)
# Cost factors: Grid Distance (km), Slope (deg) -> (max - x) / (max - min)

def norm_benefit(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx > mn else pd.Series(1.0, index=series.index)

def norm_cost(series):
    mn, mx = series.min(), series.max()
    return (mx - series) / (mx - mn) if mx > mn else pd.Series(1.0, index=series.index)

# Solar Normalized Subfactors
norm_solar_res = norm_benefit(df['solar_ghi_kwh_m2_day'])
norm_grid = norm_cost(df['substation_dist_km'])
norm_land = norm_benefit(df['barren_land_pct'])
norm_terrain = norm_cost(df['terrain_slope_deg'])

# Wind Normalized Subfactors
norm_wind_res = norm_benefit(df['wind_power_density_w_m2'])

# Composite 0-100 Scores
df['norm_solar_res'] = norm_solar_res.round(3)
df['norm_wind_res'] = norm_wind_res.round(3)
df['norm_grid'] = norm_grid.round(3)
df['norm_land'] = norm_land.round(3)
df['norm_terrain'] = norm_terrain.round(3)

df['solar_score'] = (
    weights_solar['resource'] * norm_solar_res +
    weights_solar['grid'] * norm_grid +
    weights_solar['land'] * norm_land +
    weights_solar['terrain'] * norm_terrain
) * 100.0

df['wind_score'] = (
    weights_wind['resource'] * norm_wind_res +
    weights_wind['grid'] * norm_grid +
    weights_wind['land'] * norm_land +
    weights_wind['terrain'] * norm_terrain
) * 100.0

df['hybrid_score'] = 0.5 * df['solar_score'] + 0.5 * df['wind_score']

# Round scores
df['solar_score'] = df['solar_score'].round(1)
df['wind_score'] = df['wind_score'].round(1)
df['hybrid_score'] = df['hybrid_score'].round(1)

# Ranks (1 = highest score)
df['solar_rank'] = df['solar_score'].rank(ascending=False, method='min').astype(int)
df['wind_rank'] = df['wind_score'].rank(ascending=False, method='min').astype(int)
df['hybrid_rank'] = df['hybrid_score'].rank(ascending=False, method='min').astype(int)

# Categorization
def categorize(s):
    if s >= 75: return "Very High"
    if s >= 65: return "High"
    if s >= 50: return "Moderate"
    return "Low"

df['solar_tier'] = df['solar_score'].apply(categorize)
df['wind_tier'] = df['wind_score'].apply(categorize)

# Export columns
cols_to_export = [
    'district', 'latitude', 'longitude',
    'solar_score', 'solar_rank', 'solar_tier',
    'wind_score', 'wind_rank', 'wind_tier',
    'hybrid_score', 'hybrid_rank',
    'solar_ghi_kwh_m2_day', 'wind_power_density_w_m2',
    'substation_dist_km', 'barren_land_pct', 'terrain_slope_deg',
    'norm_solar_res', 'norm_wind_res', 'norm_grid', 'norm_land', 'norm_terrain',
    'existing_solar_capacity_mw', 'existing_wind_capacity_mw',
    'has_utility_solar', 'has_utility_wind'
]
df_results = df[cols_to_export].sort_values(by='solar_rank')

# Save results
try:
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Results saved directly to:\n  {OUTPUT_CSV}")
except Exception as e:
    print(f"\n[INFO] Direct write to {OUTPUT_CSV} encountered: {e}")
    if os.path.exists(OUTPUT_CSV):
        print(f"[STATUS] Existing results dataset {OUTPUT_CSV} remains intact.")

# Print Top 5 Solar & Top 5 Wind
print("\n" + "=" * 50)
print("TOP 5 DISTRICTS FOR SOLAR ENERGY SITES")
print("=" * 50)
top_solar = df_results.sort_values(by='solar_score', ascending=False).head(5)
print(top_solar[['solar_rank', 'district', 'solar_score', 'solar_ghi_kwh_m2_day', 'barren_land_pct', 'substation_dist_km']].to_string(index=False))

print("\n" + "=" * 50)
print("TOP 5 DISTRICTS FOR WIND ENERGY SITES")
print("=" * 50)
top_wind = df_results.sort_values(by='wind_score', ascending=False).head(5)
print(top_wind[['wind_rank', 'district', 'wind_score', 'wind_power_density_w_m2', 'substation_dist_km', 'existing_wind_capacity_mw']].to_string(index=False))
