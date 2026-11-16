"""
run_all.py
==========
Master end-to-end runner for Maharashtra Renewable Energy Site Siting Pipeline.

Executes:
1. scripts/01_data_pipeline.py       -> data/processed/district_features.csv
2. scripts/02_scoring_engine.py      -> data/processed/results.csv
3. scripts/03_validation_ml.py       -> data/processed/validation_metrics.json
4. scripts/04_sensitivity_analysis.py -> data/processed/sensitivity_and_clusters.csv

Author: Antigravity Agent
Project: Maharashtra Renewable Energy Site Finder
"""

import os
import sys
import subprocess
import time

def run_step(step_name, script_path):
    print("\n" + "=" * 70)
    print(f"RUNNING: {step_name} ({os.path.basename(script_path)})")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[COMPLETED] {step_name} in {elapsed:.1f}s")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, "scripts")

    print("\n" + "#" * 70)
    print("  MAHARASHTRA RENEWABLE ENERGY SITE SUITABILITY PIPELINE")
    print("#" * 70)

    run_step("MVP 1: Data Pipeline & Spatial Feature Extraction", os.path.join(scripts_dir, "01_data_pipeline.py"))
    run_step("MVP 2: AHP Scoring Engine & District Ranking", os.path.join(scripts_dir, "02_scoring_engine.py"))
    run_step("MVP 3: Machine Learning & Empirical Validation", os.path.join(scripts_dir, "03_validation_ml.py"))
    run_step("MVP 4: Sensitivity Analysis & Cluster Tiers", os.path.join(scripts_dir, "04_sensitivity_analysis.py"))

    print("\n" + "#" * 70)
    print("  ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("  - Feature table: data/processed/district_features.csv")
    print("  - Siting results: data/processed/results.csv")
    print("  - Validation metrics: data/processed/validation_metrics.json")
    print("  - Sensitivity & clusters: data/processed/sensitivity_and_clusters.csv")
    print("  - Interactive Dashboard: Maharashtra Site Finder.html")
    print("#" * 70 + "\n")
