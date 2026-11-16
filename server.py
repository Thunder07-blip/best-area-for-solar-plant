"""
server.py
=========
FastAPI application serving the Maharashtra Renewable Energy Site Finder
interactive dashboard, 5-minute presentation slides, and technical documentation.

Routes:
  /               -> Interactive Siting Studio & Dashboard (index.html)
  /slides         -> 5-Minute Presentation Slide Deck (slides.html)
  /docs           -> Technical Documentation & Engineering Solutions (docs/index.html)
  /api/health     -> Health check & server metadata
  /api/districts  -> District-level feature & ranking dataset (JSON)
  /api/talukas    -> Sub-district taluka dataset (JSON)
  /api/validation -> Machine learning validation metrics (JSON)
  /api/swagger    -> OpenAPI Swagger UI documentation
"""

import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Initialize FastAPI with custom OpenAPI doc URL so /docs serves our technical documentation
app = FastAPI(
    title="Maharashtra Renewable Energy Site Finder API",
    description="Spatial Decision Support System for Utility-Scale Solar and Wind Siting across Maharashtra",
    version="1.0.0",
    docs_url="/api/swagger",
    redoc_url="/api/redoc"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /data/processed as static directory for downloadable CSVs/JSONs
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# -------------------------------------------------------------
# HTML Presentation & Dashboard Routes
# -------------------------------------------------------------

@app.get("/", response_class=FileResponse, summary="Interactive Siting Studio & Heatmap")
@app.get("/index.html", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    index_file = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file, media_type="text/html")


@app.get("/slides", response_class=FileResponse, summary="5-Minute Presentation Slide Deck")
@app.get("/slides.html", response_class=FileResponse, include_in_schema=False)
@app.get("/presentation", response_class=FileResponse, include_in_schema=False)
@app.get("/presentation.html", response_class=FileResponse, include_in_schema=False)
async def serve_slides():
    slides_file = os.path.join(BASE_DIR, "slides.html")
    if not os.path.exists(slides_file):
        raise HTTPException(status_code=404, detail="slides.html not found")
    return FileResponse(slides_file, media_type="text/html")


@app.get("/docs", response_class=FileResponse, summary="Technical Documentation & Engineering Solutions")
@app.get("/docs/", response_class=FileResponse, include_in_schema=False)
@app.get("/docs/index.html", response_class=FileResponse, include_in_schema=False)
async def serve_docs():
    docs_file = os.path.join(BASE_DIR, "docs", "index.html")
    if not os.path.exists(docs_file):
        raise HTTPException(status_code=404, detail="docs/index.html not found")
    return FileResponse(docs_file, media_type="text/html")


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get("/api/health", summary="Service Health & Pipeline Status")
async def health_check():
    return {
        "status": "healthy",
        "service": "maharashtra-renewable-site-finder",
        "districts_count": 34,
        "talukas_count": 150,
        "validation_status": "verified (ROC AUC: 0.9028 Solar / 0.9524 Wind)"
    }


@app.get("/api/districts", summary="District-level Siting Rankings and Features")
async def get_districts():
    results_path = os.path.join(PROCESSED_DIR, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        return df.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="results.csv not found")


@app.get("/api/talukas", summary="Granular Sub-District Taluka Features (150 Talukas)")
async def get_talukas():
    taluka_path = os.path.join(PROCESSED_DIR, "taluka_features.csv")
    if os.path.exists(taluka_path):
        df = pd.read_csv(taluka_path)
        return df.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="taluka_features.csv not found")


@app.get("/api/validation", summary="Machine Learning Validation Metrics")
async def get_validation_metrics():
    metrics_path = os.path.join(PROCESSED_DIR, "validation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="validation_metrics.json not found")


@app.get("/api/sensitivity", summary="Monte Carlo Sensitivity & Clustering Metrics")
async def get_sensitivity_summary():
    sens_path = os.path.join(PROCESSED_DIR, "sensitivity_summary.json")
    if os.path.exists(sens_path):
        with open(sens_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="sensitivity_summary.json not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n" + "=" * 65)
    print(f"🚀 Maharashtra Renewable Energy Site Finder Server Running!")
    print(f"=" * 65)
    print(f"  • Interactive Siting Studio: http://localhost:{port}/")
    print(f"  • 5-Minute Slide Deck:      http://localhost:{port}/slides")
    print(f"  • Technical Documentation:   http://localhost:{port}/docs")
    print(f"  • District API Endpoint:     http://localhost:{port}/api/districts")
    print(f"  • OpenAPI Swagger UI:        http://localhost:{port}/api/swagger")
    print(f"=" * 65 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
