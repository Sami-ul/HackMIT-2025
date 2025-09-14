# simple_server.py - Simplified FastAPI server that just serves the points data
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Startup Map API (Simple)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load points data once at startup
POINTS_PATH = os.path.join("out_full", "points.json")
_points_data = None

def load_points():
    global _points_data
    if _points_data is None:
        try:
            print(f"Loading points from: {POINTS_PATH}")
            with open(POINTS_PATH, "r", encoding="utf-8") as f:
                _points_data = json.load(f)
            print(f"Loaded {len(_points_data)} points successfully")
        except Exception as e:
            print(f"Error loading points: {e}")
            raise
    return _points_data

@app.get("/")
def root():
    return {"message": "Startup Map Simple API"}

@app.get("/points")
def points():
    """Return all points data"""
    try:
        data = load_points()
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "points_loaded": _points_data is not None}