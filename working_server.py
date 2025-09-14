#!/usr/bin/env python3
# working_server.py - Reliable FastAPI server for serving points data
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="YC Startup Map API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache points data
_points_cache = None

def load_points_data():
    """Load points data from JSON file"""
    global _points_cache
    if _points_cache is None:
        points_path = "out_full/points.json"
        try:
            with open(points_path, "r", encoding="utf-8") as f:
                _points_cache = json.load(f)
            print(f"✅ Loaded {len(_points_cache)} points successfully")
        except Exception as e:
            print(f"❌ Error loading points: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load points: {e}")
    return _points_cache

@app.on_event("startup")
def startup_event():
    """Load data on startup"""
    load_points_data()
    print("🚀 Server started successfully!")

@app.get("/")
def root():
    return {"message": "YC Startup Map API", "status": "running"}

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        data = load_points_data()
        return {
            "status": "healthy",
            "points_count": len(data),
            "sample_point_keys": list(data[0].keys()) if data else [],
            "points_with_logos": sum(1 for p in data if p.get('logo_url'))
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/points")
def get_points():
    """Get all startup points data"""
    try:
        data = load_points_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=False)