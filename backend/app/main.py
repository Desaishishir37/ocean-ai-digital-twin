"""
Main FastAPI Application Entrypoint for Ocean-AI Digital Twin.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as ocean_router

app = FastAPI(
    title="Ocean-AI: 3D Subsurface Reconstruction & Digital Twin",
    description="Near-real-time physics-informed subsurface ocean temperature reconstruction and What-If simulator for the North Indian Ocean.",
    version="1.0.0"
)

# Enable CORS for local and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(ocean_router)

# Mount frontend directory if it exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    vendor_dir = os.path.join(frontend_dir, "static", "vendor")
    if os.path.exists(vendor_dir):
        app.mount("/vendor", StaticFiles(directory=vendor_dir), name="vendor")

@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, stale-while-revalidate=86400",
                "Pragma": "no-cache"
            }
        )
    return {
        "status": "online",
        "service": "Ocean-AI 3D Reconstruction API",
        "docs_url": "/docs",
        "endpoints": {
            "status": "/api/v1/ocean/status",
            "metadata": "/api/v1/ocean/metadata",
            "depth_slice": "/api/v1/ocean/depth-slice?depth=100",
            "profile": "/api/v1/ocean/profile?lat=12.0&lon=85.0",
            "transect": "/api/v1/ocean/transect?lat=12.0",
            "validation": "/api/v1/ocean/validation",
            "simulate": "/api/v1/ocean/simulate"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
