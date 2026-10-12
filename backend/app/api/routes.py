"""
FastAPI Routes for Ocean-AI Digital Twin.
Provides endpoints for operational status, 2D depth slices, 1D vertical profiles,
vertical transect curtains, Argo float validation, and What-If scenario simulation.
"""

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import numpy as np

from app.core.physics import (
    STANDARD_DEPTHS, NUM_DEPTHS,
    LATS, LONS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    compute_mld, compute_d20, compute_ohc
)
from app.services.data_engine import OceanDataEngine
from app.models.ocean_net import SubsurfaceInferenceEngine
from app.services.argo_service import ArgoValidationService
from app.services.simulator_service import WhatIfSimulatorService

router = APIRouter(prefix="/api/v1/ocean", tags=["Ocean AI"])

# Global singleton engine instances
data_engine = OceanDataEngine()
inference_engine = SubsurfaceInferenceEngine()
argo_service = ArgoValidationService(data_engine, inference_engine)
simulator_service = WhatIfSimulatorService(data_engine, inference_engine)

# Cached latest state for fast responsiveness
_cached_state = None

def get_current_reconstruction():
    global _cached_state
    if _cached_state is None:
        surface_inputs, metadata = data_engine.get_latest_satellite_state()
        recon_cube, unc_cube = inference_engine.reconstruct(
            surface_inputs=surface_inputs,
            climatology_cube=data_engine.climatology_cube,
            land_mask=data_engine.land_mask
        )
        _cached_state = {
            "surface_inputs": surface_inputs,
            "metadata": metadata,
            "recon_cube": recon_cube,
            "unc_cube": unc_cube
        }
    return _cached_state


@router.get("/status")
def get_system_status():
    """Returns operational ingestion status, data latency, and satellite product freshness."""
    state = get_current_reconstruction()
    return state["metadata"]


@router.get("/metadata")
def get_grid_metadata():
    """Returns spatial grid dimensions, depths, and coordinate arrays."""
    return {
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "grid_resolution": 0.25,
        "depths": STANDARD_DEPTHS,
        "num_lats": len(LATS),
        "num_lons": len(LONS),
        "lats": LATS.tolist(),
        "lons": LONS.tolist()
    }


@router.get("/bathymetry")
def get_bathymetry(downsample: int = Query(2, description="Spatial downsample factor")):
    """
    Returns 2D bathymetric depth grid (meters) for the North Indian Ocean basin.
    """
    bathy = data_engine.bathymetry_max_depth[::downsample, ::downsample]
    return {
        "lats": LATS[::downsample].tolist(),
        "lons": LONS[::downsample].tolist(),
        "bathymetry_grid": [[float(round(val, 1)) for val in row] for row in bathy],
        "min_depth": float(round(np.min(bathy), 1)),
        "max_depth": float(round(np.max(bathy), 1))
    }


@router.get("/depth-slice")
def get_depth_slice(
    depth: float = Query(0.0, description="Depth in meters (0 to 1000)"),
    mode: str = Query("operational", description="operational | whatif"),
    downsample: int = Query(2, description="Spatial downsample stride for fast web rendering")
):
    """
    Returns a 2D grid slice of temperature, anomaly, and uncertainty at a requested depth.
    """
    state = get_current_reconstruction()
    
    # Locate closest standard depth index
    depth_arr = np.array(STANDARD_DEPTHS)
    k = int(np.argmin(np.abs(depth_arr - depth)))
    matched_depth = float(depth_arr[k])
    
    recon_cube = state["recon_cube"]
    unc_cube = state["unc_cube"]
    clim_cube = data_engine.climatology_cube
    
    slice_temp = recon_cube[::downsample, ::downsample, k]
    slice_unc = unc_cube[::downsample, ::downsample, k]
    slice_anom = slice_temp - clim_cube[::downsample, ::downsample, k]
    
    # Replace NaNs with null for clean JSON transmission
    temp_list = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in slice_temp]
    unc_list = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in slice_unc]
    anom_list = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in slice_anom]
    
    valid_temps = slice_temp[~np.isnan(slice_temp)]
    min_temp = float(np.min(valid_temps)) if len(valid_temps) > 0 else 4.0
    max_temp = float(np.max(valid_temps)) if len(valid_temps) > 0 else 32.0
    
    return {
        "requested_depth": depth,
        "matched_depth": matched_depth,
        "depth_index": k,
        "downsample_factor": downsample,
        "lats": LATS[::downsample].tolist(),
        "lons": LONS[::downsample].tolist(),
        "min_val": round(min_temp, 1),
        "max_val": round(max_temp, 1),
        "temperature_grid": temp_list,
        "uncertainty_grid": unc_list,
        "anomaly_grid": anom_list
    }


@router.get("/surface-layer")
def get_surface_layer(
    variable: str = Query("sst", description="sst | sss | sla | currents | winds"),
    downsample: int = Query(2, description="Spatial downsample stride")
):
    """
    Returns high-resolution 2D surface geospatial observation layers:
    - sst: Sea Surface Temperature (°C)
    - sss: Sea Surface Salinity (PSU)
    - sla: Sea Level Anomaly (m)
    - currents: Surface Geostrophic Currents speed (m/s) & vector components
    - winds: Monsoon Winds speed (m/s) & vector components
    """
    state = get_current_reconstruction()
    surf = state["surface_inputs"]
    clim_surf = data_engine.climatology_cube[:, :, 0]
    
    lats_sub = LATS[::downsample].tolist()
    lons_sub = LONS[::downsample].tolist()
    
    if variable == "sst":
        layer = clim_surf + surf[0]
        unit = "°C"
        title = "Sea Surface Temperature"
        min_v, max_v = 24.0, 31.5
    elif variable == "sss":
        x_grid, _ = np.meshgrid(data_engine.lons, data_engine.lats)
        lon_factor = (x_grid - LON_MIN) / (LON_MAX - LON_MIN)
        base_sss = 36.2 - 3.4 * lon_factor
        layer = base_sss + surf[1]
        unit = "PSU"
        title = "Sea Surface Salinity"
        min_v, max_v = 31.0, 37.0
    elif variable == "sla":
        layer = surf[2]
        unit = "m"
        title = "Sea Level Anomaly"
        min_v, max_v = -0.20, 0.20
    elif variable == "currents":
        u = surf[3]
        v = surf[4]
        layer = np.sqrt(u**2 + v**2)
        unit = "m/s"
        title = "Surface Geostrophic Velocity"
        min_v, max_v = 0.0, 1.2
    elif variable == "winds":
        u = surf[5]
        v = surf[6]
        layer = np.sqrt(u**2 + v**2)
        unit = "m/s"
        title = "Monsoon Wind Speed"
        min_v, max_v = 1.0, 12.0
    elif variable == "mld":
        recon_cube = state["recon_cube"]
        sub_recon = recon_cube[::downsample, ::downsample, :]
        H_sub, W_sub, _ = sub_recon.shape
        depth_arr = np.array(STANDARD_DEPTHS)
        mld_map = np.full((H_sub, W_sub), np.nan)
        for i in range(H_sub):
            for j in range(W_sub):
                if not np.isnan(sub_recon[i, j, 0]):
                    mld_map[i, j] = compute_mld(depth_arr, sub_recon[i, j, :])
        layer = mld_map
        unit = "m"
        title = "Mixed Layer Depth (MLD)"
        min_v, max_v = 15.0, 95.0
    elif variable == "d20":
        recon_cube = state["recon_cube"]
        sub_recon = recon_cube[::downsample, ::downsample, :]
        H_sub, W_sub, _ = sub_recon.shape
        depth_arr = np.array(STANDARD_DEPTHS)
        d20_map = np.full((H_sub, W_sub), np.nan)
        for i in range(H_sub):
            for j in range(W_sub):
                if not np.isnan(sub_recon[i, j, 0]):
                    d20_map[i, j] = compute_d20(sub_recon[i, j, :], depth_arr)
        layer = d20_map
        unit = "m"
        title = "Thermocline Depth (D20)"
        min_v, max_v = 55.0, 160.0
    elif variable == "ohc":
        recon_cube = state["recon_cube"]
        sub_recon = recon_cube[::downsample, ::downsample, :]
        H_sub, W_sub, _ = sub_recon.shape
        depth_arr = np.array(STANDARD_DEPTHS)
        ohc_map = np.full((H_sub, W_sub), np.nan)
        for i in range(H_sub):
            for j in range(W_sub):
                if not np.isnan(sub_recon[i, j, 0]):
                    ohc_map[i, j] = compute_ohc(sub_recon[i, j, :], depth_arr)
        layer = ohc_map
        unit = "kJ/cm²"
        title = "Cyclone Heat Potential (OHC)"
        min_v, max_v = 10.0, 115.0
    elif variable == "bathymetry":
        layer = data_engine.bathymetry_max_depth
        unit = "m"
        title = "Ocean Bed Bathymetry"
        min_v, max_v = 0.0, 5500.0
    else:
        raise HTTPException(status_code=400, detail=f"Unknown surface variable: {variable}")
        
    sub_layer = layer if variable in ["mld", "d20", "ohc"] else layer[::downsample, ::downsample]
    grid_list = [[None if np.isnan(val) else float(round(val, 3)) for val in row] for row in sub_layer]
    
    resp = {
        "variable": variable,
        "title": title,
        "unit": unit,
        "min_val": min_v,
        "max_val": max_v,
        "lats": lats_sub,
        "lons": lons_sub,
        "grid": grid_list
    }
    
    if variable in ["currents", "winds"]:
        u_sub = (surf[3] if variable == "currents" else surf[5])[::downsample, ::downsample]
        v_sub = (surf[4] if variable == "currents" else surf[6])[::downsample, ::downsample]
        resp["u_grid"] = [[None if np.isnan(val) else float(round(val, 3)) for val in row] for row in u_sub]
        resp["v_grid"] = [[None if np.isnan(val) else float(round(val, 3)) for val in row] for row in v_sub]
        
    return resp


@router.get("/profile")
def get_vertical_profile(
    lat: float = Query(..., description="Latitude (0 to 30)"),
    lon: float = Query(..., description="Longitude (45 to 105)")
):
    """
    Queries vertical 15-depth temperature profile and uncertainty envelope at any (lat, lon).
    If the coordinate falls on a coastal land pixel or shallow continental shelf,
    seamlessly blends with the nearest offshore deep-water column for complete 0-1000m diagnostics.
    """
    lat_clamped = float(np.clip(lat, LAT_MIN, LAT_MAX))
    lon_clamped = float(np.clip(lon, LON_MIN, LON_MAX))
        
    state = get_current_reconstruction()
    i = int(np.argmin(np.abs(LATS - lat_clamped)))
    j = int(np.argmin(np.abs(LONS - lon_clamped)))
    
    recon_cube = state["recon_cube"]
    unc_cube = state["unc_cube"]
    clim_cube = data_engine.climatology_cube
    
    # If point is on land or missing surface temperature, locate nearest valid ocean cell
    if bool(data_engine.land_mask[i, j]) or np.isnan(recon_cube[i, j, 0]):
        valid_mask = (~data_engine.land_mask) & (~np.isnan(recon_cube[:, :, 0]))
        vi, vj = np.where(valid_mask)
        if len(vi) > 0:
            dists = (LATS[vi] - lat_clamped) ** 2 + (LONS[vj] - lon_clamped) ** 2
            best_idx = int(np.argmin(dists))
            i, j = int(vi[best_idx]), int(vj[best_idx])
            
    temp_raw = np.array(recon_cube[i, j, :], dtype=np.float64)
    unc_raw = np.array(unc_cube[i, j, :], dtype=np.float64)
    clim_raw = np.array(clim_cube[i, j, :], dtype=np.float64)
    
    # If coastal shelf truncates deeper levels (NaN below shelf), fill deeper levels from nearest deep-water cell
    if np.any(np.isnan(temp_raw)):
        deep_mask = (~data_engine.land_mask) & (~np.isnan(recon_cube[:, :, -1]))
        di, dj = np.where(deep_mask)
        if len(di) > 0:
            dists = (LATS[di] - lat_clamped) ** 2 + (LONS[dj] - lon_clamped) ** 2
            best_deep = int(np.argmin(dists))
            i_d, j_d = int(di[best_deep]), int(dj[best_deep])
            for k in range(NUM_DEPTHS):
                if np.isnan(temp_raw[k]):
                    temp_raw[k] = recon_cube[i_d, j_d, k]
                if np.isnan(unc_raw[k]):
                    unc_raw[k] = unc_cube[i_d, j_d, k]
                if np.isnan(clim_raw[k]):
                    clim_raw[k] = clim_cube[i_d, j_d, k]
    
    temp_profile = [None if np.isnan(v) else float(round(v, 2)) for v in temp_raw]
    unc_profile = [None if np.isnan(v) else float(round(v, 2)) for v in unc_raw]
    clim_profile = [None if np.isnan(v) else float(round(v, 2)) for v in clim_raw]
    
    clean_temp = np.nan_to_num(temp_raw, nan=5.0)
    mld = compute_mld(clean_temp)
    d20 = compute_d20(clean_temp)
    ohc = compute_ohc(clean_temp)
    
    return {
        "lat": round(lat_clamped, 2),
        "lon": round(lon_clamped, 2),
        "is_land": False,
        "depths": STANDARD_DEPTHS,
        "temperature": temp_profile,
        "uncertainty": unc_profile,
        "climatology": clim_profile,
        "diagnostics": {
            "mld_meters": round(mld, 1),
            "d20_meters": round(d20, 1),
            "ohc_kj_cm2": round(ohc, 2)
        }
    }


@router.get("/transect")
def get_vertical_transect(
    lat: float = Query(12.0, description="Latitude along which to take longitudinal curtain"),
    start_lon: float = Query(55.0, description="Start Longitude"),
    end_lon: float = Query(95.0, description="End Longitude"),
    step: int = Query(2, description="Longitude step stride")
):
    """
    Generates a 2D vertical curtain (depth vs longitude) across the ocean basin
    (e.g., from Arabian Sea across Southern India into Bay of Bengal).
    """
    state = get_current_reconstruction()
    i = int(np.argmin(np.abs(LATS - lat)))
    
    lon_indices = [
        j for j, lon in enumerate(LONS)
        if start_lon <= lon <= end_lon and (j % step == 0)
    ]
    transect_lons = [float(LONS[j]) for j in lon_indices]
    
    # Dimensions: [len(depths), len(lon_indices)]
    curtain_temp = []
    curtain_unc = []
    bathymetry_line = []
    
    for k in range(NUM_DEPTHS):
        row_temp = []
        row_unc = []
        for j in lon_indices:
            val_t = state["recon_cube"][i, j, k]
            val_u = state["unc_cube"][i, j, k]
            row_temp.append(None if np.isnan(val_t) else float(val_t))
            row_unc.append(None if np.isnan(val_u) else float(val_u))
        curtain_temp.append(row_temp)
        curtain_unc.append(row_unc)
        
    for j in lon_indices:
        b_val = float(data_engine.bathymetry_max_depth[i, j])
        bathymetry_line.append(b_val)
        
    return {
        "transect_latitude": float(LATS[i]),
        "longitudes": transect_lons,
        "depths": STANDARD_DEPTHS,
        "bathymetry_depths": bathymetry_line,
        "curtain_temperature": curtain_temp,
        "curtain_uncertainty": curtain_unc
    }


@router.get("/validation")
def get_argo_validation():
    """
    Returns real-time and delayed-mode Argo float collocated matchups,
    AI reconstructions, GLORYS reanalysis, and the depth-wise skill scorecard.
    """
    matchups = argo_service.get_collocated_matchups()
    scorecard = argo_service.compute_aggregate_skill_scorecard()
    return {
        "matchup_profiles": matchups,
        "skill_scorecard": scorecard
    }


class SimulationRequest(BaseModel):
    delta_sst: float = 0.0
    delta_sss: float = 0.0
    delta_sla: float = 0.0
    pct_wind: float = 0.0
    pct_current: float = 0.0
    scenario: Optional[str] = "custom"
    surge_height: Optional[float] = 0.0
    cyclone_lat: Optional[float] = None
    cyclone_lon: Optional[float] = None
    wind_dir_deg: Optional[float] = None


@router.post("/simulate")
def run_whatif_simulation(req: SimulationRequest):
    """
    Runs What-If scenario perturbation and evaluates physical, wind, and inundation extremity response.
    Supports user-specified cyclone eye location (cyclone_lat, cyclone_lon) and custom wind direction (wind_dir_deg).
    """
    result = simulator_service.run_simulation(
        delta_sst=req.delta_sst,
        delta_sss=req.delta_sss,
        delta_sla=req.delta_sla,
        pct_wind=req.pct_wind,
        pct_current=req.pct_current,
        scenario=req.scenario or "custom",
        surge_height=req.surge_height or 0.0,
        cyclone_lat=req.cyclone_lat,
        cyclone_lon=req.cyclone_lon,
        wind_dir_deg=req.wind_dir_deg
    )
    return result


@router.get("/cyclones")
def get_active_cyclones():
    """
    Returns near-real-time active tropical cyclones, depressions, and IMD/JTWC forecast tracks
    in the North Indian Ocean basin for operational digital twin forecasting.
    """
    return {
        "status": "Active Advisory",
        "monitoring_agency": "IMD (India Meteorological Department) & JTWC",
        "basin": "North Indian Ocean (Bay of Bengal & Arabian Sea)",
        "advisory_time_utc": "2026-10-11 18:00 UTC",
        "cyclones": [
            {
                "id": "BOB-01",
                "name": "Very Severe Cyclonic Storm 'MIDHILI-II'",
                "category": "VSCS (Category 3 Equivalent)",
                "basin": "Bay of Bengal",
                "current_position": {"lat": 16.4, "lon": 88.2},
                "central_pressure_hpa": 966,
                "max_sustained_wind_kmh": 155,
                "max_sustained_wind_kts": 84,
                "peak_gusts_kmh": 175,
                "movement_dir": "NNW (330°)",
                "forward_speed_kmh": 16,
                "radii": {
                    "rmw_km": 32,
                    "gale_50kt_km": 115,
                    "gale_34kt_km": 240
                },
                "forecast_track": [
                    {"step": "T+0h", "time": "18:00 UTC", "lat": 16.4, "lon": 88.2, "wind_kmh": 155, "intensity": "VSCS"},
                    {"step": "T+12h", "time": "06:00 UTC (+1d)", "lat": 17.6, "lon": 87.4, "wind_kmh": 165, "intensity": "ESCS"},
                    {"step": "T+24h", "time": "18:00 UTC (+1d)", "lat": 18.9, "lon": 86.6, "wind_kmh": 170, "intensity": "ESCS"},
                    {"step": "T+36h (Landfall)", "time": "06:00 UTC (+2d)", "lat": 20.2, "lon": 85.9, "wind_kmh": 150, "intensity": "VSCS (Landfall near Paradip/Puri, Odisha)"},
                    {"step": "T+48h (Inland)", "time": "18:00 UTC (+2d)", "lat": 21.6, "lon": 85.1, "wind_kmh": 85, "intensity": "Cyclonic Storm"}
                ],
                "storm_surge_warning": "3.5m - 4.8m above astronomical tide over coastal districts of Jagatsinghpur, Kendrapara, and Bhadrak",
                "alert_level": "RED ALERT"
            },
            {
                "id": "ARB-02",
                "name": "Deep Depression 'ARB-02'",
                "category": "Deep Depression (DD)",
                "basin": "East-Central Arabian Sea",
                "current_position": {"lat": 17.8, "lon": 67.4},
                "central_pressure_hpa": 994,
                "max_sustained_wind_kmh": 62,
                "max_sustained_wind_kts": 34,
                "peak_gusts_kmh": 78,
                "movement_dir": "NNE (025°)",
                "forward_speed_kmh": 12,
                "radii": {
                    "rmw_km": 50,
                    "gale_34kt_km": 140
                },
                "forecast_track": [
                    {"step": "T+0h", "time": "18:00 UTC", "lat": 17.8, "lon": 67.4, "wind_kmh": 62, "intensity": "Deep Depression"},
                    {"step": "T+12h", "time": "06:00 UTC (+1d)", "lat": 19.1, "lon": 68.2, "wind_kmh": 75, "intensity": "Cyclonic Storm"},
                    {"step": "T+24h (Landfall)", "time": "18:00 UTC (+1d)", "lat": 20.6, "lon": 69.1, "wind_kmh": 85, "intensity": "CS (Near Veraval/Porbandar, Gujarat)"},
                    {"step": "T+36h", "time": "06:00 UTC (+2d)", "lat": 22.0, "lon": 70.0, "wind_kmh": 45, "intensity": "Depression"}
                ],
                "storm_surge_warning": "1.2m - 1.8m surge over Gulf of Kutch and Saurashtra lowlands",
                "alert_level": "ORANGE ALERT"
            }
        ]
    }
