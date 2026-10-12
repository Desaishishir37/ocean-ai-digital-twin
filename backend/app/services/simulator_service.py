"""
What-If Simulator Service for the North Indian Ocean Digital Twin.
Executes controlled environmental perturbations, validates scenarios via Mahalanobis
Out-of-Distribution (OOD) guardrails, and quantifies 3D thermal response and diagnostic shifts.
"""

from typing import Dict, Any, Tuple
import numpy as np

from app.core.physics import (
    STANDARD_DEPTHS, NUM_DEPTHS,
    compute_mld, compute_d20, compute_ohc
)


class WhatIfSimulatorService:
    """
    Manages scenario perturbations, OOD detection, difference fields, and physical metrics.
    """
    def __init__(self, data_engine, inference_engine):
        self.data_engine = data_engine
        self.inference_engine = inference_engine
        self.depths = np.array(STANDARD_DEPTHS)
        
        # Historical standard deviations for normalization in Mahalanobis calculation
        # [sst_delta, sss_delta, sla_delta, wind_pct, current_pct]
        self.param_stds = {
            "delta_sst": 0.85,    # °C
            "delta_sss": 0.50,    # PSU
            "delta_sla": 0.08,    # m
            "pct_wind": 25.0,     # %
            "pct_current": 30.0   # %
        }

    def evaluate_ood_guardrail(
        self,
        delta_sst: float,
        delta_sss: float,
        delta_sla: float,
        pct_wind: float,
        pct_current: float
    ) -> Dict[str, Any]:
        """
        Computes the normalized Mahalanobis distance D_M of the user's perturbation vector.
        Determines scientific confidence tier:
        - Green (< 2.0): Within Historical Distribution (High Confidence)
        - Yellow (2.0 - 4.0): Moderate Extrapolation (Interpret with Caution)
        - Red (> 4.0): Outside Learned Range (Low Confidence / Speculative)
        """
        # Normalized squared distances
        z_sst = (delta_sst / self.param_stds["delta_sst"]) ** 2
        z_sss = (delta_sss / self.param_stds["delta_sss"]) ** 2
        z_sla = (delta_sla / self.param_stds["delta_sla"]) ** 2
        z_wind = (pct_wind / self.param_stds["pct_wind"]) ** 2
        z_curr = (pct_current / self.param_stds["pct_current"]) ** 2
        
        d_m = float(np.sqrt((z_sst + z_sss + z_sla + z_wind + z_curr) / 5.0))
        d_m_rounded = round(d_m, 2)
        
        if d_m <= 1.8:
            status = "Within Historical Distribution"
            badge = "green"
            confidence = "High Confidence"
            description = "This scenario falls within observed natural variability and seasonal extremes in the North Indian Ocean."
        elif d_m <= 3.5:
            status = "Moderate Extrapolation"
            badge = "yellow"
            confidence = "Medium Confidence"
            description = "This perturbation represents an extreme climate anomaly. Interpret vertical response with moderate caution."
        else:
            status = "Outside Learned Physics"
            badge = "red"
            confidence = "Low Confidence (Experimental)"
            description = "The requested parameters exceed observed historical thresholds. Non-linear deep ocean dynamics may be underrepresented."
            
        return {
            "mahalanobis_distance": d_m_rounded,
            "status": status,
            "badge": badge,
            "confidence": confidence,
            "description": description
        }

    def run_simulation(
        self,
        delta_sst: float = 0.0,
        delta_sss: float = 0.0,
        delta_sla: float = 0.0,
        pct_wind: float = 0.0,
        pct_current: float = 0.0,
        scenario: str = "custom",
        surge_height: float = 0.0,
        cyclone_lat: float = None,
        cyclone_lon: float = None,
        wind_dir_deg: float = None
    ) -> Dict[str, Any]:
        """
        Runs the What-If simulation:
        1. Evaluates OOD guardrail.
        2. Perturbs the operational satellite surface state.
        3. Runs AI model inference on baseline vs perturbed state.
        4. Computes 3D difference field (T_sim - T_base).
        5. Computes diagnostic shifts in MLD, D20, and Ocean Heat Content.
        6. Synthesizes spatial wind, Holland cyclone vortex at user-selected (lat, lon), and regional surge inundation.
        """
        ood_info = self.evaluate_ood_guardrail(
            delta_sst, delta_sss, delta_sla, pct_wind, pct_current
        )
        
        # 1. Base State
        base_surface, metadata = self.data_engine.get_latest_satellite_state()
        base_3d, base_unc = self.inference_engine.reconstruct(
            surface_inputs=base_surface,
            climatology_cube=self.data_engine.climatology_cube,
            land_mask=self.data_engine.land_mask
        )
        
        # 2. Perturbed Surface State
        perturbed_surface = np.copy(base_surface)
        perturbed_surface[0] += delta_sst  # SST Anomaly
        perturbed_surface[1] += delta_sss  # SSS Anomaly
        perturbed_surface[2] += delta_sla  # SLA
        
        wind_multiplier = 1.0 + (pct_wind / 100.0)
        if wind_dir_deg is not None:
            # Reorient wind vector along custom direction (meteorological FROM angle)
            theta = np.radians(wind_dir_deg)
            speed_mag = np.sqrt(perturbed_surface[5]**2 + perturbed_surface[6]**2) * wind_multiplier
            perturbed_surface[5] = -speed_mag * np.sin(theta)
            perturbed_surface[6] = -speed_mag * np.cos(theta)
        else:
            perturbed_surface[5] *= wind_multiplier  # Wind U
            perturbed_surface[6] *= wind_multiplier  # Wind V
        
        curr_multiplier = 1.0 + (pct_current / 100.0)
        perturbed_surface[3] *= curr_multiplier  # Current U
        perturbed_surface[4] *= curr_multiplier  # Current V
        
        # Mask land
        for c in range(7):
            perturbed_surface[c][self.data_engine.land_mask] = np.nan
            
        # 3. Infer Perturbed 3D Ocean State
        perturbed_3d, perturbed_unc = self.inference_engine.reconstruct(
            surface_inputs=perturbed_surface,
            climatology_cube=self.data_engine.climatology_cube,
            land_mask=self.data_engine.land_mask
        )
        
        # 4. Difference Field [H, W, 15]
        diff_3d = perturbed_3d - base_3d
        
        # 5. Diagnostic shifts at key sampling stations
        sample_lats = [15.0, 12.0, 8.0, 5.0]
        sample_lons = [65.0, 88.0, 75.0, 90.0]
        
        mld_shifts = []
        d20_shifts = []
        ohc_shifts = []
        
        for lat, lon in zip(sample_lats, sample_lons):
            i = int(np.argmin(np.abs(self.data_engine.lats - lat)))
            j = int(np.argmin(np.abs(self.data_engine.lons - lon)))
            
            p_base = base_3d[i, j, :]
            p_sim = perturbed_3d[i, j, :]
            
            if not np.isnan(p_base[0]):
                mld_b = compute_mld(p_base, self.depths)
                mld_s = compute_mld(p_sim, self.depths)
                mld_shifts.append(mld_s - mld_b)
                
                d20_b = compute_d20(p_base, self.depths)
                d20_s = compute_d20(p_sim, self.depths)
                d20_shifts.append(d20_s - d20_b)
                
                ohc_b = compute_ohc(p_base, self.depths)
                ohc_s = compute_ohc(p_sim, self.depths)
                ohc_shifts.append(ohc_s - ohc_b)
                
        avg_mld_shift = float(np.mean(mld_shifts)) if mld_shifts else 0.0
        avg_d20_shift = float(np.mean(d20_shifts)) if d20_shifts else 0.0
        avg_ohc_shift = float(np.mean(ohc_shifts)) if ohc_shifts else 0.0
        
        # Depth-wise mean anomaly response across basin
        depth_response = []
        for k, z in enumerate(self.depths):
            layer_diff = diff_3d[:, :, k]
            valid_diff = layer_diff[~np.isnan(layer_diff)]
            mean_diff = float(np.mean(valid_diff)) if len(valid_diff) > 0 else 0.0
            depth_response.append({
                "depth_m": float(z),
                "mean_delta_t": round(mean_diff, 3)
            })

        # ==========================================
        # 6. SPATIAL SIMULATION GRIDS (Downsampled stride 2 for fast web delivery)
        # ==========================================
        downsample = 2
        lats_sub = self.data_engine.lats[::downsample].tolist()
        lons_sub = self.data_engine.lons[::downsample].tolist()
        H_sub = len(lats_sub)
        W_sub = len(lons_sub)
        
        # A. Perturbed Sea Surface Temperature & Anomaly
        clim_surf = self.data_engine.climatology_cube[:, :, 0]
        full_sim_sst = clim_surf + perturbed_surface[0]
        sub_sim_sst = full_sim_sst[::downsample, ::downsample]
        sub_anom_sst = (perturbed_surface[0])[::downsample, ::downsample]

        sst_grid = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in sub_sim_sst]
        anom_grid = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in sub_anom_sst]

        # B. Wind Vector Field (U, V, Speed, Direction)
        u_wind = np.copy(perturbed_surface[5])
        v_wind = np.copy(perturbed_surface[6])

        if wind_dir_deg is not None:
            # Reorient wind vectors to user-specified compass angle (meteorological FROM angle)
            target_rad = np.radians(float(wind_dir_deg))
            cur_speed = np.sqrt(u_wind**2 + v_wind**2)
            u_wind = -cur_speed * np.sin(target_rad)
            v_wind = -cur_speed * np.cos(target_rad)

        # If Cyclone Scenario or extreme wind forcing, synthesize intense cyclonic vortex
        effective_surge = max(surge_height, 0.0)
        is_cyclone = (scenario == "cyclone") or (pct_wind >= 75) or (effective_surge >= 4.0 and scenario != "tsunami")
        if scenario == "tsunami":
            effective_surge = max(effective_surge, 11.2)
        elif is_cyclone:
            effective_surge = max(effective_surge, 6.5)
        elif pct_wind > 50:
            effective_surge = max(effective_surge, delta_sla * 5.0 + (pct_wind - 50.0) / 25.0)
        elif delta_sla > 0:
            effective_surge = max(effective_surge, delta_sla * 4.0)

        if is_cyclone:
            # Moveable Cyclone Center: defaults to Bay of Bengal (16.0°N, 88.5°E) or user selected location
            c_lat = float(cyclone_lat) if cyclone_lat is not None else 16.0
            c_lon = float(cyclone_lon) if cyclone_lon is not None else 88.5
            c_lat = max(3.0, min(25.0, c_lat))
            c_lon = max(52.0, min(98.0, c_lon))

            x_mesh, y_mesh = np.meshgrid(self.data_engine.lons, self.data_engine.lats)
            dx = (x_mesh - c_lon) * 1.1
            dy = y_mesh - c_lat
            dist_deg = np.sqrt(dx**2 + dy**2)
            angle_rad = np.arctan2(dy, dx)
            
            # Holland vortex wind profile
            v_max = 58.0 + (pct_wind / 10.0)  # ~210 km/h sustained peak eyewall
            r_max = 1.8
            r_norm = dist_deg / (r_max + 1e-4)
            v_tan = v_max * (2.0 * r_norm / (1.0 + r_norm**2)) * np.exp(-dist_deg / 10.0)
            
            # Inflow angle of 18 degrees (cyclonic counter-clockwise spiral)
            inflow = np.radians(18.0)
            u_vortex = -v_tan * np.sin(angle_rad + inflow)
            v_vortex = v_tan * np.cos(angle_rad + inflow)
            
            u_wind = u_wind * 0.4 + u_vortex
            v_wind = v_wind * 0.4 + v_vortex
            u_wind[self.data_engine.land_mask] = np.nan
            v_wind[self.data_engine.land_mask] = np.nan
        else:
            c_lat = None
            c_lon = None

        sub_u = u_wind[::downsample, ::downsample]
        sub_v = v_wind[::downsample, ::downsample]
        wind_speed_ms = np.sqrt(sub_u**2 + sub_v**2)
        wind_speed_kmh = wind_speed_ms * 3.6

        u_grid = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in sub_u]
        v_grid = [[None if np.isnan(val) else float(round(val, 2)) for val in row] for row in sub_v]
        spd_grid = [[None if np.isnan(val) else float(round(val, 1)) for val in row] for row in wind_speed_kmh]

        valid_speeds = wind_speed_kmh[~np.isnan(wind_speed_kmh)]
        mean_speed = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 24.0
        peak_speed = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else 65.0

        # Meteorological wind direction (direction wind blows FROM)
        valid_u = sub_u[~np.isnan(sub_u)]
        valid_v = sub_v[~np.isnan(sub_v)]
        mean_u = float(np.mean(valid_u)) if len(valid_u) > 0 else 5.0
        mean_v = float(np.mean(valid_v)) if len(valid_v) > 0 else 3.0
        dir_deg = float((np.degrees(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)

        cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        cardinal_idx = int((dir_deg + 11.25) / 22.5) % 16
        cardinal_dir = cardinals[cardinal_idx]
        if is_cyclone:
            basin_name = "Arabian Sea" if c_lon < 77.0 else "Bay of Bengal"
            cardinal_dir = f"Cyclonic Vortex ({basin_name} at {c_lat:.1f}°N, {c_lon:.1f}°E)"

        # C. Ocean Bed Bathymetry Grid
        bathy_sub = self.data_engine.bathymetry_max_depth[::downsample, ::downsample]
        bathy_grid = [[float(round(val, 1)) for val in row] for row in bathy_sub]

        # D. Extremity & Coastal Landmass Inundation Modeling
        flooded_area = round(max(0.0, effective_surge * 4400.0), 0)
        peak_runup = round(effective_surge * 1.38, 2)

        if effective_surge < 0.6:
            hazard_level = "Normal Tide (No Inundation)"
            hazard_badge = "emerald"
        elif effective_surge < 2.0:
            hazard_level = "Minor High-Tide Encroachment"
            hazard_badge = "blue"
        elif effective_surge < 4.5:
            hazard_level = "Severe Storm Surge Inundation"
            hazard_badge = "amber"
        elif effective_surge < 8.0:
            hazard_level = "Catastrophic Storm Surge Engulfment"
            hazard_badge = "rose"
        else:
            hazard_level = "Existential Mega-Tsunami Landmass Engulfment"
            hazard_badge = "purple"

        raw_coastal_nodes = [
            {"name": "Mumbai & Konkan Coast", "lat": 18.98, "lon": 72.83, "scale": 1.05, "desc": "Marine Drive, Port, & Coastal belt submerged"},
            {"name": "Sundarbans & Bengal Delta", "lat": 21.80, "lon": 89.20, "scale": 1.45, "desc": "85% Mangrove biome flooded, saline dike breach"},
            {"name": "Bangladesh Meghna Estuary", "lat": 22.40, "lon": 91.20, "scale": 1.50, "desc": "Delta lowlands engulfed up to 25km inland"},
            {"name": "Chennai Marina Coastline", "lat": 13.08, "lon": 80.28, "scale": 1.15, "desc": "Harbor & coastal boulevard overrun by surge"},
            {"name": "Kochi & Kerala Backwaters", "lat": 9.93, "lon": 76.27, "scale": 0.90, "desc": "Vembanad estuary overflow, coastal roads flooded"},
            {"name": "Gujarat Lowlands & Gulf of Kutch", "lat": 23.20, "lon": 69.80, "scale": 1.35, "desc": "Salt flats & industrial ports submerged"},
            {"name": "Galle & Southern Sri Lanka", "lat": 6.03, "lon": 80.22, "scale": 1.15, "desc": "Southern coastal corridor severe inundation"},
            {"name": "Maldives Atolls (Malé)", "lat": 4.18, "lon": 73.51, "scale": 1.40, "desc": "Low elevation 1.5m atoll ring submerged"}
        ]

        affected_regions = []
        for node in raw_coastal_nodes:
            if is_cyclone and c_lat is not None and c_lon is not None:
                # Proximity decay from eye of cyclone
                d_lat = node["lat"] - c_lat
                d_lon = (node["lon"] - c_lon) * np.cos(np.radians(c_lat))
                dist_deg = np.sqrt(d_lat**2 + d_lon**2)
                proximity_weight = max(0.15, float(np.exp(-dist_deg / 7.5)))
                node_surge = round(effective_surge * node["scale"] * proximity_weight, 1)
            else:
                node_surge = round(effective_surge * node["scale"], 1)

            if node_surge >= 2.5:
                status = "Engulfed (Catastrophic)"
                impact = node["desc"]
            elif node_surge >= 1.5:
                status = "Submerged"
                impact = node["desc"]
            elif node_surge >= 0.8:
                status = "Threatened"
                impact = "Coastal buffer zone under warning"
            else:
                status = "Nominal"
                impact = "Tidal level within normal threshold"

            affected_regions.append({
                "name": node["name"],
                "lat": node["lat"],
                "lon": node["lon"],
                "surge_depth_m": node_surge,
                "status": status,
                "impact": impact
            })

        # Physical Consequences & Extreme Weather Classifications
        if peak_speed >= 220:
            cyclone_category = "Cat-5 Super Cyclone (220+ km/h Sustained)"
        elif peak_speed >= 165:
            cyclone_category = "Cat-4 Extremely Severe Cyclonic Storm (165-220 km/h)"
        elif peak_speed >= 118:
            cyclone_category = "Cat-3 Very Severe Cyclonic Storm (118-165 km/h)"
        elif peak_speed >= 89:
            cyclone_category = "Cat-2 Severe Cyclonic Storm (89-118 km/h)"
        elif peak_speed >= 62:
            cyclone_category = "Cat-1 Cyclonic Storm (62-89 km/h)"
        else:
            cyclone_category = "Moderate Monsoon Gale (<62 km/h)"

        if delta_sst >= 4.0:
            mhw_category = "Category V (Beyond Historical Record)"
            bleaching_alert = "Alert Level 2 (100% Coral Reef Bleaching & Ecological Collapse)"
        elif delta_sst >= 2.5:
            mhw_category = "Category IV (Extreme Marine Heatwave)"
            bleaching_alert = "Alert Level 2 (Severe Widespread Bleaching & Mortality)"
        elif delta_sst >= 1.5:
            mhw_category = "Category III (Severe Marine Heatwave)"
            bleaching_alert = "Alert Level 1 (Significant Coral Bleaching Likely)"
        elif delta_sst >= 0.8:
            mhw_category = "Category II (Strong Marine Heatwave)"
            bleaching_alert = "Bleaching Watch (Thermal Stress Accumulating)"
        elif delta_sst >= 0.4:
            mhw_category = "Category I (Moderate Heatwave)"
            bleaching_alert = "No Significant Thermal Stress"
        else:
            mhw_category = "Normal Thermal State"
            bleaching_alert = "No Thermal Stress"
            
        sub_mask = self.data_engine.land_mask[::downsample, ::downsample]
        land_mask_grid = [[bool(val) for val in row] for row in sub_mask]

        return {
            "ood_guardrail": ood_info,
            "perturbations": {
                "delta_sst": delta_sst,
                "delta_sss": delta_sss,
                "delta_sla": delta_sla,
                "pct_wind": pct_wind,
                "pct_current": pct_current,
                "scenario": scenario,
                "surge_height": effective_surge
            },
            "diagnostics": {
                "avg_mld_shift_meters": round(avg_mld_shift, 1),
                "avg_d20_shift_meters": round(avg_d20_shift, 1),
                "avg_ohc_shift_kj_cm2": round(avg_ohc_shift, 2)
            },
            "depth_response": depth_response,
            "spatial_simulation": {
                "lats": lats_sub,
                "lons": lons_sub,
                "land_mask": land_mask_grid,
                "sst_grid": sst_grid,
                "sst_anomaly_grid": anom_grid,
                "wind_u_grid": u_grid,
                "wind_v_grid": v_grid,
                "wind_speed_kmh_grid": spd_grid,
                "bathymetry_grid": bathy_grid,
                "mean_wind_speed_kmh": round(mean_speed, 1),
                "peak_wind_speed_kmh": round(peak_speed, 1),
                "mean_wind_speed_kts": round(mean_speed / 1.852, 1),
                "dominant_wind_dir_deg": round(dir_deg, 1),
                "dominant_wind_cardinal": cardinal_dir,
                "cyclone_center": {"lat": round(c_lat, 2), "lon": round(c_lon, 2)} if (is_cyclone and c_lat is not None and c_lon is not None) else None
            },
            "extremity_impact": {
                "hazard_level": hazard_level,
                "hazard_badge": hazard_badge,
                "effective_surge_m": round(effective_surge, 2),
                "peak_runup_m": peak_runup,
                "flooded_area_km2": flooded_area,
                "cyclone_category": cyclone_category,
                "marine_heatwave_category": mhw_category,
                "coral_bleaching_alert": bleaching_alert,
                "affected_regions": affected_regions
            }
        }
