"""
Physical oceanography constants, coordinates, and diagnostic algorithms for the
North Indian Ocean Subsurface Reconstruction & Digital Twin.
"""

from typing import List, Tuple, Union, Optional
import numpy as np

# 15 Standard Oceanographic Depth Levels (meters)
STANDARD_DEPTHS: List[float] = [
    0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0,
    300.0, 400.0, 500.0, 750.0, 1000.0
]
NUM_DEPTHS = len(STANDARD_DEPTHS)

# Geographic Domain: North Indian Ocean (Arabian Sea, Bay of Bengal, Equatorial Indian Ocean)
LAT_MIN = 0.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0
GRID_RES = 0.25  # degrees

LATS = np.arange(LAT_MIN, LAT_MAX + 1e-5, GRID_RES)
LONS = np.arange(LON_MIN, LON_MAX + 1e-5, GRID_RES)
NUM_LATS = len(LATS)  # 121
NUM_LONS = len(LONS)  # 241

# Oceanographic Thermodynamics Constants
SEAWATER_DENSITY = 1025.0  # kg/m^3
SPECIFIC_HEAT = 3990.0     # J / (kg * °C)


def compute_mld(temperature_profile: np.ndarray, depths: np.ndarray = None, threshold: float = 0.2) -> float:
    """
    Computes Mixed Layer Depth (MLD) in meters.
    Standard criterion: depth at which temperature decreases by `threshold` (°C, default 0.2°C)
    relative to the surface temperature (depth=0m).
    """
    if depths is None:
        depths = np.array(STANDARD_DEPTHS)
    
    t_surf = temperature_profile[0]
    target_t = t_surf - threshold
    
    for i in range(1, len(depths)):
        if temperature_profile[i] <= target_t:
            # Linear interpolation between depths[i-1] and depths[i]
            t_prev = temperature_profile[i - 1]
            t_curr = temperature_profile[i]
            if t_prev == t_curr:
                return float(depths[i])
            fraction = (target_t - t_prev) / (t_curr - t_prev)
            mld = depths[i - 1] + fraction * (depths[i] - depths[i - 1])
            return float(np.clip(mld, depths[0], depths[-1]))
            
    return float(depths[-1])


def compute_d20(temperature_profile: np.ndarray, depths: np.ndarray = None) -> float:
    """
    Computes the Depth of the 20°C Isotherm (D20) in meters.
    D20 serves as the primary proxy for the thermocline depth in the tropical Indian Ocean.
    """
    if depths is None:
        depths = np.array(STANDARD_DEPTHS)
        
    target_t = 20.0
    
    # If surface is already colder than 20°C (rare in North Indian Ocean), D20 is 0
    if temperature_profile[0] <= target_t:
        return 0.0
        
    for i in range(1, len(depths)):
        if temperature_profile[i] <= target_t:
            t_prev = temperature_profile[i - 1]
            t_curr = temperature_profile[i]
            if t_prev == t_curr:
                return float(depths[i])
            fraction = (target_t - t_prev) / (t_curr - t_prev)
            d20 = depths[i - 1] + fraction * (depths[i] - depths[i - 1])
            return float(np.clip(d20, depths[0], depths[-1]))
            
    return float(depths[-1])


def compute_ohc(temperature_profile: np.ndarray, depths: np.ndarray = None, t_ref: float = 26.0) -> float:
    """
    Computes Tropical Cyclone Heat Potential (TCHP) / Ocean Heat Content (OHC) in kJ/cm^2.
    Formula: OHC = integral_{0}^{D26} rho * Cp * (T(z) - 26) dz
    Only layers warmer than 26°C contribute to cyclone intensification.
    """
    if depths is None:
        depths = np.array(STANDARD_DEPTHS)
        
    if temperature_profile[0] < t_ref:
        return 0.0
        
    ohc_joules_m2 = 0.0
    for i in range(len(depths) - 1):
        z_top = depths[i]
        z_bot = depths[i + 1]
        t_top = temperature_profile[i]
        t_bot = temperature_profile[i + 1]
        
        if t_top >= t_ref and t_bot >= t_ref:
            # Entire layer is >= 26°C
            avg_excess = ((t_top - t_ref) + (t_bot - t_ref)) / 2.0
            dz = z_bot - z_top
            ohc_joules_m2 += SEAWATER_DENSITY * SPECIFIC_HEAT * avg_excess * dz
        elif t_top >= t_ref and t_bot < t_ref:
            # Crosses 26°C within this layer
            fraction = (t_ref - t_top) / (t_bot - t_top)
            z_cross = z_top + fraction * (z_bot - z_top)
            dz = z_cross - z_top
            avg_excess = (t_top - t_ref) / 2.0
            ohc_joules_m2 += SEAWATER_DENSITY * SPECIFIC_HEAT * avg_excess * dz
            break
        else:
            break
            
    # Convert J/m^2 to kJ/cm^2 (1 J/m^2 = 1e-7 kJ/cm^2)
    ohc_kj_cm2 = ohc_joules_m2 * 1e-7
    return float(ohc_kj_cm2)


def enforce_vertical_stability(temperature_profile: np.ndarray, mld_depth: float = 30.0, max_inversion: float = 0.05) -> np.ndarray:
    """
    Guarantees thermal monotonicity and vertical hydrostatic sanity.
    Below the mixed layer, water cannot be significantly warmer than the layer above it
    without causing density instability (allowing slight tolerance `max_inversion` for barrier layers).
    """
    stable_profile = np.copy(temperature_profile)
    depths = np.array(STANDARD_DEPTHS)
    
    # Locate mixed layer index
    mld_idx = np.searchsorted(depths, mld_depth)
    mld_idx = max(1, min(mld_idx, len(depths) - 2))
    
    # Enforce downward cooling below MLD
    for i in range(mld_idx, len(depths)):
        if stable_profile[i] > stable_profile[i - 1] + max_inversion:
            stable_profile[i] = stable_profile[i - 1]
            
    return stable_profile
