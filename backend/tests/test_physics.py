"""
Unit tests for physical oceanography diagnostics and stability constraints.
"""
import pytest
import numpy as np
from app.core.physics import (
    STANDARD_DEPTHS,
    compute_mld,
    compute_d20,
    compute_ohc,
    enforce_vertical_stability
)

def test_standard_depths():
    assert len(STANDARD_DEPTHS) == 15
    assert STANDARD_DEPTHS[0] == 0.0
    assert STANDARD_DEPTHS[-1] == 1000.0

def test_mld_computation():
    # Surface at 28°C, stays at 28°C until 30m, then drops to 27.5°C at 50m
    # Target T = 28.0 - 0.2 = 27.8°C (should fall between 30m and 50m)
    profile = np.array([28.0, 28.0, 28.0, 28.0, 27.5, 26.0, 24.0, 22.0, 20.0, 16.0, 12.0, 9.0, 7.0, 6.0, 5.0])
    mld = compute_mld(profile)
    assert 30.0 < mld < 50.0
    # Exact: fraction = (27.8 - 28.0) / (27.5 - 28.0) = -0.2 / -0.5 = 0.4
    # 30 + 0.4 * 20 = 38.0m
    assert abs(mld - 38.0) < 0.1

def test_d20_computation():
    # Profile crosses 20°C between 125m (22°C) and 150m (18°C)
    profile = np.array([28.5, 28.4, 28.2, 28.0, 27.0, 25.5, 24.0, 22.0, 18.0, 15.0, 11.0, 8.5, 7.0, 5.5, 4.5])
    d20 = compute_d20(profile)
    # Between 125m and 150m: fraction = (20 - 22) / (18 - 22) = 0.5
    # 125 + 0.5 * 25 = 137.5m
    assert abs(d20 - 137.5) < 0.1

def test_ohc_computation():
    # Profile with warm water down to 75m
    profile = np.array([29.0, 28.8, 28.5, 28.0, 27.0, 25.5, 23.0, 20.0, 17.0, 14.0, 10.0, 8.0, 6.5, 5.2, 4.2])
    ohc = compute_ohc(profile)
    assert ohc > 0.0
    # When surface is cold (< 26°C), OHC should be 0
    cold_profile = np.array([24.0] * 15)
    assert compute_ohc(cold_profile) == 0.0

def test_enforce_vertical_stability():
    # Create an unphysical profile with an artificial deep warm bubble at 200m
    unstable = np.array([28.0, 28.0, 27.5, 27.0, 26.0, 24.0, 22.0, 20.0, 18.0, 23.0, 12.0, 9.0, 7.0, 6.0, 5.0])
    stable = enforce_vertical_stability(unstable, mld_depth=30.0)
    # At index 9 (200m), it was 23.0°C which was higher than 18.0°C at index 8 (150m)
    assert stable[9] <= stable[8] + 0.05
