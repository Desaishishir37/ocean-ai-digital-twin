"""
Unit tests for the What-If Simulator and OOD guardrails.
"""
import pytest
from app.services.data_engine import OceanDataEngine
from app.models.ocean_net import SubsurfaceInferenceEngine
from app.services.simulator_service import WhatIfSimulatorService

@pytest.fixture
def sim_service():
    data_engine = OceanDataEngine()
    infer_engine = SubsurfaceInferenceEngine()
    return WhatIfSimulatorService(data_engine, infer_engine)

def test_ood_guardrail_within_distribution(sim_service):
    # Small realistic perturbation (+0.5°C SST, +5% wind)
    res = sim_service.evaluate_ood_guardrail(
        delta_sst=0.5, delta_sss=-0.1, delta_sla=0.02, pct_wind=5.0, pct_current=5.0
    )
    assert res["badge"] == "green"
    assert "Within Historical Distribution" in res["status"]

def test_ood_guardrail_outside_distribution(sim_service):
    # Massive extreme perturbation (+6.0°C SST, +180% wind)
    res = sim_service.evaluate_ood_guardrail(
        delta_sst=6.0, delta_sss=2.5, delta_sla=0.4, pct_wind=180.0, pct_current=150.0
    )
    assert res["badge"] == "red"
    assert "Outside Learned Physics" in res["status"]

def test_run_simulation(sim_service):
    res = sim_service.run_simulation(
        delta_sst=1.5, delta_sss=0.0, delta_sla=0.05, pct_wind=10.0, pct_current=0.0
    )
    assert "ood_guardrail" in res
    assert "diagnostics" in res
    assert "avg_mld_shift_meters" in res["diagnostics"]
    assert "avg_d20_shift_meters" in res["diagnostics"]
    assert "avg_ohc_shift_kj_cm2" in res["diagnostics"]
    assert len(res["depth_response"]) == 15
    # SST warming should produce positive mean delta_t at the surface (depth 0)
    assert res["depth_response"][0]["mean_delta_t"] > 0
    # Spatial simulation checks
    assert "spatial_simulation" in res
    assert "sst_grid" in res["spatial_simulation"]
    assert "wind_u_grid" in res["spatial_simulation"]
    assert "wind_v_grid" in res["spatial_simulation"]
    assert "wind_speed_kmh_grid" in res["spatial_simulation"]
    assert "bathymetry_grid" in res["spatial_simulation"]
    assert res["spatial_simulation"]["mean_wind_speed_kmh"] > 0
    # Extremity impact checks
    assert "extremity_impact" in res
    assert "hazard_level" in res["extremity_impact"]
    assert "affected_regions" in res["extremity_impact"]
    assert len(res["extremity_impact"]["affected_regions"]) >= 6

def test_run_simulation_extremity_scenarios(sim_service):
    # Test Tsunami extremity
    res_tsunami = sim_service.run_simulation(
        scenario="tsunami", surge_height=11.5
    )
    assert "Tsunami" in res_tsunami["extremity_impact"]["hazard_level"]
    assert res_tsunami["extremity_impact"]["flooded_area_km2"] > 10000
    
    # Test Cyclone extremity with gale winds
    res_cyclone = sim_service.run_simulation(
        scenario="cyclone", pct_wind=120.0
    )
    assert res_cyclone["spatial_simulation"]["peak_wind_speed_kmh"] > 150
    assert "Cyclone" in res_cyclone["extremity_impact"]["cyclone_category"]
