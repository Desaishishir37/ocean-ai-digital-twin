"""
Unit tests for the deep ocean reconstruction neural network and inference engine.
"""
import pytest
import numpy as np
import torch
from app.models.ocean_net import OceanAnomalyNet, SubsurfaceInferenceEngine
from app.services.data_engine import OceanDataEngine

def test_neural_network_shapes():
    net = OceanAnomalyNet(in_channels=7, num_depths=15)
    dummy_input = torch.randn(2, 7, 32, 64)
    anomaly, uncertainty = net(dummy_input)
    assert anomaly.shape == (2, 15, 32, 64)
    assert uncertainty.shape == (2, 15, 32, 64)
    # Uncertainty must be strictly positive
    assert torch.all(uncertainty > 0)

def test_inference_engine_pipeline():
    engine = OceanDataEngine()
    surface_inputs, metadata = engine.get_latest_satellite_state()
    infer_engine = SubsurfaceInferenceEngine()
    
    recon_cube, unc_cube = infer_engine.reconstruct(
        surface_inputs=surface_inputs,
        climatology_cube=engine.climatology_cube,
        land_mask=engine.land_mask
    )
    
    assert recon_cube.shape == (engine.land_mask.shape[0], engine.land_mask.shape[1], 15)
    assert unc_cube.shape == (engine.land_mask.shape[0], engine.land_mask.shape[1], 15)
    
    # Check that land points are NaN
    land_i, land_j = np.where(engine.land_mask)
    if len(land_i) > 0:
        assert np.isnan(recon_cube[land_i[0], land_j[0], 0])
        assert np.isnan(unc_cube[land_i[0], land_j[0], 0])
        
    # Check ocean point
    ocean_i, ocean_j = np.where(~engine.land_mask)
    if len(ocean_i) > 0:
        sample_profile = recon_cube[ocean_i[0], ocean_j[0], :]
        assert not np.isnan(sample_profile[0])
        # Sea surface is tropical (> 20°C)
        assert sample_profile[0] > 20.0
        # Deep ocean at 1000m is cool (< 10°C)
        if not np.isnan(sample_profile[-1]):
            assert sample_profile[-1] < 10.0
