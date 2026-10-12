"""
Physics-Informed Deep Reconstruction Model for Ocean Subsurface Thermal Structure.
Architecture: 2D-to-3D ResNet Anomaly Reconstructor with Physics-Guided Prior Skip
and Uncertainty Quantification.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Limit CPU thread pool overhead for cloud container deployment (e.g. Render 512MB tier)
torch.set_num_threads(1)

from app.core.physics import STANDARD_DEPTHS, NUM_DEPTHS, enforce_vertical_stability


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class OceanAnomalyNet(nn.Module):
    """
    Predicts 15-depth temperature anomalies (ΔT) and 15-depth uncertainty (σ)
    from 7 surface satellite input channels.
    Combines a physics-guided hydrostatic prior with a deep residual neural network.
    """
    def __init__(self, in_channels: int = 7, num_depths: int = NUM_DEPTHS):
        super().__init__()
        self.num_depths = num_depths
        
        # Register standard depth decay tensor as buffer [15, 1, 1]
        depths_arr = np.array(STANDARD_DEPTHS, dtype=np.float32)
        # Thermal penetration decay: surface anomaly decays into thermocline
        sst_decay = np.exp(-depths_arr / 160.0).reshape(num_depths, 1, 1)
        # SLA dynamic topography penetration: steric height signals penetrate down to ~350m
        sla_coupling = (18.0 * np.exp(-depths_arr / 220.0)).reshape(num_depths, 1, 1)
        
        self.register_buffer("sst_decay", torch.from_numpy(sst_decay))
        self.register_buffer("sla_coupling", torch.from_numpy(sla_coupling))
        
        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Feature extraction residual backbone
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.layer3 = ResidualBlock(64)
        
        # Anomaly residual head (predicts fine-scale mesoscale eddies and non-linear shifts)
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_depths, kernel_size=1)
        )
        # Small initialization for residual head so physical prior dominates baseline
        nn.init.normal_(self.anomaly_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.anomaly_head[-1].bias)
        
        # Uncertainty prediction head (15 depth uncertainty channels, constrained > 0 via Softplus)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, num_depths, kernel_size=1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 7, H, W]
        sst_anom = x[:, 0:1, :, :]  # Channel 0: SST Anomaly
        sla = x[:, 2:3, :, :]       # Channel 2: Sea Level Anomaly
        
        # Physics-guided baseline: surface heating directly translates into mixed layer anomaly
        # and steric expansion from SLA deepens/shoals the thermocline
        physics_prior = sst_anom * self.sst_decay + sla * self.sla_coupling
        
        features = self.init_conv(x)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        
        residual_anomaly = self.anomaly_head(features)
        anomaly = physics_prior + residual_anomaly
        
        uncertainty = self.uncertainty_head(features) + 0.15
        
        return anomaly, uncertainty


class SubsurfaceInferenceEngine:
    """
    Wrapper around the neural model for inference, climatology addition,
    and physical stability validation.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = OceanAnomalyNet(in_channels=7, num_depths=NUM_DEPTHS)
        self.model.to(self.device)
        self.model.eval()
        
        # Calibrate depth-decay factors (uncertainty grows with depth)
        depth_arr = np.array(STANDARD_DEPTHS)
        self.depth_uncertainty_scale = 0.25 + 0.95 * (depth_arr / 1000.0) ** 0.65

    def reconstruct(
        self,
        surface_inputs: np.ndarray,
        climatology_cube: np.ndarray,
        land_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs neural model inference on surface inputs and reconstructs full 3D temperature volume.
        Inputs:
        - surface_inputs: [7, H, W]
        - climatology_cube: [H, W, 15]
        - land_mask: [H, W] (bool)
        
        Returns:
        - reconstructed_cube: [H, W, 15] (absolute °C)
        - uncertainty_cube: [H, W, 15] (±°C)
        """
        H, W = land_mask.shape
        
        # Prepare tensor, replacing NaNs on land with zeros for conv layers
        input_clean = np.nan_to_num(surface_inputs, nan=0.0)
        input_tensor = torch.from_numpy(input_clean).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            anomaly_tensor, unc_tensor = self.model(input_tensor)
            
        # Convert to numpy: [15, H, W] -> [H, W, 15]
        pred_anomaly = anomaly_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_unc = unc_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Scale uncertainty by depth physics
        pred_unc = pred_unc * self.depth_uncertainty_scale[np.newaxis, np.newaxis, :]
        
        # Combine anomaly with background climatology: T = T_clim + ΔT
        reconstructed = climatology_cube + pred_anomaly
        
        # Enforce physical vertical stability profile by profile
        for i in range(H):
            for j in range(W):
                if land_mask[i, j]:
                    reconstructed[i, j, :] = np.nan
                    pred_unc[i, j, :] = np.nan
                else:
                    valid_mask = ~np.isnan(reconstructed[i, j, :])
                    if np.any(valid_mask):
                        profile = reconstructed[i, j, :]
                        profile_valid = np.nan_to_num(profile, nan=4.0)
                        stable = enforce_vertical_stability(profile_valid)
                        reconstructed[i, j, valid_mask] = np.round(stable[valid_mask], 2)
                        pred_unc[i, j, valid_mask] = np.round(pred_unc[i, j, valid_mask], 2)
                        
        return reconstructed, pred_unc
