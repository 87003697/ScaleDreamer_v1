import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import *
from jaxtyping import Float
from omegaconf import OmegaConf

def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)

def scale_tensor(
    dat: Float[Tensor, "... D"], 
    inp_scale: Union[Tuple[float, float], Float[Tensor, "2 D"]], 
    tgt_scale: Union[Tuple[float, float], Float[Tensor, "2 D"]]
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def contract_to_unisphere_custom(
    x: Float[Tensor, "... 3"], 
    bbox: Float[Tensor, "2 3"],
    radius: float = 1.0
) -> Float[Tensor, "... 3"]:
    """Custom version of contract_to_unisphere for triplane representation"""
    x = scale_tensor(x, bbox, (0, 1))
    x = x * 2 - 1  # Scale to [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > radius
    x[mask] = (2 - radius / mag[mask]) * (x[mask] / mag[mask])
    return x

def sample_from_planes(
    plane_features: Float[Tensor, "B P C H W"],
    coordinates: Float[Tensor, "B N 3"],
    interpolate_feat: str = "v1"
) -> Float[Tensor, "B N C"]:
    """Sample features from triplane representation at given coordinates.
    
    Args:
        plane_features: Tensor of shape [batch, num_planes, channels, height, width]
        coordinates: Tensor of shape [batch, num_points, 3] in [-1, 1]
        interpolate_feat: Interpolation method ("v1" or "v2")
        
    Returns:
        Sampled features tensor of shape [batch, num_points, channels]
    """
    B, P, C, H, W = plane_features.shape
    N = coordinates.shape[1]
    
    # Sample from each plane
    features = []
    
    # XY plane (Z axis)
    xy_plane = plane_features[:, 0]  # [B, C, H, W]
    xy_coords = coordinates[..., [0, 1]]  # [B, N, 2]
    xy_features = F.grid_sample(
        xy_plane, 
        xy_coords.view(B, 1, -1, 2),
        mode='bilinear',
        align_corners=True
    ).view(B, C, N)
    features.append(xy_features)
    
    # XZ plane (Y axis)
    xz_plane = plane_features[:, 1]
    xz_coords = coordinates[..., [0, 2]]
    xz_features = F.grid_sample(
        xz_plane,
        xz_coords.view(B, 1, -1, 2),
        mode='bilinear',
        align_corners=True
    ).view(B, C, N)
    features.append(xz_features)
    
    # YZ plane (X axis)
    yz_plane = plane_features[:, 2]
    yz_coords = coordinates[..., [1, 2]]
    yz_features = F.grid_sample(
        yz_plane,
        yz_coords.view(B, 1, -1, 2),
        mode='bilinear',
        align_corners=True
    ).view(B, C, N)
    features.append(yz_features)
    
    # Combine features based on interpolation method
    if interpolate_feat == "v1":
        # Average pooling
        features = torch.stack(features, dim=0).mean(0)
    elif interpolate_feat == "v2":
        # Concatenate features
        features = torch.cat(features, dim=1)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolate_feat}")
        
    return features.permute(0, 2, 1)  # [B, N, C] 
