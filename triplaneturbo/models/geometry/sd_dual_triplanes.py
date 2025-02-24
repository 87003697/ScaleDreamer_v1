import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from torch import Tensor
from typing import *

from ...utils.general_utils import contract_to_unisphere_custom, sample_from_planes
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel

from ..networks import get_mlp
from ...utils.general_utils import config_to_primitive
@dataclass
class StableDiffusionTriplaneDualAttentionConfig:
    space_generator_config: dict = field(
        default_factory=lambda: {
            "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
            "training_type": "self_lora_rank_16-cross_lora_rank_16-locon_rank_16",
            "output_dim": 32,
            "gradient_checkpoint": False,
            "self_lora_type": "hexa_v1",
            "cross_lora_type": "hexa_v1",
            "locon_type": "vanilla_v1",
            "vae_attn_type": "basic",
            "prompt_bias": False,
        }
    )

    mlp_network_config: dict = field(
        default_factory=lambda: {
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
    )

    n_feature_dims: int = 3
    backbone: str = "one_step_triplane_dual_stable_diffusion"
    normal_type: Optional[str] = "analytic"
    finite_difference_normal_eps: Union[float, str] = 0.01
    sdf_bias: Union[float, str] = 0.0
    sdf_bias_params: Optional[Any] = None

    isosurface_remove_outliers: bool = False
    rotate_planes: Optional[str] = None
    split_channels: Optional[str] = None
    
    geo_interpolate: str = "v1"
    tex_interpolate: str = "v1"


class StableDiffusionTriplaneDualAttention(nn.Module):
    def __init__(
        self, 
        config: StableDiffusionTriplaneDualAttentionConfig,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
    ):
        super().__init__()
        self.cfg = config
        
        # set up the space generator
        from ...extern.sd_dual_triplane_modules import OneStepTriplaneDualStableDiffusion as Generator
        self.space_generator = Generator(
            self.cfg.space_generator_config,
            vae=vae,
            unet=unet,
        )

        # Convert dict to StableDiffusionTriplaneDualAttentionConfig if needed
        self.cfg = StableDiffusionTriplaneDualAttentionConfig(**config) if isinstance(config, dict) else config

        input_dim = self.space_generator.output_dim
        assert self.cfg.split_channels in [None, "v1"]
        if self.cfg.split_channels in ["v1"]:
            input_dim = input_dim // 2

        assert self.cfg.geo_interpolate in ["v1", "v2"]
        if self.cfg.geo_interpolate in ["v2"]:
            geo_input_dim = input_dim * 3
        else:
            geo_input_dim = input_dim

        assert self.cfg.tex_interpolate in ["v1", "v2"]
        if self.cfg.tex_interpolate in ["v2"]:
            tex_input_dim = input_dim * 3
        else:
            tex_input_dim = input_dim

        self.sdf_network = get_mlp(
            geo_input_dim,
            1,
            self.cfg.mlp_network_config
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                tex_input_dim,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        self.finite_difference_normal_eps: Optional[float] = None

    def generate_space_cache(
        self,
        styles: Float[Tensor, "B Z"],
        text_embed: Float[Tensor, "B C"],
    ) -> Any:
        output = self.space_generator(
            text_embed=text_embed,
            styles=styles,
        )
        return output
    
    def denoise(
        self,
        noisy_input: Any,
        text_embed: Float[Tensor, "B C"],
        timestep
    ) -> Any:
        output = self.space_generator.forward_denoise(
            text_embed=text_embed,
            noisy_input=noisy_input,
            t=timestep
        )
        return output

    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(
            latents=latents
        )
        if self.cfg.split_channels == None:
            return triplane
        elif self.cfg.split_channels == "v1":
            B, _, C, H, W = triplane.shape
            used_indices_geo = torch.tensor([True] * (self.space_generator.output_dim// 2) + [False] * (self.space_generator.output_dim // 2))
            used_indices_tex = torch.tensor([False] * (self.space_generator.output_dim // 2) + [True] * (self.space_generator.output_dim // 2))
            used_indices = torch.stack([used_indices_geo] * 3 + [used_indices_tex] * 3, dim=0).to(triplane.device)
            return triplane[:, used_indices].view(B, 6, C//2, H, W)

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        only_geo: bool = False,
    ):
        batch_size, n_points, n_dims = points.shape
        
        assert self.cfg.rotate_planes in [None, "v1", "v2"]

        if self.cfg.rotate_planes == None:
            raise NotImplementedError("rotate_planes == None is not implemented yet.")

        space_cache_rotated = torch.zeros_like(space_cache)
        if self.cfg.rotate_planes == "v1":
            space_cache_rotated[:, 0::3] = torch.transpose(
                space_cache[:, 0::3], 3, 4
            )
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )
        elif self.cfg.rotate_planes == "v2":
            space_cache_rotated[:, 0::3] = torch.flip(
                space_cache[:, 0::3], dims=(4,)
            )
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )

        geo_feat = sample_from_planes(
            plane_features=space_cache_rotated[:, 0:3].contiguous(),
            coordinates=points,
            interpolate_feat=self.cfg.geo_interpolate
        ).view(*points.shape[:-1],-1)

        if only_geo:
            return geo_feat
        else:
            tex_feat = sample_from_planes(
                plane_features=space_cache_rotated[:, 3:6].contiguous(),
                coordinates=points,
                interpolate_feat=self.cfg.tex_interpolate
            ).view(*points.shape[:-1],-1)

            return geo_feat, tex_feat

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Any,
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, n_points, n_dims = points.shape

        enc_geo, enc_tex = self.interpolate_encodings(points, space_cache)
        sdf = self.sdf_network(enc_geo).view(*points.shape[:-1], 1)

        output = {
            "sdf": sdf.view(batch_size * n_points, 1)
        }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc_tex).view(
                *points.shape[:-1], self.cfg.n_feature_dims)
            output.update({
                "features": features.view(batch_size * n_points, self.cfg.n_feature_dims)
            })

        return output

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()

    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(
            latents = latents
        )
        if self.cfg.split_channels == None:
            return triplane
        elif self.cfg.split_channels == "v1":
            B, _, C, H, W = triplane.shape
            # geometry triplane uses the first n_feature_dims // 2 channels
            # texture triplane uses the last n_feature_dims // 2 channels
            used_indices_geo = torch.tensor([True] * (self.space_generator.output_dim// 2) + [False] * (self.space_generator.output_dim // 2))
            used_indices_tex = torch.tensor([False] * (self.space_generator.output_dim // 2) + [True] * (self.space_generator.output_dim // 2))
            used_indices = torch.stack([used_indices_geo] * 3 + [used_indices_tex] * 3, dim=0).to(triplane.device)
            return triplane[:, used_indices].view(B, 6, C//2, H, W)
