from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None
    use_latent_history: bool = False
    latent_dim: int = 128


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.use_latent_history = cfg.use_latent_history
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        
        # Base conditioning projection
        cond_input_dim = cfg.cond_channels
        if cfg.use_latent_history:
            cond_input_dim += cfg.latent_dim
        
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(
        self, 
        noisy_next_obs: Tensor, 
        c_noise: Tensor, 
        obs: Tensor, 
        act: Tensor,
        latent_history: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            noisy_next_obs: (B, C, H, W) - noisy next observation
            c_noise: (B, 1) - noise conditioning
            obs: (B, T*C, H, W) - stacked history observations
            act: (B, T) - history actions
            latent_history: (B, latent_dim) - optional latent history encoding
        """
        # Combine noise and action embeddings
        cond = self.noise_emb(c_noise) + self.act_emb(act)  # (B, cond_channels)
        
        # Add latent history conditioning if provided
        if self.use_latent_history and latent_history is not None:
            cond = torch.cat((cond, latent_history), dim=1)  # (B, cond_channels + latent_dim)
        
        cond = self.cond_proj(cond)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
