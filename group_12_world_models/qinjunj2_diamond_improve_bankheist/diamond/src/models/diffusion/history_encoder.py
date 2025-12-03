"""
Latent encoder for history (frames and actions) using GRU.
Uses DIAMOND's conv encoder per-frame, then processes temporal sequence with GRU.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, SmallResBlock


@dataclass
class HistoryEncoderConfig:
    img_channels: int
    latent_dim: int  # GRU hidden dimension
    num_steps_conditioning: int  # Number of past frames to encode (T)
    # Conv encoder architecture (reuses DIAMOND's pattern)
    channels: List[int]
    down: List[int]
    gru_num_layers: int = 1  # Number of GRU layers


class FrameEncoder(nn.Module):
    """Encodes a single frame using conv blocks (reuses DIAMOND's encoder pattern)."""
    
    def __init__(self, img_channels: int, channels: List[int], down: List[int]) -> None:
        super().__init__()
        assert len(channels) == len(down)
        
        encoder_layers = [Conv3x3(img_channels, channels[0])]
        for i in range(len(channels)):
            encoder_layers.append(SmallResBlock(channels[max(0, i - 1)], channels[i]))
            if down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.feature_dim = channels[-1]
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) - single frame
        
        Returns:
            features: (B, feature_dim) - flattened spatial features
        """
        x = self.encoder(x)
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # (B, feature_dim, 1, 1)
        x = x.flatten(start_dim=1)  # (B, feature_dim)
        return x


class HistoryEncoder(nn.Module):
    """
    Encodes past observations and actions into a latent state vector using GRU.
    
    Process:
    1. For each frame: use conv encoder to extract features
    2. For each step: concatenate frame features + action embedding
    3. Feed sequence through GRU
    4. Return final GRU hidden state as latent
    
    Input:
        - obs: (B, T, C, H, W) - T consecutive frames
        - act: (B, T) - T actions
    Output:
        - latent: (B, latent_dim) - final GRU hidden state
    """
    
    def __init__(
        self, 
        cfg: HistoryEncoderConfig, 
        num_actions: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.latent_dim = cfg.latent_dim
        self.num_steps = cfg.num_steps_conditioning
        
        # Frame encoder (reuses DIAMOND's conv architecture)
        self.frame_encoder = FrameEncoder(
            cfg.img_channels,
            cfg.channels,
            cfg.down
        )
        
        # Action embedding
        action_emb_dim = 16
        self.act_emb = nn.Embedding(num_actions, action_emb_dim)
        
        # GRU input dimension: frame features + action embedding
        gru_input_dim = self.frame_encoder.feature_dim + action_emb_dim
        
        # GRU to process temporal sequence
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=cfg.latent_dim,
            num_layers=cfg.gru_num_layers,
            batch_first=True,
            dropout=0.0 if cfg.gru_num_layers == 1 else 0.1
        )
    
    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        """
        Args:
            obs: (B, T, C, H, W) - T consecutive frames
            act: (B, T) - T actions
        
        Returns:
            latent: (B, latent_dim) - final hidden state from GRU
        """
        b, t, c, h, w = obs.shape
        
        # Process each frame with conv encoder
        # Reshape to (B*T, C, H, W) for batch processing
        obs_flat = obs.reshape(b * t, c, h, w)
        frame_features = self.frame_encoder(obs_flat)  # (B*T, feature_dim)
        
        # Reshape back to (B, T, feature_dim)
        frame_features = frame_features.reshape(b, t, -1)
        
        # Embed actions: (B, T) -> (B, T, action_emb_dim)
        act_emb = self.act_emb(act)
        
        # Concatenate frame features and action embeddings
        # (B, T, feature_dim + action_emb_dim)
        gru_input = torch.cat([frame_features, act_emb], dim=-1)
        
        # Process through GRU
        # Output: (B, T, latent_dim)
        # Hidden: (num_layers, B, latent_dim)
        _, hidden = self.gru(gru_input)
        
        # Extract final hidden state from last layer
        # (num_layers, B, latent_dim) -> (B, latent_dim)
        latent = hidden[-1]
        
        return latent
