import torch
import torch.nn as nn
from torch import Tensor

def mlp(sizes, act=nn.ReLU, use_layernorm: bool = False):
    assert len(sizes) == 4
    in_dim, h1, h2, out_dim = sizes
    layers = [nn.Linear(in_dim, h1), act(),
              nn.Linear(h1, h2), act(),
              nn.Linear(h2, out_dim)]
    if use_layernorm:
        layers += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)

class GNSConfig:
    latent_dim: int = 128
    encoder_mode: str = "relative"     # "relative" or "absolute"
    num_message_passing_steps: int = 10
    shared_processor_weights: bool = False
    connect_radius: float = 0.015
    max_neighbors: int = 512
    node_feature_dim: int = 18         # walls(8 vec) + vel_hist(10)
    global_feature_dim: int = 2        # [dt_frame, g_y]
    out_dim: int = 2                   # 2D acceleration

class Encoder(nn.Module):
    """
    - Absolute: node_in = [pos, node_feats, globals]; edges = trainable bias
    - Relative: node_in = [node_feats, globals]; edges = MLP([vec, ||vec||])
    """
    def __init__(self, cfg: GNSConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.latent_dim
        if cfg.encoder_mode not in ("absolute", "relative"):
            raise ValueError("encoder_mode must be 'absolute' or 'relative'")
        if cfg.encoder_mode == "absolute":
            node_in_dim = 2 + cfg.node_feature_dim + cfg.global_feature_dim
        else:
            node_in_dim = cfg.node_feature_dim + cfg.global_feature_dim
        self.node_encoder = mlp((node_in_dim, D, D, D), use_layernorm=True)
        if cfg.encoder_mode == "absolute":
            self.edge_encoder = None
            # trainable edge bias
            self.edge_bias = nn.Parameter(torch.zeros(D))
            nn.init.xavier_uniform_(self.edge_bias.unsqueeze(0))
        else:
            self.edge_encoder = mlp((3, D, D, D), use_layernorm=True)  
            self.edge_bias = None

    def forward(self,
        pos: Tensor, node_feats: Tensor, edge_index: Tensor, edge_feats: Tensor, globals_g: Tensor
    ):
        N = pos.shape[0]
        if self.cfg.encoder_mode == "absolute":
            node_in = torch.cat([pos, node_feats, globals_g.unsqueeze(0).expand(N, -1)], dim=-1)
        else:
            node_in = torch.cat([node_feats, globals_g.unsqueeze(0).expand(N, -1)], dim=-1)
        v0 = self.node_encoder(node_in)
        if self.cfg.encoder_mode == "absolute":
            E = edge_index.size(1)
            e0 = self.edge_bias.expand(E, -1)
        else:
            e0 = self.edge_encoder(edge_feats)
        return v0, e0

class GNBlock(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        D = latent_dim
        self.phi_e = mlp((3*D, D, D, D), use_layernorm=True)
        self.phi_v = mlp((2*D, D, D, D), use_layernorm=True)
    def forward(self, v: Tensor, e: Tensor, edge_index: Tensor):
        src, dst = edge_index
        e_in = torch.cat([v[src], v[dst], e], dim=-1)
        e = e + self.phi_e(e_in)
        N, D = v.size()
        m = torch.zeros((N, D), device=v.device, dtype=v.dtype)
        m.index_add_(0, dst, e)
        v = v + self.phi_v(torch.cat([v, m], dim=-1))
        return v, e

class Processor(nn.Module):
    def __init__(self, cfg: GNSConfig):
        super().__init__()
        if cfg.shared_processor_weights:
            blk = GNBlock(cfg.latent_dim)
            self.blocks = nn.ModuleList([blk for _ in range(cfg.num_message_passing_steps)])
        else:
            self.blocks = nn.ModuleList([GNBlock(cfg.latent_dim) for _ in range(cfg.num_message_passing_steps)])
    def forward(self, v: Tensor, e: Tensor, edge_index: Tensor):
        for blk in self.blocks:
            v, e = blk(v, e, edge_index)
        return v, e

class Decoder(nn.Module):
    def __init__(self, cfg: GNSConfig):
        super().__init__()
        D = cfg.latent_dim
        # do not use layernorm in the decoder
        self.out = mlp((D, D, D, cfg.out_dim), use_layernorm=False)
    def forward(self, v: Tensor):
        return self.out(v)

class GNSModel(nn.Module):
    def __init__(self, cfg: GNSConfig = None):
        super().__init__()
        self.cfg = cfg or GNSConfig()
        self.encoder = Encoder(self.cfg)
        self.processor = Processor(self.cfg)
        self.decoder = Decoder(self.cfg)
    def forward(self, pos_t: Tensor, node_feats: Tensor, edge_index: Tensor, edge_feats: Tensor, globals_g: Tensor):
        v0, e0 = self.encoder(pos_t, node_feats, edge_index, edge_feats, globals_g)
        vL, _ = self.processor(v0, e0, edge_index)
        return self.decoder(vL)
