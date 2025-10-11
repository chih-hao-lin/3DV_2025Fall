import os, glob
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def sort_rollout_paths(root: str) -> List[str]:
    paths = glob.glob(os.path.join(root, "rollout_*.npz"))
    def _key(p):
        s = os.path.splitext(os.path.basename(p))[0]
        return int(s.split("_")[-1])
    return sorted(paths, key=_key)

def get_dist(pt: torch.Tensor,
            box_minx: float, box_miny: float,
            box_maxx: float, box_maxy: float,
            R: float) -> torch.Tensor:
    """
    Return vector distance to 4 walls of a box.
    Each component is clamped to [-R, R] to mimic boundary.
    left  vector:  [xmin - x, 0]
    right vector:  [xmax - x, 0]
    bottom vector: [0, ymin - y]
    top   vector:  [0, ymax - y]
    Output shape: [N, 8] = concat of (left, right, bottom, top) vectors
    """
    x = pt[:, 0]; y = pt[:, 1]
    vx_l = torch.clamp(box_minx - x, min=-R, max=0.0)
    vx_r = torch.clamp(box_maxx - x, min=0.0,  max=R)
    vy_b = torch.clamp(box_miny - y, min=-R, max=0.0)
    vy_t = torch.clamp(box_maxy - y, min=0.0,  max=R)
    v_left   = torch.stack([vx_l, torch.zeros_like(vx_l)], dim=-1)
    v_right  = torch.stack([vx_r, torch.zeros_like(vx_r)], dim=-1)
    v_bottom = torch.stack([torch.zeros_like(vy_b), vy_b], dim=-1)
    v_top    = torch.stack([torch.zeros_like(vy_t), vy_t], dim=-1)
    return torch.cat([v_left, v_right, v_bottom, v_top], dim=-1)  

class RolloutDataset(Dataset):
    def __init__(self,
                 root: str,
                 box_minx: float = 0.2, box_miny: float = 0.2,
                 box_maxx: float = 0.8, box_maxy: float = 0.8,
                 dt_frame: float = 2e-3,
                 g_y: float = -50.0,
                 connect_radius: float = 0.015,
                 preload_points: bool = False,
                 device: str = "cuda"):
        super().__init__()
        self.root = root
        self.device = device
        self.default_box = (box_minx, box_miny, box_maxx, box_maxy)
        self.dt = float(dt_frame)
        self.g = float(g_y)
        self.R = float(connect_radius)

        self.paths = sort_rollout_paths(root)
        if not self.paths:
            raise FileNotFoundError(f"No files at {root}/rollout_*.npz")

        with np.load(self.paths[0]) as Z:
            self.N = int(Z["positions"].shape[1])

        self.index: List[Tuple[int, int]] = []
        self.preload_P: Optional[List[torch.Tensor]] = [] if preload_points else None
        for ridx, p in tqdm(enumerate(self.paths), desc="Indexing rollouts", total=len(self.paths)):
            with np.load(p) as Z:
                P = Z["positions"]
                T = int(P.shape[0])
                for t in range(5, T - 1):
                    self.index.append((ridx, t))
                if preload_points:
                    self.preload_P.append(torch.from_numpy(P).to(self.device))

    def __len__(self) -> int:
        return len(self.index)

    # get single sample, with particles in time t, t-1, and t+1 (for computing target acceleration)
    def __getitem__(self, k: int) -> Dict[str, Any]:
        ridx, t = self.index[k]
        bx0, by0, bx1, by1 = self.default_box
        if self.preload_P is None:
            with np.load(self.paths[ridx]) as Z:
                P = torch.from_numpy(Z["positions"]).to(self.device)  # [T,N,2]
        else:
            P = self.preload_P[ridx]

        p_t   = P[t]
        p_tm1 = P[t-1]
        p_tm2 = P[t-2]
        p_tm3 = P[t-3]
        p_tm4 = P[t-4]
        p_tm5 = P[t-5]
        p_tp1 = P[t+1]

        # finite difference velocity (in unit space, so dt = 1)
        v_tm4 = p_tm4 - p_tm5
        v_tm3 = p_tm3 - p_tm4
        v_tm2 = p_tm2 - p_tm3
        v_tm1 = p_tm1 - p_tm2
        v_t   = p_t   - p_tm1
        vel_hist = torch.stack([v_tm4, v_tm3, v_tm2, v_tm1, v_t], dim=1)

        # distance to walls
        node_feats = get_dist(p_t, bx0, by0, bx1, by1, R=self.R)
        globals_vec = torch.tensor([self.dt, self.g], device=self.device, dtype=torch.float32)  # [2]

        return {
            "path": self.paths[ridx],
            "t": int(t),
            "pos_t": p_t, "pos_tm1": p_tm1, "pos_tp1": p_tp1,
            "vel_hist": vel_hist, "node_feats": node_feats,
            "globals": globals_vec,
        }
