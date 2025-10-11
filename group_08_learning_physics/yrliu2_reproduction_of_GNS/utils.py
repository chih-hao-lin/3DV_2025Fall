import torch
import os
import imageio
import numpy as np
import shutil
from typing import Optional, Tuple

# Training utils

class OnlineNormalizer:
    def __init__(self, dim: int, device="cuda", eps=1e-8):
        self.device = device; self.eps = eps
        self.n = torch.zeros((), device=device)
        self.sum = torch.zeros(dim, device=device)
        self.sumsq = torch.zeros(dim, device=device)
    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        self.n += x.shape[0]
        self.sum += x.sum(dim=0)
        self.sumsq += (x * x).sum(dim=0)
    def stats(self):
        n = torch.clamp(self.n, min=1.0)
        mean = self.sum / n
        var = torch.clamp(self.sumsq / n - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        return mean, std
    def normalize(self, x: torch.Tensor):
        mean, std = self.stats()
        return (x - mean) / std

def lr_at_step(step: int, total_steps: int, lr_start=1e-4, lr_final=1e-6, decades: int = 4):
    # mimic the exponential decay used in the paper
    denom = max(total_steps / decades, 1)
    return lr_final + (lr_start - lr_final) * (0.1 ** (step / denom))

@torch.no_grad()
def apply_random_walk_noise_unit(vel_hist_unit: torch.Tensor, sigma_v: float = 3e-4, add_noise=True):
    if not add_noise:
        # for ablation
        N = vel_hist_unit.shape[0]
        vel_noisy = vel_hist_unit
        final_noise = torch.zeros((N, 2), device=vel_hist_unit.device)
        pos_adjust = torch.zeros((N, 6, 2), device=vel_hist_unit.device)
        return vel_noisy, final_noise, pos_adjust
    eps = torch.randn_like(vel_hist_unit) * sigma_v
    rw = torch.cumsum(eps, dim=1)
    vel_noisy = vel_hist_unit + rw
    final_noise = rw[:, -1, :]
    N = vel_hist_unit.shape[0]
    pos_adjust = torch.zeros((N, 6, 2), device=vel_hist_unit.device)
    acc = torch.zeros((N, 2), device=vel_hist_unit.device)
    # return the position adjustment at each of the next 5 frames
    for s in range(5):
        acc = acc + rw[:, s, :]
        pos_adjust[:, s+1, :] = acc
    return vel_noisy, final_noise, pos_adjust

def build_edges_and_attr(pos: torch.Tensor, R: float, max_nn: Optional[int] = None):
    from torch_geometric.nn import radius_graph
    edge_index = radius_graph(
        x=pos, r=R, loop=False,
        max_num_neighbors=(max_nn if max_nn is not None else 2**31 - 1)
    )
    src, dst = edge_index
    rel = pos[src] - pos[dst]
    rel_norm = rel.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([rel, rel_norm], dim=-1)
    return edge_index, edge_attr

# Saving and loading utils

def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: str):
    for state in opt.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_training_state(
    resume_from: Optional[str],
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device: str,
    node_in_dim: int = 18,   
) -> Tuple[int, "OnlineNormalizer", "OnlineNormalizer", "OnlineNormalizer"]:
    
    # load existing ckpt if provided, else return fresh state
    
    norm_node = OnlineNormalizer(node_in_dim, device=device)
    norm_edge = OnlineNormalizer(3, device=device)
    norm_tgt  = OnlineNormalizer(2, device=device)

    start_step = 1
    if not resume_from:
        print("[resume] No checkpoint provided; starting fresh.")
        return start_step, norm_node, norm_edge, norm_tgt

    try:
        ckpt = torch.load(resume_from, map_location=device)

        if "model_state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"[resume] Loaded model. missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[resume] WARNING: no 'model_state_dict' in checkpoint.")

        if "opt_state_dict" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt_state_dict"])
                _move_optimizer_state_to_device(opt, device)
                print("[resume] Loaded optimizer state.")
            except Exception as e:
                print(f"[resume] WARNING: failed to load optimizer state: {e}")
        else:
            print("[resume] No optimizer state found; starting optimizer fresh.")

        def _load_norm(dst_norm: OnlineNormalizer, src: dict, name: str):
            dst_norm.n     = src["n"].to(device)
            dst_norm.sum   = src["sum"].to(device)
            dst_norm.sumsq = src["sumsq"].to(device)
            print(f"[resume] Restored {name} normalizer.")

        if "norm_node" in ckpt:
            try: _load_norm(norm_node, ckpt["norm_node"], "node")
            except Exception as e: print(f"[resume] WARNING: could not restore node normalizer: {e}")
        else:
            print("[resume] No norm_node in checkpoint; reinitializing.")

        if "norm_edge" in ckpt:
            try: _load_norm(norm_edge, ckpt["norm_edge"], "edge")
            except Exception as e: print(f"[resume] WARNING: could not restore edge normalizer: {e}")
        else:
            print("[resume] No norm_edge in checkpoint; reinitializing.")

        if "norm_tgt" in ckpt:
            try: _load_norm(norm_tgt, ckpt["norm_tgt"], "tgt")
            except Exception as e: print(f"[resume] WARNING: could not restore tgt normalizer: {e}")
        else:
            print("[resume] No norm_tgt in checkpoint; reinitializing.")

        # Step
        if isinstance(ckpt.get("step"), int):
            start_step = ckpt["step"] + 1
            print(f"[resume] Resuming from step {ckpt['step']} -> start at {start_step}.")
        else:
            print("[resume] No valid 'step' found; starting from step 1.")

    except FileNotFoundError:
        print(f"[resume] WARNING: checkpoint '{resume_from}' not found. Starting fresh.")
    except Exception as e:
        print(f"[resume] WARNING: failed to load checkpoint '{resume_from}': {e}")
        print("[resume] Continuing with fresh training state.")

    return start_step, norm_node, norm_edge, norm_tgt

# Visualization utils

# to prevent re-initializing Taichi GUI multiple times
_ti_mod = None
_gui = None
_gui_res = None

def _get_taichi_and_gui(res: int, bg=0xF6F6F6):
    global _ti_mod, _gui, _gui_res
    if _ti_mod is None:
        import taichi as ti
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
        _ti_mod = ti
    if _gui is None or _gui_res != res:
        _gui = _ti_mod.GUI("GNS: GT (left) vs Pred (right)", res=res, background_color=bg, show_gui=False)
        _gui_res = res
    return _ti_mod, _gui

class TaichiVisualizer:
    """
    Renders two side-by-side viewports in a single Taichi GUI:
      - left  = Ground Truth
      - right = Prediction
    """
    def __init__(self, box: tuple, target_H: int = 512, bg=0xF6F6F6,
                 pad_x: float = 0.02, pad_y: float = 0.02):
        """
        box: (xmin, ymin, xmax, ymax) in world coords
        pad_x, pad_y: margins around each viewport (in normalized screen coords)
        """
        self.box = tuple(map(float, box))
        self.bg = bg
        self.target_H = target_H
        self.pad_x = float(pad_x)
        self.pad_y = float(pad_y)

        target_H = target_H
        ratio = (1 - 2*pad_y) / (0.5 - 2*pad_x)
        target_W = int(round(ratio * target_H))
        self.ti, self.gui = _get_taichi_and_gui((target_W, target_H), self.bg)  # res=(W,H)

        # Viewport rectangles (normalized screen coords)
        self.left_x0  = self.pad_x
        self.left_x1  = 0.5 - self.pad_x
        self.right_x0 = 0.5 + self.pad_x
        self.right_x1 = 1.0 - self.pad_x
        self.y0 = self.pad_y
        self.y1 = 1.0 - self.pad_y

        self.left_w  = self.left_x1  - self.left_x0
        self.right_w = self.right_x1 - self.right_x0
        self.vh = self.y1 - self.y0  

        xmin, ymin, xmax, ymax = self.box
        xs = np.linspace(xmin, xmax, 100, dtype=np.float32)
        ys = np.linspace(ymin, ymax, 100, dtype=np.float32)
        self._border_world = {
            "bottom": np.stack([xs, np.full_like(xs, ymin)], axis=1),
            "top":    np.stack([xs, np.full_like(xs, ymax)], axis=1),
            "left":   np.stack([np.full_like(ys, xmin), ys], axis=1),
            "right":  np.stack([np.full_like(ys, xmax), ys], axis=1),
        }

        self._palette = [
            0xE53935, 0x8E24AA, 0x3949AB, 0x1E88E5, 0x00897B,
            0x43A047, 0xFDD835, 0xFB8C00, 0x6D4C41, 0x00ACC1,
            0x7CB342, 0xD81B60, 0x5E35B1, 0x039BE5, 0xC0CA33,
            0xF4511E, 0x546E7A, 0x26C6DA, 0x9CCC65, 0xEC407A
        ]

    def _map_world_to_left(self, P: np.ndarray) -> np.ndarray:
        # P: [*, 2] in world [0,1]^2 → screen coords inside left viewport
        out = np.empty_like(P)
        out[:, 0] = self.left_x0 + self.left_w * P[:, 0]
        out[:, 1] = self.y0 + self.vh * P[:, 1]
        return out

    def _map_world_to_right(self, P: np.ndarray) -> np.ndarray:
        out = np.empty_like(P)
        out[:, 0] = self.right_x0 + self.right_w * P[:, 0]
        out[:, 1] = self.y0 + self.vh * P[:, 1]
        return out

    def _draw_border_left(self):
        for key in ("bottom", "top", "left", "right"):
            self.gui.circles(self._map_world_to_left(self._border_world[key]),
                             radius=1, color=0x666666)

    def _draw_border_right(self):
        for key in ("bottom", "top", "left", "right"):
            self.gui.circles(self._map_world_to_right(self._border_world[key]),
                             radius=1, color=0x666666)

    def _draw_polyline(self, pts_world: np.ndarray, map_fn, color: int, radius: float = 1.0):
        """
        Draw a polyline as consecutive segments through pts_world[T, 2].
        """
        if pts_world.shape[0] < 2:
            return
        a = pts_world[:-1]
        b = pts_world[1:]
        a_m = map_fn(a)
        b_m = map_fn(b)
        self.gui.lines(a_m, b_m, radius=radius, color=color)

    @torch.no_grad()
    def render_seq(self,
        P_pred: torch.Tensor,     # [F, N, 2]
        P_gt: torch.Tensor,       # [F, N, 2]
        save_path: str,
        tmp_dir: str,
        fps: int = 30,
        max_points_vis: int = 5000,
        track_samples: bool = False,
        sample_count: int = 10,
        sample_indices: np.ndarray | None = None,
        sample_seed: int | None = 0,
        show_sample_labels: bool = False,
        traj_radius: float = 1.0,
        traj_every_k: int = 1,
    ):
        # clean old tmp files
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        P_pred = P_pred.detach().cpu().numpy().astype(np.float32)
        P_gt   = P_gt.detach().cpu().numpy().astype(np.float32)
        F, N, _ = P_pred.shape

        if max_points_vis is not None and N > max_points_vis:
            idx_vis = np.random.permutation(N)[:max_points_vis]
            P_pred_vis = P_pred[:, idx_vis, :]
            P_gt_vis   = P_gt[:, idx_vis, :]
        else:
            P_pred_vis = P_pred
            P_gt_vis   = P_gt

        if track_samples:
            print("[viz] Tracking sample trajectories...")
            # choose fixed IDs
            if sample_indices is None:
                k = min(sample_count, N)
                rng = np.random.default_rng(sample_seed) if sample_seed is not None else np.random.default_rng()
                sample_indices = np.sort(rng.choice(N, size=k, replace=False))
            else:
                sample_indices = np.array(sample_indices, dtype=int)
                sample_indices = sample_indices[(sample_indices >= 0) & (sample_indices < N)]
                if sample_indices.size == 0:
                    raise ValueError("sample_indices ended up empty after filtering.")
            colors = [self._palette[i % len(self._palette)] for i in range(sample_indices.size)]
            frame_idxs = np.arange(F)[::max(1, int(traj_every_k))]
            gt_trajs   = P_gt[frame_idxs][:, sample_indices, :]
            pred_trajs = P_pred[frame_idxs][:, sample_indices, :]

        for t in range(F):
            self.gui.clear(self.bg)

            self._draw_border_left()
            self._draw_border_right()

            gt_l   = self._map_world_to_left(P_gt_vis[t])
            pred_r = self._map_world_to_right(P_pred_vis[t])
            self.gui.circles(gt_l,   radius=1.0, color=0x0055FF)  # GT (left)
            self.gui.circles(pred_r, radius=1.0, color=0xFF6600)  # Pred (right)

            if track_samples:
                t_mask = (np.arange(F)[::max(1, int(traj_every_k))] <= t)
                if np.any(t_mask):
                    for j, color in enumerate(colors):
                        gt_path   = gt_trajs[t_mask, j, :]
                        pred_path = pred_trajs[t_mask, j, :]
                        self._draw_polyline(gt_path,   self._map_world_to_left,  color=color, radius=traj_radius)
                        self._draw_polyline(pred_path, self._map_world_to_right, color=color, radius=traj_radius)

                gt_now_l   = self._map_world_to_left(P_gt[t, sample_indices, :])
                pred_now_r = self._map_world_to_right(P_pred[t, sample_indices, :])
                for j, color in enumerate(colors):
                    self.gui.circles(gt_now_l[j:j+1],   radius=3.0, color=color)
                    self.gui.circles(pred_now_r[j:j+1], radius=3.0, color=color)
                    if show_sample_labels:
                        try:
                            self.gui.text(f"{sample_indices[j]}", pos=(gt_now_l[j,0]+0.005, gt_now_l[j,1]+0.005), color=color)
                            self.gui.text(f"{sample_indices[j]}", pos=(pred_now_r[j,0]+0.005, pred_now_r[j,1]+0.005), color=color)
                        except Exception:
                            pass

            
            try:
                self.gui.text("GT",   pos=(self.left_x0 + 0.01,  self.y1 - 0.02),  color=0x000000)
                self.gui.text("Pred", pos=(self.right_x0 + 0.01, self.y1 - 0.02), color=0x000000)
            except Exception:
                pass

            self.gui.show(os.path.join(tmp_dir, f"frame_{t:06d}.png"))

        # Compile MP4
        frames = [imageio.imread(os.path.join(tmp_dir, f))
                for f in sorted(os.listdir(tmp_dir)) if f.endswith(".png")]
        imageio.mimwrite(save_path, frames, fps=fps)
        print(f"[viz] wrote {save_path}")
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            raise RuntimeError(f"Failed to remove temp dir '{tmp_dir}'")