import os, shutil, tempfile
import imageio
from typing import Dict, Any, Optional, Tuple
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GNS import GNSModel, GNSConfig
from rollout_dataset import RolloutDataset, get_dist
from utils import (
    OnlineNormalizer, 
    apply_random_walk_noise_unit, 
    build_edges_and_attr, 
    lr_at_step, 
    load_training_state,
    TaichiVisualizer,
)


@torch.no_grad()
def validate(
    model: GNSModel,
    val_dataset: RolloutDataset,
    norm_node: Optional[OnlineNormalizer],
    norm_edge: Optional[OnlineNormalizer],
    norm_tgt: Optional[OnlineNormalizer],
    device: str = "cuda",
    max_batches: Optional[int] = 100,
    visualize_taichi: bool = False,
    vis_save_path: Optional[str] = None,
    vis_tmpdir: Optional[str] = None,
    vis_fps: int = 30,
    vis_res: int = 512,
    vis_max_points: int = 20000,
    viz: Optional[TaichiVisualizer] = None,
    vis_index: int = 0,
    draw_traj: bool = False,
) -> Dict[str, Any]:
    model.eval()
    if norm_tgt is not None:
        tgt_mean, tgt_std = norm_tgt.stats()
        denorm = lambda a: a * tgt_std + tgt_mean
    else:
        denorm = lambda a: a

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: b[0])

    acc_mse_sum = 0.0
    n_mse = 0
    box = val_dataset.default_box; R = val_dataset.R

    print(f"[val] Running validation on {len(val_dataset)} samples" + (f", max {max_batches} batches" if max_batches is not None else "") )
    for b_idx, s in tqdm(enumerate(loader), total=(len(val_dataset) if max_batches is None else min(len(val_dataset), max_batches))):
        # if max_batches is set, only run that many batches for efficiency
        if max_batches is not None and b_idx >= max_batches: break
        
        pos_t, pos_tm1, pos_tp1 = s["pos_t"], s["pos_tm1"], s["pos_tp1"]
        vel_hist_u = s["vel_hist"]; globals_g = s["globals"]  # [2]

        # finite difference target acceleration
        a_tgt = pos_tp1 - 2.0*pos_t + pos_tm1
        nf    = get_dist(pos_t, *box, R=R)
        vel_f = vel_hist_u.reshape(vel_hist_u.shape[0], -1)
        node_in = torch.cat([nf, vel_f], dim=-1)              # [N,18]
        edge_index, edge_attr = build_edges_and_attr(pos_t, R, max_nn=model.cfg.max_neighbors)

        node_in_n   = norm_node.normalize(node_in) if norm_node else node_in
        edge_attr_n = norm_edge.normalize(edge_attr) if norm_edge else edge_attr
        a_tgt_n     = norm_tgt.normalize(a_tgt) if norm_tgt else a_tgt

        a_pred_n = model(pos_t=pos_t, node_feats=node_in_n,
                         edge_index=edge_index, edge_feats=edge_attr_n,
                         globals_g=globals_g)
        acc_mse_sum += float(F.mse_loss(a_pred_n, a_tgt_n).item()); n_mse += 1


    out = {
        "one_step_accel_mse_norm": acc_mse_sum / max(n_mse, 1),
        "evaluated_batches":       int(min(len(val_dataset), max_batches) if max_batches is not None else len(val_dataset)),
    }

    if visualize_taichi:
        if vis_index < len(val_dataset.paths):
            path = val_dataset.paths[vis_index]
        else:
            # default: visualize the first rollout in the dataset
            path = val_dataset.paths[0]
        with np.load(path) as Z:
            P_gt = torch.from_numpy(Z["positions"]).to(device)  # [T,N,2]
        T, N = int(P_gt.shape[0]), int(P_gt.shape[1])

        t0 = 5
        H = T - (t0 + 1) # horizon, actual time steps to simulate

        p = P_gt[t0].clone()
        v_tm4 = P_gt[t0-4] - P_gt[t0-5]
        v_tm3 = P_gt[t0-3] - P_gt[t0-4]
        v_tm2 = P_gt[t0-2] - P_gt[t0-3]
        v_tm1 = P_gt[t0-1] - P_gt[t0-2]
        v_t   = P_gt[t0]   - P_gt[t0-1]
        vbuf  = torch.stack([v_tm4, v_tm3, v_tm2, v_tm1, v_t], dim=1)
        v     = vbuf[:, -1, :].clone()
        globals_g = torch.tensor([val_dataset.dt, val_dataset.g], device=device, dtype=torch.float32)

        P_pred_frames = [p.clone()]
        P_gt_frames   = [P_gt[t0].clone()]
        cur_t = t0

        for _ in tqdm(range(H), desc="[viz] Simulating rollout for visualization"):
            edge_index, edge_attr = build_edges_and_attr(p, val_dataset.R, max_nn=model.cfg.max_neighbors)
            nf   = get_dist(p, *box, R=val_dataset.R)
            velf = vbuf.reshape(vbuf.shape[0], -1)
            node_in = torch.cat([nf, velf], dim=-1)  # [N,18]

            node_in_n   = norm_node.normalize(node_in) if norm_node else node_in
            edge_attr_n = norm_edge.normalize(edge_attr) if norm_edge else edge_attr
            a_pred_n    = model(pos_t=p, node_feats=node_in_n,
                                edge_index=edge_index, edge_feats=edge_attr_n,
                                globals_g=globals_g)
            a_pred = denorm(a_pred_n)

            # use simple explicit Euler integration
            v = v + a_pred
            p = p + v

            P_pred_frames.append(p.clone())
            cur_t += 1
            P_gt_frames.append(P_gt[cur_t].clone())

            vbuf = torch.cat([vbuf[:, 1:, :], v.unsqueeze(1)], dim=1)

        P_pred_seq = torch.stack(P_pred_frames, dim=0)  # [F,N,2]
        P_gt_seq   = torch.stack(P_gt_frames,   dim=0)  # [F,N,2]

        final_mp4 = vis_save_path or "validate_taichi.mp4"
        tmp_dir = vis_tmpdir or (os.path.splitext(final_mp4)[0] + "_frames")
        

        if viz is None:
            viz = TaichiVisualizer(box=box, target_H=vis_res)
        viz.render_seq(P_pred_seq, P_gt_seq, final_mp4, tmp_dir,
                       fps=vis_fps, max_points_vis=vis_max_points,
                       track_samples=draw_traj,)

    return out


def train(
    model: GNSModel,
    train_dataset: RolloutDataset,
    val_dataset: Optional[RolloutDataset] = None,
    start_step: int = 1,
    norm_node: Optional[OnlineNormalizer] = None,
    norm_edge: Optional[OnlineNormalizer] = None,
    norm_tgt: Optional[OnlineNormalizer] = None,
    max_steps: int = 200_000,
    lr_start: float = 1e-4, lr_final: float = 1e-6,
    log_every: int = 200, val_every: int = 10_000,
    save_every: int = 0,
    vis_every: int = 0,
    device: str = "cuda", output_dir: Optional[str] = None,
    viz: Optional[TaichiVisualizer] = None,
    no_apply_noise: bool = False,
    draw_traj: bool = False,
) -> Dict[str, OnlineNormalizer]:
    if norm_node is None or norm_edge is None or norm_tgt is None:
        raise ValueError("All normalizers (norm_node, norm_edge, norm_tgt) must be provided to train()")
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr_start)
    writer = SummaryWriter(output_dir) if output_dir else None

    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda b: b[0])
    it = iter(loader)

    # Early exit if already done
    if start_step > max_steps:
        print(f"[train] start_step ({start_step}) > max_steps ({max_steps}). Nothing to train.")
        if writer: writer.close()
        return {"norm_node": norm_node, "norm_edge": norm_edge, "norm_tgt": norm_tgt}

    ema = 0.0
    print(f"[train] Starting training from step {start_step} to {max_steps} on device '{device}'")
    for step in range(start_step, max_steps + 1):
        
        # iterate all data loader, restart if needed
        try: s = next(it)
        except StopIteration:
            it = iter(loader); s = next(it)
    
        pos_t, pos_tm1, pos_tp1 = s["pos_t"], s["pos_tm1"], s["pos_tp1"]
        vel_hist_u = s["vel_hist"]; globals_g = s["globals"]
        bx0, by0, bx1, by1 = train_dataset.default_box; R = train_dataset.R
        a_tgt_clean = pos_tp1 - 2.0*pos_t + pos_tm1
        
        # purturb the position with random-walk noise
        vel_noisy_u, final_noise, pos_adj = apply_random_walk_noise_unit(vel_hist_u, sigma_v=3e-4, add_noise=(not no_apply_noise))
        pos_t_noisy = pos_t + pos_adj[:, -1, :]

        node_feats_noisy = get_dist(pos_t_noisy, bx0, by0, bx1, by1, R=R)
        vel_flat = vel_noisy_u.reshape(vel_noisy_u.shape[0], -1)
        node_in = torch.cat([node_feats_noisy, vel_flat], dim=-1)  # [N,18]
        a_target = a_tgt_clean - final_noise

        edge_index, edge_attr = build_edges_and_attr(pos_t_noisy, R, max_nn=model.cfg.max_neighbors)

        norm_node.update(node_in); norm_edge.update(edge_attr); norm_tgt.update(a_target)
        node_in_n   = norm_node.normalize(node_in)
        edge_attr_n = norm_edge.normalize(edge_attr)
        a_target_n  = norm_tgt.normalize(a_target)

        a_pred_n = model(pos_t=pos_t_noisy, node_feats=node_in_n,
                        edge_index=edge_index, edge_feats=edge_attr_n,
                        globals_g=globals_g)
        loss = F.mse_loss(a_pred_n, a_target_n)

        # lr schedule
        for g in opt.param_groups:
            g["lr"] = lr_at_step(step, total_steps=max_steps, lr_start=lr_start, lr_final=lr_final)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        ema = 0.98*ema + 0.02*loss.item()
        if step % log_every == 0:
            print(f"[step {step}/{max_steps}] loss={loss.item():.6e}  ema={ema:.6e}  lr={opt.param_groups[0]['lr']:.2e}")
            if writer:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/ema_loss", ema, step)
                writer.add_scalar("train/lr", opt.param_groups[0]['lr'], step)

        if val_dataset is not None and (step % val_every == 0 or step == max_steps or step == 1):
            if vis_every > 0 and (step % vis_every == 0 or step == max_steps or step == 1):
                vis_tmpdir = None if output_dir is None else os.path.join(output_dir, f"val_vis_step{step}")
                stats = validate(model, val_dataset, norm_node, norm_edge, norm_tgt,
                                device=device, max_batches=100,
                                visualize_taichi=True,
                                vis_save_path=(os.path.join(output_dir, f"val_vis_step{step}.mp4") if output_dir else None),
                                vis_tmpdir=vis_tmpdir,
                                vis_fps=30, vis_res=512, vis_max_points=20000, viz=viz,
                                draw_traj=draw_traj,)
            else:
                stats = validate(model, val_dataset, norm_node, norm_edge, norm_tgt,
                                device=device, max_batches=100,
                                )
            print("[val] " + ", ".join(f"{k}={v:.6e}" if isinstance(v,float) else f"{k}={v}" for k,v in stats.items()))
            if writer:
                for k, v in stats.items():
                    if isinstance(v, float):
                        writer.add_scalar(f"val/{k}", v, step)
            torch.cuda.empty_cache()
            model.train()
            
        if save_every > 0 and (step % save_every == 0 or step == max_steps):
            save_path = os.path.join(output_dir if output_dir else ".", f"model_step{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "norm_node": {
                    "n": norm_node.n.detach().cpu(), "sum": norm_node.sum.detach().cpu(), "sumsq": norm_node.sumsq.detach().cpu()
                },
                "norm_edge": {
                    "n": norm_edge.n.detach().cpu(), "sum": norm_edge.sum.detach().cpu(), "sumsq": norm_edge.sumsq.detach().cpu()
                },
                "norm_tgt": {
                    "n": norm_tgt.n.detach().cpu(), "sum": norm_tgt.sum.detach().cpu(), "sumsq": norm_tgt.sumsq.detach().cpu()
                },
            }, save_path)
            print(f"[save] Saved model checkpoint to {save_path}")

    if writer: writer.close()
    return {"norm_node": norm_node, "norm_edge": norm_edge, "norm_tgt": norm_tgt}


def test_only(
    checkpoint_path: str,
    test_dataset: RolloutDataset,
    cfg: GNSConfig,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    output_dir: Optional[str] = None,
    visualize: bool = True,
    vis_index: int = 0,
    vis_fps: int = 30,
    vis_res: int = 512,
    vis_max_points: int = 20000,
    draw_traj: bool = False,
) -> Dict[str, Any]:
    
    model = GNSModel(cfg).to(device)
    # dummy
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    node_in_dim = 18
    step, norm_node, norm_edge, norm_tgt = load_training_state(
        checkpoint_path,
        model=model,
        opt=opt,
        device=device,
        node_in_dim=node_in_dim
    )
    
    print(f"\n[test] Running test evaluation on checkpoint from step {step - 1}")
    
    viz = None
    if visualize:
        viz = TaichiVisualizer(
            box=test_dataset.default_box,
            target_H=vis_res,
        )
    
    stats = validate(
        model, test_dataset,
        norm_node=norm_node, norm_edge=norm_edge, norm_tgt=norm_tgt,
        device=device, max_batches=max_batches,
        visualize_taichi=visualize,
        vis_save_path=(os.path.join(output_dir, f"test_vis_{vis_index}.mp4") if output_dir else f"test_vis_{vis_index}.mp4"),
        vis_tmpdir=(os.path.join(output_dir, f"test_vis_tmp") if output_dir else "test_vis_tmp"),
        vis_fps=vis_fps, vis_res=vis_res, vis_max_points=vis_max_points,
        viz=viz, vis_index=vis_index, draw_traj=draw_traj
    )
    
    print("\n[test] Test Results:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, "test_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"\n[save] Saved test stats to {stats_path}")
    
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train_root", type=str, default="rollouts_train")
    ap.add_argument("--val_root", type=str, default="rollouts_val")
    ap.add_argument("--test_root", type=str, default="rollouts_test")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=10000)
    ap.add_argument("--vis_every", type=int, default=20000)
    ap.add_argument("--save_every", type=int, default=50000)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--test_only", action="store_true", help="Run test-only evaluation")
    ap.add_argument("--test_checkpoint", type=str, default=None, help="Checkpoint path for test-only mode")
    ap.add_argument("--test_vis_index", type=int, default=0)
    ap.add_argument("--test_max_batches", type=int, default=100)
    ap.add_argument("--box_minx", type=float, default=0.2)
    ap.add_argument("--box_miny", type=float, default=0.2)
    ap.add_argument("--box_maxx", type=float, default=0.8)
    ap.add_argument("--box_maxy", type=float, default=0.8)
    ap.add_argument("--dt_frame", type=float, default=2e-3)
    ap.add_argument("--g_y", type=float, default=-50.0)
    ap.add_argument("--no_apply_noise", action="store_true", help="Disable random-walk noise in training")
    ap.add_argument("--draw_traj", action="store_true", help="In visualization, draw trajectories of some tracked particles")
    args = ap.parse_args()

    # Adjust config in GNS.py if needed
    cfg = GNSConfig()
    
    ds_kwargs = dict(
        box_minx=args.box_minx, box_miny=args.box_miny,
        box_maxx=args.box_maxx, box_maxy=args.box_maxy,
        dt_frame=args.dt_frame, g_y=args.g_y,
        connect_radius=cfg.connect_radius, preload_points=False,
        device=args.device
    )

    if args.test_only:
        if args.test_checkpoint is None:
            raise ValueError("--test_checkpoint must be provided when using --test_only")
        
        print("Test-only mode")
        
        print("Initializing test dataset...")
        test_ds = RolloutDataset(root=args.test_root, **ds_kwargs)
        
        test_only(
            checkpoint_path=args.test_checkpoint,
            test_dataset=test_ds,
            cfg=cfg,
            device=args.device,
            max_batches=args.test_max_batches,
            output_dir=args.output_dir,
            visualize=True,
            vis_index=args.test_vis_index,
            vis_fps=30,
            vis_res=512,
            vis_max_points=20000,
            draw_traj=args.draw_traj,
        )
        
        print("\nTest-only evaluation complete!")
        exit(0)

    print("Training mode")
    
    model = GNSModel(cfg).to(args.device)
    
    print("Initializing datasets...")
    train_ds = RolloutDataset(root=args.train_root, **ds_kwargs)
    val_ds   = RolloutDataset(root=args.val_root, **ds_kwargs)
    test_ds  = RolloutDataset(root=args.test_root, **ds_kwargs)

    viz = TaichiVisualizer(
        box=(args.box_minx, args.box_miny, args.box_maxx, args.box_maxy),
        target_H=512,
    ) if args.vis_every > 0 else None

    node_in_dim = 18
    opt_temp = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_step, norm_node, norm_edge, norm_tgt = load_training_state(
        args.resume_from,
        model=model,
        opt=opt_temp,
        device=args.device,
        node_in_dim=node_in_dim
    )
    del opt_temp  

    norms = train(
        model, train_ds, val_ds,
        start_step=start_step,
        norm_node=norm_node,
        norm_edge=norm_edge,
        norm_tgt=norm_tgt,
        max_steps=args.max_steps, 
        log_every=args.log_every, 
        val_every=args.val_every, 
        save_every=args.save_every,
        vis_every=args.vis_every,
        device=args.device,
        output_dir=args.output_dir,
        viz=viz,
        no_apply_noise=args.no_apply_noise,
        draw_traj=args.draw_traj,
    )

    # Save final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        final_path = os.path.join(args.output_dir, f"model_final.pt")
        torch.save({
            "step": args.max_steps,
            "model_state_dict": model.state_dict(),
            "norm_node": {
                "n": norms["norm_node"].n.detach().cpu(),
                "sum": norms["norm_node"].sum.detach().cpu(),
                "sumsq": norms["norm_node"].sumsq.detach().cpu()
            },
            "norm_edge": {
                "n": norms["norm_edge"].n.detach().cpu(),
                "sum": norms["norm_edge"].sum.detach().cpu(),
                "sumsq": norms["norm_edge"].sumsq.detach().cpu()
            },
            "norm_tgt": {
                "n": norms["norm_tgt"].n.detach().cpu(),
                "sum": norms["norm_tgt"].sum.detach().cpu(),
                "sumsq": norms["norm_tgt"].sumsq.detach().cpu()
            },
        }, final_path)
        print(f"[save] Saved final model checkpoint to {final_path}")

    print("Final evaluation on test set...")
    stats = validate(
        model, test_ds,
        norm_node=norms["norm_node"], norm_edge=norms["norm_edge"], norm_tgt=norms["norm_tgt"],
        device=args.device, max_batches=args.test_max_batches,
        visualize_taichi=True,
        vis_save_path=(os.path.join(args.output_dir, f"test_vis_{args.test_vis_index}.mp4") if args.output_dir else f"test_vis_{args.test_vis_index}.mp4"),
        vis_tmpdir=(os.path.join(args.output_dir, f"test_vis_tmp") if args.output_dir else "test_vis_tmp"),
        vis_fps=30, vis_res=512, vis_max_points=20000, viz=viz, vis_index=args.test_vis_index, draw_traj=args.draw_traj
    )
    print("[test] " + ", ".join(f"{k}={v:.6e}" if isinstance(v,float) else f"{k}={v}" for k,v in stats.items()))
    
    # Save test stats as json
    if args.output_dir:
        stats_path = os.path.join(args.output_dir, f"test_stats_{args.test_vis_index}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[save] Saved final test stats to {stats_path}")
    
    print("\nTraining and evaluation complete!")