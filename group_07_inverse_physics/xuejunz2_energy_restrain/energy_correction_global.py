#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global energy-aware post-correction (both x & y), very gentle.

Idea:
  - At each step t, compute mean KE over non-kinematic particles: KE_t = mean(0.5*||v||^2)
  - Reference KE_ref = mean of first K steps of KE_t
  - Raw scale s_raw = sqrt(KE_ref / (KE_t + eps))
  - Blend: s = 1 + beta * (s_raw - 1)    (beta << 1 makes it gentle)
  - Clamp asymmetrically: s in [1 - gmax_down, 1 + gmax_up]
  - Apply to ALL non-kinematic particles at step t (direction preserved)
  - Align per-step mean velocity back to baseline (keep gravity / drift)
  - Light EMA smoothing on velocities (both axes)
  - Integrate back to positions. Kinematic particles untouched.

Default is extremely conservative: beta=0.35, gmax_up=0.01, gmax_down=0.02, ema=0.03
"""

import os, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt

KINEMATIC_PARTICLE_ID = 3

def _squeeze(x):
    if x is None: return None
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    return x

def load_roll(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        pred = d.get("predicted_rollout_mean", d.get("predicted_rollout"))
        if pred is None:
            raise ValueError("npz missing predicted_rollout(_mean)")
        gt   = d.get("ground_truth_rollout", None)
        ptyp = d.get("particle_types", None)
    else:
        with open(path, "rb") as f:
            ex = pickle.load(f)
        pred = ex["predicted_rollout"]
        gt   = ex.get("ground_truth_rollout", None)
        ptyp = ex.get("particle_types", None)
    return {"pred": _squeeze(pred), "gt": _squeeze(gt), "ptype": ptyp}

def velocities(pos):
    return pos[1:] - pos[:-1]   # (T-1,N,D)

def mse_t(gt, pred):
    return ((gt - pred) ** 2).mean(axis=(1, 2))

def global_energy_correction(pred, ptype=None,
                             K=10, beta=0.35,
                             gmax_up=0.01, gmax_down=0.02,
                             ema=0.03, eps=1e-12):
    """
    pred: (T,N,D) baseline rollout
    returns: positions_corr (T,N,D), scales s_t (T-1,)
    """
    T, N, D = pred.shape
    v_base = velocities(pred)                 # (T-1,N,D)
    nonkin = np.ones(N, bool) if ptype is None else (ptype != KINEMATIC_PARTICLE_ID)

    # mean KE per step over non-kinematics
    speed2 = (v_base[:, nonkin, :] ** 2).sum(axis=-1)             # (T-1, N_nonkin)
    KE = 0.5 * speed2.mean(axis=1)                                 # (T-1,)
    K = max(1, min(K, KE.shape[0]))
    KE_ref = float(np.maximum(KE[:K].mean(), eps))                 # scalar

    # per-step global scale
    s_raw = np.sqrt(KE_ref / (KE + eps))                           # (T-1,)
    s = 1.0 + beta * (s_raw - 1.0)                                 # gentle blend
    # asymmetric clamp
    s = np.minimum(s, 1.0 + gmax_up)
    s = np.maximum(s, 1.0 - gmax_down)

    # apply scale to non-kinematic particles (direction preserved)
    v_scaled = v_base.copy()
    v_scaled[:, nonkin, :] *= s[:, None, None]

    # align step-wise mean velocity back to baseline (keep gravity/drift)
    mean_base  = v_base.mean(axis=1, keepdims=True)                # (T-1,1,D)
    mean_scaled= v_scaled.mean(axis=1, keepdims=True)
    v_corr = v_scaled + (mean_base - mean_scaled)

    # light EMA smoothing on both axes
    if ema > 0.0:
        out = np.empty_like(v_corr)
        out[0] = v_corr[0]
        a = float(ema)
        for t in range(1, v_corr.shape[0]):
            out[t] = (1 - a) * out[t-1] + a * v_corr[t]
        v_corr = out

    # integrate back
    pos = np.empty_like(pred)
    pos[0] = pred[0]
    for t in range(1, T):
        pos[t] = pos[t-1] + v_corr[t-1]

    # keep kinematic particles identical to baseline
    if np.any(~nonkin):
        pos[:, ~nonkin, :] = pred[:, ~nonkin, :]

    return pos, s, KE, KE_ref

def quick_plot(gt, base, corr, scales, KE, KE_ref, out_png):
    # choose a representative particle: largest displacement
    disp = np.linalg.norm(base[1:] - base[:-1], axis=-1).sum(axis=0)
    p = int(disp.argmax())

    fig = plt.figure(figsize=(14,8))
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax1.plot(base[:,p,0], base[:,p,1], '--', label='Base')
    ax1.plot(corr[:,p,0], corr[:,p,1], ':', label='Global-corr')
    if gt is not None: ax1.plot(gt[:,p,0], gt[:,p,1], label='GT', lw=2)
    ax1.set_title(f"Trajectory (particle {p})"); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.legend()

    # MSE curve
    ax2 = plt.subplot2grid((2,3),(0,1))
    if gt is not None:
        ax2.plot(mse_t(gt, base), label='Base MSE')
        ax2.plot(mse_t(gt, corr), label='Global-corr MSE')
        ax2.set_ylabel('MSE')
    else:
        vnorm = lambda x: np.linalg.norm(velocities(x), axis=-1).mean(axis=1)
        ax2.plot(vnorm(base), label='Base mean|v|')
        ax2.plot(vnorm(corr), label='Global mean|v|')
        ax2.set_ylabel('mean|v|')
    ax2.set_xlabel('t'); ax2.set_title('Quality over time'); ax2.legend()

    # scales & energy timeline
    ax3 = plt.subplot2grid((2,3),(0,2))
    ax3.plot(scales, label='scale s_t')
    ax3.axhline(1.0, color='k', lw=1, ls='--')
    ax3.set_title('Applied global scale'); ax3.set_xlabel('t-1'); ax3.legend()

    ax4 = plt.subplot2grid((2,3),(1,0), colspan=3)
    ax4.plot(KE, label='mean KE (base, t-1)')
    ax4.axhline(KE_ref, color='k', lw=1, ls='--', label='KE_ref')
    ax4.set_title('Mean kinetic energy per step'); ax4.set_xlabel('t-1'); ax4.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved:", out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_png", required=True)
    # hyper-params
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--gmax_up", type=float, default=0.01)
    ap.add_argument("--gmax_down", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.03)
    args = ap.parse_args()

    data = load_roll(args.roll)
    base, gt, ptype = data["pred"], data["gt"], data["ptype"]

    corr, scales, KE, KE_ref = global_energy_correction(
        base, ptype=ptype,
        K=args.K, beta=args.beta,
        gmax_up=args.gmax_up, gmax_down=args.gmax_down,
        ema=args.ema
    )

    # sanity & summary
    v_b = velocities(base); v_c = velocities(corr)
    mean_delta = float(np.linalg.norm(v_b.mean(1) - v_c.mean(1), axis=1).max())
    print(f"Sanity: max ||mean_vel(base)-mean_vel(corr)|| = {mean_delta:.4e}")
    print(f"Scales: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")
    print(f"KE_ref = {KE_ref:.6e}")

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz,
        predicted_rollout=corr,
        ground_truth_rollout=gt if gt is not None else np.array([]),
        particle_types=ptype if ptype is not None else np.array([]),
        scales=scales, KE=KE, KE_ref=np.array([KE_ref])
    )
    print("Saved corrected rollout:", args.out_npz)

    quick_plot(gt, base, corr, scales, KE, KE_ref, args.out_png)

if __name__ == "__main__":
    main()
