from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

@dataclass
class Person:
    V: np.ndarray
    gamma: np.ndarray
    meta: Optional[Dict] = None

def collect_ground_candidates(P_list, Q_list, q_quantile = 0.60, lowest_quantile = 0.30, up_axis = 1, max_points = 800000):
    q_vals = np.asarray([q for q in Q_list if q is not None and np.isfinite(q)], dtype = np.float32)
    q_th = float(np.quantile(q_vals, q_quantile))

    tmp_pts = []
    for P, q in zip(P_list, Q_list):
        if (q is None) or (not np.isfinite(q)) or (q < q_th):
            continue
        
        P = np.asarray(P).reshape(3)
        if P is None:
            continue
        
        tmp_pts.append(P)

    # CAUTION: the sign is REVERSED!!(bigger y value with small height)
    P = np.asarray(tmp_pts)
    ht = P[:, up_axis]
    htq = np.quantile(ht, 1.00 - lowest_quantile)
    P = P[P[:, up_axis] >= htq]

    if len(P) > max_points:
        idx = np.random.choice(len(P), max_points, replace=False)
        P = P[idx]
       
    return P

def ransac_plane(P: np.ndarray, iters = 1500, inlier_thresh = 0.02, min_inlier_ratio = 0.6, up_axis = 1):
    rng = np.random.RandomState(42)
    best_inliers = None
    N = len(P)
    for _ in range(iters):
        idx = rng.choice(N, 3, replace = False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        nrm = np.linalg.norm(n)
        if nrm < 1e-9:
            continue
        n = n / nrm
        d = -np.dot(n, a)
        dist = np.abs(P @ n + d)
        inliers = dist < inlier_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            
    if best_inliers is None or best_inliers.sum() < max(50, int(min_inlier_ratio * N)):
        return None
    
    G = P[best_inliers]
    p_g = G.mean(0)
    _, _, Vt = np.linalg.svd(G - p_g, full_matrices = False)
    n_g = Vt[-1]
    n_g = n_g / (np.linalg.norm(n_g) + 1e-12)
    if up_axis is not None and n_g[up_axis] < 0:
        n_g = -n_g
        
    return n_g.astype(np.float32), p_g.astype(np.float32), best_inliers

def derive_foot_roi(V, up_axis = 1, k = 10):
    if V is None or len(V) == 0:
        return np.array([], np.int64)

    k = min(k, len(V))
    idx_sorted = np.argsort(V[:, up_axis])
    idx = idx_sorted[-k:]
    return idx.astype(np.int64)

def compute_shift_for_person(person: Person, n_g, p_g, up_axis = 1, jump_threshold = 0.2, eps = 0.001):
    V = np.asarray(person.V, np.float32)
    foot_idx = derive_foot_roi(V, up_axis = up_axis)
    info = {"status": "skip", "d_med": None, "delta": 0.0}
    if foot_idx is None or len(foot_idx) == 0:
        return np.zeros(3, np.float32), info
    
    d = (V[foot_idx] - p_g[None,:]) @ n_g
    d_med = float(np.median(d))
    if abs(d_med) > jump_threshold:
        info.update({"status": "extreme_skip", "d_med":d_med})
        return np.zeros(3, np.float32), info
    
    delta, status = 0.0, "none"
    if d_med < -eps:
        delta, status = -(d_med + eps), "floating"
    elif d_med > eps:
        delta, status = -(d_med - eps), "penetration"
        
    if abs(delta) < 1e-4:
        info.update({"status": status, "d_med": d_med, "delta": 0.0})
        return np.zeros(3, np.float32), info
    
    shift = (delta * n_g).astype(np.float32)
    info.update({"status": status, "d_med": d_med, "delta": abs(delta)})
    return shift, info

def fix_feet_ground(
    S_list, Q_list, persons: List[Person],
    lowest_quantile = 0.3, q_quantile = 0.6, up_axis = 1, jump_threshold = 0.2,
    plane_ransac_iters = 1500, plane_inlier_thresh = 0.02, plane_min_inlier_ratio = 0.6
):
    P = collect_ground_candidates(S_list, Q_list, q_quantile = q_quantile, lowest_quantile = lowest_quantile, up_axis = up_axis)
    if len(P) < 3:
        print("[foot_fix] Not enough ground candidates; skip.")
        return {"plane": None, "persons": []}
    
    res = ransac_plane(P, iters = plane_ransac_iters, inlier_thresh = plane_inlier_thresh, min_inlier_ratio = plane_min_inlier_ratio, up_axis = up_axis)
    
    if res is None:
        print("[foot_fix] Plane RANSAC failed; skip.")
        return {"plane": None, "persons": []}
    
    n_g, p_g, inliers = res
    print(f"[foot_fix] Plane OK. Inliers {inliers.sum()}/{len(P)}; n={n_g}, p={p_g}")
    out = {"plane": {"n_g": n_g, "p_g": p_g, "n_inliers": int(inliers.sum())}, "persons": []}

    for person in persons:
        shift, info = compute_shift_for_person(person, n_g, p_g, up_axis = up_axis, jump_threshold = jump_threshold)
        new_gamma = person.gamma.astype(np.float32) + shift
        if np.linalg.norm(shift) > 0:
            person.V = person.V + shift
            person.gamma = new_gamma
            
        print(f"[foot_fix] Person meta = {person.meta} status = {info['status']} delta = {info['delta']:.4f}m")
        out["persons"].append({"shift": shift, "new_gamma": new_gamma, "info": info})
    
    return out
