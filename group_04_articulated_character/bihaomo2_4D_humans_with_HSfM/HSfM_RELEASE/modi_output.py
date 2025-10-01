import os, pickle, numpy as np, torch

pkl_in  = "./demo_output/test/hsfm_output_smpl.pkl"   # 改成你的
pkl_out = "./demo_output/test/hsfm_output_smpl_aligned.pkl"
device  = "cuda"

data = pickle.load(open(pkl_in, "rb"))
cams   = data["hsfm_places_cameras"]
people = data["hsfm_people(smplx_params)"]

fn = sorted(cams.keys())[0]
P = np.asarray(cams[fn]["pts3d"]).reshape(-1,3)
finite = np.isfinite(P).all(1)
P = P[finite]
pc_center = np.median(P, axis=0)
pc_radius = np.median(np.linalg.norm(P - pc_center, axis=1))
print("pointcloud center:", pc_center, " radius:", pc_radius)

# ---- 对齐人体 ----
for pid, params in people.items():
    t = params["root_transl"][0]  # numpy (3,)
    human_size = np.linalg.norm(t)
    # scale factor: 让人体的“位移尺度”接近点云半径
    scale = pc_radius / (human_size + 1e-8)
    # scale /= 1000.0   # 进一步缩小一些
    print(f"pid {pid} original transl {t}, norm={human_size}, scale={scale:.3f}")
    
    # 缩放 root_transl
    t = t * scale
    # 再平移到点云中心
    t[:] = pc_center
    params["root_transl"][0] = t
    print(f"pid {pid} new root_transl:", t)

manual_scale = 0.1
params["global_scale"] = np.array([manual_scale], dtype=np.float32)

pickle.dump(data, open(pkl_out, "wb"))
print("wrote:", pkl_out)

