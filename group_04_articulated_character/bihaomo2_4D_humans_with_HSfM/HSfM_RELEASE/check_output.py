import os, pickle, numpy as np

out_dir = "./demo_output/test"          # 改成你的
body_model_name = "smpl"               # 或 "smplx"
pkl_path = os.path.join(out_dir, f"hsfm_output_{body_model_name}_aligned.pkl")
print("loading:", pkl_path)
data = pickle.load(open(pkl_path, "rb"))

cams   = data["hsfm_places_cameras"]                 # 相机/点云
people = data["hsfm_people(smplx_params)"]           # 人体参数（优化后的）

# 选一帧（单镜头就是唯一一帧）
fn = sorted(cams.keys())[0]

# --- 点云的尺度/中心 ---
P = cams[fn].get("pts3d", None)
if P is None:
    print("No pts3d in this file.")
else:
    P = np.asarray(P).reshape(-1, 3)
    finite = np.isfinite(P).all(1)
    P = P[finite]
    pc_center = np.nanmedian(P, axis=0)
    pc_medR   = np.nanmedian(np.linalg.norm(P - pc_center, axis=1))
    print("[PointCloud] center:", pc_center, " median radius:", pc_medR)

# --- 各人的 root 平移（世界坐标系）---
for pid, params in people.items():
    t = params["root_transl"][0]     # (3,)
    print(f"[Human {pid}] root_transl:", t, " | norm:", np.linalg.norm(t), " | z:", t[2])

# --- 检查 root 在该相机坐标系下的 z 是否 > 0（应该 > 0）---
K      = np.asarray(cams[fn]["intrinsic"])
T_c2w  = np.asarray(cams[fn]["cam2world"])
T_w2c  = np.linalg.inv(T_c2w)

pid0 = sorted(people.keys())[0]
pelvis_w = people[pid0]["root_transl"][0]
pelvis_c = T_w2c[:3,:3] @ pelvis_w + T_w2c[:3,3]
print(f"[Human {pid0}] pelvis in camera coords:", pelvis_c, " (z should be > 0)")

# --- 简单的“尺度比”粗估（用人体 root 距离与点云半径比对）---
if P is not None:
    scale_smpl = np.linalg.norm(pelvis_w)
    print("Rough scale ratio  pc_medR / |root_transl|  =", pc_medR / (scale_smpl + 1e-8))
