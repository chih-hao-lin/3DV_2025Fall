import argparse, pickle, os
import numpy as np
from foot_ground_fix import Person, fix_feet_ground
import torch
import smplx
    
def get_world_pts(pkl_world):
    key_cameras = "hsfm_places_cameras"
    if key_cameras not in pkl_world or not isinstance(pkl_world[key_cameras], dict):
        raise RuntimeError(f"Cannot find dict '{key_cameras}' in HSFM PKL!")
    
    container = pkl_world[key_cameras]
    P_list, Q_list = [], []
    for k in sorted(container.keys()):
        view = container[k]
        
        if not isinstance(view, dict):
            continue
        
        if "pts3d" not in view or "conf" not in view:
            continue
        
        x, y, z = view["pts3d"].shape
        assert view["conf"].shape[0] == x and view["conf"].shape[1] == y
        for i in range(x):
            for j in range(y):
                P_list.append(view["pts3d"][i][j])
                Q_list.append(view["conf"][i][j])

    return P_list, Q_list

def extract_people_from_hsfm(pkl_hsfm, smplx_model_folder: str, device: str = "cpu"):
    key_people = "hsfm_people(smplx_params)"
    if key_people not in pkl_hsfm or not isinstance(pkl_hsfm[key_people], dict):
        raise RuntimeError(f"Cannot find dict '{key_people}' in HSFM PKL.")
    
    container = pkl_hsfm[key_people]
    if smplx_model_folder is None:
        raise RuntimeError("smplx_model_folder is required!")

    model = smplx.create(model_path = smplx_model_folder,
                         model_type = 'smplx', 
                         gender = 'neutral', 
                         use_pca = False, 
                         num_pca_comps = 45, 
                         flat_hand_mean = True, 
                         use_face_contour = True, 
                         num_betas = 10, 
                         batch_size = 1).to(device)
    model.eval()

    persons = []
    idx_map = []
    for pid in sorted(container.keys()):
        d = container[pid]
        if not isinstance(d, dict):
            continue
        
        global_orient = np.asarray(d["global_orient"], dtype = np.float32)
        body_pose = np.asarray(d["body_pose"], dtype = np.float32)
        betas = np.asarray(d["betas"], dtype = np.float32)
        root_transl = np.asarray(d["root_transl"], dtype = np.float32)
        left_hand_pose = np.asarray(d["left_hand_pose"], dtype = np.float32)
        right_hand_pose = np.asarray(d["right_hand_pose"], dtype = np.float32)

        go = torch.from_numpy(global_orient).to(device)
        bp = torch.from_numpy(body_pose).to(device)
        bt = torch.from_numpy(betas).to(device)
        lh = torch.from_numpy(left_hand_pose).to(device)
        rh = torch.from_numpy(right_hand_pose).to(device)
        
        with torch.no_grad():
            out = model(body_pose = bp, 
                        betas = bt, 
                        global_orient = go, 
                        left_hand_pose = lh,
                        right_hand_pose = rh)
            verts = out.vertices[0].detach().cpu().numpy()
            joints = out.joints[0].detach().cpu().numpy()
            verts = verts - joints[0 : 1] + root_transl

        gamma = root_transl.reshape(-1)[:3].copy()

        persons.append(Person(
            V = verts.copy(),
            gamma = gamma,
            meta = {"id": pid}
        ))
        idx_map.append((pid, "root_transl"))

    def apply_inplace(updated_persons):
        for (pid, write_key), newp in zip(idx_map, updated_persons):
            if pid not in container:
                continue
            
            arr = np.asarray(newp.gamma, dtype = np.float32).reshape(1, 3)
            container[pid][write_key] = arr

    return persons, apply_inplace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsfm-pkl", required = True, help = "path to HSfM output pkl")
    ap.add_argument("--out-pkl", default = None, help = "output path")
    ap.add_argument("--up-axis", default = "z", choices = ["x", "y", "z"], help = "world up axis")
    ap.add_argument("--q-quantile", type = float, default = 0.6)
    ap.add_argument("--lowest-quantile", type = float, default = 0.3)
    ap.add_argument("--plane-inlier-thresh", type = float, default = 0.02)
    ap.add_argument("--plane-min-inlier-ratio", type = float, default = 0.6)
    ap.add_argument("--plane-ransac-iters", type = int, default = 1500)
    ap.add_argument("--jump-threshold", type = float, default = 0.2)
    ap.add_argument("--smplx-model-folder", type = str, default = "body_models/smplx")
    args = ap.parse_args()

    up_axis = {"x": 0, "y": 1, "z": 2}[args.up_axis]

    with open(args.hsfm_pkl, "rb") as f:
        pkl_hsfm = pickle.load(f)

    P_list, Q_list = get_world_pts(pkl_hsfm)
    
    if not P_list:
        raise RuntimeError("Could not find pointclouds.")
    
    persons, apply_back = extract_people_from_hsfm(
        pkl_hsfm = pkl_hsfm,
        smplx_model_folder = args.smplx_model_folder
    )

    out = fix_feet_ground(
        P_list, Q_list, persons,
        lowest_quantile = args.lowest_quantile, q_quantile = args.q_quantile, up_axis = up_axis,
        jump_threshold = args.jump_threshold, plane_ransac_iters = args.plane_ransac_iters, 
        plane_inlier_thresh = args.plane_inlier_thresh, plane_min_inlier_ratio = args.plane_min_inlier_ratio
    )
    print("out: ", out)
    
    apply_back(persons)
    out_pkl = args.out_pkl or os.path.splitext(args.hsfm_pkl)[0] + "_footfix.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(pkl_hsfm, f)
    print(f"done wrote: {out_pkl}")

if __name__ == "__main__":
    main()
