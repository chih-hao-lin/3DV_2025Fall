import time
import numpy as np
import open3d as o3d
import scipy.sparse
import scipy.sparse.linalg
from pathlib import Path
from itertools import permutations
from collections import defaultdict
import copy


N_STEPS_SEG = 30                 
SLEEP       = 0.06               
ARAP_ITERS  = 80

# Running gait offsets for the cow and horse
FWD_SCALE   = 0.01
BWD_SCALE   = 0.01
UP_SCALE    = 0.01
DOWN_SCALE  = 0.01
OUT_SCALE   = 0.09
IN_SCALE    = 0.09
          
PREVIEW_MARKERS = True

def nearest_vertex_id(vertices: np.ndarray, query_pt: np.ndarray) -> int:
    dif = vertices - query_pt.reshape(1, 3)
    return int(np.argmin(np.einsum("ij,ij->i", dif, dif)))

# Make handle constraints
def make_constraints(current_handle_pos, static_ids, handle_id, static_pos):
        ids = o3d.utility.IntVector(static_ids + [handle_id])
        pos = o3d.utility.Vector3dVector(static_pos + [current_handle_pos])
        return ids, pos

def update_mesh_geometry(mesh_handle, new_vertices, new_colors=None):
        mesh_handle.vertices = o3d.utility.Vector3dVector(new_vertices)
        if new_colors is not None:
            mesh_handle.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        mesh_handle.compute_vertex_normals()

def prepare_vis(mesh, R):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1100, height=800, visible=True)
        m = o3d.geometry.TriangleMesh(mesh)   # copy
        m.rotate(R, center=m.get_center())
        vis.add_geometry(m)

        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.array([1, 1, 1])
        return vis, m

# Following functions are to find hooves of the animals
def side_split(idx_array, lr_vals, tries=(0.5, 0.45, 0.55, 0.4, 0.6)):
    if idx_array.size == 0:
        return np.array([]), np.array([])
    for q in tries:
        cut = np.quantile(lr_vals[idx_array], q)
        left  = idx_array[lr_vals[idx_array] <  cut]
        right = idx_array[lr_vals[idx_array] >= cut]
        if left.size > 0 and right.size > 0:
            return left, right
    # fallback: median
    cut = np.median(lr_vals[idx_array])
    return idx_array[lr_vals[idx_array] < cut], idx_array[lr_vals[idx_array] >= cut]

def pick_lowest_in_group(idx_array, up_vals):
    if idx_array.size == 0:
        return None
    return int(idx_array[np.argmin(up_vals[idx_array])])

def score_hoof_set(V, up_i, fb_i, lr_i, front_sign, handles):
    """High score = good LR separation per half, correct front/back ordering, near ground, all distinct."""
    ids = list(handles.values())
    if any(h is None for h in ids) or len(set(ids)) < 4:
        return -1e12
    up = V[:, up_i]; fb = V[:, fb_i]; lr = V[:, lr_i]
    FL, FR, HL, HR = ids
    # Separation along LR for front and hind
    sep_lr_front = abs(lr[FL] - lr[FR])
    sep_lr_hind  = abs(lr[HL] - lr[HR])
    # Front vs back ordering along FB (with front_sign)
    fb_c = (fb - np.median(fb)) * front_sign
    fb_front_mean = 0.5 * (fb_c[FL] + fb_c[FR])
    fb_back_mean  = 0.5 * (fb_c[HL] + fb_c[HR])
    fb_gap = fb_front_mean - fb_back_mean  # should be positive
    # Close to ground
    groundness = -np.mean(up[[FL, FR, HL, HR]])  # lower is better
    
    up_span = -(up[[FL, FR, HL, HR]].max() - up[[FL, FR, HL, HR]].min())
    return 2.0*(sep_lr_front + sep_lr_hind) + 1.5*fb_gap + 0.5*groundness + 0.3*up_span

def auto_axes_and_hooves_split_halves(V):
    """
    Try all axis permutations and both front signs.
    For each candidate, pick hooves by:
      - take ground band
      - split into front/back halves by FB (with sign)
      - inside each half, split left/right by LR and pick the lowest (min up)
    Keep the mapping with the best score.
    This method made sure I picked the correct hooves and not head or backside due to axes change. 
    """
    best = None
    for up_i, fb_i, lr_i in permutations([0,1,2], 3):
        up_vals = V[:, up_i]
        fb_vals = V[:, fb_i]
        lr_vals = V[:, lr_i]
        
        def ground_idx_for(pct):
            thr = np.quantile(up_vals, pct)
            return np.where(up_vals <= thr)[0]
        ground_idx = ground_idx_for(0.30)
        if ground_idx.size < 50: ground_idx = ground_idx_for(0.45)

        fb_med = np.median(fb_vals)
        lr_med = np.median(lr_vals)

        for front_sign in (+1.0, -1.0):
            fb_centered = (fb_vals - fb_med) * front_sign
            front_idx = ground_idx[fb_centered[ground_idx] >= 0]
            back_idx  = ground_idx[fb_centered[ground_idx] <  0]
            if front_idx.size == 0: front_idx = ground_idx
            if back_idx.size  == 0: back_idx  = ground_idx

            front_left,  front_right = side_split(front_idx, lr_vals)
            back_left,   back_right  = side_split(back_idx, lr_vals)

            FL = pick_lowest_in_group(front_left, up_vals)
            FR = pick_lowest_in_group(front_right, up_vals)
            HL = pick_lowest_in_group(back_left, up_vals)
            HR = pick_lowest_in_group(back_right, up_vals)

            ids = [FL, FR, HL, HR]
            
            if any(i is None for i in ids):
                used = {i for i in ids if i is not None}
                for j, val in enumerate(ids):
                    if val is None:
                        for k in ground_idx[np.argsort(up_vals[ground_idx])]:
                            if k not in used:
                                ids[j] = int(k); used.add(int(k)); break
                FL, FR, HL, HR = ids

            handles = {"FL": FL, "FR": FR, "HL": HL, "HR": HR}
            sc = score_hoof_set(V, up_i, fb_i, lr_i, front_sign, handles)
            if (best is None) or (sc > best[0]):
                best = (sc, up_i, fb_i, lr_i, front_sign, handles)

    if best is None:
        raise RuntimeError("Failed to auto-detect axes.")
    _, up_i, fb_i, lr_i, front_sign, handles = best
    return up_i, fb_i, lr_i, front_sign, handles

# Visualize the handles
def make_handle_markers(mesh_in, handles, radius_scale=0.02):
    Vv = np.asarray(mesh_in.vertices)
    bbox = mesh_in.get_axis_aligned_bounding_box()
    r = radius_scale * np.linalg.norm(bbox.get_extent())
    colors = {"FL":[1,0,0], "FR":[0,1,0], "HL":[0,0,1], "HR":[1,0.6,0]}
    markers = {}
    for name, vid in handles.items():
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.compute_vertex_normals()
        s.paint_uniform_color(colors.get(name, [0.2,0.2,0.2]))
        s.translate(Vv[vid])
        markers[name] = s
    return markers

def rotate_pts(V, Rmat, center):
    return (V - center) @ Rmat.T + center

# Function to solve ARAP using Gauss Newton
def arap_gauss_newton(mesh, targets_dict, static_ids, static_pos, max_iter=30):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    neighbors = defaultdict(set)
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            neighbors[a].add(b)
            neighbors[b].add(a)

    W = defaultdict(dict)
    for i in range(len(V)):
        for j in neighbors[i]:
            W[i][j] = 1.0

    V_def = V.copy()
    n = len(V)

    constraint_ids = static_ids + list(targets_dict.keys())
    constraint_targets = static_pos + [targets_dict[i] for i in targets_dict]

    for it in range(max_iter):
        # Estimate per-vertex local rotations via SVD
        R = []
        for i in range(n):
            S = np.zeros((3, 3))
            for j in neighbors[i]:
                w = W[i][j]
                p_ij = V[i] - V[j]
                q_ij = V_def[i] - V_def[j]
                S += w * np.outer(q_ij, p_ij)
            U, _, VT = np.linalg.svd(S)
            Ri = U @ VT
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ VT
            R.append(Ri)

        # Linear system (Lx = b)
        row = []
        col = []
        data = []
        b = np.zeros((n, 3))

        for i in range(n):
            b_i = np.zeros(3)
            for j in neighbors[i]:
                w = W[i][j]
                row.extend([i, i])
                col.extend([i, j])
                data.extend([w, -w])
                b_i += w * (R[i] @ (V[i] - V[j]))
            b[i] = b_i

        # Hard constraints
        big_weight = 1e8
        for cid, target in zip(constraint_ids, constraint_targets):
            row.append(cid)
            col.append(cid)
            data.append(big_weight)
            b[cid] = big_weight * target

        # Solve linear system
        L = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        for dim in range(3):
            V_def[:, dim] = scipy.sparse.linalg.spsolve(L, b[:, dim])

    # Create deformed mesh
    mesh_def = copy.deepcopy(mesh)
    mesh_def.vertices = o3d.utility.Vector3dVector(V_def)
    return mesh_def


def make_preview_markers(mesh_in, handles, radius_scale=0.015):
    Vv = np.asarray(mesh_in.vertices)
    bbox = mesh_in.get_axis_aligned_bounding_box()
    r = radius_scale * np.linalg.norm(bbox.get_extent())
    colors = {"FL":[1,0,0], "FR":[0,1,0], "HL":[0,0,1], "HR":[1,0.6,0]}
    ms = []
    for name, vid in handles.items():
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r).paint_uniform_color(colors[name])
        s.translate(Vv[vid]); ms.append(s)
    return ms

def morph_sequence_visualizer(mesh0, meshA, meshB, handles, steps=N_STEPS_SEG, sleep=SLEEP):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1100, height=800, visible=True)

    Rmat = mesh0.get_rotation_matrix_from_xyz((0, np.pi, 0))
    center = mesh0.get_center()

    V0 = rotate_pts(np.asarray(mesh0.vertices), Rmat, center)
    VA = rotate_pts(np.asarray(meshA.vertices), Rmat, center)
    VB = rotate_pts(np.asarray(meshB.vertices), Rmat, center)

    m = o3d.geometry.TriangleMesh(mesh0)
    m.vertices = o3d.utility.Vector3dVector(V0)
    m.compute_vertex_normals()
    vis.add_geometry(m)

    # Handle markers (rotated)
    markers = make_handle_markers(mesh0, handles, radius_scale=0.015)
    for name, s in markers.items():
        Vs = np.asarray(s.vertices)
        Vs_rot = rotate_pts(Vs, Rmat, center)
        s.vertices = o3d.utility.Vector3dVector(Vs_rot)
        s.compute_vertex_normals()
        vis.add_geometry(s)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1, 1, 1])
    def update_markers(Vt):
        for name, vid in handles.items():
            s = markers[name]
            cur_center = s.get_center()
            tgt = Vt[vid]
            s.translate(tgt - cur_center, relative=True)
            vis.update_geometry(s)

    def blend(a, b, s):
        for i in range(s):
            t = i / (s - 1) if s > 1 else 1.0
            Vt = (1 - t) * a + t * b
            m.vertices = o3d.utility.Vector3dVector(Vt)
            m.compute_vertex_normals()
            vis.update_geometry(m)

            update_markers(Vt)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(sleep)

    blend(V0, VA, steps)
    blend(VA, VB, steps)
    blend(VB, V0, steps)

    print("Loop done — close the window to exit.")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()


# Inspiration taken from open3d example
def torus(mesh):
    assert mesh.has_triangles(), "Mesh must be a triangle mesh."
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.85, 0.2]]), (np.asarray(mesh.vertices).shape[0], 1))
    )

    V = np.asarray(mesh.vertices)

    # Static anchors + one moving handle
    y_vals = V[:, 1]
    y_thresh = np.quantile(y_vals, 0.15)
    static_ids = np.where(y_vals <= y_thresh)[0].tolist()
    static_pos = [V[i].copy() for i in static_ids]

    approx_handle_pos = np.array([60.0, 20.0, 0.0])
    handle_id = nearest_vertex_id(V, approx_handle_pos)
    handle_start = V[handle_id].copy()

    # Handle movement
    handle_target = handle_start + np.array([-30.0, -40.0, -15.0])

    R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))

    # Progressive ARAP (solve per frame as the handle moves)
    n_steps=50 
    sleep=0.02
    max_iter=30
    vis, m = prepare_vis(mesh, R)
    orig_mesh = mesh

    base_colors = np.asarray(mesh.vertex_colors)
    highlight_colors = base_colors.copy()
    highlight_colors[handle_id] = np.array([1.0, 0.2, 0.2])

    for i in range(n_steps + 1):
        alpha = i / n_steps
        current_handle = (1 - alpha) * handle_start + alpha * handle_target
        cids, cpos = make_constraints(current_handle, static_ids, handle_id, static_pos)

        # Solve ARAP at this handle position
        deformed = orig_mesh.deform_as_rigid_as_possible(cids, cpos, max_iter=max_iter)

        colors = np.asarray(deformed.vertex_colors)
        if colors.shape[0] == 0:
            colors = np.tile(np.array([[1.0, 0.85, 0.2]]), (np.asarray(deformed.vertices).shape[0], 1))
        colors[handle_id] = np.array([1.0, 0.2, 0.2])

        deformed.rotate(R, center=deformed.get_center())
        update_mesh_geometry(m, np.asarray(deformed.vertices), colors)

        vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sleep)

    print("Animation finished. Close the window to exit.")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()

# Taken from open3d example
def armadillo(armadillo):
    mesh = o3d.io.read_triangle_mesh(armadillo.path)
    assert mesh.has_triangles(), "Mesh must be a triangle mesh."
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1., 0., 0.2]]), (np.asarray(mesh.vertices).shape[0], 1))
    )
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # Static anchors (keep base fixed)
    static_ids = np.where(vertices[:, 1] < -30)[0].tolist()
    static_pos = [vertices[i] for i in static_ids]

    # Handle keypoint & target
    handle_ids = [2490]
    handle_start = vertices[2490].copy()
    handle_target = handle_start + np.array([-40.0, -40.0, -40.0])

    constraint_ids_final, constraint_pos_final = make_constraints(handle_target, static_ids, handle_ids, static_pos)
    mesh_final = mesh.deform_as_rigid_as_possible(
        constraint_ids_final, constraint_pos_final, max_iter=50
    )
    mesh_final.compute_vertex_normals()

    R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)) 

    vis, m = prepare_vis()
    orig_mesh = mesh 

    n_steps=80
    sleep=0.015

    for i in range(n_steps + 1):
        alpha = i / n_steps
        # Move the handle along a straight path
        current_handle = (1 - alpha) * handle_start + alpha * handle_target
        cids, cpos = make_constraints(current_handle)

        # Solve ARAP at this handle position
        deformed = orig_mesh.deform_as_rigid_as_possible(
            cids, cpos, max_iter=30
        )
        V = np.asarray(deformed.vertices)
        
        deformed.rotate(R, center=deformed.get_center())
        update_mesh_geometry(m, np.asarray(deformed.vertices))

        vis.update_geometry(m)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sleep)

    print("Animation finished. Close the window to exit.")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()

# Function for animal mesh deformation
def animal(mesh):
    mesh = o3d.io.read_triangle_mesh(mesh)
    assert mesh.has_triangles(), "Mesh must be a triangle mesh."
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.48, 0.24, 0.]]), (np.asarray(mesh.vertices).shape[0], 1))
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)

    up_axis, fb_axis, lr_axis, front_dir_sign, handles = auto_axes_and_hooves_split_halves(V)
    print("Axes (up, fb, lr):", up_axis, fb_axis, lr_axis, "| front_sign:", front_dir_sign)
    print("Hoof vertex IDs:", handles)

    up_vals = V[:, up_axis]
    high_thresh = np.quantile(up_vals, 0.75)
    static_ids = np.where(up_vals >= high_thresh)[0].tolist()

    # Anchoring a subset around body mid-section for stability
    mid_low, mid_high = np.quantile(up_vals, 0.40), np.quantile(up_vals, 0.70)
    mid_ids = np.where((up_vals >= mid_low) & (up_vals <= mid_high))[0]
    if len(mid_ids) > 0:
        rng = np.random.default_rng(42)
        extra = rng.choice(mid_ids, size=min(300, len(mid_ids)), replace=False)
        static_ids += extra.tolist()

    for vid in handles.values():
        if vid in static_ids:
            static_ids.remove(vid)
    static_pos = [V[i].copy() for i in static_ids]

    ranges = V.max(0) - V.min(0)
    size_up, size_fb, size_lr = ranges[up_axis], ranges[fb_axis], ranges[lr_axis]
    fwd =  FWD_SCALE  * size_fb * front_dir_sign
    bwd = -BWD_SCALE  * size_fb * front_dir_sign
    up_ofs  =  UP_SCALE   * size_up
    dn_ofs  = -DOWN_SCALE * size_up
    out     =  OUT_SCALE * size_lr
    ins     = -IN_SCALE  * size_lr

    def targets_for_pose(pose="A"):
        T = {}
        def add(tag, fb_ofs, up_ofs_, lr_ofs):
            vid = handles[tag]
            base = V[vid].copy()
            ofs = np.zeros(3)
            ofs[fb_axis] = fb_ofs
            ofs[up_axis] = up_ofs_
            ofs[lr_axis] = lr_ofs
            T[vid] = base + ofs

        if pose == "A":
            add("FL", fwd, up_ofs,  ins)
            add("HR", fwd, dn_ofs,  out)
            add("FR", bwd, dn_ofs,  out)
            add("HL", bwd, up_ofs,  ins)
        else:  # pose B (swap)
            add("FR", fwd, up_ofs,  ins)
            add("HL", fwd, dn_ofs,  out)
            add("FL", bwd, dn_ofs,  out)
            add("HR", bwd, up_ofs,  ins)

        return T

    targets_A = targets_for_pose("A")
    targets_B = targets_for_pose("B")

    try:    mesh_A = arap_gauss_newton(mesh, targets_A, static_ids, static_pos, ARAP_ITERS)
    except: mesh_A = arap_gauss_newton(mesh, targets_A, static_ids, static_pos, static_ids, 30)
    try:    mesh_B = arap_gauss_newton(mesh, targets_B, static_ids, static_pos, ARAP_ITERS)
    except: mesh_B = arap_gauss_newton(mesh, targets_B, static_ids, static_pos, 30)

    mesh_A.compute_vertex_normals()
    mesh_B.compute_vertex_normals()

    if PREVIEW_MARKERS:
        o3d.visualization.draw_geometries([mesh, *make_preview_markers(mesh, handles)])

    morph_sequence_visualizer(mesh, mesh_A, mesh_B, handles)


if __name__ == "__main__":

    mesh_object = "horse"

    mesh_dict = {
        "cow" : "/mesh/cow.obj",
        "horse" : "/mesh/horse.obj",
        "armadillo" : o3d.data.ArmadilloMesh(),
        "torus" : o3d.geometry.TriangleMesh.create_torus(
                    torus_radius=60.0,     # distance from center to tube center
                    tube_radius=18.0,      # tube thickness
                    radial_resolution=80,  # more = smoother
                    tubular_resolution=40
                )
    }

    if mesh_object == "torus":
        torus(mesh_dict["torus"])

    elif mesh_object == "armadillo":
         armadillo(mesh_dict["armadillo"])

    else:
        animal(mesh_dict[mesh_object])
         
