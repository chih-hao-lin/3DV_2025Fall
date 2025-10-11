import taichi as ti
import numpy as np
import os
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

# Modified from mpm_99.py in taichi python example

MAX_BLOBS = 12   
BAND = 3         

x = v = C = F = Jp = material = grid_v = grid_m = None
blob_center = blob_axis = blob_theta = blob_vdrift = blob_hole = None

box_minx_g = box_miny_g = box_maxx_g = box_maxy_g = None  
gi0_g = gi1_g = gj0_g = gj1_g = None                      

n_particles = n_grid = None
dx = inv_dx = dt = None
p_vol = p_rho = p_mass = None
E = nu = mu_0 = lambda_0 = None

def parse_args():
    ap = ArgumentParser()
    ap.add_argument("--n_particles", type=int, default=19000)
    ap.add_argument("--n_frames", type=int, default=500)
    ap.add_argument("--quality", type=int, default=1)   
    ap.add_argument("--seed", type=int, default=0)      
    ap.add_argument("--out_path", type=str, default="rollout_000.npz")
    ap.add_argument("--show_gui", action="store_true")

    ap.add_argument("--base_g", type=float, default=50.0)
    ap.add_argument("--blobs_min", type=int, default=2)
    ap.add_argument("--blobs_max", type=int, default=6)

    # inner box in normalized coordinates [0,1]^2
    ap.add_argument("--box_minx", type=float, default=0.2)
    ap.add_argument("--box_miny", type=float, default=0.2)
    ap.add_argument("--box_maxx", type=float, default=0.8)
    ap.add_argument("--box_maxy", type=float, default=0.8)
    return ap.parse_args()


# initialize taichi field and other global variables
def init_params(n_particles_arg: int, quality: int, *, bx0: float, by0: float, bx1: float, by1: float):
    global n_particles, n_grid, dx, inv_dx, dt, p_vol, p_rho, p_mass
    global E, nu, mu_0, lambda_0
    global x, v, C, F, Jp, material, grid_v, grid_m
    global blob_center, blob_axis, blob_theta, blob_vdrift, blob_hole
    global box_minx_g, box_miny_g, box_maxx_g, box_maxy_g, gi0_g, gi1_g, gj0_g, gj1_g

    n_particles = int(n_particles_arg)
    n_grid = 128 * quality
    dx, inv_dx = 1.0 / n_grid, float(n_grid)
    dt = 1e-4 / quality
    p_vol, p_rho = (dx * 0.5) ** 2, 1.0
    p_mass = p_vol * p_rho
    E, nu = 0.1e4, 0.2
    mu_0 = E / (2 * (1 + nu))
    lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

    x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
    Jp = ti.field(dtype=ti.f32, shape=n_particles)
    material = ti.field(dtype=ti.i32, shape=n_particles)

    grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

    blob_center = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BLOBS)
    blob_axis   = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BLOBS)
    blob_theta  = ti.field(dtype=ti.f32, shape=MAX_BLOBS)
    blob_vdrift = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BLOBS)
    blob_hole   = ti.field(dtype=ti.f32, shape=MAX_BLOBS)

    box_minx_g = ti.field(dtype=ti.f32, shape=())
    box_miny_g = ti.field(dtype=ti.f32, shape=())
    box_maxx_g = ti.field(dtype=ti.f32, shape=())
    box_maxy_g = ti.field(dtype=ti.f32, shape=())

    gi0_g = ti.field(dtype=ti.i32, shape=())
    gi1_g = ti.field(dtype=ti.i32, shape=())
    gj0_g = ti.field(dtype=ti.i32, shape=())
    gj1_g = ti.field(dtype=ti.i32, shape=())

    assert 0.0 <= bx0 < bx1 <= 1.0 and 0.0 <= by0 < by1 <= 1.0, "inner box must be within [0,1]"
    box_minx_g[None] = float(bx0)
    box_miny_g[None] = float(by0)
    box_maxx_g[None] = float(bx1)
    box_maxy_g[None] = float(by1)

    gi0 = int(np.floor(bx0 * n_grid))
    gi1 = int(np.ceil (bx1 * n_grid)) - 1
    gj0 = int(np.floor(by0 * n_grid))
    gj1 = int(np.ceil (by1 * n_grid)) - 1
    
    gi0_g[None] = gi0
    gi1_g[None] = gi1
    gj0_g[None] = gj0
    gj1_g[None] = gj1


@ti.kernel
def zero_all_fields():
    for i in range(n_particles):
        x[i] = ti.Vector([0.0, 0.0])
        v[i] = ti.Vector([0.0, 0.0])
        C[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        F[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[i] = 1.0
        material[i] = 0
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector([0.0, 0.0])
        grid_m[i, j] = 0.0

# we use several blobs to randomly initialize particles
@ti.kernel
def init_blob_params(n_blobs: ti.i32):
    for b in range(n_blobs):
        
        margin = 0.04
        cx = ti.random() * (box_maxx_g[None] - box_minx_g[None] - 2.0 * margin) + (box_minx_g[None] + margin)
        cy = ti.random() * (box_maxy_g[None] - box_miny_g[None] - 2.0 * margin) + (box_miny_g[None] + margin)
        blob_center[b] = ti.Vector([cx, cy])

        
        a = 0.07 + 0.12 * ti.random()
        bb = 0.07 + 0.12 * ti.random()
        blob_axis[b] = ti.Vector([a, bb])

        
        blob_theta[b] = 2.0 * 3.14159265 * ti.random()

        
        blob_hole[b] = 0.0
        if ti.random() < 0.25:
            blob_hole[b] = 0.2 + 0.5 * ti.random()

        
        blob_vdrift[b] = 0.1 * ti.Vector([ti.random() - 0.5, ti.random() - 0.5])

# initialize liquid particles within the blobs
@ti.kernel
def initialize_liquid(n_blobs: ti.i32):
    
    for i in range(n_particles):
        
        b = (i + ti.cast(ti.random() * 1e6, ti.i32)) % n_blobs

        
        u1 = ti.random()
        u2 = ti.random()
        r = ti.sqrt(u1)  
        r = blob_hole[b] + (1.0 - blob_hole[b]) * r  
        ang = 2.0 * 3.14159265 * u2
        local = ti.Vector([r * ti.cos(ang), r * ti.sin(ang)])

        a, bb = blob_axis[b][0], blob_axis[b][1]
        scaled = ti.Vector([local[0] * a, local[1] * bb])
        ct = ti.cos(blob_theta[b]); st = ti.sin(blob_theta[b])
        R = ti.Matrix([[ct, -st], [st,  ct]])
        pos = blob_center[b] + R @ scaled

        
        pos += 0.02 * ti.Vector([ti.random() - 0.5, ti.random() - 0.5])

        eps = 0.01
        pos = ti.max(pos, ti.Vector([box_minx_g[None] + eps, box_miny_g[None] + eps]))
        pos = ti.min(pos, ti.Vector([box_maxx_g[None] - eps, box_maxy_g[None] - eps]))
        x[i] = pos
        
        v[i] = blob_vdrift[b] * 0.2 + 0.05 * ti.Vector([ti.random() - 0.5, ti.random() - 0.5])

        F[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        C[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        Jp[i] = 1.0
        material[i] = 0

@ti.kernel
def substep(gravity_y: ti.f32):
    # Clear grid
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector([0.0, 0.0])
        grid_m[i, j] = 0.0

    # P2G
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        # Deformation gradient update
        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)

        # Liquid: mu=0, reset F to sqrt(J) I (stability)
        J = sig[0, 0] * sig[1, 1]
        F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)

        la = lambda_0
        U2, sig2, V2 = ti.svd(F[p])
        J2 = sig2[0, 0] * sig2[1, 1]
        stress = ti.Matrix.identity(ti.f32, 2) * la * J2 * (J2 - 1.0)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j].y += dt * gravity_y  # negative for downward

            if i < gi0_g[None] + BAND and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0

            if i > gi1_g[None] - BAND and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0

            if j < gj0_g[None] + BAND and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0

            if j > gj1_g[None] - BAND and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0

    # G2P
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


def main():
    args = parse_args()

    ti.init(arch=ti.gpu, random_seed=args.seed)

    rng = np.random.default_rng(args.seed)
    n_blobs = int(rng.integers(args.blobs_min, args.blobs_max + 1))
    if n_blobs > MAX_BLOBS:
        raise ValueError(f"n_blobs={n_blobs} exceeds MAX_BLOBS={MAX_BLOBS}; raise MAX_BLOBS.")
    g_vec = np.array([0.0, -args.base_g], dtype=np.float32)

    init_params(
        args.n_particles, args.quality,
        bx0=args.box_minx, by0=args.box_miny,
        bx1=args.box_maxx, by1=args.box_maxy
    )

    zero_all_fields()
    init_blob_params(n_blobs)
    initialize_liquid(n_blobs)

    gui = ti.GUI("Water MPM (single rollout)", res=512, background_color=0xFFFFFF, show_gui=False) if args.show_gui else None

    P = np.zeros((args.n_frames, n_particles, 2), dtype=np.float32)

    for frame in tqdm(range(args.n_frames)):
        n_steps = max(1, int(2e-3 // dt))  # 2ms of physical time per recorded frame
        for _ in range(n_steps):
            substep(g_vec[1])

        P[frame] = x.to_numpy()

        if gui is not None:
            xs = np.linspace(args.box_minx, args.box_maxx, 100, dtype=np.float32)
            ys = np.linspace(args.box_miny, args.box_maxy, 100, dtype=np.float32)
            gui.circles(np.stack([xs, np.full_like(xs, args.box_miny)], 1), radius=1, color=0x000000)
            gui.circles(np.stack([xs, np.full_like(xs, args.box_maxy)], 1), radius=1, color=0x000000)
            gui.circles(np.stack([np.full_like(ys, args.box_minx), ys], 1), radius=1, color=0x000000)
            gui.circles(np.stack([np.full_like(ys, args.box_maxx), ys], 1), radius=1, color=0x000000)

            gui.circles(P[frame], radius=1.0, color=0x0000FF)
            vis_path = os.path.splitext(args.out_path)[0] + f"_{frame:06d}.png"
            gui.show(vis_path)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_path,
        positions=P
    )

    if gui is not None:
        print("[single] compiling video...")
        images = []
        for frame in range(args.n_frames):
            vis_path = os.path.splitext(args.out_path)[0] + f"_{frame:06d}.png"
            images.append(imageio.imread(vis_path))
            os.remove(vis_path)
        mp4_path = os.path.splitext(args.out_path)[0] + ".mp4"
        imageio.mimwrite(mp4_path, images, fps=30)

    print(f"[single] saved {args.out_path}  P{P.shape} blobs={n_blobs}  box=({args.box_minx},{args.box_miny})-({args.box_maxx},{args.box_maxy})")

if __name__ == "__main__":
    main()
