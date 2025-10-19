import taichi as ti
import argparse
import random
import os
import math
import numpy as np

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_GB=4.0)

# ------------------------ config ------------------------
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = 1.0 / dx
dt = 1e-4
num_particles=2000

# elastic params (kept simple/stable)
E = 1000.0
mu = E
la = E

# steps & buffers
max_steps = 1000
steps = 500  # training horizon (you can increase after it stabilizes)

# ------------------------ fields ------------------------
scalar = lambda: ti.field(dtype=real)
vec    = lambda: ti.Vector.field(dim, dtype=real)
mat    = lambda: ti.Matrix.field(dim, dim, dtype=real)

# particle bookkeeping
particle_type = ti.field(ti.i32)   # 0: fluid-like, 1: solid-like (snow)
x, v = vec(), vec()
C, F = mat(), mat()

# grid (out-of-place op)
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()

# loss & target
loss = scalar()
# at top with other scalars
# loss_wsum = scalar()
x_target = vec()

# learnable gravity (scalar; applied along -y as in original diffmpm)
g = scalar()  # g >= 0, used as v_out[1] -= dt * g * 30

# counts (decided after building scene)
n_particles = 0
n_solid_particles = 0

# ------------------------ allocation ------------------------
def allocate_fields():
    global n_particles, n_solid_particles
    # NOTE: we place big time-varying particle arrays first
    ti.root.dense(ti.ij, (max_steps, n_particles)).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.dense(ti.i, n_particles).place(particle_type)
    ti.root.dense(ti.ij, (max_steps, n_particles)).place(x_target)

    ti.root.place(loss, g)
    ti.root.lazy_grad()

# ------------------------ kernels ------------------------
@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = ti.Vector([0.0, 0.0])
        grid_m_in[i, j] = 0.0
        # also clear adjoints touched by custom backward
        grid_v_in.grad[i, j] = ti.Vector([0.0, 0.0])
        grid_m_in.grad[i, j] = 0.0
        grid_v_out.grad[i, j] = ti.Vector([0.0, 0.0])

@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = ti.Vector([0.0, 0.0])
        v.grad[f, i] = ti.Vector([0.0, 0.0])
        C.grad[f, i] = ti.Matrix.zero(real, 2, 2)
        F.grad[f, i] = ti.Matrix.zero(real, 2, 2)

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx   = x[f, p] * inv_dx - ti.cast(base, real)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = new_F.determinant()

        # Optional "fluid-like" shortcut (same as your original)
        if particle_type[p] == 0:
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0.0], [0.0, sqrtJ]])

        F[f + 1, p] = new_F
        r, _ = ti.polar_decompose(new_F)

        cauchy = ti.Matrix.zero(real, 2, 2)
        mass   = ti.cast(0.0, real)
        mu, la = E, E
        if particle_type[p] == 1:
            h = 0.3
            mu *= h
            la *= h
        elif particle_type[p] == 0:
            mu = 0
        # stress (no actuation)
        if particle_type[p] == 0:
            mass = 4.0
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1.0) * E
        else:
            mass = 1.0
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1.0) * J)
        stress = -(dt * 1.0 * 4.0 * inv_dx * inv_dx) * cauchy  # p_vol=1 for simplicity
        affine = stress + mass * C[f, p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos   = (ti.cast(offset, real) - fx) * dx
                weight = w[i][0] * w[j][1]
                ti.atomic_add(grid_v_in[base + offset], weight * (mass * v[f, p] + affine @ dpos))
                ti.atomic_add(grid_m_in[base + offset], weight * mass)

bound = 3

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        if grid_m_in[i, j] > 0:
            v_out = grid_v_in[i, j] / (grid_m_in[i, j] + 1e-10)
            v_out[1] -= dt * g[None] * 30  # diffmpm-style scale
            if i < bound and v_out[0] < 0: v_out = ti.Vector([0.0, 0.0])
            if i > n_grid - bound and v_out[0] > 0: v_out = ti.Vector([0.0, 0.0])
            if j < bound and v_out[1] < 0: v_out = ti.Vector([0.0, 0.0])
            if j > n_grid - bound and v_out[1] > 0: v_out = ti.Vector([0.0, 0.0])
            grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx   = x[f, p] * inv_dx - ti.cast(base, real)
        w    = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos   = ti.cast(ti.Vector([i, j]), real) - fx
                g_v    = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4.0 * weight * g_v.outer_product(dpos) * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

# ------------------------ loss ------------------------
@ti.kernel
def traj_loss(T: ti.i32):

    for t in range(1, T):
        for p in range(n_particles):
            # if v[t, p][1] <= 0.1:
            d2 = (x[t, p] - x_target[t, p]).norm_sqr()
            ti.atomic_add(loss[None], d2 / (T * n_particles))
            

# ------------------------ custom step with grad ------------------------
@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)

@ti.ad.grad_for(advance)
def advance_grad(s):
    # re-run forward side to build the same intermediates for grads
    clear_grid()
    p2g(s)
    grid_op()
    # backward through the sub-steps
    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)

# ------------------------ scene ------------------------
class Scene:
    def __init__(self):
        self.x = []
        self.ptype = []

    def add_rect(self, x0, y0, w, h, ptype=1):
        # simple dense packing: 2x per cell like original
        w_cnt = int(w / dx) * 2
        h_cnt = int(h / dx) * 2
        real_dx = w / max(1, w_cnt)
        real_dy = h / max(1, h_cnt)
        for i in range(w_cnt):
            for j in range(h_cnt):
                self.x.append([x0 + (i + 0.5) * real_dx, y0 + (j + 0.5) * real_dy])
                self.ptype.append(ptype)
    
    def add_ball(self, n_particles=1000, center_x=0.5, center_y=0.5, radius=0.1, ptype=1):
        """
        Adds a ball of randomly and uniformly distributed particles.

        Args:
            center_x (float): The x-coordinate of the ball's center.
            center_y (float): The y-coordinate of the ball's center.
            radius (float): The radius of the ball.
            n_particles (int): The number of particles to generate.
            ptype (int): The particle type identifier.
        """
        for _ in range(n_particles):
            # To ensure uniform density by area, we use the square root of a random number for the radius.
            r = radius * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            
            px = center_x + r * math.cos(theta)
            py = center_y + r * math.sin(theta)
            
            self.x.append([px, py])
            self.ptype.append(ptype)

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = len(self.x)
        n_solid_particles = sum(1 for t in self.ptype if t == 1)
        print('n_particles', n_particles, '| n_solid', n_solid_particles)

def make_snowball_scene():
    s = Scene()
    # a compact block as "snowball" near top-center
    # s.add_rect(0.45, 0.70, 0.10, 0.10, ptype=1)
    s.add_ball(n_particles=num_particles, ptype=1)
    s.finalize()
    return s

# ------------------------ forward driver ------------------------
def forward(total_steps=steps):
    for s in range(total_steps - 1):
        advance(s)
    # for s in range(total_steps - 1):
    #     print(f"Step, {s}, x of particle 0: {x[s, 30]}, v of particle 0: {v[s, 0]}")
    # import ipdb; ipdb.set_trace()
    traj_loss(total_steps)

@ti.kernel
def init0():
    for i in range(n_particles):
        x[0, i] = x_target[0, i]
        # x[0, i] = ti.Vector([x_target[i, 0], x_target[i, 1]])
        v[0, i] = ti.Vector([0.0, 0.0])
        C[0, i] = ti.Matrix.zero(real, 2, 2)      # ✅ allowed in kernel
        F[0, i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        particle_type[i] = 1

@ti.kernel
def viz_init0():
    for i in range(n_particles):
        x[0, i] = x_target[0, i]
        x[0, i].x -= 0.2
        # x[0, i] = ti.Vector([x_target[i, 0], x_target[i, 1]])
        v[0, i] = ti.Vector([0.0, 0.0])
        C[0, i] = ti.Matrix.zero(real, 2, 2)      # ✅ allowed in kernel
        F[0, i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        particle_type[i] = 1

def visualize(opt_iter, x_gt, dir='mpm128/optimize'):
    
    # 1. Define your "ground truth" physics parameters
    # In this case, we use the default E and nu values.
    os.makedirs(dir, exist_ok=True)
    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41, show_gui=False)
    # 2. Initialize the simulation state
    viz_init0()
    # for t in range(max_steps - 1):
    # x_history_np = x.to_numpy()
    # 3. Run the forward simulation for the full duration
    for t in range(max_steps - 1):
        advance(t)
        if t % 20 == 0: # Render every 4 steps
            gui.clear()
            gui.circles(
                x.to_numpy()[t],
                radius=5,
                color=0xD3A8FF
            )
            gui.circles(
                x_gt[t],
                radius=5,
                color=0xEEEEF0
            )
            gui.text(f"optimization steps: {opt_iter}", pos=(0.02, 0.96), color=0xFFFFFF, font_size=20)
            gui.text(f"ground truth G: 9.8", pos=(0.02, 0.92), color=0xFFFFFF, font_size=20)
            gui.text(f"estimated G: {float(g[None]):.1f}", pos=(0.02, 0.88), color=0xD3A8FF, font_size=20)
           
            gui.show(f'{dir}/{opt_iter:04d}_{t:04d}.png')
        
# ------------------------ main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--target', type=str, default='mpm128_simplified/target_positions.npy',
                        help='GT trajectory file: (T, N, 2) float32')
    parser.add_argument('--horizon', type=int, default=steps)
    args = parser.parse_args()

    # Build scene & allocate fields
    scene = make_snowball_scene()
    allocate_fields()

    # --- after allocate_fields(), load GT first ---
    gt = np.load(args.target).astype(np.float32)   # shape (Tgt, N, 2)
    Tgt, Ngt = gt.shape[0], gt.shape[1]
    # if Ngt != n_particles:
    #     raise RuntimeError(f"GT particle count {Ngt} != scene {n_particles}.")

    # Fit GT to horizon/max_steps and upload to field x_target
    T = min(args.horizon, min(Tgt, max_steps))
    # gt = gt[::4]
    gt_trim = gt[:T]
    for t in range(T):
        print(gt_trim[t, 0])
    if T < max_steps:
        pad = np.repeat(gt_trim[-1:], max_steps - T, axis=0)
        gt_full = np.concatenate([gt_trim, pad], axis=0)
    else:
        gt_full = gt_trim
    x_target.from_numpy(gt_full)
    init0()
    # Init gravity (positive number; forward uses -y)
    g[None] = 0.0

    # Optimize gravity
    lr = 100
    losses = []

    for it in range(args.iters):
        loss[None] = 0.0
        clear_particle_grad()  # clean adjoints from previous iter
        with ti.ad.Tape(loss=loss):
            forward(T)
        L = float(loss[None])
        losses.append(L)
        if it % 10 == 0 or it == args.iters - 1:
            print(f'iter {it:03d}  loss={L:.6e}  g={float(g[None]):.4f}  g.grad={float(g.grad[None]):.6e}')
        if it % 5 == 1:
            visualize(it, gt, dir="mpm128_simplified/optimize")
            init0()

        # gradient step (clip)
        g_clamped_grad = max(min(g.grad[None], 1.0), -1.0)
        g[None] -= lr * g_clamped_grad

    print('Done. Estimated gravity:', float(g[None]))

if __name__ == '__main__':
    main()