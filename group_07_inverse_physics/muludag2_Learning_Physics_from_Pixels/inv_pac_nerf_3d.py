import taichi as ti
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import os
import cv2

ti.init(arch=ti.cpu)

# Simulation parameters
n_particles = 100
n_grid = 64
dx = 1 / n_grid
dt = 1e-4
steps = 50

# Taichi fields
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
Jp = ti.field(dtype=ti.f32, shape=n_particles)

# Grid
grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# Ground truth physics parameters
theta_gt = torch.tensor([0.3, 400.0], requires_grad=False)  # [mu, lambda]

# Optimized physics parameters
theta = torch.tensor([0.5, 200.0], requires_grad=True)  # start guess
optimizer = torch.optim.Adam([theta], lr=0.01)


@ti.kernel
def init_particles():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3, ti.random() * 0.2 + 0.3]
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(ti.f32, 2, 2)


@ti.kernel
def substep(mu: ti.f32, la: ti.f32):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    for p in range(n_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        r, s = ti.polar_decompose(F[p])
        stress = 2 * mu * (F[p] - r) @ F[p].transpose() + ti.Matrix.identity(ti.f32, 2) * la * (J - 1) * J
        stress = (-dt * 4 * stress) / (dx * dx)
        affine = stress + Jp[p] * C[p]
        mass = 1.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.8
            if i < 3 or i > n_grid - 3 or j < 3:
                grid_v[i, j] = [0, 0]

    for p in range(n_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * ti.outer_product(g_v, dpos) / (dx * dx)
        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C


def run_mpm(theta_val):
    init_particles()
    for s in range(steps):
        substep(float(theta_val[0]), float(theta_val[1]))
    return x.to_numpy()


def silhouette_loss(pred, gt):
    grid_size = 64
    pred_img = np.zeros((grid_size, grid_size))
    gt_img = np.zeros((grid_size, grid_size))
    pred_img[(pred * grid_size).astype(int)[:, 0], (pred * grid_size).astype(int)[:, 1]] = 1
    gt_img[(gt * grid_size).astype(int)[:, 0], (gt * grid_size).astype(int)[:, 1]] = 1
    return torch.tensor(np.mean((pred_img - gt_img) ** 2), dtype=torch.float32, requires_grad=True)


frames = []
gt_pos = run_mpm(theta_gt)

for it in tqdm(range(80), desc="Optimization"):
    pred_pos = run_mpm(theta.detach().numpy())
    loss = silhouette_loss(pred_pos, gt_pos)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 10 == 0:
        print(f"[Iter {it}] Loss={loss.item():.6f}, mu={theta[0].item():.4f}, lambda={theta[1].item():.4f}")

    fig, ax = plt.subplots()
    ax.scatter(gt_pos[:, 0], gt_pos[:, 1], s=5, label="GT", c="blue")
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], s=5, label="Pred", c="red")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)
    plt.close(fig)

# SAve vid
video_path = "mpm_inverse_final.mp4"
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames[0].shape[1], frames[0].shape[0]))
for f in frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
out.release()
print(f"✅ Saved video: {video_path}")
