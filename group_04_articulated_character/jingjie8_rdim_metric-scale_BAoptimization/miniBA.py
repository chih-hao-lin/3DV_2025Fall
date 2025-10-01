import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

np.random.seed(42)
focal_length = 0.3
image_plane_width = 0.6
N_cams_desired = 8


def rotmat(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def camera_dirs(angle):
    forward = np.array([np.cos(angle), np.sin(angle)])
    right = np.array([-forward[1], forward[0]])
    return forward, right

def project_to_camera(X, cam_pos, cam_angle, f=0.3):
    forward, right = camera_dirs(cam_angle)
    rel = X - cam_pos
    x_cam = rel @ right
    z_cam = rel @ forward
    return f * (x_cam / z_cam)


triangle_ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
theta_mesh_gt = np.pi / 6
trans_mesh_gt = np.array([0.3, -0.4])
triangle_gt = triangle_ref @ rotmat(theta_mesh_gt).T + trans_mesh_gt


camera_gt, observations = [], []
triangle_center = triangle_gt.mean(axis=0)
angles = np.linspace(0, 2 * np.pi, N_cams_desired, endpoint=False)

for angle in angles:
    radius = np.random.uniform(2, 3)
    cam_pos = triangle_center + radius * np.array([np.cos(angle), np.sin(angle)])
    cam_angle = np.arctan2(triangle_center[1] - cam_pos[1], triangle_center[0] - cam_pos[0])
    proj = project_to_camera(triangle_gt, cam_pos, cam_angle, f=focal_length)
    observations.append(proj)
    camera_gt.append((cam_pos, cam_angle))

camera_gt = np.array(camera_gt, dtype=object)
observations = np.array(observations)
N_cams = len(camera_gt)


cam_angles_init = np.array([theta for _, theta in camera_gt])+ np.random.randn(N_cams) * 0.5
cam_pos_init = np.array([pos for pos, _ in camera_gt]) + np.random.randn(N_cams, 2) * 0.5
x0 = np.stack([cam_angles_init, cam_pos_init[:, 0], cam_pos_init[:, 1]], axis=1).flatten()

def unpack_camera_params(x):
    cam_params = x.reshape(N_cams, 3)
    cam_angles = cam_params[:, 0]
    cam_pos = cam_params[:, 1:3]
    return cam_angles, cam_pos

def triangulate_point(obs, cam_angles, cam_pos):
    A = []
    b = []
    for i in range(N_cams):
        theta = cam_angles[i]
        pos = cam_pos[i]
        fwd, right = camera_dirs(theta)
        ray_dir = fwd * focal_length + right * obs[i]
        n = np.array([[0, -1], [1, 0]]) @ ray_dir  # normal to ray
        A.append(n)
        b.append(n @ pos)
    A = np.stack(A)
    b = np.array(b)
    pt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pt

def loss_fn(x):
    cam_angles, cam_pos = unpack_camera_params(x)
    loss = 0.0
    for pt_idx in range(3):
        X = triangulate_point(observations[:, pt_idx], cam_angles, cam_pos)
        for i in range(N_cams):
            proj = project_to_camera(X, cam_pos[i], cam_angles[i], f=focal_length)
            
            loss += (proj - observations[i, pt_idx]) ** 2
        loss +=   np.sum((X - triangle_gt[pt_idx]) ** 2)
    return loss


history = []
def callback(xk):
    history.append(xk.copy())
    print(f"[BA Step {len(history)}] Loss: {loss_fn(xk):.6f}")

minimize(loss_fn, x0, method='L-BFGS-B', callback=callback)
if len(history) == 0:
    history.append(x0.copy())


fig, ax = plt.subplots(figsize=(7, 7))
colors = ['r', 'g', 'b']
gt_triangle_plot, = ax.plot([], [], 'k-', label='Default Mesh')
opt_triangle_plot, = ax.plot([], [], 'g--', label='Triangulated Mesh')

gt_pts = ax.scatter([0]*3, [0]*3, c=colors, s=60)
opt_pts = ax.scatter([0]*3, [0]*3, marker='x', c=colors, s=60)
cam_lines, obs_keypoints, frustum_lines = [], [], []

for _ in range(N_cams):
    cam_line, = ax.plot([], [], 'gray', linestyle=':')
    obs_pts = ax.scatter([0]*3, [0]*3, c=colors, marker='o', s=40, alpha=0.6)
    frustum_line, = ax.plot([], [], 'k--', lw=0.7)
    cam_lines.append(cam_line)
    obs_keypoints.append(obs_pts)
    frustum_lines.append(frustum_line)

def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Triangulation-Based Bundle Adjustment")
    ax.legend()
    return [gt_triangle_plot, opt_triangle_plot, gt_pts, opt_pts] + cam_lines + obs_keypoints + frustum_lines

def update(step_idx):
    x = history[step_idx]
    cam_angles, cam_pos = unpack_camera_params(x)
    tri_opt = np.stack([
        triangulate_point(observations[:, i], cam_angles, cam_pos)
        for i in range(3)
    ])

    gt_triangle_plot.set_data(*triangle_gt[[0,1,2,0]].T)
    opt_triangle_plot.set_data(*tri_opt[[0,1,2,0]].T)
    gt_pts.set_offsets(triangle_gt)
    opt_pts.set_offsets(tri_opt)

    for i in range(N_cams):
        fwd, right = camera_dirs(cam_angles[i])
        image_center = cam_pos[i] + focal_length * fwd
        cam_lines[i].set_data([cam_pos[i][0], image_center[0]],
                              [cam_pos[i][1], image_center[1]])

        # Draw image plane frustum
        left_edge = image_center - (image_plane_width / 2) * right
        right_edge = image_center + (image_plane_width / 2) * right
        frustum_lines[i].set_data([cam_pos[i][0], left_edge[0], right_edge[0], cam_pos[i][0]],
                                  [cam_pos[i][1], left_edge[1], right_edge[1], cam_pos[i][1]])

        obs_pts = observations[i][:, None] * right[None, :] + image_center
        obs_keypoints[i].set_offsets(obs_pts)

    ax.set_title(f"Step {step_idx+1}/{len(history)}")
    return [gt_triangle_plot, opt_triangle_plot, gt_pts, opt_pts] + cam_lines + obs_keypoints + frustum_lines

anim = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=False, interval=300)
#anim.save("bundle_adjustment_triangles.gif", fps=5)
plt.show()
