import numpy as np
import matplotlib.pyplot as plt
from math import pi
from visual_kinematics.RobotSerial import RobotSerial, Frame

# Orginal Source: https://github.com/dbddqy/visual_kinematics?tab=readme-ov-file

# Disclosure of Usage of AI Assistant ---------------------------------------------------------------------------------------------------
# I used Gemini (version 2.5 Pro) for this assignment. 
# Access link: https://gemini.google.com/ 
# The tool was used several times in September 2025.
#  
# Initially, I used Gemini to understand the code.
# Then I used AI for drafting code to generate various kinds of results.

# All AI-generated content was thoroughly reviewed and revised by me to ensure its accuracy and relevance. 
# I can provide the unedited transcripts with prompts, interactions. 
# I take full responsibility for the final work, ensuring that it is my own.
# ----------------------------------------------------------------------------------------------------------------------------------------

np.set_printoptions(precision=3, suppress=True)

def plot_path_segmented(ax, pts, color='#ADD8E6', lw=2.2, gap=0.10):

    if len(pts) < 2: 
        return
    seg = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[i-1]) <= gap:
            seg.append(pts[i])
        else:
            seg = np.array(seg)
            ax.plot(seg[:,0], seg[:,1], seg[:,2], color=color, linewidth=lw, alpha=0.95)
            seg = [pts[i]]

    seg = np.array(seg)
    ax.plot(seg[:,0], seg[:,1], seg[:,2], color=color, linewidth=lw, alpha=0.95)


def heart_curve(n_pts=180, scale=0.08, center=(0.30, 0.00), z=0.95):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=True)
    x = scale * (16*np.sin(t)**3) + center[0]
    y = scale * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) + center[1]
    z = np.full_like(x, z)
    return np.stack([x, y, z], axis=1)  # (N,3)


dh_params = np.array([
    [0.163,  0.0,   0.5*pi,  0.0],
    [0.0,    0.632, pi,      0.5*pi],
    [0.0,    0.6005,pi,      0.0],
    [0.2013, 0.0,  -0.5*pi, -0.5*pi],
    [0.1025, 0.0,   0.5*pi,  0.0],
    [0.094,  0.0,   0.0,     0.0],
], dtype=float)

robot = RobotSerial(dh_params.copy())

heart = heart_curve(
    n_pts=320, 
    scale=0.04,          
    center=(0.26, 0.00),  
    z=0.90                
)

abc = np.array([0.0, -0.25, 0.0])  

robot.step_size = 0.6
robot.max_iter  = 400
robot.final_loss = 1e-3

robot.axis_values = np.zeros(robot.num_axis)
succeeded = []
for p in heart:
    target = Frame.from_euler_3(abc, p.reshape(3,1))
    robot.inverse(target)
    if robot.is_reachable_inverse:
        succeeded.append(p)

fig = plt.figure(figsize=(6.4, 6.0))
ax = fig.add_subplot(111, projection='3d')

if succeeded:
    succeeded = np.array(succeeded)
    plot_path_segmented(ax, succeeded, color="#F06996", gap=0.14)  

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=22, azim=-60)

xs, ys, zs = heart[:,0], heart[:,1], heart[:,2]
m = 0.15
ax.set_xlim([xs.min()-m, xs.max()+m])
ax.set_ylim([ys.min()-m, ys.max()+m])
ax.set_zlim([zs.min()-m, zs.max()+m])

plt.tight_layout()
plt.show()