#!/usr/bin/env python3

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi
from visual_kinematics.RobotSerial import RobotSerial, Frame

np.set_printoptions(precision=3, suppress=True)

fig = plt.figure(figsize=(6.4, 6.0))
ax  = fig.add_subplot(111, projection='3d')

GAP = 0.12 
started = False
trail_segments = []      # [np.ndarray(N,3), ...] 
curr_seg = []

# ---------------- Heart path ----------------
def heart_curve(n_pts=300, scale=0.07, center=(0.30, 0.00), z=0.92):
    t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)    
    x = scale * (16*np.sin(t)**3) + center[0]
    y = scale * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) + center[1]
    z = np.full_like(x, z)
    return np.stack([x, y, z], axis=1)                   

# ---------------- Robot model (DH) ----------------
dh_params = np.array([
    [0.163,  0.0,   0.5*pi,  0.0],
    [0.0,    0.632, pi,      0.5*pi],
    [0.0,    0.6005,pi,      0.0],
    [0.2013, 0.0,  -0.5*pi, -0.5*pi],
    [0.1025, 0.0,   0.5*pi,  0.0],
    [0.094,  0.0,   0.0,     0.0],
], dtype=float)

robot = RobotSerial(dh_params.copy())

# base_p = np.array([0.0, 0.0, 0.0])  
# # ax.scatter([base_p[0]], [base_p[1]], [base_p[2]], s=30, c='#666666')

path = heart_curve(scale=0.04, center=(0.28, 0.00), z=0.90)
abc  = np.array([0.0, -0.25, 0.0])   # [yaw, pitch, roll] = ZYX

robot.axis_values = np.zeros(robot.num_axis)
robot.step_size   = 0.5
robot.max_iter    = 600
robot.final_loss  = 1e-3

# ---------------- Figure ----------------
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=22, azim=-60)

# 범위 고정
m=0.15
xs, ys, zs = path[:,0], path[:,1], path[:,2]
ax.set_xlim([xs.min()-m, xs.max()+m])
ax.set_ylim([ys.min()-m, ys.max()+m])
ax.set_zlim([0, zs.max()+m])

link_line, = ax.plot([], [], [], color="#55585A", marker='o', linewidth=2.0, markersize=3)
trail_line, = ax.plot([], [], [], color= "#F06996", linewidth=2.2, alpha=0.95)

trail_points = []  

def fk_polyline(rbt):
    pts = [f.t_3_1.flatten() for f in rbt.axis_frames]
    return np.array(pts)  # (num_axis+1, 3)

def init():
    link_line.set_data([], []); link_line.set_3d_properties([])
    trail_line.set_data([], []); trail_line.set_3d_properties([])
    return link_line, trail_line

def add_point_to_trail(pt):
    global started, curr_seg, trail_segments
    if not started:
        curr_seg = [pt]
        started = True
        return
    if np.linalg.norm(pt - curr_seg[-1]) > GAP:
        if len(curr_seg) >= 2:
            trail_segments.append(np.array(curr_seg))
        curr_seg = [pt]
    else:
        curr_seg.append(pt)

def redraw_trail():
    global trail_segments, curr_seg
    [l.remove() for l in getattr(redraw_trail, "lines", [])] if hasattr(redraw_trail, "lines") else None
    for seg in trail_segments + ([np.array(curr_seg)] if len(curr_seg) >= 2 else []):
        ln, = ax.plot(seg[:,0], seg[:,1], seg[:,2], color="#F06996", linewidth=2.2, alpha=0.95)
        redraw_trail.lines.append(ln)

def update(i):
    target = Frame.from_euler_3(abc, path[i].reshape(3,1))
    robot.inverse(target)

    if robot.is_reachable_inverse:
        poly = np.array([f.t_3_1.flatten() for f in robot.axis_frames])
        link_line.set_data(poly[:,0], poly[:,1])
        link_line.set_3d_properties(poly[:,2])

        add_point_to_trail(path[i])
        redraw_trail()

    return link_line, *getattr(redraw_trail, "lines", [])

ani = FuncAnimation(fig, update, frames=len(path), init_func=init,
                    interval=25, blit=False, repeat=True)  

plt.tight_layout()
plt.show()