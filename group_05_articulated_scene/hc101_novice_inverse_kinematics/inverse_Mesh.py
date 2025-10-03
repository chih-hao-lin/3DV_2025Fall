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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.set_printoptions(precision=3, suppress=True)

fig = plt.figure(figsize=(6.4, 6.0))
ax  = fig.add_subplot(111, projection='3d')

GAP = 0.12  
started = False
trail_segments = []     
curr_seg = []
link_meshes = []

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

path = heart_curve(scale=0.04, center=(0.28, 0.00), z=0.90)
abc  = np.array([0.0, -0.25, 0.0])   # [yaw, pitch, roll] = ZYX

robot.axis_values = np.zeros(robot.num_axis)
robot.step_size   = 0.5
robot.max_iter    = 600
robot.final_loss  = 1e-3
# robot.inv_m = "jac_t"  

# ---------------- Figure ----------------
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=22, azim=-60)

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

def cylinder_between(ax, p1, p2, r=0.02, n=24, color='#87CEFA'):

    p1, p2 = np.asarray(p1), np.asarray(p2)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1e-9:
        return None
    v = v / L


    a = np.array([1,0,0]) if abs(v[0]) < 0.9 else np.array([0,1,0])
    n1 = np.cross(v, a); n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    th = np.linspace(0, 2*np.pi, n, endpoint=True)
    circle1 = p1 + r*(np.outer(np.cos(th), n1) + np.outer(np.sin(th), n2))
    circle2 = p2 + r*(np.outer(np.cos(th), n1) + np.outer(np.sin(th), n2))

    faces = []
    for i in range(n-1):
        quad = [circle1[i], circle1[i+1], circle2[i+1], circle2[i]]
        faces.append(quad)

    faces.append(circle1)
    faces.append(circle2[::-1])

    coll = Poly3DCollection(faces, facecolor=color, edgecolor='none')
    ax.add_collection3d(coll)
    return coll

def redraw_trail():

    global trail_segments, curr_seg
    [l.remove() for l in getattr(redraw_trail, "lines", [])] if hasattr(redraw_trail, "lines") else None
    redraw_trail.lines = []
    for seg in trail_segments + ([np.array(curr_seg)] if len(curr_seg) >= 2 else []):
        ln, = ax.plot(seg[:,0], seg[:,1], seg[:,2], color="#F06996", linewidth=2.2, alpha=0.95)
        redraw_trail.lines.append(ln)

def update(i):
    global link_meshes
    target = Frame.from_euler_3(abc, path[i].reshape(3,1))
    robot.inverse(target)

    for m in link_meshes:
        try: m.remove()
        except: pass
    link_meshes = []
    link_colors = ["#000000",  
               "#4D4D4D",  
               "#000000", 
               "#585858", 
               "#000000", 
               "#4E4E4E"] 

    if robot.is_reachable_inverse:
        poly = np.array([f.t_3_1.flatten() for f in robot.axis_frames])

        link_line.set_data(poly[:,0], poly[:,1])
        link_line.set_3d_properties(poly[:,2])

        for j in range(len(poly)-1):
            color = link_colors[j % len(link_colors)]
            coll = cylinder_between(ax, poly[j], poly[j+1],
                                    r=0.04, n=24, color=color)
            if coll is not None:
                link_meshes.append(coll)

        add_point_to_trail(path[i])
        redraw_trail()

    return link_line, *getattr(redraw_trail, "lines", [])

ani = FuncAnimation(fig, update, frames=len(path), init_func=init,
                    interval=25, blit=False, repeat=True)

plt.tight_layout()
plt.show()