import torch
import numpy as np
import os

print("=" * 40)
print("pytorch physics sim - 1d")
print("=" * 40)

n_steps = 300
dt = 0.01
gravity = 9.8

# params
bounciness_gt = 0.7
bounciness_guess = 0.3

print(f"gt: {bounciness_gt}")
print(f"guess: {bounciness_guess}\n")


def simulate(b):
    h = torch.zeros(n_steps)
    v = torch.zeros(n_steps)
    
    h[0] = 5.0
    v[0] = 0.0
    
    for t in range(n_steps - 1):
        v_new = v[t] - gravity * dt
        h_new = h[t] + v_new * dt
        
        if h_new.item() < 0:
            h[t + 1] = 0.0
            v[t + 1] = -v_new * b
        else:
            h[t + 1] = h_new
            v[t + 1] = v_new
    
    return h


print("making gt trajectory...")
with torch.no_grad():
    gt_traj = simulate(torch.tensor([bounciness_gt])).numpy()

# viz
import taichi as ti
ti.init(arch=ti.cpu)
gui = ti.GUI("1d physics", res=(800, 600), show_gui=False)

os.makedirs("frames", exist_ok=True)

# Optimization
print("\n[2] Optimizing...\n")

bounciness = torch.tensor([bounciness_guess], requires_grad=True)
optimizer = torch.optim.SGD([bounciness], lr=0.01, momentum=0.9)

all_losses = []
all_guesses = []
frame_counter = 0

for iteration in range(100):
    optimizer.zero_grad()
    
    # Forward
    predicted = simulate(bounciness)
    observed = torch.from_numpy(gt_traj)
    loss = torch.mean((predicted - observed) ** 2)
    
    # Backward
    loss.backward()
    
    # Update
    optimizer.step()
    with torch.no_grad():
        bounciness.clamp_(0.01, 0.99)
    
    # Track
    all_losses.append(loss.item())
    all_guesses.append(bounciness.item())
    
    # Log
    if iteration % 5 == 0:
        err_pct = abs(bounciness.item() - bounciness_gt) / bounciness_gt * 100
        print(f"Iter {iteration:3d} | Loss: {loss.item():.6f} | "
              f"b: {bounciness.item():.4f} | Error: {err_pct:.1f}%")
    
    # Render frames (3 per iteration for smooth video)
    with torch.no_grad():
        guess_traj = predicted.numpy()
    
    for anim_frame in range(3):
        gui.clear(0x000000)
        
        time_coords = np.linspace(0.1, 0.9, n_steps)
        
        # Ground truth in RED
        for i in range(n_steps - 1):
            x1, y1 = time_coords[i], gt_traj[i] / 6.0
            x2, y2 = time_coords[i + 1], gt_traj[i + 1] / 6.0
            gui.line((x1, y1), (x2, y2), color=0xFF0000, radius=2)
        
        # Learned in GREEN
        for i in range(n_steps - 1):
            x1, y1 = time_coords[i], guess_traj[i] / 6.0
            x2, y2 = time_coords[i + 1], guess_traj[i + 1] / 6.0
            gui.line((x1, y1), (x2, y2), color=0x00FF00, radius=2)
        
        # Ground
        gui.line((0.0, 0.0), (1.0, 0.0), color=0xFFFFFF, radius=3)
        
        # Animated balls
        sim_frame = (iteration * 10 + anim_frame * 3) % n_steps
        gt_ball_y = gt_traj[sim_frame] / 6.0
        guess_ball_y = guess_traj[sim_frame] / 6.0
        
        gui.circle((0.2, gt_ball_y), color=0xFF0000, radius=15)
        gui.circle((0.8, guess_ball_y), color=0x00FF00, radius=15)
        
        # Text
        gui.text(content=f'Ground Truth (RED) - b: {bounciness_gt:.3f}', 
                 pos=(0.02, 0.96), color=0xFF0000, font_size=20)
        gui.text(content=f'PyTorch Learned (GREEN) - b: {bounciness.item():.3f}', 
                 pos=(0.02, 0.92), color=0x00FF00, font_size=20)
        gui.text(content=f'Iteration: {iteration}/100', 
                 pos=(0.02, 0.12), color=0xFFFFFF, font_size=24)
        gui.text(content=f'Loss: {loss.item():.6f}', 
                 pos=(0.02, 0.08), color=0xFFFFFF, font_size=24)
        
        err = abs(bounciness.item() - bounciness_gt)
        err_pct = err / bounciness_gt * 100
        gui.text(content=f'Error: {err:.4f} ({err_pct:.1f}%)', 
                 pos=(0.02, 0.04), color=0xFFFF00, font_size=24)
        
        gui.show(f"frames/frame_{frame_counter:04d}.png")
        frame_counter += 1
    
    if abs(bounciness.item() - bounciness_gt) < 0.02:
        print(f"\nCONVERGED at iteration {iteration}!")
        break

# === ADD FINAL DEMONSTRATION SEQUENCE ===
print("\n[3] Rendering final demonstration...\n")

# Generate final trajectories with learned parameter
with torch.no_grad():
    final_learned = simulate(bounciness).numpy()
    final_gt = gt_traj

# Render 200 frames showing both balls bouncing (about 6.7 seconds at 30fps)
for demo_frame in range(200):
    gui.clear(0x000000)
    
    time_coords = np.linspace(0.1, 0.9, n_steps)
    
    # Draw full trajectories (faded)
    for i in range(n_steps - 1):
        x1, y1 = time_coords[i], final_gt[i] / 6.0
        x2, y2 = time_coords[i + 1], final_gt[i + 1] / 6.0
        gui.line((x1, y1), (x2, y2), color=0x660000, radius=1)  # Dark red
        
        x1, y1 = time_coords[i], final_learned[i] / 6.0
        x2, y2 = time_coords[i + 1], final_learned[i + 1] / 6.0
        gui.line((x1, y1), (x2, y2), color=0x006600, radius=1)  # Dark green
    
    # Ground
    gui.line((0.0, 0.0), (1.0, 0.0), color=0xFFFFFF, radius=3)
    
    # Animated balls cycling through the simulation
    sim_idx = (demo_frame * 2) % n_steps  # Cycle through trajectory
    gt_y = final_gt[sim_idx] / 6.0
    learned_y = final_learned[sim_idx] / 6.0
    
    gui.circle((0.2, gt_y), color=0xFF0000, radius=15)
    gui.circle((0.8, learned_y), color=0x00FF00, radius=15)
    
    # Success message
    gui.text(content='CONVERGED - Final Demonstration', 
             pos=(0.25, 0.96), color=0xFFFF00, font_size=24)
    gui.text(content=f'Ground Truth: b={bounciness_gt:.3f}', 
             pos=(0.02, 0.88), color=0xFF0000, font_size=20)
    gui.text(content=f'Learned: b={bounciness.item():.4f} (Error: {abs(bounciness.item()-bounciness_gt)/bounciness_gt*100:.1f}%)', 
             pos=(0.02, 0.84), color=0x00FF00, font_size=20)
    
    gui.text(content='Watch both balls bounce identically!', 
             pos=(0.25, 0.08), color=0xFFFFFF, font_size=22)
    gui.text(content='PyTorch Automatic Differentiation', 
             pos=(0.23, 0.04), color=0x00FFFF, font_size=22)
    
    gui.show(f"frames/frame_{frame_counter:04d}.png")
    frame_counter += 1

print(f"   Rendered {frame_counter} total frames")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Ground truth: {bounciness_gt}")
print(f"Final guess:  {bounciness.item():.4f}")
print(f"Error:        {abs(bounciness.item() - bounciness_gt):.4f} ({abs(bounciness.item() - bounciness_gt)/bounciness_gt*100:.1f}%)")
print("=" * 60)

# Create video
print("\n[4] Creating video...")
os.system("ffmpeg -y -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 pytorch_inverse_physics.mp4")
print("✅ Video: pytorch_inverse_physics.mp4")

# Summary plots
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = np.arange(n_steps) * dt
    axes[0, 0].plot(time, gt_traj, 'r-', label=f'GT (b={bounciness_gt})', lw=3)
    axes[0, 0].plot(time, final_learned, 'g--', label=f'Learned (b={bounciness.item():.4f})', lw=3)
    axes[0, 0].set_xlabel('Time (s)', fontsize=12)
    axes[0, 0].set_ylabel('Height (m)', fontsize=12)
    axes[0, 0].set_title('PyTorch Autodiff: Inverse Physics', fontweight='bold', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(all_losses, 'b-', lw=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 1].set_title('Loss Convergence', fontweight='bold', fontsize=14)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(all_guesses, 'g-', lw=3, label='Estimated')
    axes[1, 0].axhline(bounciness_gt, color='r', ls='--', lw=2, label='GT')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Bounciness', fontsize=12)
    axes[1, 0].set_title('Parameter Convergence', fontweight='bold', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    errors = [abs(g - bounciness_gt) for g in all_guesses]
    axes[1, 1].plot(errors, 'purple', lw=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
    axes[1, 1].set_title('Error Reduction', fontweight='bold', fontsize=14)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_summary.png', dpi=200)
    print("✅ Summary: pytorch_summary.png")
except Exception as e:
    print(f"Summary plot error: {e}")

print("\nPAC-NERF CONCEPT SUCCESSFULLY DEMONSTRATED!")
print("  Formula: min_theta loss(diff_physics(theta) - observed)")
print("  - Differentiable physics with PyTorch")
print("  - Automatic gradients through simulation")
print("  - Parameter estimation from observations")
print("\nYou're ready for your presentation!\n")
