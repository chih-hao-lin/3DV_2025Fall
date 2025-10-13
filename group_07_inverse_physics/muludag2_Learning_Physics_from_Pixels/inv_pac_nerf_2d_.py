import torch
import numpy as np
import os

print("=" * 50)
print("2d pytorch physics sim")
print("=" * 50)

n_steps = 300
dt = 0.01

# params
gravity_gt = 9.8
bounciness_gt = 0.6

# guesses
gravity_guess = 5.0
bounciness_guess = 0.3

print(f"gt: g={gravity_gt}, b={bounciness_gt}")
print(f"guess: g={gravity_guess}, b={bounciness_guess}\n")


def simulate_2d(g, b):
    pos = torch.zeros(n_steps, 2)
    vel = torch.zeros(n_steps, 2)
    
    # Initial conditions - throw ball at an angle
    pos[0] = torch.tensor([0.2, 0.5])
    vel[0] = torch.tensor([3.0, 4.0])
    
    for t in range(n_steps - 1):
        # Physics: gravity only affects y
        vel_new = vel[t].clone()
        vel_new[1] -= g * dt
        
        # Update position
        pos_new = pos[t] + vel_new * dt
        
        # Ground collision (y = 0)
        if pos_new[1].item() < 0:
            pos_new[1] = 0.0
            vel_new[1] = -vel_new[1] * b
            vel_new[0] *= 0.95  # friction
        
        # Wall collisions
        if pos_new[0].item() < 0:
            pos_new[0] = 0.0
            vel_new[0] = -vel_new[0] * b
        elif pos_new[0].item() > 1.0:
            pos_new[0] = 1.0
            vel_new[0] = -vel_new[0] * b
        
        pos[t + 1] = pos_new
        vel[t + 1] = vel_new
    
    return pos


print("making gt trajectory...")
with torch.no_grad():
    gt_traj = simulate_2d(torch.tensor(gravity_gt), torch.tensor(bounciness_gt)).numpy()

# viz
import taichi as ti
ti.init(arch=ti.cpu)
gui = ti.GUI("2d physics", res=(800, 600), show_gui=False)

os.makedirs("frames", exist_ok=True)

# Learnable parameters
gravity = torch.tensor([gravity_guess], requires_grad=True)
bounciness = torch.tensor([bounciness_guess], requires_grad=True)

# Optimizer
optimizer = torch.optim.Adam([gravity, bounciness], lr=0.1)

# Optimization
print("\n[2] Optimizing 2 parameters with PyTorch autodiff...\n")

all_losses = []
all_gravity = []
all_bounciness = []
frame_counter = 0

for iteration in range(150):
    optimizer.zero_grad()
    
    # Forward pass
    predicted = simulate_2d(gravity[0], bounciness[0])
    observed = torch.from_numpy(gt_traj).float()
    
    # Loss: MSE on positions
    loss = torch.mean((predicted - observed) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Update
    optimizer.step()
    
    # Clamp to valid ranges
    with torch.no_grad():
        gravity.clamp_(1.0, 15.0)
        bounciness.clamp_(0.01, 0.99)
    
    # Track
    all_losses.append(loss.item())
    all_gravity.append(gravity.item())
    all_bounciness.append(bounciness.item())
    
    # Log
    if iteration % 10 == 0:
        g_err = abs(gravity.item() - gravity_gt) / gravity_gt * 100
        b_err = abs(bounciness.item() - bounciness_gt) / bounciness_gt * 100
        print(f"Iter {iteration:3d} | Loss: {loss.item():.6f} | "
              f"g: {gravity.item():5.2f} ({g_err:4.1f}%) | "
              f"b: {bounciness.item():.3f} ({b_err:4.1f}%)")
    
    # Visualize every 2 iterations
    if iteration % 2 == 0:
        with torch.no_grad():
            pred_traj = predicted.numpy()
        
        # Render 2 frames per iteration
        for anim in range(2):
            gui.clear(0x112233)
            
            # Scale for visualization
            max_height = max(gt_traj[:, 1].max(), pred_traj[:, 1].max())
            scale_y = 0.7 / (max_height + 0.1)
            
            # Draw ground
            gui.line((0, 0), (1, 0), color=0xFFFFFF, radius=3)
            
            # Draw trajectories
            for i in range(n_steps - 1):
                # GT in RED
                x1, y1 = gt_traj[i, 0], gt_traj[i, 1] * scale_y
                x2, y2 = gt_traj[i + 1, 0], gt_traj[i + 1, 1] * scale_y
                gui.line((x1, y1), (x2, y2), color=0xFF3333, radius=2)
                
                # Learned in GREEN
                x1, y1 = pred_traj[i, 0], pred_traj[i, 1] * scale_y
                x2, y2 = pred_traj[i + 1, 0], pred_traj[i + 1, 1] * scale_y
                gui.line((x1, y1), (x2, y2), color=0x33FF33, radius=2)
            
            # Animated balls
            frame = (iteration * 10 + anim * 5) % n_steps
            gt_pos = gt_traj[frame]
            pred_pos = pred_traj[frame]
            
            # Draw trails
            for j in range(30):
                idx = max(0, frame - j)
                alpha = 1.0 - j / 30.0
                size = int(8 * alpha + 3)
                
                gui.circle((gt_traj[idx, 0], gt_traj[idx, 1] * scale_y), 
                          color=0xFF0000, radius=size)
                gui.circle((pred_traj[idx, 0], pred_traj[idx, 1] * scale_y), 
                          color=0x00FF00, radius=size)
            
            # Text
            gui.text(content='PAC-NeRF: 2D Inverse Physics', 
                    pos=(0.28, 0.95), color=0xFFFF00, font_size=26)
            
            gui.text(content=f'Ground Truth (RED)', 
                    pos=(0.02, 0.88), color=0xFF3333, font_size=18)
            gui.text(content=f'  g={gravity_gt:.1f} m/s², b={bounciness_gt:.2f}', 
                    pos=(0.02, 0.84), color=0xFF3333, font_size=16)
            
            gui.text(content=f'Learned (GREEN)', 
                    pos=(0.02, 0.76), color=0x33FF33, font_size=18)
            gui.text(content=f'  g={gravity.item():.2f} m/s², b={bounciness.item():.2f}', 
                    pos=(0.02, 0.72), color=0x33FF33, font_size=16)
            
            gui.text(content=f'Iteration: {iteration}/150', 
                    pos=(0.02, 0.12), color=0xFFFFFF, font_size=24)
            gui.text(content=f'Loss: {loss.item():.6f}', 
                    pos=(0.02, 0.08), color=0xFFFFFF, font_size=24)
            
            g_err = abs(gravity.item() - gravity_gt)
            b_err = abs(bounciness.item() - bounciness_gt)
            gui.text(content=f'Errors: g={g_err:.2f} ({g_err/gravity_gt*100:.1f}%), '
                            f'b={b_err:.3f} ({b_err/bounciness_gt*100:.1f}%)', 
                    pos=(0.02, 0.04), color=0xFF8800, font_size=20)
            
            gui.show(f"frames/frame_{frame_counter:04d}.png")
            frame_counter += 1
    
    # Early stopping
    if (abs(gravity.item() - gravity_gt) < 0.3 and 
        abs(bounciness.item() - bounciness_gt) < 0.03):
        print(f"\nCONVERGED at iteration {iteration}!")
        break

# Final demonstration
print("\n[3] Rendering final demonstration...\n")

with torch.no_grad():
    final_pred = simulate_2d(gravity[0], bounciness[0]).numpy()

max_height = max(gt_traj[:, 1].max(), final_pred[:, 1].max())
scale_y = 0.7 / (max_height + 0.1)

for demo_frame in range(200):
    gui.clear(0x112233)
    
    # Ground
    gui.line((0, 0), (1, 0), color=0xFFFFFF, radius=3)
    
    # Draw full trajectories (faded)
    for i in range(n_steps - 1):
        gui.line((gt_traj[i, 0], gt_traj[i, 1] * scale_y),
                (gt_traj[i + 1, 0], gt_traj[i + 1, 1] * scale_y),
                color=0x660000, radius=1)
        gui.line((final_pred[i, 0], final_pred[i, 1] * scale_y),
                (final_pred[i + 1, 0], final_pred[i + 1, 1] * scale_y),
                color=0x006600, radius=1)
    
    # Animated balls
    idx = (demo_frame * 2) % n_steps
    
    for j in range(30):
        trail_idx = max(0, idx - j)
        alpha = 1.0 - j / 30.0
        size = int(10 * alpha + 4)
        
        gui.circle((gt_traj[trail_idx, 0], gt_traj[trail_idx, 1] * scale_y), 
                  color=0xFF0000, radius=size)
        gui.circle((final_pred[trail_idx, 0], final_pred[trail_idx, 1] * scale_y), 
                  color=0x00FF00, radius=size)
    
    # Success message
    gui.text(content='CONVERGED - Final Demonstration', 
            pos=(0.22, 0.95), color=0xFFFF00, font_size=26)
    gui.text(content=f'Ground Truth: g={gravity_gt:.1f}, b={bounciness_gt:.2f}', 
            pos=(0.02, 0.88), color=0xFF3333, font_size=20)
    gui.text(content=f'Learned: g={gravity.item():.2f}, b={bounciness.item():.2f}', 
            pos=(0.02, 0.84), color=0x33FF33, font_size=20)
    gui.text(content='PyTorch Automatic Differentiation', 
            pos=(0.22, 0.04), color=0x00FFFF, font_size=22)
    
    gui.show(f"frames/frame_{frame_counter:04d}.png")
    frame_counter += 1

print(f"   Rendered {frame_counter} total frames")

print("\nmaking video...")
os.system("ffmpeg -y -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 pytorch_physics.mp4")
print("saved video")

# Summary plots
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Trajectories
    axes[0, 0].plot(gt_traj[:, 0], gt_traj[:, 1], 'r-', lw=3, label='Ground Truth')
    axes[0, 0].plot(final_pred[:, 0], final_pred[:, 1], 'g--', lw=3, label='Learned')
    axes[0, 0].axhline(0, color='k', lw=2, alpha=0.3)
    axes[0, 0].set_xlabel('X Position', fontsize=12)
    axes[0, 0].set_ylabel('Y Position', fontsize=12)
    axes[0, 0].set_title('2D Trajectories', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Loss
    axes[0, 1].plot(all_losses, 'b-', lw=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 1].set_title('Loss Convergence', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error
    g_errors = [abs(g - gravity_gt) / gravity_gt * 100 for g in all_gravity]
    b_errors = [abs(b - bounciness_gt) / bounciness_gt * 100 for b in all_bounciness]
    axes[0, 2].plot(g_errors, 'purple', lw=2, label='Gravity %')
    axes[0, 2].plot(b_errors, 'orange', lw=2, label='Bounciness %')
    axes[0, 2].set_xlabel('Iteration', fontsize=12)
    axes[0, 2].set_ylabel('Error (%)', fontsize=12)
    axes[0, 2].set_title('Error Reduction', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Gravity convergence
    axes[1, 0].plot(all_gravity, 'purple', lw=3, label='Estimated')
    axes[1, 0].axhline(gravity_gt, color='r', ls='--', lw=2, label='Ground Truth')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Gravity (m/s²)', fontsize=12)
    axes[1, 0].set_title('Gravity Parameter', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bounciness convergence
    axes[1, 1].plot(all_bounciness, 'orange', lw=3, label='Estimated')
    axes[1, 1].axhline(bounciness_gt, color='r', ls='--', lw=2, label='Ground Truth')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Bounciness', fontsize=12)
    axes[1, 1].set_title('Bounciness Parameter', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final comparison
    axes[1, 2].text(0.1, 0.7, 'Ground Truth:', fontsize=14, fontweight='bold', color='red')
    axes[1, 2].text(0.1, 0.6, f'g = {gravity_gt:.1f} m/s²', fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'b = {bounciness_gt:.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.3, 'Learned:', fontsize=14, fontweight='bold', color='green')
    axes[1, 2].text(0.1, 0.2, f'g = {gravity.item():.2f} m/s²', fontsize=12)
    axes[1, 2].text(0.1, 0.1, f'b = {bounciness.item():.3f}', fontsize=12)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Final Results', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pacnerf_2d_summary.png', dpi=200)
    print("✅ Summary: pacnerf_2d_summary.png")
except Exception as e:
    print(f"Summary error: {e}")

print("\ndone!")
