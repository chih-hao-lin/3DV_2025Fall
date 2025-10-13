import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu)

# Simulation parameters
n_steps = 300
dt = 0.01
gravity = 9.8

# Ground truth bounciness
bounciness_gt = 0.8
bounciness_guess = 0.3

# State
pos_gt = ti.field(dtype=ti.f32, shape=n_steps)
vel_gt = ti.field(dtype=ti.f32, shape=n_steps)
pos_guess = ti.field(dtype=ti.f32, shape=n_steps)
vel_guess = ti.field(dtype=ti.f32, shape=n_steps)


@ti.kernel
def simulate(pos: ti.template(), vel: ti.template(), bounciness: ti.f32):
    pos[0] = 5.0
    vel[0] = 0.0
    
    for i in range(n_steps - 1):
        vel[i + 1] = vel[i] - gravity * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt
        
        if pos[i + 1] < 0.0:
            pos[i + 1] = 0.0
            vel[i + 1] = -vel[i + 1] * bounciness


def compute_loss(pos1, pos2):
    diff = pos1 - pos2
    return np.mean(diff ** 2)


def main():
    global bounciness_guess
    
    print("=" * 60)
    print("1D BOUNCING BALL - INVERSE PHYSICS")
    print("=" * 60)
    print(f"\nGround truth bounciness: {bounciness_gt}")
    print(f"Initial guess:           {bounciness_guess}")
    print("\nGenerating video...\n")
    
    # Generate ground truth
    simulate(pos_gt, vel_gt, bounciness_gt)
    gt_trajectory = pos_gt.to_numpy()
    
    # Setup GUI with show_gui=False for WSL
    gui = ti.GUI("Inverse Physics - 1D Bouncing Ball", res=(800, 600), show_gui=False)
    
    # Create output directory
    os.makedirs("frames", exist_ok=True)
    
    learning_rate = 0.01
    epsilon = 0.001
    iteration = 0
    max_iterations = 100
    
    # Store trajectories for each iteration
    all_losses = []
    all_guesses = []
    
    while iteration < max_iterations:
        # Simulate with current guess
        simulate(pos_guess, vel_guess, bounciness_guess)
        guess_trajectory = pos_guess.to_numpy()
        
        # Compute loss and gradient
        loss = compute_loss(gt_trajectory, guess_trajectory)
        all_losses.append(loss)
        all_guesses.append(bounciness_guess)
        
        simulate(pos_guess, vel_guess, bounciness_guess + epsilon)
        trajectory_plus = pos_guess.to_numpy()
        loss_plus = compute_loss(gt_trajectory, trajectory_plus)
        
        gradient = (loss_plus - loss) / epsilon
        bounciness_guess -= learning_rate * gradient
        bounciness_guess = np.clip(bounciness_guess, 0.0, 0.99)
        
        # Print progress
        if iteration % 5 == 0:
            error_percent = abs(bounciness_guess - bounciness_gt) / bounciness_gt * 100
            print(f"Iter {iteration:3d} | Loss: {loss:.6f} | Guess: {bounciness_guess:.4f} | Error: {error_percent:.1f}%")
        
        # Create visualization frame - render multiple times per iteration for smooth animation
        for animation_frame in range(3):  # 3 frames per iteration
            gui.clear(0x000000)
            
            # Draw trajectories
            time_coords = np.linspace(0.1, 0.9, n_steps)
            
            # Ground truth in RED
            for i in range(n_steps - 1):
                x1, y1 = time_coords[i], gt_trajectory[i] / 6.0
                x2, y2 = time_coords[i + 1], gt_trajectory[i + 1] / 6.0
                gui.line((x1, y1), (x2, y2), color=0xFF0000, radius=2)
            
            # Guess in GREEN
            for i in range(n_steps - 1):
                x1, y1 = time_coords[i], guess_trajectory[i] / 6.0
                x2, y2 = time_coords[i + 1], guess_trajectory[i + 1] / 6.0
                gui.line((x1, y1), (x2, y2), color=0x00FF00, radius=2)
            
            # Draw ground
            gui.line((0.0, 0.0), (1.0, 0.0), color=0xFFFFFF, radius=3)
            
            # Draw current ball positions (animated)
            # Cycle through the simulation to create bouncing animation
            sim_frame = (iteration * 10 + animation_frame * 3) % n_steps
            gt_ball_y = gt_trajectory[sim_frame] / 6.0
            guess_ball_y = guess_trajectory[sim_frame] / 6.0
            
            gui.circle((0.2, gt_ball_y), color=0xFF0000, radius=15)
            gui.circle((0.8, guess_ball_y), color=0x00FF00, radius=15)
            
            # Text overlay
            gui.text(content=f'Ground Truth (RED) - Bounciness: {bounciness_gt:.3f}', 
                     pos=(0.02, 0.96), color=0xFF0000, font_size=20)
            gui.text(content=f'Learned (GREEN) - Bounciness: {bounciness_guess:.3f}', 
                     pos=(0.02, 0.92), color=0x00FF00, font_size=20)
            gui.text(content=f'Iteration: {iteration}/{max_iterations}', 
                     pos=(0.02, 0.12), color=0xFFFFFF, font_size=24)
            gui.text(content=f'Loss: {loss:.6f}', 
                     pos=(0.02, 0.08), color=0xFFFFFF, font_size=24)
            
            error = abs(bounciness_guess - bounciness_gt)
            error_pct = error / bounciness_gt * 100
            gui.text(content=f'Error: {error:.4f} ({error_pct:.1f}%)', 
                     pos=(0.02, 0.04), color=0xFFFF00, font_size=24)
            
            # Save frame
            frame_num = iteration * 3 + animation_frame
            gui.show(f"frames/frame_{frame_num:04d}.png")
        
        iteration += 1
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Ground truth: {bounciness_gt}")
    print(f"Final guess:  {bounciness_guess:.4f}")
    print(f"Error:        {abs(bounciness_guess - bounciness_gt):.4f}")
    print("=" * 60)
    
    # Create video using ffmpeg
    print("\nCreating video from frames...")
    os.system("ffmpeg -y -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 inverse_physics_demo.mp4")
    
    print("\n✅ Video saved as 'inverse_physics_demo.mp4'")
    print("   You can open this in Windows to view the results!")
    
    # Also create a summary plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Final trajectories
        time = np.arange(n_steps) * dt
        axes[0, 0].plot(time, gt_trajectory, 'r-', label=f'Ground Truth (b={bounciness_gt})', linewidth=3)
        axes[0, 0].plot(time, guess_trajectory, 'g--', label=f'Learned (b={bounciness_guess:.4f})', linewidth=3)
        axes[0, 0].set_xlabel('Time (s)', fontsize=12)
        axes[0, 0].set_ylabel('Height (m)', fontsize=12)
        axes[0, 0].set_title('Ball Trajectories - Final Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([-0.2, 5.5])
        
        # Plot 2: Loss convergence
        axes[0, 1].plot(all_losses, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0, 1].set_title('Optimization Progress', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Parameter convergence
        axes[1, 0].plot(all_guesses, 'g-', linewidth=3, label='Estimated Bounciness')
        axes[1, 0].axhline(y=bounciness_gt, color='r', linestyle='--', linewidth=2, label='Ground Truth')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Bounciness Coefficient', fontsize=12)
        axes[1, 0].set_title('Parameter Convergence', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.2, 0.9])
        
        # Plot 4: Error reduction
        errors = [abs(g - bounciness_gt) for g in all_guesses]
        axes[1, 1].plot(errors, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteration', fontsize=12)
        axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
        axes[1, 1].set_title('Error Reduction Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('summary_plots.png', dpi=200, bbox_inches='tight')
        print("✅ Summary plots saved as 'summary_plots.png'")
        
    except Exception as e:
        print(f"Could not create summary plots: {e}")


if __name__ == "__main__":
    main()
