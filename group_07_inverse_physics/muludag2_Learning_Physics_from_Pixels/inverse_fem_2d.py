import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu)

n_steps = 400
dt = 0.01

# params
gravity_gt = 9.8
bounciness_gt = 0.25

# guesses
gravity_guess = 9.8
bounciness_guess = 0.4

# 2D State: position and velocity
pos_gt = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)
vel_gt = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)

pos_guess = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)
vel_guess = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)


@ti.kernel
def simulate_2d(pos: ti.template(), vel: ti.template(), gravity: ti.f32, bounciness: ti.f32):
    pos[0] = ti.Vector([0.2, 0.5])
    vel[0] = ti.Vector([3.0, 4.0])
    
    # Simulate
    for i in range(n_steps - 1):
        # Update velocity (gravity only affects y)
        vel[i + 1] = vel[i]
        vel[i + 1][1] -= gravity * dt  # Gravity pulls down
        
        # Update position
        pos[i + 1] = pos[i] + vel[i + 1] * dt
        
        # Ground collision (y = 0)
        if pos[i + 1][1] < 0.0:
            pos[i + 1][1] = 0.0
            vel[i + 1][1] = -vel[i + 1][1] * bounciness  # Bounce vertically
            vel[i + 1][0] *= 0.95  # Some friction on x
        
        # Wall collisions (keep ball in bounds)
        if pos[i + 1][0] < 0.0:
            pos[i + 1][0] = 0.0
            vel[i + 1][0] = -vel[i + 1][0] * bounciness
        if pos[i + 1][0] > 1.0:
            pos[i + 1][0] = 1.0
            vel[i + 1][0] = -vel[i + 1][0] * bounciness


def compute_loss(pos1, pos2):
    pos1_np = pos1.to_numpy()
    pos2_np = pos2.to_numpy()
    diff = pos1_np - pos2_np
    return np.mean(np.sum(diff ** 2, axis=1))


def main():
    global gravity_guess, bounciness_guess
    
    print("=" * 50)
    print("2d bouncing ball - inverse physics")
    print("=" * 50)
    print(f"gt:   g={gravity_gt}, b={bounciness_gt}")
    print(f"guess: g={gravity_guess}, b={bounciness_guess}")
    
    # Generate ground truth
    simulate_2d(pos_gt, vel_gt, gravity_gt, bounciness_gt)
    
    # Setup GUI
    gui = ti.GUI("Inverse Physics - 2D Bouncing Ball", res=(800, 600), show_gui=False)
    
    # Create output directory
    os.makedirs("frames", exist_ok=True)
    
    # optim params
    learning_rate_gravity = 0.7
    learning_rate_bounciness = 0.0075
    epsilon_gravity = 0.1
    epsilon_bounciness = 0.001
    
    momentum = 0.9
    velocity_gravity = 0.0
    velocity_bounciness = 0.0
    
    iteration = 0
    max_iterations = 150
    
    # History tracking
    all_losses = []
    all_gravity_guesses = []
    all_bounciness_guesses = []
    
    while iteration < max_iterations:
        # Simulate with current guesses
        simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess)
        loss = compute_loss(pos_gt, pos_guess)
        
        all_losses.append(loss)
        all_gravity_guesses.append(gravity_guess)
        all_bounciness_guesses.append(bounciness_guess)
        
        # Compute gradient w.r.t. gravity using finite differences
        simulate_2d(pos_guess, vel_guess, gravity_guess + epsilon_gravity, bounciness_guess)
        loss_gravity_plus = compute_loss(pos_gt, pos_guess)
        grad_gravity = (loss_gravity_plus - loss) / epsilon_gravity
        
        # Compute gradient w.r.t. bounciness
        simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess + epsilon_bounciness)
        loss_bounciness_plus = compute_loss(pos_gt, pos_guess)
        grad_bounciness = (loss_bounciness_plus - loss) / epsilon_bounciness
        
        # Momentum-based gradient descent (much more aggressive!)
        velocity_gravity = momentum * velocity_gravity - learning_rate_gravity * grad_gravity
        velocity_bounciness = momentum * velocity_bounciness - learning_rate_bounciness * grad_bounciness
        
        gravity_guess += velocity_gravity
        bounciness_guess += velocity_bounciness
        
        # Clamp to valid ranges
        gravity_guess = np.clip(gravity_guess, 0.1, 20.0)
        bounciness_guess = np.clip(bounciness_guess, 0.0, 0.99)
        
        # Re-simulate with updated parameters for visualization
        simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess)
        
        # Print progress MORE FREQUENTLY to show the dramatic changes
        if iteration % 2 == 0 or iteration < 10:
            gravity_error = abs(gravity_guess - gravity_gt) / gravity_gt * 100
            bounciness_error = abs(bounciness_guess - bounciness_gt) / bounciness_gt * 100
            print(f"Iter {iteration:3d} | Loss: {loss:.6f} | "
                  f"g: {gravity_guess:5.2f} ({gravity_error:5.1f}%) | "
                  f"b: {bounciness_guess:.3f} ({bounciness_error:5.1f}%) | "
                  f"Δg: {velocity_gravity:+6.3f} Δb: {velocity_bounciness:+7.4f}")
        
        # Create visualization frames - multiple frames per iteration for smooth animation
        for anim_frame in range(3):  # 3 frames per iteration for smoother video
            gui.clear(0x112233)
            
            # Get trajectories
            gt_traj = pos_gt.to_numpy()
            guess_traj = pos_guess.to_numpy()
            
            # Draw ground
            gui.line((0.0, 0.0), (1.0, 0.0), color=0xFFFFFF, radius=3)
            
            # Draw trajectories as trails
            # Ground truth in RED
            for i in range(0, n_steps - 1, 2):
                x1, y1 = gt_traj[i][0], gt_traj[i][1]
                x2, y2 = gt_traj[i + 1][0], gt_traj[i + 1][1]
                gui.line((x1, y1), (x2, y2), color=0xFF3333, radius=2)
            
            # Guess in GREEN (thicker to show difference)
            for i in range(0, n_steps - 1, 2):
                x1, y1 = guess_traj[i][0], guess_traj[i][1]
                x2, y2 = guess_traj[i + 1][0], guess_traj[i + 1][1]
                gui.line((x1, y1), (x2, y2), color=0x33FF33, radius=2)
            
            # Animate the balls along their trajectories
            sim_frame = (iteration * 10 + anim_frame * 3) % n_steps
            gt_ball_pos = gt_traj[sim_frame]
            guess_ball_pos = guess_traj[sim_frame]
            
            # Draw balls with trails showing recent history
            trail_length = 30
            for j in range(trail_length):
                trail_idx = max(0, sim_frame - j)
                alpha = 1.0 - (j / trail_length)
                size = int(10 * alpha + 4)
                
                # GT trail
                gt_trail_pos = gt_traj[trail_idx]
                gui.circle((gt_trail_pos[0], gt_trail_pos[1]), 
                          color=0xFF0000, radius=size)
                
                # Guess trail
                guess_trail_pos = guess_traj[trail_idx]
                gui.circle((guess_trail_pos[0], guess_trail_pos[1]), 
                          color=0x00FF00, radius=size)
            
            # Text overlay - Title
            gui.text(content='AGGRESSIVE 2D INVERSE PHYSICS', 
                    pos=(0.22, 0.95), color=0xFFFF00, font_size=30)
            
            # Ground truth parameters
            gui.text(content=f'Ground Truth (RED):', 
                    pos=(0.02, 0.88), color=0xFF3333, font_size=20)
            gui.text(content=f'  Gravity: {gravity_gt:.1f} m/s²', 
                    pos=(0.02, 0.84), color=0xFF3333, font_size=18)
            gui.text(content=f'  Bounciness: {bounciness_gt:.3f}', 
                    pos=(0.02, 0.80), color=0xFF3333, font_size=18)
            
            # Learned parameters
            gui.text(content=f'Learned (GREEN):', 
                    pos=(0.02, 0.72), color=0x33FF33, font_size=20)
            gui.text(content=f'  Gravity: {gravity_guess:.2f} m/s²', 
                    pos=(0.02, 0.68), color=0x33FF33, font_size=18)
            gui.text(content=f'  Bounciness: {bounciness_guess:.3f}', 
                    pos=(0.02, 0.64), color=0x33FF33, font_size=18)
            
            # Show the velocity/momentum (how fast it's learning)
            gui.text(content=f'Learning Rate: AGGRESSIVE', 
                    pos=(0.02, 0.56), color=0xFFAA00, font_size=20)
            gui.text(content=f'  Δ Gravity: {velocity_gravity:+.3f}', 
                    pos=(0.02, 0.52), color=0xFFAA00, font_size=16)
            gui.text(content=f'  Δ Bounciness: {velocity_bounciness:+.4f}', 
                    pos=(0.02, 0.48), color=0xFFAA00, font_size=16)
            
            # Optimization info
            gui.text(content=f'Iteration: {iteration}/{max_iterations}', 
                    pos=(0.02, 0.12), color=0xFFFFFF, font_size=26)
            gui.text(content=f'Loss: {loss:.6f}', 
                    pos=(0.02, 0.08), color=0xFFFF00, font_size=26)
            
            # Errors
            g_err = abs(gravity_guess - gravity_gt)
            b_err = abs(bounciness_guess - bounciness_gt)
            gui.text(content=f'Errors: g={g_err:.2f} ({g_err/gravity_gt*100:.1f}%), b={b_err:.3f} ({b_err/bounciness_gt*100:.1f}%)', 
                    pos=(0.02, 0.04), color=0xFF8800, font_size=22)
            
            # Save frame
            frame_num = iteration * 3 + anim_frame
            gui.show(f"frames/frame_{frame_num:04d}.png")
        
        iteration += 1
        
        # Early stopping if converged
        if loss < 1e-6:
            print(f"\n🎯 CONVERGED at iteration {iteration}! Loss < 1e-6")
            # Continue a bit more to show the final result
            if iteration > max_iterations - 10:
                break
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Ground truth:  Gravity = {gravity_gt:.1f} m/s², Bounciness = {bounciness_gt:.3f}")
    print(f"Final guess:   Gravity = {gravity_guess:.2f} m/s², Bounciness = {bounciness_guess:.3f}")
    print(f"Errors:        Gravity = {abs(gravity_guess - gravity_gt):.2f} m/s² ({abs(gravity_guess - gravity_gt)/gravity_gt*100:.1f}%)")
    print(f"               Bounciness = {abs(bounciness_guess - bounciness_gt):.3f} ({abs(bounciness_guess - bounciness_gt)/bounciness_gt*100:.1f}%)")
    print(f"Converged in:  {iteration} iterations")
    print("=" * 70)
    
    # Create video using OpenCV
    print("\nCreating video from frames...")
    import cv2
    import glob
    
    frame_files = sorted(glob.glob("frames/frame_*.png"))
    if len(frame_files) > 0:
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('inverse_physics_2d_aggressive.mp4', fourcc, 30, (width, height))
        
        for frame_file in frame_files:
            img = cv2.imread(frame_file)
            video.write(img)
        
        video.release()
        print("✅ Video saved as 'inverse_physics_2d_aggressive.mp4'")
    
    # Create summary plots
    try:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: 2D Trajectories (larger, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        gt_traj = pos_gt.to_numpy()
        guess_traj = pos_guess.to_numpy()
        
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'r-', 
                label=f'Ground Truth (g={gravity_gt}, b={bounciness_gt})', linewidth=4)
        ax1.plot(guess_traj[:, 0], guess_traj[:, 1], 'g--', 
                label=f'Learned (g={gravity_guess:.2f}, b={bounciness_guess:.3f})', linewidth=4)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=3, label='Ground')
        ax1.set_xlabel('X Position', fontsize=16)
        ax1.set_ylabel('Y Position', fontsize=16)
        ax1.set_title('2D Ball Trajectories - AGGRESSIVE Optimization', fontsize=18, fontweight='bold')
        ax1.legend(fontsize=13, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Loss convergence
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(all_losses, 'b-', linewidth=3)
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Loss (MSE)', fontsize=14)
        ax2.set_title('Loss Convergence', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Gravity convergence
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(all_gravity_guesses, 'purple', linewidth=4, label='Estimated')
        ax3.axhline(y=gravity_gt, color='r', linestyle='--', linewidth=3, label='Ground Truth')
        ax3.set_xlabel('Iteration', fontsize=14)
        ax3.set_ylabel('Gravity (m/s²)', fontsize=14)
        ax3.set_title('Gravity Convergence', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(range(len(all_gravity_guesses)), all_gravity_guesses, gravity_gt, 
                         alpha=0.3, color='purple')
        
        # Plot 4: Bounciness convergence
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(all_bounciness_guesses, 'orange', linewidth=4, label='Estimated')
        ax4.axhline(y=bounciness_gt, color='r', linestyle='--', linewidth=3, label='Ground Truth')
        ax4.set_xlabel('Iteration', fontsize=14)
        ax4.set_ylabel('Bounciness', fontsize=14)
        ax4.set_title('Bounciness Convergence', fontsize=16, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(range(len(all_bounciness_guesses)), all_bounciness_guesses, bounciness_gt, 
                         alpha=0.3, color='orange')
        
        # Plot 5: Combined error
        ax5 = fig.add_subplot(gs[1, 2])
        gravity_errors = [abs(g - gravity_gt)/gravity_gt * 100 for g in all_gravity_guesses]
        bounciness_errors = [abs(b - bounciness_gt)/bounciness_gt * 100 for b in all_bounciness_guesses]
        ax5.plot(gravity_errors, 'purple', linewidth=3, label='Gravity Error %')
        ax5.plot(bounciness_errors, 'orange', linewidth=3, label='Bounciness Error %')
        ax5.set_xlabel('Iteration', fontsize=14)
        ax5.set_ylabel('Error (%)', fontsize=14)
        ax5.set_title('Error Reduction', fontsize=16, fontweight='bold')
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
        
        # Plot 6: Learning dynamics (velocities)
        ax6 = fig.add_subplot(gs[2, :])
        # Reconstruct the velocities from the parameter changes
        gravity_velocities = np.diff(all_gravity_guesses, prepend=gravity_guess - all_gravity_guesses[0])
        bounciness_velocities = np.diff(all_bounciness_guesses, prepend=bounciness_guess - all_bounciness_guesses[0])
        
        ax6_twin = ax6.twinx()
        line1 = ax6.plot(gravity_velocities, 'purple', linewidth=2, label='Δ Gravity', alpha=0.7)
        line2 = ax6_twin.plot(bounciness_velocities, 'orange', linewidth=2, label='Δ Bounciness', alpha=0.7)
        
        ax6.set_xlabel('Iteration', fontsize=14)
        ax6.set_ylabel('Δ Gravity (m/s²)', fontsize=14, color='purple')
        ax6_twin.set_ylabel('Δ Bounciness', fontsize=14, color='orange')
        ax6.set_title('Learning Dynamics (Parameter Updates per Iteration)', fontsize=16, fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='purple')
        ax6_twin.tick_params(axis='y', labelcolor='orange')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, fontsize=12, loc='upper right')
        
        plt.suptitle('AGGRESSIVE Inverse Physics Optimization Results', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        plt.savefig('summary_plots_2d_aggressive.png', dpi=200, bbox_inches='tight')
        print("✅ Summary plots saved as 'summary_plots_2d_aggressive.png'")
        
    except Exception as e:
        print(f"Could not create summary plots: {e}")


if __name__ == "__main__":
    main()
