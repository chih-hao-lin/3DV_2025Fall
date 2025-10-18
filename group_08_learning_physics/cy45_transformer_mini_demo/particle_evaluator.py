import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from particle_models import ParticleDataset, PhysicsInformedParticleTransformer

def evaluate_rollout_accuracy(model, dataset, num_rollouts=5, rollout_length=300, device='cpu'):
    """
    Evaluate model accuracy on multi-step rollouts
    """
    model.eval()
    model.to(device)
    
    position_errors = []
    velocity_errors = []
    
    print(f"Evaluating {rollout_length}-step rollouts on {num_rollouts} trajectories...")
    
    for i in range(min(num_rollouts, len(dataset.trajectories))):
        # Get ground truth trajectory
        true_trajectory = dataset.get_full_trajectory(i)
        
        if true_trajectory.shape[0] <= rollout_length:
            print(f"  Skipping trajectory {i} (too short: {true_trajectory.shape[0]} steps)")
            continue
            
        # Initial state
        initial_state = true_trajectory[0:1]  # (1, particles, 4)
        
        # Ground truth for comparison
        true_rollout = true_trajectory[:rollout_length+1]  # (rollout_length+1, particles, 4)
        
        # Model prediction
        print(f"  Generating {rollout_length}-step prediction for trajectory {i}...")
        pred_rollout = model.multi_step_rollout(initial_state, rollout_length, device)
        
        # Compute errors
        pred_positions = pred_rollout[:, :, :2]  # (steps, particles, 2)
        true_positions = true_rollout[:, :, :2]
        
        pred_velocities = pred_rollout[:, :, 2:]  # (steps, particles, 2)
        true_velocities = true_rollout[:, :, 2:]
        
        # MSE at each timestep
        pos_mse = torch.mean((pred_positions - true_positions) ** 2, dim=(1, 2))  # (steps,)
        vel_mse = torch.mean((pred_velocities - true_velocities) ** 2, dim=(1, 2))  # (steps,)
        
        position_errors.append(pos_mse)
        velocity_errors.append(vel_mse)
    
    if not position_errors:
        raise ValueError("No valid trajectories found for evaluation")
    
    # Average across rollouts
    avg_pos_error = torch.stack(position_errors).mean(dim=0)  # (steps,)
    avg_vel_error = torch.stack(velocity_errors).mean(dim=0)  # (steps,)
    
    return {
        'position_mse': avg_pos_error,
        'velocity_mse': avg_vel_error,
        'final_position_mse': avg_pos_error[-1].item(),
        'final_velocity_mse': avg_vel_error[-1].item()
    }

def plot_rollout_errors(metrics, rollout_length, save_path='rollout_errors.png'):
    """Plot rollout error evolution over time"""
    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    timesteps = range(len(metrics['position_mse']))
    
    # Position error
    ax1.plot(timesteps, metrics['position_mse'], 'b-', linewidth=2, label='Position MSE')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Position MSE')
    ax1.set_title(f'Position Error Over Time ({rollout_length} steps)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity error
    ax2.plot(timesteps, metrics['velocity_mse'], 'r-', linewidth=2, label='Velocity MSE')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Velocity MSE')
    ax2.set_title(f'Velocity Error Over Time ({rollout_length} steps)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error plot saved to {save_path}")
    plt.close()

def compare_trajectories_static(true_trajectory, pred_trajectory, timesteps_to_show=[0, 50, 150, 300],
                               bounds=(0.1, 0.9), save_path='trajectory_comparison.png'):
    """
    Create static comparison of trajectories at different timesteps
    """
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 8))
    
    true_pos = true_trajectory[:, :, :2].numpy()
    pred_pos = pred_trajectory[:, :, :2].numpy()
    
    min_bound, max_bound = bounds
    
    for i, t in enumerate(timesteps_to_show):
        if t >= len(true_pos) or t >= len(pred_pos):
            continue
            
        # Ground truth
        ax_true = axes[0, i]
        ax_true.scatter(true_pos[t, :, 0], true_pos[t, :, 1], s=30, alpha=0.7, c='blue')
        ax_true.add_patch(Rectangle((min_bound, min_bound), 
                                  max_bound - min_bound, max_bound - min_bound,
                                  fill=False, edgecolor='black', linewidth=2))
        ax_true.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax_true.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax_true.set_aspect('equal')
        ax_true.set_title(f'Ground Truth t={t}')
        
        # Prediction
        ax_pred = axes[1, i]
        ax_pred.scatter(pred_pos[t, :, 0], pred_pos[t, :, 1], s=30, alpha=0.7, c='red')
        ax_pred.add_patch(Rectangle((min_bound, min_bound), 
                                  max_bound - min_bound, max_bound - min_bound,
                                  fill=False, edgecolor='black', linewidth=2))
        ax_pred.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax_pred.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax_pred.set_aspect('equal')
        ax_pred.set_title(f'Prediction t={t}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()

def visualize_particle_trajectory(true_trajectory, pred_trajectory, bounds=(0.1, 0.9), 
                                 save_path='particle_animation.gif', show_every=5):
    """
    Create an animated visualization of particle trajectories
    """
    plt.switch_backend('Agg')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract positions
    true_pos = true_trajectory[:, :, :2].numpy()
    pred_pos = pred_trajectory[:, :, :2].numpy()
    
    min_bound, max_bound = bounds
    
    def setup_axis(ax, title):
        ax.set_xlim(min_bound - 0.05, max_bound + 0.05)
        ax.set_ylim(min_bound - 0.05, max_bound + 0.05)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.add_patch(Rectangle((min_bound, min_bound), 
                              max_bound - min_bound, max_bound - min_bound,
                              fill=False, edgecolor='black', linewidth=2))
        return ax.scatter([], [], s=20, alpha=0.7)
    
    scat1 = setup_axis(ax1, 'Ground Truth')
    scat2 = setup_axis(ax2, 'Model Prediction')
    
    def animate(frame_idx):
        frame = frames[frame_idx] if frame_idx < len(frames) else frames[-1]
        if frame >= len(true_pos):
            frame = len(true_pos) - 1
            
        # Update scatter plots
        scat1.set_offsets(true_pos[frame])
        scat2.set_offsets(pred_pos[frame])
        
        # Color particles by velocity magnitude for visual appeal
        if frame < len(true_trajectory):
            true_vel = true_trajectory[frame, :, 2:].numpy()
            pred_vel = pred_trajectory[frame, :, 2:].numpy()
            
            true_speed = np.linalg.norm(true_vel, axis=1)
            pred_speed = np.linalg.norm(pred_vel, axis=1)
            
            scat1.set_array(true_speed)
            scat2.set_array(pred_speed)
        
        return scat1, scat2
    
    frames = list(range(0, min(len(true_pos), len(pred_pos)), show_every))
    
    try:
        print(f"Creating animation with {len(frames)} frames...")
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=200, blit=False, repeat=True)
        
        plt.tight_layout()
        
        # Try different writers
        try:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save with pillow: {e}")
            try:
                anim.save(save_path, writer='imagemagick', fps=5)
                print(f"Animation saved to {save_path} (using imagemagick)")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")
                print("Skipping animation creation")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
    
    plt.close(fig)

def evaluate_multiple_rollout_lengths(model, dataset, device='cpu'):
    """
    Evaluate model at different rollout lengths to see degradation over time
    """
    rollout_lengths = [50, 100, 200, 300]
    results = {}
    
    print("\n=== Multi-Length Rollout Evaluation ===")
    
    for length in rollout_lengths:
        print(f"\nEvaluating {length}-step rollouts...")
        metrics = evaluate_rollout_accuracy(model, dataset, num_rollouts=3, 
                                          rollout_length=length, device=device)
        results[length] = metrics
        print(f"  Final Position MSE: {metrics['final_position_mse']:.6f}")
        print(f"  Final Velocity MSE: {metrics['final_velocity_mse']:.6f}")
    
    # Plot comparison
    plt.switch_backend('Agg')
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for length in rollout_lengths:
        timesteps = range(len(results[length]['position_mse']))
        plt.plot(timesteps, results[length]['position_mse'], 
                label=f'{length} steps', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Position MSE')
    plt.title('Position Error vs Rollout Length')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for length in rollout_lengths:
        timesteps = range(len(results[length]['velocity_mse']))
        plt.plot(timesteps, results[length]['velocity_mse'], 
                label=f'{length} steps', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Velocity MSE')
    plt.title('Velocity Error vs Rollout Length')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_length_rollout_comparison.png', dpi=300, bbox_inches='tight')
    print("Multi-length comparison saved to 'multi_length_rollout_comparison.png'")
    plt.close()
    
    return results

def load_trained_model(model_path='particle_transformer.pth', info_path='training_info.pth'):
    """Load trained model and training information"""
    
    # Load training info if available
    training_info = None
    try:
        training_info = torch.load(info_path, map_location='cpu')
        print("Loaded training information")
    except FileNotFoundError:
        print("Training info not found, using default model config")
    
    # Initialize model with config from training info or defaults
    if training_info and 'model_config' in training_info:
        config = training_info['model_config']
        model = PhysicsInformedParticleTransformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            bounds=config['bounds'],
            dt=config['dt']
        )
    else:
        # Default config
        model = PhysicsInformedParticleTransformer(
            d_model=128,
            n_heads=8,
            n_layers=3,
            dropout=0.1,
            bounds=(0.1, 0.9),
            dt=0.01
        )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Loaded model from {model_path}")
    
    return model, training_info

# Main evaluation script
if __name__ == "__main__":
    plt.switch_backend('Agg')
    
    print("=== Particle Physics Transformer Evaluation ===")
    
    # Load trained model
    try:
        model, training_info = load_trained_model()
    except FileNotFoundError:
        print("Error: Could not find trained model. Please run particle_trainer.py first.")
        exit(1)
    
    # Print training info if available
    if training_info:
        print(f"Final training loss: {training_info['train_losses'][-1]:.6f}")
        print(f"Final validation loss: {training_info['val_losses'][-1]:.6f}")
        print(f"Learned gravity: {training_info['final_gravity']}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ParticleDataset('sample_data/water_drop/combined_dataset.npz', sequence_length=15)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on device: {device}")
    
    # === MAIN EVALUATION ===
    print("\n=== Main Evaluation (300-step rollout) ===")
    metrics_300 = evaluate_rollout_accuracy(model, dataset, num_rollouts=3, rollout_length=300, device=device)
    
    print(f"Final Position MSE (300 steps): {metrics_300['final_position_mse']:.6f}")
    print(f"Final Velocity MSE (300 steps): {metrics_300['final_velocity_mse']:.6f}")
    
    # Plot 300-step rollout errors
    plot_rollout_errors(metrics_300, 300, save_path='rollout_errors_300.png')
    
    # === MULTI-LENGTH EVALUATION ===
    multi_results = evaluate_multiple_rollout_lengths(model, dataset, device=device)
    
    # === VISUALIZATION ===
    print("\n=== Creating Visualizations ===")
    
    # Get a sample trajectory for visualization
    sample_traj = dataset.get_full_trajectory(0)
    initial_state = sample_traj[0:1]
    
    # Generate 300-step prediction
    print("Generating 300-step prediction for visualization...")
    pred_traj = model.multi_step_rollout(initial_state, 300, device=device)
    true_traj = sample_traj[:301]  # Match length
    
    # Create static comparison
    compare_trajectories_static(true_traj, pred_traj, 
                              timesteps_to_show=[0, 50, 150, 300],
                              save_path='trajectory_comparison_300.png')
    
    # Create animated visualization
    try:
        visualize_particle_trajectory(true_traj, pred_traj, 
                                    save_path='particle_animation_300.gif',
                                    show_every=5)
    except Exception as e:
        print(f"Animation creation failed: {e}")
        print("Static comparison created instead")
    
    # === SUMMARY ===
    print("\n=== Evaluation Summary ===")
    print("Rollout Performance Summary:")
    for length in [50, 100, 200, 300]:
        if length in multi_results:
            pos_mse = multi_results[length]['final_position_mse']
            vel_mse = multi_results[length]['final_velocity_mse']
            print(f"  {length:3d} steps - Pos MSE: {pos_mse:.6f}, Vel MSE: {vel_mse:.6f}")
    
    print("\nGenerated files:")
    print("  - rollout_errors_300.png")
    print("  - trajectory_comparison_300.png") 
    print("  - particle_animation_300.gif (if successful)")
    print("  - multi_length_rollout_comparison.png")
    
    print("\n=== Evaluation Complete ===")