import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')
print(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================
N = 6
N_STEPS = 40
DT = 0.02
GRAVITY = -5.0

TRUE_STIFFNESS = 80.0 #100.0
INIT_STIFFNESS = 400.0 #500.0

IMG_SIZE = 32

# ============================================================================
# MASS-SPRING
# ============================================================================
class MassSpring:
    def __init__(self, n=N, size=0.5):
        self.n = n
        
        grid = torch.linspace(-size/2, size/2, n)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        self.x0 = torch.stack([x.flatten() + 0.5, y.flatten() + 0.65], dim=1)
        
        self.fixed = torch.zeros(n * n, dtype=torch.bool)
        self.fixed[:n] = True
        
        springs = []
        rest_lengths = []
        dx = size / (n - 1)
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if j < n - 1:
                    springs.append([idx, i * n + j + 1])
                    rest_lengths.append(dx)
                if i < n - 1:
                    springs.append([idx, (i + 1) * n + j])
                    rest_lengths.append(dx)
        
        self.springs = torch.tensor(springs, dtype=torch.long)
        self.rest_lengths = torch.tensor(rest_lengths)
        
    def simulate(self, k):
        x = self.x0.clone()
        v = torch.zeros_like(x)
        mass = 0.8
        
        for step in range(N_STEPS):
            p1 = x[self.springs[:, 0]]
            p2 = x[self.springs[:, 1]]
            
            diff = p2 - p1
            dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)
            direction = diff / dist.unsqueeze(1)
            
            stretch = dist - self.rest_lengths
            force = k * stretch.unsqueeze(1) * direction
            
            f = torch.zeros_like(x)
            for i, (s0, s1) in enumerate(self.springs):
                f[s0] += force[i]
                f[s1] -= force[i]
            
            f[:, 1] += GRAVITY * mass
            
            a = f / mass
            v = v + a * DT
            v = v * 0.93
            x = x + v * DT
            
            x[self.fixed] = self.x0[self.fixed]
            v[self.fixed] = 0.0
            
            ground = 0.05
            below = x[:, 1] < ground
            if below.any():
                penetration = ground - x[:, 1]
                correction = torch.clamp(penetration, min=0.0)
                x[:, 1] += correction
                v[:, 1] = torch.where(v[:, 1] < 0, v[:, 1] * -0.3, v[:, 1])
        
        return x
    
    # NEW: Simulate with step-by-step recording
    def simulate_animated(self, k, record_every=2):
        x = self.x0.clone()
        v = torch.zeros_like(x)
        mass = 0.8
        
        trajectory = []
        
        for step in range(N_STEPS):
            p1 = x[self.springs[:, 0]]
            p2 = x[self.springs[:, 1]]
            
            diff = p2 - p1
            dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)
            direction = diff / dist.unsqueeze(1)
            
            stretch = dist - self.rest_lengths
            force = k * stretch.unsqueeze(1) * direction
            
            f = torch.zeros_like(x)
            for i, (s0, s1) in enumerate(self.springs):
                f[s0] += force[i]
                f[s1] -= force[i]
            
            f[:, 1] += GRAVITY * mass
            
            a = f / mass
            v = v + a * DT
            v = v * 0.93
            x = x + v * DT
            
            x[self.fixed] = self.x0[self.fixed]
            v[self.fixed] = 0.0
            
            ground = 0.05
            below = x[:, 1] < ground
            if below.any():
                penetration = ground - x[:, 1]
                correction = torch.clamp(penetration, min=0.0)
                x[:, 1] += correction
                v[:, 1] = torch.where(v[:, 1] < 0, v[:, 1] * -0.3, v[:, 1])
            
            # Record positions
            if step % record_every == 0:
                trajectory.append(x.clone())
        
        return trajectory

# ============================================================================
# RENDERING
# ============================================================================
def render(vertices):
    coords = torch.linspace(0, 1, IMG_SIZE)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    grid = grid.unsqueeze(2)
    verts = vertices.unsqueeze(0).unsqueeze(0)
    
    dist_sq = ((grid - verts) ** 2).sum(dim=-1)
    
    sigma = 0.025
    weights = torch.exp(-dist_sq / (2 * sigma ** 2))
    img = weights.sum(dim=-1)
    img = torch.clamp(img, 0, 1)
    
    return img

# ============================================================================
# OPTIMIZATION
# ============================================================================
def optimize(system, target):
    log_k = torch.tensor([np.log(INIT_STIFFNESS)], requires_grad=True)
    
    optimizer = torch.optim.Adam([log_k], lr=0.1)
    
    losses = []
    k_history = []
    images = []
    vertices_history = []
    
    print(f"\n{'='*60}")
    print(f"True stiffness: {TRUE_STIFFNESS:.1f}")
    print(f"Initial guess: {INIT_STIFFNESS:.1f}")
    print(f"{'='*60}\n")
    
    for it in range(200):
        optimizer.zero_grad()
        
        k = torch.exp(log_k)
        
        verts = system.simulate(k)
        pred = render(verts)
        
        loss = F.mse_loss(pred, target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_([log_k], max_norm=0.8)
        
        optimizer.step()
        
        with torch.no_grad():
            log_k.clamp_(np.log(20.0), np.log(800.0))
        
        losses.append(loss.item())
        k_history.append(k.item())
        
        if it % 5 == 0:
            images.append(pred.detach().numpy())
            vertices_history.append(verts.detach().numpy())
        
        if it % 30 == 0:
            grad = log_k.grad.item() if log_k.grad is not None else 0.0
            error = abs(k.item() - TRUE_STIFFNESS) / TRUE_STIFFNESS * 100
            print(f"Iter {it:3d} | Loss: {loss.item():.6f} | k: {k.item():6.1f} | "
                  f"Grad: {grad:8.4f} | Error: {error:5.1f}%")
    
    final_k = torch.exp(log_k)
    
    print(f"\n{'='*60}")
    print(f"Final stiffness: {final_k.item():.1f}")
    print(f"True stiffness: {TRUE_STIFFNESS:.1f}")
    print(f"Error: {abs(final_k.item()-TRUE_STIFFNESS)/TRUE_STIFFNESS*100:.1f}%")
    print(f"{'='*60}\n")
    
    return losses, k_history, images, vertices_history, final_k

# ============================================================================
# NEW: FINAL DEMONSTRATION
# ============================================================================
def create_demo_frames(system, final_k):
    print("\n" + "="*60)
    print("Creating final demonstration...")
    print("="*60)
    
    demo_images = []
    demo_vertices_gt = []
    demo_vertices_learned = []
    
    with torch.no_grad():
        # Get animated trajectories
        print("  Simulating GT trajectory...")
        traj_gt = system.simulate_animated(torch.tensor([TRUE_STIFFNESS]), record_every=1)
        
        print("  Simulating learned trajectory...")
        traj_learned = system.simulate_animated(final_k, record_every=1)
        
        # Create demo frames - loop through simulation 3 times
        n_loops = 3
        total_frames = len(traj_gt) * n_loops
        
        print(f"  Rendering {total_frames} demonstration frames...")
        
        for loop in range(n_loops):
            for frame_idx in range(len(traj_gt)):
                # Get current state
                verts_gt = traj_gt[frame_idx]
                verts_learned = traj_learned[frame_idx]
                
                # Render silhouettes
                img_gt = render(verts_gt).numpy()
                img_learned = render(verts_learned).numpy()
                
                demo_images.append((img_gt, img_learned))
                demo_vertices_gt.append(verts_gt.numpy())
                demo_vertices_learned.append(verts_learned.numpy())
    
    print(f"  ✓ Created {len(demo_images)} demonstration frames\n")
    
    return demo_images, demo_vertices_gt, demo_vertices_learned

# ============================================================================
# SAVE WITH DEMO
# ============================================================================
def save(system, losses, k_history, images, vertices_history, final_k, target, demo_data):
    demo_images, demo_verts_gt, demo_verts_learned = demo_data
    
    with torch.no_grad():
        verts_true = system.simulate(torch.tensor([TRUE_STIFFNESS]))
        img_true = render(verts_true).numpy()
        
        verts_pred = system.simulate(final_k)
        img_pred = render(verts_pred).numpy()
        
        verts_init = system.simulate(torch.tensor([INIT_STIFFNESS]))
        img_init = render(verts_init).numpy()
    
    target_np = target.numpy()
    
    # Static result
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(target_np, cmap='plasma', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Target\n(k={TRUE_STIFFNESS:.0f}, SOFT)', fontsize=18, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_init, cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Initial\n(k={INIT_STIFFNESS:.0f}, STIFF)', fontsize=18, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_pred, cmap='plasma', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Final\n(k={k_history[-1]:.0f})', fontsize=18, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    axes[1, 0].scatter(verts_true[:, 0], verts_true[:, 1], c='red', s=100, alpha=0.7, 
                      label=f'Target (k={TRUE_STIFFNESS:.0f})', edgecolors='darkred', linewidth=2)
    axes[1, 0].scatter(verts_pred[:, 0], verts_pred[:, 1], c='green', s=100, alpha=0.6, 
                      label=f'Final (k={k_history[-1]:.0f})', marker='s', edgecolors='darkgreen', linewidth=2)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_title('Mesh Comparison', fontsize=18, fontweight='bold')
    axes[1, 0].legend(fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(losses, 'b-', linewidth=3)
    axes[1, 1].set_xlabel('Iteration', fontsize=15)
    axes[1, 1].set_ylabel('Loss (MSE)', fontsize=15)
    axes[1, 1].set_title('Loss Curve', fontsize=18, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    axes[1, 2].plot(k_history, 'g-', linewidth=3, label='Predicted')
    axes[1, 2].axhline(TRUE_STIFFNESS, color='r', linestyle='--', linewidth=3, label='True')
    axes[1, 2].fill_between(range(len(k_history)), TRUE_STIFFNESS*0.9, TRUE_STIFFNESS*1.1, 
                           alpha=0.2, color='red', label='±10% target')
    axes[1, 2].set_xlabel('Iteration', fontsize=15)
    axes[1, 2].set_ylabel('Stiffness', fontsize=15)
    axes[1, 2].set_title('Parameter Convergence', fontsize=18, fontweight='bold')
    axes[1, 2].legend(fontsize=13)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result.png', dpi=200, bbox_inches='tight')
    print("✓ Saved result.png")
    plt.close()
    
    # Create video with optimization + demo
    print("Creating video with demonstration...")
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    total_frames = len(images) + len(demo_images)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # OPTIMIZATION PHASE
        if frame < len(images):
            idx = frame
            iter_num = idx * 5
            
            ax1.imshow(target_np, cmap='plasma', vmin=0, vmax=1)
            ax1.set_title(f'TARGET (k={TRUE_STIFFNESS:.0f})', fontsize=18, fontweight='bold', color='red')
            ax1.axis('off')
            
            ax2.imshow(images[idx], cmap='plasma', vmin=0, vmax=1)
            k_val = k_history[iter_num]
            error = abs(k_val - TRUE_STIFFNESS) / TRUE_STIFFNESS * 100
            ax2.set_title(f'LEARNING...\nIter {iter_num}: k={k_val:.1f}\nError={error:.1f}%', 
                         fontsize=16, fontweight='bold', color='green')
            ax2.axis('off')
            
            verts = vertices_history[idx]
            ax3.scatter(verts[:, 0], verts[:, 1], c='cyan', s=90, alpha=0.8, 
                       edgecolors='blue', linewidth=2)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_aspect('equal')
            ax3.set_title('OPTIMIZATION', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_facecolor('#f0f0f0')
        
        # DEMONSTRATION PHASE
        else:
            demo_idx = frame - len(images)
            img_gt, img_learned = demo_images[demo_idx]
            verts_gt = demo_verts_gt[demo_idx]
            verts_learned = demo_verts_learned[demo_idx]
            
            # Side-by-side silhouettes
            ax1.imshow(img_gt, cmap='plasma', vmin=0, vmax=1)
            ax1.set_title(f'GROUND TRUTH\n(k={TRUE_STIFFNESS:.0f})', 
                         fontsize=18, fontweight='bold', color='red')
            ax1.axis('off')
            
            ax2.imshow(img_learned, cmap='plasma', vmin=0, vmax=1)
            error_final = abs(final_k.item() - TRUE_STIFFNESS) / TRUE_STIFFNESS * 100
            ax2.set_title(f'LEARNED\n(k={final_k.item():.1f})\nError: {error_final:.1f}%', 
                         fontsize=18, fontweight='bold', color='green')
            ax2.axis('off')
            
            # Overlaid meshes
            ax3.scatter(verts_gt[:, 0], verts_gt[:, 1], c='red', s=120, alpha=0.6, 
                       label='GT', edgecolors='darkred', linewidth=2)
            ax3.scatter(verts_learned[:, 0], verts_learned[:, 1], c='lime', s=90, alpha=0.7, 
                       label='Learned', marker='s', edgecolors='darkgreen', linewidth=2)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_aspect('equal')
            ax3.set_title('DEMONSTRATION\nWatch them deform identically!', 
                         fontsize=16, fontweight='bold', color='gold')
            ax3.legend(fontsize=12, loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.set_facecolor('#2a2a2a')
        
        plt.tight_layout()
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=80)
    
    try:
        writer = FFMpegWriter(fps=15, bitrate=3000)
        anim.save('optimization.mp4', writer=writer)
        print("✓ Saved optimization.mp4")
    except:
        try:
            writer = PillowWriter(fps=15)
            anim.save('optimization.gif', writer=writer)
            print("✓ Saved optimization.gif")
        except Exception as e:
            print(f"✗ Animation failed: {e}")
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    system = MassSpring()
    
    print("Generating target...")
    with torch.no_grad():
        verts_target = system.simulate(torch.tensor([TRUE_STIFFNESS]))
        target = render(verts_target)
    
    print("Optimizing...")
    losses, k_history, images, vertices_history, final_k = optimize(system, target)
    
    # NEW: Create demonstration
    demo_data = create_demo_frames(system, final_k)
    
    print("Saving results...")
    save(system, losses, k_history, images, vertices_history, final_k, target, demo_data)
    
    print("\n" + "="*60)
    print("✓ COMPLETE! Inverse FEM with silhouette loss demo")
    print("  - Optimization: 200 iterations")
    print(f"  - Demonstration: {len(demo_data[0])} frames showing learned parameter")
    print("  - Formula: min_theta loss(diff_render(diff_physics(theta)) - observed)")
    print("="*60)
