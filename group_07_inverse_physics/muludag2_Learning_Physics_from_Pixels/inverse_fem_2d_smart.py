import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu)

# Simulation parameters
n_steps = 400
dt = 0.01

# Ground truth parameters
gravity_gt = 9.8
bounciness_gt = 0.5

# Initial guesses
gravity_guess = 5.0
bounciness_guess = 0.2

# State
pos_gt = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)
vel_gt = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)
pos_guess = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)
vel_guess = ti.Vector.field(2, dtype=ti.f32, shape=n_steps)


@ti.kernel
def simulate_2d(pos: ti.template(), vel: ti.template(), gravity: ti.f32, bounciness: ti.f32):
    pos[0] = ti.Vector([0.2, 0.5])
    vel[0] = ti.Vector([3.0, 4.0])
    
    for i in range(n_steps - 1):
        vel[i + 1] = vel[i]
        vel[i + 1][1] -= gravity * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt
        
        if pos[i + 1][1] < 0.0:
            pos[i + 1][1] = 0.0
            vel[i + 1][1] = -vel[i + 1][1] * bounciness
            vel[i + 1][0] *= 0.95
        
        if pos[i + 1][0] < 0.0:
            pos[i + 1][0] = 0.0
            vel[i + 1][0] = -vel[i + 1][0] * bounciness
        if pos[i + 1][0] > 1.0:
            pos[i + 1][0] = 1.0
            vel[i + 1][0] = -vel[i + 1][0] * bounciness


def find_bounce_heights(pos):
    """Extract bounce peak heights"""
    pos_np = pos.to_numpy()
    heights = []
    
    for i in range(1, len(pos_np) - 1):
        if pos_np[i, 1] > pos_np[i-1, 1] and pos_np[i, 1] > pos_np[i+1, 1]:
            if pos_np[i, 1] > 0.02:  # Lower threshold to catch more bounces
                heights.append(pos_np[i, 1])
    
    return np.array(heights) if len(heights) > 0 else np.array([0.0])


def compute_smart_bounce_loss(pos1, pos2):
    """Smart bounce loss that penalizes wrong number of bounces AND wrong heights"""
    h1 = find_bounce_heights(pos1)
    h2 = find_bounce_heights(pos2)
    
    # Part 1: Penalize different number of bounces (CRITICAL!)
    bounce_count_loss = (len(h1) - len(h2)) ** 2 * 10.0  # Big penalty
    
    # Part 2: Compare heights that exist
    n = min(len(h1), len(h2))
    if n > 0:
        height_loss = np.sum((h1[:n] - h2[:n]) ** 2) * 100
    else:
        height_loss = 1.0
    
    # Part 3: Bounce decay ratio (key for bounciness!)
    # Ratio between consecutive bounces reveals bounciness directly
    decay_loss = 0.0
    if len(h1) >= 2 and len(h2) >= 2:
        # Compare ratios: h[1]/h[0], h[2]/h[1], etc.
        for i in range(min(len(h1)-1, len(h2)-1)):
            ratio1 = h1[i+1] / (h1[i] + 1e-6)
            ratio2 = h2[i+1] / (h2[i] + 1e-6)
            decay_loss += (ratio1 - ratio2) ** 2 * 1000  # Very sensitive!
    
    total_loss = bounce_count_loss + height_loss + decay_loss
    return total_loss


def compute_trajectory_loss(pos1, pos2):
    """Full trajectory loss"""
    pos1_np = pos1.to_numpy()
    pos2_np = pos2.to_numpy()
    diff = pos1_np - pos2_np
    return np.mean(np.sum(diff ** 2, axis=1))


def main():
    global gravity_guess, bounciness_guess
    
    print("=" * 70)
    print("INVERSE PHYSICS - SMART BOUNCE-COUNT METHOD")
    print("=" * 70)
    print(f"\nGround truth: g={gravity_gt} m/s², b={bounciness_gt}")
    print(f"Initial:      g={gravity_guess} m/s², b={bounciness_guess}\n")
    
    # Generate ground truth
    simulate_2d(pos_gt, vel_gt, gravity_gt, bounciness_gt)
    gt_traj = pos_gt.to_numpy()
    max_height = np.max(gt_traj[:, 1])
    gt_bounces = find_bounce_heights(pos_gt)
    
    print(f"Ground truth: {len(gt_bounces)} bounces with heights: {gt_bounces[:5]}")
    print(f"Max height: {max_height:.2f} m\n")
    
    # Setup
    gui = ti.GUI("Inverse Physics - Smart Bounce", res=(800, 600), show_gui=False)
    os.makedirs("frames", exist_ok=True)
    
    # Optimization
    iteration = 0
    max_iterations = 150
    
    # History
    all_losses = []
    all_gravity = []
    all_bounciness = []
    
    # Adam state
    m_g, v_g = 0.0, 0.0
    m_b, v_b = 0.0, 0.0
    beta1, beta2 = 0.9, 0.999
    eps_adam = 1e-8
    
    gravity_locked = False
    bounciness_locked = False
    
    epsilon_g = 0.05
    epsilon_b = 0.01  # Reasonable epsilon
    
    print("=" * 70)
    print("STRATEGY:")
    print("  Phase 1 (0-50):   Learn gravity (trajectory loss)")
    print("  Phase 2 (51-120): Learn bounciness (SMART bounce loss)")
    print("  Phase 3 (121-150): Joint refinement")
    print("=" * 70 + "\n")
    
    while iteration < max_iterations:
        # === PHASE & LOSS FUNCTION ===
        if iteration < 50:
            phase = "GRAVITY"
            lr_g = 1.0
            lr_b = 0.0
            loss_fn = compute_trajectory_loss
        elif iteration < 120:
            phase = "BOUNCINESS"
            lr_g = 0.05  # Keep gravity very stable
            lr_b = 0.2  # AGGRESSIVE for bounciness
            loss_fn = compute_smart_bounce_loss  # Use smart loss!
        else:
            phase = "JOINT"
            lr_g = 0.2
            lr_b = 0.05
            loss_fn = compute_trajectory_loss
        
        # === LOCKING ===
        g_err_pct = abs(gravity_guess - gravity_gt) / gravity_gt * 100
        b_err_pct = abs(bounciness_guess - bounciness_gt) / bounciness_gt * 100
        
        if not gravity_locked and g_err_pct < 3.0 and iteration > 30:
            gravity_locked = True
            lr_g = 0.0
            print(f"\n🔒 GRAVITY LOCKED at {gravity_guess:.2f} (error {g_err_pct:.1f}%)\n")
        
        if not bounciness_locked and b_err_pct < 5.0 and iteration > 70:
            bounciness_locked = True
            lr_b = 0.0
            print(f"\n🔒 BOUNCINESS LOCKED at {bounciness_guess:.3f} (error {b_err_pct:.1f}%)\n")
        
        if gravity_locked and bounciness_locked and iteration > 130:
            print("\n🎯 CONVERGED!\n")
            break
        
        if gravity_locked:
            lr_g = 0.0
        if bounciness_locked:
            lr_b = 0.0
        
        # === OPTIMIZATION ===
        simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess)
        loss = loss_fn(pos_gt, pos_guess)
        
        all_losses.append(loss)
        all_gravity.append(gravity_guess)
        all_bounciness.append(bounciness_guess)
        
        # Gradients
        if lr_g > 0:
            simulate_2d(pos_guess, vel_guess, gravity_guess + epsilon_g, bounciness_guess)
            loss_g_plus = loss_fn(pos_gt, pos_guess)
            grad_g = (loss_g_plus - loss) / epsilon_g
        else:
            grad_g = 0.0
        
        if lr_b > 0:
            simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess + epsilon_b)
            loss_b_plus = loss_fn(pos_gt, pos_guess)
            grad_b = (loss_b_plus - loss) / epsilon_b
        else:
            grad_b = 0.0
        
        # Adam
        m_g = beta1 * m_g + (1 - beta1) * grad_g
        v_g = beta2 * v_g + (1 - beta2) * (grad_g ** 2)
        m_g_hat = m_g / (1 - beta1 ** (iteration + 1))
        v_g_hat = v_g / (1 - beta2 ** (iteration + 1))
        
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
        m_b_hat = m_b / (1 - beta1 ** (iteration + 1))
        v_b_hat = v_b / (1 - beta2 ** (iteration + 1))
        
        # Update
        gravity_guess -= lr_g * m_g_hat / (np.sqrt(v_g_hat) + eps_adam)
        bounciness_guess -= lr_b * m_b_hat / (np.sqrt(v_b_hat) + eps_adam)
        
        # Clamp
        gravity_guess = np.clip(gravity_guess, 1.0, 15.0)
        bounciness_guess = np.clip(bounciness_guess, 0.0, 0.99)
        
        # Log
        if iteration % 5 == 0:
            locks = ""
            if gravity_locked: locks += "🔒g "
            if bounciness_locked: locks += "🔒b "
            
            simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess)
            guess_bounces = find_bounce_heights(pos_guess)
            
            print(f"[{phase:10s}] {iteration:3d} | Loss:{loss:8.2f} | "
                  f"g:{gravity_guess:5.2f}({g_err_pct:4.1f}%) | "
                  f"b:{bounciness_guess:.3f}({b_err_pct:4.1f}%) | "
                  f"Bounces: GT={len(gt_bounces)} vs Guess={len(guess_bounces)} {locks}")
        
        # Visualize
        simulate_2d(pos_guess, vel_guess, gravity_guess, bounciness_guess)
        
        for anim in range(2):
            gui.clear(0x112233)
            
            gt_traj = pos_gt.to_numpy()
            guess_traj = pos_guess.to_numpy()
            scale_y = 0.7 / max_height
            
            gui.line((0, 0), (1, 0), color=0xFFFFFF, radius=3)
            
            for i in range(0, n_steps-1, 2):
                gui.line((gt_traj[i][0], gt_traj[i][1]*scale_y),
                        (gt_traj[i+1][0], gt_traj[i+1][1]*scale_y),
                        color=0xFF3333, radius=2)
                gui.line((guess_traj[i][0], guess_traj[i][1]*scale_y),
                        (guess_traj[i+1][0], guess_traj[i+1][1]*scale_y),
                        color=0x33FF33, radius=2)
            
            # Mark bounce peaks
            gt_bounces_arr = find_bounce_heights(pos_gt)
            for idx, h in enumerate(gt_bounces_arr[:5]):
                gui.circle((0.05 + idx*0.03, h * scale_y), color=0xFF0000, radius=8)
            
            guess_bounces_arr = find_bounce_heights(pos_guess)
            for idx, h in enumerate(guess_bounces_arr[:5]):
                gui.circle((0.85 + idx*0.03, h * scale_y), color=0x00FF00, radius=8)
            
            frame = (iteration * 10 + anim * 5) % n_steps
            for j in range(30):
                idx = max(0, frame - j)
                a = 1 - j/30.0
                s = int(10*a + 4)
                gui.circle((gt_traj[idx][0], gt_traj[idx][1]*scale_y), 
                          color=0xFF0000, radius=s)
                gui.circle((guess_traj[idx][0], guess_traj[idx][1]*scale_y), 
                          color=0x00FF00, radius=s)
            
            phase_color = {"GRAVITY": 0xFF00FF, "BOUNCINESS": 0x00FFFF, "JOINT": 0xFFFF00}
            gui.text(content=f'SMART BOUNCE-COUNT - {phase}', 
                    pos=(0.18, 0.95), color=phase_color.get(phase, 0xFFFFFF), font_size=26)
            
            gui.text(content=f'GT (RED): g={gravity_gt:.1f}, b={bounciness_gt:.2f}, {len(gt_bounces_arr)} bounces', 
                    pos=(0.02, 0.88), color=0xFF3333, font_size=18)
            
            g_lock = " 🔒" if gravity_locked else ""
            b_lock = " 🔒" if bounciness_locked else ""
            gui.text(content=f'Learned (GREEN): g={gravity_guess:.2f}{g_lock}, b={bounciness_guess:.2f}{b_lock}, {len(guess_bounces_arr)} bounces', 
                    pos=(0.02, 0.82), color=0x33FF33, font_size=18)
            
            gui.text(content=f'Iter {iteration}/{max_iterations} | Loss {loss:.2f}', 
                    pos=(0.02, 0.08), color=0xFFFFFF, font_size=24)
            gui.text(content=f'Errors: g={abs(gravity_guess-gravity_gt):.2f}({g_err_pct:.1f}%), '
                            f'b={abs(bounciness_guess-bounciness_gt):.3f}({b_err_pct:.1f}%)', 
                    pos=(0.02, 0.04), color=0xFF8800, font_size=20)
            
            gui.show(f"frames/frame_{iteration*2+anim:04d}.png")
        
        iteration += 1
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Ground truth: g={gravity_gt:.1f}, b={bounciness_gt:.3f}")
    print(f"Final guess:  g={gravity_guess:.2f}, b={bounciness_guess:.3f}")
    print(f"Errors:       g={abs(gravity_guess-gravity_gt):.2f} ({abs(gravity_guess-gravity_gt)/gravity_gt*100:.1f}%)")
    print(f"              b={abs(bounciness_guess-bounciness_gt):.3f} ({abs(bounciness_guess-bounciness_gt)/bounciness_gt*100:.1f}%)")
    print("=" * 70)
    
    # Video
    import cv2, glob
    frames = sorted(glob.glob("frames/*.png"))
    if frames:
        img = cv2.imread(frames[0])
        h, w = img.shape[:2]
        out = cv2.VideoWriter('inverse_physics_FINAL.mp4', 
                             cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames:
            out.write(cv2.imread(f))
        out.release()
        print("\n✅ Video: inverse_physics_FINAL.mp4")
    
    # Plots
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].plot(gt_traj[:, 0], gt_traj[:, 1], 'r-', lw=4, label='GT')
        axes[0, 0].plot(guess_traj[:, 0], guess_traj[:, 1], 'g--', lw=4, label='Learned')
        axes[0, 0].set_title('Trajectories', fontweight='bold', fontsize=16)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(all_losses, 'b-', lw=3)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Loss', fontweight='bold', fontsize=16)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(all_gravity, 'purple', lw=3)
        axes[1, 0].axhline(gravity_gt, color='r', ls='--', lw=2)
        axes[1, 0].set_title('Gravity', fontweight='bold', fontsize=16)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(all_bounciness, 'orange', lw=3)
        axes[1, 1].axhline(bounciness_gt, color='r', ls='--', lw=2)
        axes[1, 1].set_title('Bounciness', fontweight='bold', fontsize=16)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('summary_FINAL.png', dpi=200)
        print("✅ Plots: summary_FINAL.png\n")
    except: pass


if __name__ == "__main__":
    main()
