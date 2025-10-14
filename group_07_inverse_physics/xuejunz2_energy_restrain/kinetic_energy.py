IDX = 1
pkl_path = f"{BASE_DIR}/rollout_test_{IDX}.pkl"
gt, base, ptype, meta = load_rollout_pkl(pkl_path)


corr, scales, KE, KE_ref = global_energy_correction(
    base, ptype=ptype, K=10, beta=0.35, gmax_up=0.01, gmax_down=0.02, ema=0.03
)


npz_path = f"{CORR_DIR}/rollout_test_{IDX}_global.npz"
np.savez_compressed(npz_path, predicted_rollout=corr, ground_truth_rollout=gt, particle_types=(ptype if ptype is not None else np.array([])))
print("saved:", npz_path)


def plot_energy_velocity(gt, base, corr, out_png):
    v = lambda x: x[1:] - x[:-1]
    ke = lambda v: 0.5 * np.sum(v**2, axis=-1).mean(axis=1)
    meanv = lambda v: np.linalg.norm(v, axis=-1).mean(axis=1)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    if gt is not None:
        ax[0].plot(ke(v(gt)), 'g-', label='GT')
    ax[0].plot(ke(v(base)), 'b--', label='Base')
    ax[0].plot(ke(v(corr)), 'r-.', label='Corr')
    ax[0].set_title('Kinetic Energy'); ax[0].legend()

    if gt is not None:
        ax[1].plot(meanv(v(gt)), 'g-', label='GT')
    ax[1].plot(meanv(v(base)), 'b--', label='Base')
    ax[1].plot(meanv(v(corr)), 'r-.', label='Corr')
    ax[1].set_title('Mean |v|'); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)

png1 = f"{OUT_DIR}/energy_velocity_compare_{IDX}.png"
plot_energy_velocity(gt, base, corr, png1)
display(Image(png1))

def plot_error_heatmap(gt, pred, out_png, title='Error Heatmap'):
    err = np.linalg.norm(pred - gt, axis=-1)  # (T,N)
    plt.figure(figsize=(8,4))
    plt.imshow(err.T, cmap='magma', aspect='auto', origin='lower')
    plt.colorbar(label='|Pred - GT|'); plt.xlabel('timestep'); plt.ylabel('particle')
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

if gt is not None:
    png2a = f"{OUT_DIR}/errmap_base_{IDX}.png"
    png2b = f"{OUT_DIR}/errmap_corr_{IDX}.png"
    plot_error_heatmap(gt, base, png2a, "Error Heatmap (Base)")
    plot_error_heatmap(gt, corr, png2b, "Error Heatmap (Energy-corr)")
    display(Image(png2a)); display(Image(png2b))


def plot_energy_distribution(gt, base, corr, out_png):
    def ke_flat(x): 
        v = x[1:] - x[:-1]
        return (0.5 * np.sum(v**2, axis=-1)).flatten()
    plt.figure(figsize=(6,4))
    if gt is not None: plt.hist(ke_flat(gt), bins=50, alpha=0.5, label='GT')
    plt.hist(ke_flat(base), bins=50, alpha=0.5, label='Base')
    plt.hist(ke_flat(corr), bins=50, alpha=0.5, label='Corr')
    plt.xlabel('Kinetic Energy'); plt.ylabel('Count'); plt.legend()
    plt.title('Energy Distribution'); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

png3 = f"{OUT_DIR}/energy_hist_{IDX}.png"
plot_energy_distribution(gt, base, corr, png3)
display(Image(png3))


def plot_stability(gt, base, corr, out_png):
    def vstd(x):
        v = x[1:] - x[:-1]
        return np.std(np.linalg.norm(v, axis=-1), axis=1)
    plt.figure(figsize=(7,4))
    if gt is not None: plt.plot(vstd(gt), label='GT')
    plt.plot(vstd(base), label='Base')
    plt.plot(vstd(corr), label='Corr')
    plt.title('Velocity Std Over Time (Stability)')
    plt.legend(); plt.xlabel('timestep'); plt.ylabel('std(|v|)')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

png4 = f"{OUT_DIR}/stability_{IDX}.png"
plot_stability(gt, base, corr, png4)
display(Image(png4))
