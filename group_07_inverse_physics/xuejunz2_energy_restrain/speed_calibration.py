def plot_speed_calibration(idx):
    pkl = f"{BASE_DIR}/rollout_test_{idx}.pkl"
    gt, base, _, _ = load_rollout_pkl(pkl)
    vg = np.linalg.norm(velocities(gt), axis=-1).flatten()
    vb = np.linalg.norm(velocities(base), axis=-1).flatten()
    a = (vb @ vg) / (vb @ vb + 1e-12)
    xs = np.linspace(0, np.percentile(vb, 99), 100)
    plt.figure(figsize=(6,6))
    plt.scatter(vb, vg, s=4, alpha=0.15, label='points')
    plt.plot(xs, xs, 'k--', label='ideal y=x')
    plt.plot(xs, a*xs, 'r-', label=f'fit y={a:.3f}x')
    plt.xlabel('Pred speed |v|'); plt.ylabel('GT speed |v|')
    plt.title(f'Speed Calibration (idx={idx})'); plt.legend()
    out = f"{OUT}/calibration_{idx}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    display(Image(out))

plot_speed_calibration(1)
