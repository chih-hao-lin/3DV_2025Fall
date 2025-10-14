def plot_error_vs_boundary(idx):
    pkl = f"{BASE_DIR}/rollout_test_{idx}.pkl"
    gt, base, _, meta = load_rollout_pkl(pkl)
    bounds = np.array(meta["bounds"])  # shape (dim,2)

    def dist_to_box(p):
        # p: (..., 2)
        dx = np.minimum(np.abs(p[...,0]-bounds[0,0]), np.abs(bounds[0,1]-p[...,0]))
        dy = np.minimum(np.abs(p[...,1]-bounds[1,0]), np.abs(bounds[1,1]-p[...,1]))
        return np.minimum(dx, dy)

    err = np.linalg.norm(base - gt, axis=-1)         # (T,N)
    dist = dist_to_box(gt)                            # (T,N) 用 GT 定位

    d = dist.flatten(); e = err.flatten()
    plt.figure(figsize=(7,4))
    plt.scatter(d, e, s=2, alpha=0.1, label='pts')

    bins = np.linspace(0, d.max(), 20)
    inds = np.digitize(d, bins)
    bin_m = [e[inds==i].mean() if np.any(inds==i) else np.nan for i in range(1,len(bins)+1)]
    plt.plot(bins, bin_m, 'r-o', lw=2, ms=3, label='bin mean')
    plt.xlabel('Distance to boundary'); plt.ylabel('Position error')
    plt.title(f'Error vs Proximity to Walls (idx={idx})'); plt.legend()
    out = f"{OUT}/err_vs_boundary_{idx}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    display(Image(out))

plot_error_vs_boundary(1)
