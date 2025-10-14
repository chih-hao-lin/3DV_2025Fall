def plot_spatial_error_heatmap(idx, bins=64):
    pkl = f"{BASE_DIR}/rollout_test_{idx}.pkl"
    gt, base, _, meta = load_rollout_pkl(pkl)
    bounds = np.array(meta["bounds"])
    err = np.linalg.norm(base - gt, axis=-1)   # (T,N)
    pos = gt                                 

    X = pos.reshape(-1,2); E = err.reshape(-1)

    xs = np.linspace(bounds[0,0], bounds[0,1], bins+1)
    ys = np.linspace(bounds[1,0], bounds[1,1], bins+1)

    H = np.zeros((bins,bins)); C = np.zeros((bins,bins))
    ix = np.clip(np.digitize(X[:,0], xs)-1, 0, bins-1)
    iy = np.clip(np.digitize(X[:,1], ys)-1, 0, bins-1)
    for a,b,v in zip(ix,iy,E):
        H[b,a] += v; C[b,a] += 1
    M = np.divide(H, C, out=np.zeros_like(H), where=C>0)

    plt.figure(figsize=(6,6))
    plt.imshow(M, origin='lower', extent=[xs[0],xs[-1],ys[0],ys[-1]], cmap='inferno', aspect='equal')
    plt.colorbar(label='mean position error')
    plt.title(f"Spatial Error Heatmap (idx={idx})")
    out = f"{OUT}/spatial_err_{idx}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    display(Image(out))

plot_spatial_error_heatmap(1, bins=64)
