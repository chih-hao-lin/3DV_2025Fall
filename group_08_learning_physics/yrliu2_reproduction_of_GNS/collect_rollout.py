import os, subprocess
from argparse import ArgumentParser

def parse_args():
    ap = ArgumentParser()
    ap.add_argument("--n_rollouts", type=int, default=100)
    ap.add_argument("--start_seed", type=int, default=0)
    ap.add_argument("--out_folder", type=str, default="data_ti_single")
    ap.add_argument("--n_particles", type=int, default=19000)
    ap.add_argument("--n_frames", type=int, default=500)
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument("--base_g", type=float, default=50.0)
    ap.add_argument("--blobs_min", type=int, default=2)
    ap.add_argument("--blobs_max", type=int, default=6)
    ap.add_argument("--show_gui", action="store_true")
    ap.add_argument("--python", type=str, default="python", help="python executable")
    ap.add_argument("--single_script", type=str, default="collect_single_rollout.py")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_folder, exist_ok=True)

    for k in range(args.n_rollouts):
        print(f"[collect] rollout {k+1}/{args.n_rollouts}")
        seed = args.start_seed + k
        out_path = os.path.join(args.out_folder, f"rollout_{k:03d}.npz")

        cmd = [
            args.python, args.single_script,
            "--n_particles", str(args.n_particles),
            "--n_frames", str(args.n_frames),
            "--quality", str(args.quality),
            "--seed", str(seed),
            "--out_path", out_path,
            "--base_g", str(args.base_g),
            "--blobs_min", str(args.blobs_min),
            "--blobs_max", str(args.blobs_max),
        ]
        if args.show_gui:
            cmd.append("--show_gui")
            
        subprocess.run(cmd, check=True)

    print(f"[collect] done. rollouts in: {args.out_folder}")

if __name__ == "__main__":
    main()
