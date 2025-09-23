#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import cv2


def find_frames(dir_path: Path, regex=r"^camera(\d{4})\.png$"):
    pat = re.compile(regex)
    frames = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            idx = int(m.group(1))
            frames.append((idx, p))
    frames.sort(key=lambda x: x[0])
    return frames


def main():
    ap = argparse.ArgumentParser(
        description="Convert frames named cameraXXXX.png into a video."
    )
    ap.add_argument("input_dir", type=Path, help="Directory containing frames")
    ap.add_argument(
        "-o", "--output", type=Path, default=Path("output.mp4"),
        help="Output video path (default: output.mp4)"
    )
    ap.add_argument(
        "--fps", type=float, default=60.0, help="Frames per second (default: 60.0)"
    )
    ap.add_argument(
        "--codec", type=str, default="mp4v",
        help="FourCC codec (default: mp4v). Examples: mp4v, avc1, XVID"
    )
    ap.add_argument(
        "--stride", type=int, default=1,
        help="Use every Nth frame (default: 1 = every frame)"
    )
    args = ap.parse_args()

    frames = find_frames(args.input_dir / "RENDER")
    if not frames:
        frames = find_frames(args.input_dir / "RENDER", regex=r"^(\d{6})\.png$")
    
    if not frames:
        raise SystemExit(f"No frames matching cameraXXXX.png found in {args.input_dir  / 'RENDER'}")

    # Apply stride
    frames = frames[:: max(1, args.stride)]

    # Read first frame to determine size
    first_img = cv2.imread(str(frames[0][1]), cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise SystemExit(f"Failed to read first frame: {frames[0][1]}")
    # Ensure 3-channel BGR for VideoWriter
    if len(first_img.shape) == 2:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_GRAY2BGR)
    elif first_img.shape[2] == 4:
        # Drop alpha
        first_img = cv2.cvtColor(first_img, cv2.COLOR_BGRA2BGR)

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(str(args.input_dir / args.output), fourcc, args.fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("Failed to open VideoWriter. Try a different --codec or output extension.")

    written = 0

    # Write the first already-read frame
    vw.write(first_img)
    written += 1

    # Write the rest
    for _, path in frames[1:]:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {path}")
            continue

        # Convert to 3-channel BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if img.shape[1] != w or img.shape[0] != h:
            print(f"[WARN] Resizing frame {path.name} from {img.shape[1]}x{img.shape[0]} to {w}x{h}")
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        vw.write(img)
        written += 1

    vw.release()
    print(f"Done. Wrote {written} frames to {args.input_dir / args.output}")


if __name__ == "__main__":
    main()
