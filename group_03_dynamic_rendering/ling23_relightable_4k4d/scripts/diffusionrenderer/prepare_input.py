from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=Path)
    parser.add_argument('--output_root', type=Path)
    parser.add_argument('--max_frame', type=int, default=57)
    parser.add_argument('--image_dirname', type=str, default='images_libx265')
    args = parser.parse_args()

    target_w, target_h = 1280, 704
    for view_path in tqdm(sorted((args.data_root / args.image_dirname).iterdir())):
        part_idx = 0
        frame_idx = 0
        final_img = None
        for img_path in sorted(view_path.glob('*.png')):
            img = cv2.imread(img_path)
            ratio = min(target_h / img.shape[0], target_w / img.shape[1])
            img = cv2.resize(
                img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
            final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            final_img[final_img.shape[0]//2 - img.shape[0]//2:final_img.shape[0]//2 - img.shape[0]//2 + img.shape[0],
                      final_img.shape[1]//2 - img.shape[1]//2:final_img.shape[1]//2 - img.shape[1]//2 + img.shape[1], :] = img
            out_path = args.output_root / \
                f'{view_path.name}_{part_idx:03d}' / \
                f'frame_{frame_idx:05d}.jpg'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, final_img)
            frame_idx += 1
            if frame_idx >= args.max_frame:
                frame_idx = 0
                part_idx += 1
        if frame_idx != 0: # finish this part
            while frame_idx < args.max_frame:
                out_path = args.output_root / \
                    f'{view_path.name}_{part_idx:03d}' / \
                    f'frame_{frame_idx:05d}.jpg'
                cv2.imwrite(out_path, final_img)
                frame_idx += 1
