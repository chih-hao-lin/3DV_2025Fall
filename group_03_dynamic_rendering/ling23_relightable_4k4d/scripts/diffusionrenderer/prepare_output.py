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

    image_types = ['basecolor', 'metallic', 'normal', 'roughness']
    target_w, target_h = 1280, 704
    for view_path in tqdm(sorted((args.data_root / args.image_dirname).iterdir())):
        part_idx = 0
        frame_idx = 0
        for img_path in sorted(view_path.glob('*.png')):
            img = cv2.imread(img_path)
            orig_img_shape = img.shape
            ratio = min(target_h / img.shape[0], target_w / img.shape[1])
            img = cv2.resize(
                img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
            for image_type in image_types:
                intrinsic_img_path = args.output_root / 'gbuffer_frames' / f'{view_path.name}_{part_idx:03d}' / f'0000.{frame_idx:04d}.{image_type}.jpg'
                intrinsic_img = cv2.imread(intrinsic_img_path)
                intrinsic_img = intrinsic_img[intrinsic_img.shape[0]//2 - img.shape[0]//2:intrinsic_img.shape[0]//2 - img.shape[0]//2 + img.shape[0],
                      intrinsic_img.shape[1]//2 - img.shape[1]//2:intrinsic_img.shape[1]//2 - img.shape[1]//2 + img.shape[1], :]
                intrinsic_img = cv2.resize(intrinsic_img, (orig_img_shape[1], orig_img_shape[0]))
                out_path = args.data_root / image_type / view_path.name / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(out_path, intrinsic_img)

            frame_idx += 1
            if frame_idx >= args.max_frame:
                frame_idx = 0
                part_idx += 1