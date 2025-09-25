from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=Path)
    args = parser.parse_args()
    (args.data_root / 'videos').mkdir(parents=True, exist_ok=True)

    image_types = ['basecolor', 'normal', 'metallic',
                   'roughness', 'global_normal', 'images_libx265']
    for view_path in tqdm(sorted((args.data_root / image_types[-1]).iterdir())):
        for image_type in image_types:
            if image_type == 'global_normal':
                pattern = args.data_root / 'normal' / view_path.name / "%06d.png.global.png"
            else:
                pattern = args.data_root / image_type / view_path.name / "%06d.png"
            cmd = f'''
            ffmpeg -hwaccel cuda -hide_banner -loglevel error -framerate 60 -f image2 -r 60 -nostdin -y -i "{pattern}" -c:v hevc_nvenc -preset p7 -cq:v 19 -rc:v vbr -tag:v hvc1 -crf 17 -pix_fmt yuv420p -rc-lookahead 20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {args.data_root / 'videos' / f'{view_path.name}_{image_type}.mp4'}
            '''
            # print(cmd)
            os.system(cmd)
