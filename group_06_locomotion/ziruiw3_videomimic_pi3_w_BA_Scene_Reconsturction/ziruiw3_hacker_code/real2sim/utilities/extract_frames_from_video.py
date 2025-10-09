# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import cv2
import tyro
import os
from pathlib import Path


def main(video_path: str,
         output_dir: str,
         start_frame: int,
         end_frame: int):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_count = 0
    extracted_frames = 0
    while True:
        # Read next frame
        ret, frame = video.read()
        
        if not ret:
            break
            
        # Save frame as image
        if frame_count >= start_frame and frame_count <= end_frame:
            frame_count += 1
            extracted_frames += 1
            output_path = os.path.join(output_dir, f"{extracted_frames:05d}.jpg")
            cv2.imwrite(output_path, frame)
        else:
            frame_count += 1
            # Skip frames outside the specified range
            continue
        
    # Release video capture
    video.release()
    
    print(f"Extracted {extracted_frames} frames to {output_dir}")


# Example usage
#extract_frame("/path/to/your/video.mp4", "output_frame.jpg", 50)
if __name__ == '__main__':
    tyro.cli(main)
             