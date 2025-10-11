import numpy as np
from tqdm import tqdm
import os
import cv2
import json

accumulated_new_keypoints = []  # will hold rows [t, x, y]

prev_kp_count = 0

video_name = "video_name"
cam = "cam01"

pose_dir = f"/home/rsathua2/VideoMimic/real2sim/demo_data/input_2d_poses/{video_name}/{cam}"
img_dir = f"/home/rsathua2/VideoMimic/real2sim/demo_data/input_images/{video_name}/{cam}"
out_dir = f"/home/rsathua2/VideoMimic/real2sim/demo_data/annotated_images/{video_name}/{cam}"
os.makedirs(out_dir, exist_ok=True)

# --- Annotate images + track keypoints ---
json_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".json")])

for idx, json_file in enumerate(tqdm(json_files, desc="Processing frames")):
    frame_id = json_file.replace("pose_", "").replace(".json", "")
    img_path = os.path.join(img_dir, f"{frame_id}.jpg")
    json_path = os.path.join(pose_dir, json_file)
    out_path = os.path.join(out_dir, f"{frame_id}.jpg")

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    current_keypoints = []
    for person_id, person_data in data.items():
        keypoints = person_data.get("keypoints", [])
        for (x, y, conf) in keypoints:
            if conf > 0.9:
                current_keypoints.append([idx, x, y])  # row format
                # Draw points
                cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.circle(img, (int(x), int(y)), 10, (0, 0, 0), 2)

    cv2.imwrite(out_path, img)

    # Check if number of keypoints increased
    if len(current_keypoints) > prev_kp_count:
        accumulated_new_keypoints.extend(current_keypoints)
        print(f"[INFO] Increase detected at frame {idx}: {len(current_keypoints)} keypoints")

    prev_kp_count = len(current_keypoints)

# Convert to numpy array of shape (N,3)
accumulated_array = np.array(accumulated_new_keypoints)
print(f"Final accumulated array shape: {accumulated_array.shape}")
# Save as .npy
np.save(os.path.join(out_dir, "new_keypoints.npy"), accumulated_array)
