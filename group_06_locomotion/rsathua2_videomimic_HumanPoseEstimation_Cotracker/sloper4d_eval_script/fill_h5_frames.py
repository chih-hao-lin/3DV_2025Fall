import h5py
import numpy as np
import argparse
import os
from pathlib import Path

def nearest(idx, pool):
    """Find the index in pool closest to idx."""
    pool = np.asarray(pool)
    dist = np.abs(pool - idx)
    i    = np.argmin(dist)
    # Handle ties by choosing the smaller index among tied elements
    tie  = np.where(dist == dist[i])[0]
    return pool[min(tie)]

def fill_h5_frames(h5_path, target_frames, save_path=None):
    """
    Pads or truncates SMPL parameters in an HDF5 file to a target number of frames
    using nearest neighbor interpolation for padding.

    Args:
        h5_path (str): Path to the input HDF5 file.
        target_frames (int): The desired total number of frames.
        save_path (str, optional): Path to save the modified HDF5 file.
                                   If None, overwrites the input file. Defaults to None.
    """
    h5_path = Path(h5_path)
    if save_path is None:
        save_path = h5_path
    else:
        save_path = Path(save_path)

    temp_save_path = save_path.with_suffix(save_path.suffix + '.temp')

    print(f"Processing HDF5 file: {h5_path}")
    print(f"Target frames: {target_frames}")

    with h5py.File(h5_path, 'r') as f_in:
        # --- Check existing frames ---
        human_ids = list(f_in.get('our_pred_humans_smplx_params', {}).keys())
        if not human_ids:
            print("Error: 'our_pred_humans_smplx_params' group not found or empty.")
            return

        # Assume all humans have the same number of frames initially
        first_human_id = human_ids[0]
        try:
            existing_frames = f_in['our_pred_humans_smplx_params'][first_human_id]['betas'].shape[0]
            print(f"Existing frames found: {existing_frames}")
        except KeyError:
            print(f"Error: Could not determine existing frame count for human {first_human_id}.")
            return

        # --- Open output file ---
        with h5py.File(temp_save_path, 'w') as f_out:
            # --- Copy non-human/non-frame data ---
            print("Copying non-human/frame data...")
            for key in f_in.keys():
                if key not in ['our_pred_humans_smplx_params', 'person_frame_info_list']:
                    print(f"  Copying group/dataset: {key}")
                    f_in.copy(key, f_out)

            # --- Process and pad/truncate human SMPL parameters ---
            print("Processing 'our_pred_humans_smplx_params'...")
            human_params_out = f_out.create_group('our_pred_humans_smplx_params')
            existing_indices = np.arange(existing_frames)

            for human_id in human_ids:
                print(f"  Processing Human ID: {human_id}")
                human_in_group = f_in['our_pred_humans_smplx_params'][human_id]
                human_out_group = human_params_out.create_group(human_id)

                for param_name, param_data_in in human_in_group.items():
                    print(f"    Processing parameter: {param_name}")
                    existing_data = param_data_in[...] # Load data into memory

                    if existing_frames == target_frames:
                        # No padding/truncation needed
                        padded_data = existing_data
                    elif existing_frames > target_frames:
                        # Truncate
                        print(f"      Truncating from {existing_frames} to {target_frames} frames.")
                        padded_data = existing_data[:target_frames]
                    else: # existing_frames < target_frames
                        # Pad using nearest neighbor
                        print(f"      Padding from {existing_frames} to {target_frames} frames.")
                        new_shape = (target_frames,) + existing_data.shape[1:]
                        padded_data = np.empty(new_shape, dtype=existing_data.dtype)
                        # Copy existing data
                        padded_data[:existing_frames] = existing_data
                        # Fill missing frames
                        for frame_idx in range(existing_frames, target_frames):
                            nearest_idx = nearest(frame_idx, existing_indices)
                            padded_data[frame_idx] = existing_data[nearest_idx]

                    # Write padded/truncated data to output file
                    human_out_group.create_dataset(param_name, data=padded_data, dtype=existing_data.dtype)
                    print(f"      New shape: {padded_data.shape}")


            # --- Recreate person_frame_info_list based on target frames ---
            print("Recreating 'person_frame_info_list'...")
            frame_info_out = f_out.create_group('person_frame_info_list')
            for frame_idx in range(target_frames):
                frame_name = f"{frame_idx:05d}"
                frame_group_out = frame_info_out.create_group(frame_name)
                # Add all human IDs to every frame in the padded sequence
                for human_id in human_ids:
                    human_group_out = frame_group_out.create_group(human_id)
                    # You can copy attributes from the original if needed and available,
                    # or just set defaults. Here, setting defaults.
                    human_group_out.attrs['visible'] = True
                    human_group_out.attrs['tracked'] = True
            print(f"  Created frame info for {target_frames} frames.")

    # --- Replace original file with the temporary file ---
    try:
        os.replace(temp_save_path, save_path)
        print(f"\nSuccessfully saved updated HDF5 file to: {save_path}")
    except OSError as e:
        print(f"\nError replacing file: {e}")
        print(f"Temporary file saved at: {temp_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pad or truncate SMPL parameters in an HDF5 file to a target number of frames.')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to the input HDF5 file.')
    parser.add_argument('--target_frames', type=int, required=True,
                        help='The desired total number of frames.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the modified HDF5 file. If None, overwrites the input file.')

    args = parser.parse_args()
    fill_h5_frames(args.h5_path, args.target_frames, args.save_path) 